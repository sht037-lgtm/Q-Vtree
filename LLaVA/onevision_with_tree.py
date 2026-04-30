"""
onevision_with_tree.py
Q-VTree core logic for LLaVA-OneVision-Qwen2-7B.
Official HuggingFace usage: processor(images=..., text=...).to("cuda:0", float16)
"""
from __future__ import annotations
import os, re, sys
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn.functional as F
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in [_HERE, _ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from module import QVTree

PATCH_SIZE     = 14
HIDDEN_DIM     = 3584
GENERIC_PROMPT = "Write a general description of the image."
_OV_IMAGE_TOKEN_FALLBACK = 151646


# =========================================================
# Config helpers
# =========================================================

def get_score_size(processor) -> int:
    ip = getattr(processor, "image_processor", processor)
    val = getattr(ip, "size", None)
    if val is not None:
        for key in ("height", "width", "shortest_edge"):
            v = getattr(val, key, None)
            if v is not None:
                return int(v)
    return 384


def get_grid_size(processor) -> int:
    return get_score_size(processor) // PATCH_SIZE


def _get_patch_size(model_or_processor):
    cfg = getattr(model_or_processor, "config", None)
    if cfg is not None:
        vc = getattr(cfg, "vision_config", None)
        if vc is not None:
            return int(getattr(vc, "patch_size", 14))
    ip = getattr(model_or_processor, "image_processor", model_or_processor)
    return int(getattr(ip, "patch_size", 14))


def _get_image_token_id(processor) -> int:
    tid = getattr(processor, "image_token_id", None)
    if tid is not None:
        return int(tid)
    tok = getattr(processor, "tokenizer", processor)
    for name in ("<image>", "<|image_pad|>", "<img>"):
        tid = tok.convert_tokens_to_ids(name)
        if tid is not None and tid != getattr(tok, "unk_token_id", None):
            return int(tid)
    return _OV_IMAGE_TOKEN_FALLBACK


# =========================================================
# Image preprocessing
# =========================================================

@dataclass
class PadResizeMeta:
    scale     : float
    pad_x     : int
    pad_y     : int
    orig_w    : int
    orig_h    : int
    score_size: int


def _do_pad_resize(image, target):
    img = image.copy().convert("RGB")
    img.thumbnail((target, target), Image.BILINEAR)
    new_w, new_h = img.size
    pad_x = (target - new_w) // 2
    pad_y = (target - new_h) // 2
    out = Image.new("RGB", (target, target), (0, 0, 0))
    out.paste(img, (pad_x, pad_y))
    return out, new_w, new_h, pad_x, pad_y


def pad_resize(image, target):
    out, *_ = _do_pad_resize(image, target)
    return out


def pad_resize_with_meta(image, target):
    orig_w, orig_h = image.size
    out, new_w, _, pad_x, pad_y = _do_pad_resize(image, target)
    return out, PadResizeMeta(
        scale=new_w / orig_w, pad_x=pad_x, pad_y=pad_y,
        orig_w=orig_w, orig_h=orig_h, score_size=target,
    )


# =========================================================
# Coordinate mapping
# =========================================================

def patch_ids_to_bboxes(patch_ids, grid_size, meta=None, patch_size=None):
    ps = patch_size or PATCH_SIZE
    bboxes = []
    for idx in patch_ids.tolist():
        r, c = int(idx) // grid_size, int(idx) % grid_size
        x0, y0 = c * ps, r * ps
        x1, y1 = x0 + ps, y0 + ps
        if meta is not None:
            x0 = max(0,           int((x0 - meta.pad_x) / meta.scale))
            y0 = max(0,           int((y0 - meta.pad_y) / meta.scale))
            x1 = min(meta.orig_w, int((x1 - meta.pad_x) / meta.scale))
            y1 = min(meta.orig_h, int((y1 - meta.pad_y) / meta.scale))
            if x1 <= x0 or y1 <= y0:
                continue
        bboxes.append((x0, y0, x1, y1))
    return bboxes


def _clean_question(question):
    return re.split(r'\n\(A\)', question)[0].strip()


# =========================================================
# Model loading
# =========================================================

def load_model(model_id, device="auto"):
    from transformers import LlavaOnevisionForConditionalGeneration, AutoProcessor
    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to("cuda:0")
    model.eval()
    return model, processor


# =========================================================
# Attention scoring
# =========================================================

def _extract_attn_scores(outputs, text_pos, image_pos, layers):
    scores = []
    for l in layers:
        attn = outputs.attentions[l]
        a = attn[0, :, :, image_pos.to(attn.device)][:, text_pos.to(attn.device), :]
        scores.append(a.mean(0).mean(0).cpu().float())
    return torch.stack(scores).mean(0)


def _make_score_inputs(processor, image, text, score_size):
    """Build inputs for scoring pass — single tile only.
    Force exact score_size x score_size so processor picks exactly 1 tile.
    """
    # Force exact square so no tiling occurs
    canvas = Image.new("RGB", (score_size, score_size), (0, 0, 0))
    img = image.copy().convert("RGB")
    img.thumbnail((score_size, score_size), Image.BILINEAR)
    canvas.paste(img, ((score_size - img.width) // 2, (score_size - img.height) // 2))

    conversation = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": text},
    ]}]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(
        images=canvas,
        text=prompt,
        return_tensors="pt",
    ).to("cuda:0", torch.float16)
    return inputs


def compute_patch_scores(
    model, processor, image, question,
    target_layers=(14, 15, 16, 17),
    gaussian_sigma=1.0, gaussian_ks=3,
):
    score_size     = get_score_size(processor)
    grid_size      = score_size // PATCH_SIZE
    n_patches      = grid_size * grid_size
    image_token_id = _get_image_token_id(processor)

    padded   = pad_resize(image, score_size)
    clean_q  = _clean_question(question)

    # Pass 1: question-specific
    inputs_q    = _make_score_inputs(processor, padded, clean_q, score_size)
    full_ids    = inputs_q["input_ids"]
    seq_len     = full_ids.shape[1]
    # Find image pad token — it's the most frequent token with id > 150000
    ids_list = full_ids[0].tolist()
    from collections import Counter
    high_tokens = Counter(t for t in ids_list if t > 150000)
    image_token_id_actual = high_tokens.most_common(1)[0][0] if high_tokens else image_token_id
    image_pos   = (full_ids[0] == image_token_id_actual).nonzero(as_tuple=True)[0].cpu()
    img_end     = int(image_pos[-1])
    text_pos    = torch.arange(img_end + 1, seq_len, dtype=torch.long)
    pixel_vals  = inputs_q["pixel_values"]

    fwd_kw = dict(pixel_values=pixel_vals,
                  attention_mask=torch.ones_like(full_ids),
                  output_attentions=True)
    if "image_sizes" in inputs_q:
        fwd_kw["image_sizes"] = inputs_q["image_sizes"]

    # Temporarily switch to eager attention for output_attentions support
    orig_attn = model.config._attn_implementation if hasattr(model.config, '_attn_implementation') else None
    model.config._attn_implementation = "eager"
    with torch.inference_mode():
        out_q = model(input_ids=full_ids, **fwd_kw)
    A_q = _extract_attn_scores(out_q, text_pos, image_pos, target_layers)
    del out_q; torch.cuda.empty_cache()

    # Pass 2: generic
    inputs_g     = _make_score_inputs(processor, padded, GENERIC_PROMPT, score_size)
    gen_ids      = inputs_g["input_ids"]
    gen_text_pos = torch.arange(img_end + 1, gen_ids.shape[1], dtype=torch.long)

    fwd_g = dict(pixel_values=pixel_vals,
                 attention_mask=torch.ones_like(gen_ids),
                 output_attentions=True)
    if "image_sizes" in inputs_q:
        fwd_g["image_sizes"] = inputs_q["image_sizes"]

    with torch.inference_mode():
        out_g = model(input_ids=gen_ids, **fwd_g)
    A_g = _extract_attn_scores(out_g, gen_text_pos, image_pos, target_layers)
    del out_g; torch.cuda.empty_cache()
    # Restore original attention implementation
    if orig_attn is not None:
        model.config._attn_implementation = orig_attn

    patch_scores = A_q / (A_g + 1e-8)

    def _norm(t):
        lo, hi = t.min(), t.max()
        return (t - lo) / (hi - lo + 1e-6)

    patch_scores = _norm(patch_scores)
    ax     = torch.arange(gaussian_ks, dtype=torch.float32) - gaussian_ks // 2
    g1d    = torch.exp(-ax ** 2 / (2 * gaussian_sigma ** 2)); g1d /= g1d.sum()
    kernel = (g1d.unsqueeze(1) * g1d.unsqueeze(0)).view(1, 1, gaussian_ks, gaussian_ks)
    patch_scores = F.conv2d(
        patch_scores[:n_patches].view(1, 1, grid_size, grid_size),
        kernel, padding=gaussian_ks // 2,
    ).view(-1)
    return _norm(patch_scores), grid_size, score_size


# =========================================================
# QuadTree selection
# =========================================================

def select_patches(patch_scores, grid_size, split_threshold=0.3, softmax_temperature=0.2):
    n       = grid_size * grid_size
    qvtree  = QVTree(D=HIDDEN_DIM, split_threshold=split_threshold,
                     softmax_temperature=softmax_temperature)
    x_dummy = torch.zeros(1, n, HIDDEN_DIM)
    built   = qvtree.builder.build(x_dummy, grid_size, grid_size)
    sel, _  = qvtree.navigator.select_nodes(
        nodes=built["nodes"],
        patch_scores=patch_scores.unsqueeze(0).float(),
        W=grid_size,
    )
    token_out = qvtree.navigator.nodes_to_tokens(
        built["nodes"], H=grid_size, W=grid_size,
        selected_node_ids=sel, x=x_dummy,
    )
    patch_ids = token_out["selected_token_indices"][0]
    if patch_ids.numel() == 0:
        patch_ids = torch.arange(n)
    return torch.unique(patch_ids.clamp(0, n - 1))


# =========================================================
# LPD
# =========================================================

def _merge_bboxes(bboxes):
    if not bboxes:
        return []
    def overlaps_or_adjacent(a, b):
        return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])
    merged, changed = list(bboxes), True
    while changed:
        changed = False
        result, used = [], [False] * len(merged)
        for i in range(len(merged)):
            if used[i]: continue
            cur = list(merged[i])
            for j in range(i + 1, len(merged)):
                if not used[j] and overlaps_or_adjacent(cur, merged[j]):
                    cur[:2] = [min(cur[0], merged[j][0]), min(cur[1], merged[j][1])]
                    cur[2:] = [max(cur[2], merged[j][2]), max(cur[3], merged[j][3])]
                    used[j] = True; changed = True
            result.append(tuple(cur))
        merged = result
    return merged


def _build_compact_image(image, bboxes):
    if not bboxes:
        return image
    img_w, img_h = image.size
    x_coords = sorted(set([0, img_w] + [x for b in bboxes for x in (b[0], b[2])]))
    y_coords = sorted(set([0, img_h] + [y for b in bboxes for y in (b[1], b[3])]))
    def has_content(x0, y0, x1, y1):
        return any(x0 < b[2] and x1 > b[0] and y0 < b[3] and y1 > b[1] for b in bboxes)
    new_w, x_map = 0, {}
    for i in range(len(x_coords) - 1):
        if any(has_content(x_coords[i], y_coords[j], x_coords[i+1], y_coords[j+1])
               for j in range(len(y_coords) - 1)):
            x_map[i] = new_w; new_w += x_coords[i+1] - x_coords[i]
    new_h, y_map = 0, {}
    for j in range(len(y_coords) - 1):
        if any(has_content(x_coords[i], y_coords[j], x_coords[i+1], y_coords[j+1])
               for i in range(len(x_coords) - 1)):
            y_map[j] = new_h; new_h += y_coords[j+1] - y_coords[j]
    if new_w == 0 or new_h == 0:
        return image
    compact = Image.new("RGB", (new_w, new_h), color=(255, 255, 255))
    for i in range(len(x_coords) - 1):
        for j in range(len(y_coords) - 1):
            if has_content(x_coords[i], y_coords[j], x_coords[i+1], y_coords[j+1]):
                if i not in x_map or j not in y_map: continue
                compact.paste(
                    image.crop((x_coords[i], y_coords[j], x_coords[i+1], y_coords[j+1])),
                    (x_map[i], y_map[j]),
                )
    return compact


def run_lpd_on_original(patch_ids, original_image, meta, grid_size, patch_size=None):
    raw    = patch_ids_to_bboxes(patch_ids.cpu(), grid_size, meta, patch_size)
    merged = _merge_bboxes(raw)
    return _build_compact_image(original_image, merged), merged


# =========================================================
# Inference — official HuggingFace usage
# Key: processor(images=..., text=...).to("cuda:0", torch.float16)
# =========================================================

def _ov_generate(model, processor, images, question, max_new_tokens):
    content_list = []
    for _ in images:
        content_list.append({"type": "image"})
    content_list.append({"type": "text", "text": question})
    conversation = [{"role": "user", "content": content_list}]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    # Limit each image to max 6 sub-tiles + 1 global = 7 tiles
    # by capping the longest edge so processor picks a valid pinpoint <= 6 sub-tiles
    tile_size = 384
    max_long_edge = tile_size * 6  # 2304 — fits in a 1x6 or 2x3 grid = 6 sub-tiles
    capped = []
    for img in images:
        w, h = img.size
        if max(w, h) > max_long_edge:
            scale = max_long_edge / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
        capped.append(img)

    img_input = capped if len(capped) > 1 else capped[0]
    inputs = processor(images=img_input, text=prompt, return_tensors="pt").to(0, torch.float16)
    eos_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=eos_id)
    return processor.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def run_baseline_inference(model, processor, image, question, max_new_tokens=32):
    """Baseline: original image with native OneVision anyres, max 7 tiles (6 sub + 1 global)."""
    return _ov_generate(model, processor, [image.convert("RGB")], question, max_new_tokens)


def run_tree_inference(model, processor, compact_image, question,
                       original_image=None, max_new_tokens=32):
    """local only or dual mode."""
    if original_image is None:
        images = [compact_image.convert("RGB")]
    else:
        images = [original_image.convert("RGB"), compact_image.convert("RGB")]
    return _ov_generate(model, processor, images, question, max_new_tokens)