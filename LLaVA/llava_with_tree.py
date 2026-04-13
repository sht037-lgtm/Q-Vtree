"""
llava_with_tree.py

Q-VTree core logic for LLaVA-1.5-7b.

Pipeline:
  Pass 1 : pad_resize(original) + clean question  -> A_q
  Pass 2 : pad_resize(original) + generic prompt  -> A_g
  patch_scores = A_q / A_g -> normalize -> Gaussian smooth
  QuadTree -> patch_ids (in 336x336 space)
  patch_ids mapped back to original image coordinates via PadResizeMeta
  LPD crops selected regions from original high-res image -> compact image
  Pass 3 : compact image -> final inference
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from module import QVTree


# =========================================================
# Constants
# =========================================================

IMAGE_TOKEN_ID = 32000
GRID_SIZE      = 24
PATCH_SIZE     = 14
HIDDEN_DIM     = 4096
CLIP_SIZE      = 336
GENERIC_PROMPT = "Write a general description of the image."


# =========================================================
# Image preprocessing
# =========================================================

@dataclass
class PadResizeMeta:
    """
    Records scale and padding from pad_resize_with_meta.
    Used to map 336x336 patch coords back to original image space:
        orig_coord = (padded_coord - pad_offset) / scale
    """
    scale  : float
    pad_x  : int
    pad_y  : int
    orig_w : int
    orig_h : int


def _do_pad_resize(image: Image.Image, target: int = CLIP_SIZE):
    """
    Core pad-resize logic shared by pad_resize and pad_resize_with_meta.
    Returns (padded_image, new_w, new_h, pad_x, pad_y).
    """
    img = image.copy().convert("RGB")
    img.thumbnail((target, target), Image.BILINEAR)
    new_w, new_h = img.size
    pad_x = (target - new_w) // 2
    pad_y = (target - new_h) // 2
    out   = Image.new("RGB", (target, target), (0, 0, 0))
    out.paste(img, (pad_x, pad_y))
    return out, new_w, new_h, pad_x, pad_y


def pad_resize(image: Image.Image, target: int = CLIP_SIZE) -> Image.Image:
    """Resize to target x target preserving aspect ratio, pad with black."""
    out, *_ = _do_pad_resize(image, target)
    return out


def pad_resize_with_meta(image: Image.Image, target: int = CLIP_SIZE):
    """
    Same as pad_resize but also returns PadResizeMeta for coordinate mapping.

    Returns:
        padded : target x target PIL image
        meta   : PadResizeMeta
    """
    orig_w, orig_h = image.size
    out, new_w, _, pad_x, pad_y = _do_pad_resize(image, target)
    return out, PadResizeMeta(
        scale=new_w / orig_w,
        pad_x=pad_x, pad_y=pad_y,
        orig_w=orig_w, orig_h=orig_h,
    )


# =========================================================
# Coordinate mapping
# =========================================================

def patch_ids_to_bboxes(
    patch_ids: torch.Tensor,
    meta: PadResizeMeta = None,
) -> list:
    """
    Convert patch indices to pixel bounding boxes.

    If meta is provided, maps from 336x336 pad-resized space back to
    original image coordinates (clamps to image bounds, skips padding patches).
    If meta is None, returns boxes in 336x336 space.

    Returns list of (x0, y0, x1, y1).
    """
    bboxes = []
    for idx in patch_ids.tolist():
        r, c  = int(idx) // GRID_SIZE, int(idx) % GRID_SIZE
        x0 = c * PATCH_SIZE
        y0 = r * PATCH_SIZE
        x1 = x0 + PATCH_SIZE
        y1 = y0 + PATCH_SIZE

        if meta is not None:
            x0 = max(0,            int((x0 - meta.pad_x) / meta.scale))
            y0 = max(0,            int((y0 - meta.pad_y) / meta.scale))
            x1 = min(meta.orig_w,  int((x1 - meta.pad_x) / meta.scale))
            y1 = min(meta.orig_h,  int((y1 - meta.pad_y) / meta.scale))
            if x1 <= x0 or y1 <= y0:
                continue  # entirely in padding area

        bboxes.append((x0, y0, x1, y1))
    return bboxes


# keep old name as alias for visualization callers
def patch_ids_to_orig_bboxes(patch_ids: torch.Tensor, meta: PadResizeMeta) -> list:
    return patch_ids_to_bboxes(patch_ids, meta)


# =========================================================
# Helpers
# =========================================================

def _clean_question(question: str) -> str:
    """Strip MCQ options, return only the core question text."""
    return re.split(r'\n\(A\)', question)[0].strip()


def recover_clip_image(processor, image: Image.Image, question: str) -> Image.Image:
    """Return the 336x336 pad-resized image sent to CLIP (for visualization)."""
    return pad_resize(image)


# =========================================================
# Model loading
# =========================================================

def load_model(model_id: str, device: str = "auto"):
    from transformers import LlavaForConditionalGeneration, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device,
        attn_implementation="eager",
    )
    model.eval()
    return model, processor


# =========================================================
# Attention scoring
# =========================================================

def _extract_attn_scores(outputs, text_pos, image_pos, layers) -> torch.Tensor:
    """Mean attention from text_pos -> image_pos, averaged over heads and layers."""
    scores = []
    for l in layers:
        attn = outputs.attentions[l]
        a    = attn[0, :, :, image_pos.to(attn.device)][:, text_pos.to(attn.device), :]
        scores.append(a.mean(0).mean(0).cpu().float())
    return torch.stack(scores).mean(0)


def compute_patch_scores(
    model,
    processor,
    image: Image.Image,
    question: str,
    target_layers: tuple = (14, 15, 16, 17),
    gaussian_sigma: float = 1.0,
    gaussian_ks: int = 3,
) -> torch.Tensor:
    """
    Compute relative patch attention scores.
    question should be clean_q (no MCQ options).
    Returns [576] normalized float32 tensor.
    """
    device  = next(model.parameters()).device
    padded  = pad_resize(image)

    # ---- Pass 1: clean question ----
    prompt_q = "USER: <image>\n{}\nASSISTANT:".format(_clean_question(question))
    inputs_q = processor(text=prompt_q, images=padded, return_tensors="pt").to(device)

    full_ids        = inputs_q["input_ids"]
    seq_len         = full_ids.shape[1]
    image_positions = (full_ids[0] == IMAGE_TOKEN_ID).nonzero(as_tuple=True)[0].cpu()
    img_end         = int(image_positions[-1])
    text_positions  = torch.arange(img_end + 1, seq_len, dtype=torch.long)
    pixel_values    = inputs_q["pixel_values"]

    with torch.inference_mode():
        out_q = model(input_ids=full_ids, pixel_values=pixel_values,
                      attention_mask=torch.ones_like(full_ids), output_attentions=True)
    A_q = _extract_attn_scores(out_q, text_positions, image_positions, target_layers)
    del out_q
    torch.cuda.empty_cache()

    # ---- Pass 2: generic description ----
    generic_ids   = processor.tokenizer(
        GENERIC_PROMPT, return_tensors="pt", add_special_tokens=False,
    ).input_ids.to(device)
    gen_input_ids = torch.cat(
        [full_ids[:, :img_end + 1], generic_ids, full_ids[:, seq_len - 6:]], dim=1
    )
    gen_seq_len  = gen_input_ids.shape[1]
    gen_text_pos = torch.arange(img_end + 1, gen_seq_len, dtype=torch.long)

    with torch.inference_mode():
        out_g = model(input_ids=gen_input_ids, pixel_values=pixel_values,
                      attention_mask=torch.ones_like(gen_input_ids), output_attentions=True)
    A_g = _extract_attn_scores(out_g, gen_text_pos, image_positions, target_layers)
    del out_g
    torch.cuda.empty_cache()

    # ---- relative salience -> normalize -> Gaussian smooth -> normalize ----
    patch_scores = A_q / (A_g + 1e-8)

    def _normalize(t):
        lo, hi = t.min(), t.max()
        return (t - lo) / (hi - lo + 1e-6)

    patch_scores = _normalize(patch_scores)

    n      = GRID_SIZE * GRID_SIZE
    ax     = torch.arange(gaussian_ks, dtype=torch.float32) - gaussian_ks // 2
    g1d    = torch.exp(-ax ** 2 / (2 * gaussian_sigma ** 2)); g1d /= g1d.sum()
    kernel = (g1d.unsqueeze(1) * g1d.unsqueeze(0)).view(1, 1, gaussian_ks, gaussian_ks)
    patch_scores = F.conv2d(
        patch_scores[:n].view(1, 1, GRID_SIZE, GRID_SIZE),
        kernel, padding=gaussian_ks // 2,
    ).view(-1)

    return _normalize(patch_scores)


# =========================================================
# QuadTree selection
# =========================================================

def select_patches(
    patch_scores: torch.Tensor,
    split_threshold: float = 0.3,
    softmax_temperature: float = 0.2,
) -> torch.Tensor:
    n      = GRID_SIZE * GRID_SIZE
    qvtree = QVTree(D=HIDDEN_DIM, split_threshold=split_threshold,
                    softmax_temperature=softmax_temperature)
    x_dummy = torch.zeros(1, n, HIDDEN_DIM)
    built   = qvtree.builder.build(x_dummy, GRID_SIZE, GRID_SIZE)
    sel, _  = qvtree.navigator.select_nodes(
        nodes=built["nodes"], patch_scores=patch_scores.unsqueeze(0).float(), W=GRID_SIZE
    )
    token_out = qvtree.navigator.nodes_to_tokens(
        built["nodes"], H=GRID_SIZE, W=GRID_SIZE, selected_node_ids=sel, x=x_dummy,
    )
    patch_ids = token_out["selected_token_indices"][0]
    if patch_ids.numel() == 0:
        patch_ids = torch.arange(n)
    return torch.unique(patch_ids.clamp(0, n - 1))


# =========================================================
# LPD  (Layout-Preserving Downsample)
# =========================================================

def _merge_bboxes(bboxes: list) -> list:
    if not bboxes:
        return []

    def overlaps_or_adjacent(a, b):
        return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])

    merged, changed = list(bboxes), True
    while changed:
        changed = False
        result, used = [], [False] * len(merged)
        for i in range(len(merged)):
            if used[i]:
                continue
            cur = list(merged[i])
            for j in range(i + 1, len(merged)):
                if not used[j] and overlaps_or_adjacent(cur, merged[j]):
                    cur[:2] = [min(cur[0], merged[j][0]), min(cur[1], merged[j][1])]
                    cur[2:] = [max(cur[2], merged[j][2]), max(cur[3], merged[j][3])]
                    used[j] = True
                    changed = True
            result.append(tuple(cur))
        merged = result
    return merged


def _build_compact_image(image: Image.Image, bboxes: list) -> Image.Image:
    """Recompose selected regions preserving relative spatial layout."""
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
            x_map[i] = new_w
            new_w   += x_coords[i+1] - x_coords[i]

    new_h, y_map = 0, {}
    for j in range(len(y_coords) - 1):
        if any(has_content(x_coords[i], y_coords[j], x_coords[i+1], y_coords[j+1])
               for i in range(len(x_coords) - 1)):
            y_map[j] = new_h
            new_h   += y_coords[j+1] - y_coords[j]

    if new_w == 0 or new_h == 0:
        return image

    compact = Image.new("RGB", (new_w, new_h), color=(255, 255, 255))
    for i in range(len(x_coords) - 1):
        for j in range(len(y_coords) - 1):
            if has_content(x_coords[i], y_coords[j], x_coords[i+1], y_coords[j+1]):
                if i not in x_map or j not in y_map:
                    continue
                compact.paste(
                    image.crop((x_coords[i], y_coords[j], x_coords[i+1], y_coords[j+1])),
                    (x_map[i], y_map[j]),
                )
    return compact


def run_lpd_on_original(patch_ids: torch.Tensor, original_image: Image.Image,
                        meta: PadResizeMeta):
    """
    Main LPD function: crops from original high-res image.
    patch_ids are in 336x336 space and are mapped back via meta.
    """
    raw      = patch_ids_to_bboxes(patch_ids.cpu(), meta)
    merged   = _merge_bboxes(raw)
    compact  = _build_compact_image(original_image, merged)
    return compact, merged


def run_lpd(patch_ids: torch.Tensor, clip_image: Image.Image):
    """LPD on 336x336 pad-resized image (for visualization only)."""
    raw    = patch_ids_to_bboxes(patch_ids.cpu())   # no meta -> 336x336 space
    merged = _merge_bboxes(raw)
    return _build_compact_image(clip_image, merged), merged


# =========================================================
# Inference
# =========================================================

def _generate(model, processor, image: Image.Image, prompt: str,
              max_new_tokens: int) -> str:
    """Shared generate helper."""
    device  = next(model.parameters()).device
    inputs  = processor(text=prompt, images=image.convert("RGB"),
                        return_tensors="pt").to(device)
    with torch.inference_mode():
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.batch_decode(
        out_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )[0].strip()


def run_baseline_inference(model, processor, image: Image.Image, question: str,
                           max_new_tokens: int = 32) -> str:
    """Baseline: pad-resize original image, standard inference."""
    return _generate(model, processor, pad_resize(image),
                     "USER: <image>\n{}\nASSISTANT:".format(question), max_new_tokens)


def run_tree_inference(model, processor, compact_image: Image.Image, question: str,
                       max_new_tokens: int = 32) -> str:
    """Tree second-pass: compact image is already high-res, no extra pad_resize."""
    return _generate(model, processor, compact_image,
                     "USER: <image>\n{}\nASSISTANT:".format(question), max_new_tokens)