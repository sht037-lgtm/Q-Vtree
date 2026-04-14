from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from qwen_vl_utils import process_vision_info

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from module import QVTree


# =========================================================
# Constants
# =========================================================

IMAGE_TOKEN_ID = 151655      # vision patch token id in Qwen2.5-VL
VISION_START   = 151652
VISION_END     = 151653
PATCH_SIZE     = 28          # Qwen2.5-VL patch size in pixels
GENERIC_PROMPT = "Write a general description of the image."


# =========================================================
# Attention scoring
# =========================================================

def compute_patch_scores(
    model,
    processor,
    image: Image.Image,
    question: str,
    target_layers: tuple = (20,),
    gaussian_sigma: float = 1.0,
    gaussian_ks: int = 3,
) -> tuple[torch.Tensor, int, int]:
    """
    Compute relative patch attention scores for one image/question pair.

    Pipeline:
      Pass 1 : forward with the specific question          -> A_q
      Pass 2 : forward with a generic description          -> A_g
      scores  = A_q / A_g
      -> normalize [0,1] -> Gaussian smooth -> normalize [0,1]

    Args:
        model, processor : loaded Qwen2.5-VL model and processor
        image            : PIL image
        question         : question string
        target_layers    : transformer layer indices to hook
        gaussian_sigma   : smoothing kernel sigma
        gaussian_ks      : smoothing kernel size

    Returns:
        patch_scores : [grid_h * grid_w] float32 CPU tensor
        grid_h       : patch grid height
        grid_w       : patch grid width
    """
    device = next(model.parameters()).device

    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text": question},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    )
    inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

    input_ids = inputs["input_ids"]
    _, grid_h_raw, grid_w_raw = inputs["image_grid_thw"][0].tolist()
    grid_h, grid_w = grid_h_raw // 2, grid_w_raw // 2

    vis_positions = (input_ids[0] == IMAGE_TOKEN_ID).nonzero(as_tuple=True)[0].cpu()
    img_end   = vis_positions[-1].item()
    seq_len   = input_ids.shape[1]
    text_start, text_end = img_end + 1, seq_len
    text_positions = (
        torch.arange(text_start, text_end, dtype=torch.long)
        if text_end > text_start
        else torch.tensor([seq_len - 1], dtype=torch.long)
    )

    def _get_attn_scores(model_inputs, query_positions):
        lm = model.model.language_model
        lm_layers = getattr(lm, "layers", None) or getattr(lm, "model", lm).layers

        captured = {}
        handles  = []

        def make_hook(layer_id):
            def hook_fn(module, input, output):
                if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                    captured[layer_id] = output[1].detach().cpu()
            return hook_fn

        for lid in target_layers:
            handles.append(lm_layers[lid].self_attn.register_forward_hook(make_hook(lid)))

        try:
            with torch.no_grad():
                model(**model_inputs, output_attentions=True, return_dict=True)
        finally:
            for h in handles:
                h.remove()

        if not captured:
            raise RuntimeError("Hook did not capture attention. "
                               "Ensure attn_implementation='eager'.")

        vp = vis_positions
        qp = query_positions
        scores = []
        for lid in target_layers:
            if lid not in captured:
                continue
            layer_attn = captured[lid]
            attn = layer_attn[0, :, :, vp][:, qp, :]
            scores.append(attn.mean(0).mean(0).float())
        result = torch.stack(scores).mean(0)
        del captured
        torch.cuda.empty_cache()
        return result

    # Pass 1: question-specific
    A_q = _get_attn_scores(inputs, text_positions)

    # Pass 2: generic
    generic_ids = processor.tokenizer(
        GENERIC_PROMPT, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)
    generic_input_ids = torch.cat(
        [input_ids[:, :img_end + 1], generic_ids, input_ids[:, -3:]], dim=1
    )
    generic_inputs = {k: v for k, v in inputs.items()
                      if k not in ("input_ids", "attention_mask", "position_ids", "mm_token_type_ids")}
    generic_inputs["input_ids"] = generic_input_ids
    generic_inputs["attention_mask"] = torch.ones(
        1, generic_input_ids.shape[1], dtype=inputs["attention_mask"].dtype, device=device
    )
    generic_text_positions = torch.arange(
        img_end + 1, img_end + 1 + generic_ids.shape[1], dtype=torch.long
    )

    A_g = _get_attn_scores(generic_inputs, generic_text_positions)
    del generic_inputs
    torch.cuda.empty_cache()

    # relative salience + normalize
    patch_scores = A_q / (A_g + 1e-8)
    s_min, s_max = patch_scores.min(), patch_scores.max()
    patch_scores = (patch_scores - s_min) / (s_max - s_min + 1e-6)

    # Gaussian smoothing
    if patch_scores.shape[0] == grid_h * grid_w:
        ax   = torch.arange(gaussian_ks, dtype=torch.float32) - gaussian_ks // 2
        g1d  = torch.exp(-ax ** 2 / (2 * gaussian_sigma ** 2))
        g1d /= g1d.sum()
        kernel = (g1d.unsqueeze(1) * g1d.unsqueeze(0)).view(1, 1, gaussian_ks, gaussian_ks)
        patch_scores = F.conv2d(
            patch_scores.float().view(1, 1, grid_h, grid_w),
            kernel, padding=gaussian_ks // 2,
        ).view(-1)
        s_min, s_max = patch_scores.min(), patch_scores.max()
        patch_scores = (patch_scores - s_min) / (s_max - s_min + 1e-6)

    return patch_scores, grid_h, grid_w


# =========================================================
# QuadTree selection
# =========================================================

def select_patches(
    patch_scores: torch.Tensor,
    grid_h: int,
    grid_w: int,
    hidden_dim: int,
    split_threshold: float = 0.3,
    softmax_temperature: float = 0.2,
) -> torch.Tensor:
    """
    Run QuadTree builder + navigator to select salient patches.

    Returns:
        patch_ids : [M] sorted unique selected patch indices
    """
    n      = grid_h * grid_w
    qvtree = QVTree(
        D=hidden_dim,
        split_threshold=split_threshold,
        softmax_temperature=softmax_temperature,
    )

    x_dummy = torch.zeros(1, n, hidden_dim)
    ps      = patch_scores.unsqueeze(0).float()

    built  = qvtree.builder.build(x_dummy, grid_h, grid_w)
    nodes  = built["nodes"]
    sel, _ = qvtree.navigator.select_nodes(nodes=nodes, patch_scores=ps, W=grid_w)

    token_out = qvtree.navigator.nodes_to_tokens(
        nodes, H=grid_h, W=grid_w, selected_node_ids=sel, x=x_dummy,
    )
    patch_ids = token_out["selected_token_indices"][0]

    if patch_ids.numel() == 0:
        patch_ids = torch.arange(n)

    return torch.unique(patch_ids.clamp(0, n - 1))


# =========================================================
# LPD (Layout-Preserving Downsample)
# =========================================================

def _patch_ids_to_bboxes(patch_ids: torch.Tensor, grid_w: int, patch_size: int) -> list:
    bboxes = []
    for idx in patch_ids.tolist():
        r, c = int(idx) // grid_w, int(idx) % grid_w
        x0, y0 = c * patch_size, r * patch_size
        bboxes.append((x0, y0, x0 + patch_size, y0 + patch_size))
    return bboxes


def _merge_bboxes(bboxes: list) -> list:
    if not bboxes:
        return []

    def overlaps_or_adjacent(a, b):
        return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])

    merged, changed = list(bboxes), True
    while changed:
        changed = False
        result  = []
        used    = [False] * len(merged)
        for i in range(len(merged)):
            if used[i]:
                continue
            cur = list(merged[i])
            for j in range(i + 1, len(merged)):
                if not used[j] and overlaps_or_adjacent(cur, merged[j]):
                    cur[0] = min(cur[0], merged[j][0])
                    cur[1] = min(cur[1], merged[j][1])
                    cur[2] = max(cur[2], merged[j][2])
                    cur[3] = max(cur[3], merged[j][3])
                    used[j] = True
                    changed = True
            result.append(tuple(cur))
        merged = result
    return merged


def _build_compact_image(image: Image.Image, bboxes: list) -> Image.Image:
    if not bboxes:
        return image

    img_w, img_h = image.size
    x_coords = sorted(set([0, img_w] + [x for b in bboxes for x in (b[0], b[2])]))
    y_coords = sorted(set([0, img_h] + [y for b in bboxes for y in (b[1], b[3])]))

    def cell_has_content(x0, y0, x1, y1):
        return any(x0 < b[2] and x1 > b[0] and y0 < b[3] and y1 > b[1] for b in bboxes)

    new_w, x_map = 0, {}
    for i in range(len(x_coords) - 1):
        if any(cell_has_content(x_coords[i], y_coords[j], x_coords[i+1], y_coords[j+1])
               for j in range(len(y_coords) - 1)):
            x_map[i] = new_w
            new_w += x_coords[i+1] - x_coords[i]

    new_h, y_map = 0, {}
    for j in range(len(y_coords) - 1):
        if any(cell_has_content(x_coords[i], y_coords[j], x_coords[i+1], y_coords[j+1])
               for i in range(len(x_coords) - 1)):
            y_map[j] = new_h
            new_h += y_coords[j+1] - y_coords[j]

    if new_w == 0 or new_h == 0:
        return image

    compact = Image.new("RGB", (new_w, new_h), color=(255, 255, 255))
    for i in range(len(x_coords) - 1):
        for j in range(len(y_coords) - 1):
            if cell_has_content(x_coords[i], y_coords[j], x_coords[i+1], y_coords[j+1]):
                if i not in x_map or j not in y_map:
                    continue
                patch = image.crop((x_coords[i], y_coords[j], x_coords[i+1], y_coords[j+1]))
                compact.paste(patch, (x_map[i], y_map[j]))
    return compact


def run_lpd(
    patch_ids: torch.Tensor,
    image: Image.Image,
    grid_h: int,
    grid_w: int,
) -> tuple[Image.Image, list]:
    """
    Full LPD pipeline.

    Args:
        patch_ids : [M] selected patch indices
        image     : original PIL image (will be resized to grid pixel size)
        grid_h    : patch grid height
        grid_w    : patch grid width

    Returns:
        compact_image : PIL.Image
        merged_bboxes : list of (x0, y0, x1, y1)
    """
    pil_img = image.convert("RGB").resize(
        (grid_w * PATCH_SIZE, grid_h * PATCH_SIZE), Image.BILINEAR
    )
    raw_bboxes    = _patch_ids_to_bboxes(patch_ids.cpu(), grid_w, PATCH_SIZE)
    merged_bboxes = _merge_bboxes(raw_bboxes)
    compact_image = _build_compact_image(pil_img, merged_bboxes)
    return compact_image, merged_bboxes


# =========================================================
# Inference
# =========================================================

def run_baseline_inference(
    model,
    processor,
    image: Image.Image,
    question: str,
    max_new_tokens: int = 16,
) -> str:
    """Standard Qwen2.5-VL inference without any token selection."""
    device = next(model.parameters()).device
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text": question},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    )
    inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

    with torch.inference_mode():
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    return processor.batch_decode(
        out_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )[0].strip()


def run_tree_inference(
    model,
    processor,
    image: Image.Image,
    question: str,
    split_threshold: float = 0.3,
    softmax_temperature: float = 0.2,
    max_new_tokens: int = 16,
) -> tuple[str, torch.Tensor, Image.Image, list, int, int, float]:
    """
    Full Q-VTree inference pipeline for Qwen2.5-VL.

    Returns:
        pred_text     : decoded prediction string
        patch_scores  : [grid_h * grid_w] attention scores
        compact_image : PIL.Image after LPD
        merged_bboxes : list of merged bboxes
        n_selected    : number of selected tokens
        n_total       : total number of tokens
        select_ratio  : n_selected / n_total
    """
    device = next(model.parameters()).device
    hidden_dim = model.config.text_config.hidden_size

    # Step 1: compute patch scores
    patch_scores, grid_h, grid_w = compute_patch_scores(
        model, processor, image, question
    )

    # Step 2: QuadTree selection
    patch_ids = select_patches(
        patch_scores, grid_h, grid_w, hidden_dim,
        split_threshold=split_threshold,
        softmax_temperature=softmax_temperature,
    )
    n_selected = int(patch_ids.numel())
    n_total    = grid_h * grid_w
    select_ratio = n_selected / n_total if n_total > 0 else 0.0
    print(f"[TREE] selected tokens: {n_selected}, original: {n_total}")

    # Step 3: LPD → compact image
    compact_image, merged_bboxes = run_lpd(patch_ids, image, grid_h, grid_w)

    # Step 4: inference with compact image
    messages = [{"role": "user", "content": [
        {"type": "image", "image": compact_image},
        {"type": "text",  "text": question},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    )
    inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

    with torch.inference_mode():
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    pred_text = processor.batch_decode(
        out_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )[0].strip()

    return pred_text, patch_scores, compact_image, merged_bboxes, n_selected, n_total, select_ratio