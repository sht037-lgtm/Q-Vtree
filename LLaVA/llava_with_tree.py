"""
llava_with_tree.py

Core logic for Q-VTree on LLaVA-1.5-7b.
  - Attention scoring  : relative multi-layer (aligned with qwen/model.py)
  - QuadTree selection : via module.QVTree
  - LPD                : Layout-Preserving Downsample (aligned with qwen/model.py)
  - Inference          : baseline and tree second-pass

The notebook only calls the public functions defined here.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F
from PIL import Image

# Make project root importable so `module.py` can be found
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from module import QVTree


# =========================================================
# Constants
# =========================================================

IMAGE_TOKEN_ID = 32000       # <image> special token id in LLaVA-1.5
GRID_SIZE      = 24          # 24 x 24 = 576 patches  (336 / 14)
PATCH_SIZE     = 14          # CLIP ViT-L/14 patch size in pixels
HIDDEN_DIM     = 4096        # LLaVA-1.5-7b language model hidden dim
GENERIC_PROMPT = "Write a general description of the image."


# =========================================================
# Model loading
# =========================================================

def load_model(model_id: str, device: str = "auto"):
    """
    Load LLaVA-1.5-7b in fp16 without quantization.

    Args:
        model_id : HF hub id or local checkpoint path
        device   : passed to device_map

    Returns:
        model, processor
    """
    from transformers import LlavaForConditionalGeneration, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device,
        attn_implementation="eager",   # required to get attention weights
    )
    model.eval()
    return model, processor


# =========================================================
# Attention scoring  (aligned with qwen/model.py)
# =========================================================

def _extract_attn_scores(
    outputs,
    text_positions: torch.Tensor,
    image_positions: torch.Tensor,
    target_layers: tuple,
) -> torch.Tensor:
    """
    Mean attention from text_positions -> image_positions,
    averaged over heads and target_layers.

    Returns:
        [N_img] float32 on CPU
    """
    scores = []
    for l in target_layers:
        attn = outputs.attentions[l]               # [1, H, seq, seq]
        vp   = image_positions.to(attn.device)
        qp   = text_positions.to(attn.device)
        a    = attn[0, :, :, vp][:, qp, :]        # [H, Q, N_img]
        scores.append(a.mean(0).mean(0).cpu().float())
    return torch.stack(scores).mean(0)             # [N_img]


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
    Compute relative patch attention scores for one image/question pair.

    Pipeline (aligned with qwen/model.py):
      Pass 1 : forward with the specific question          -> A_q
      Pass 2 : forward with a generic description          -> A_g
      scores  = A_q / A_g   (relative salience)
      -> normalize [0,1] -> Gaussian smooth -> normalize [0,1]

    Args:
        model, processor : loaded LLaVA model and processor
        image            : original PIL image
        question         : question string
        target_layers    : transformer layer indices to average over
        gaussian_sigma   : smoothing kernel sigma
        gaussian_ks      : smoothing kernel size

    Returns:
        patch_scores : [GRID_SIZE * GRID_SIZE] float32 CPU tensor
    """
    device = next(model.parameters()).device

    # ---- Pass 1: specific question ----
    prompt_q  = f"USER: <image>\n{question}\nASSISTANT:"
    inputs_q  = processor(
        text=prompt_q,
        images=image.convert("RGB"),
        return_tensors="pt",
    ).to(device)

    full_ids        = inputs_q["input_ids"]
    seq_len         = full_ids.shape[1]
    is_image        = (full_ids[0] == IMAGE_TOKEN_ID)
    image_positions = is_image.nonzero(as_tuple=True)[0].cpu()
    img_end         = int(image_positions[-1])
    text_positions  = torch.arange(img_end + 1, seq_len, dtype=torch.long)

    with torch.inference_mode():
        out_q = model(
            input_ids=full_ids,
            pixel_values=inputs_q["pixel_values"],
            attention_mask=torch.ones_like(full_ids),
            output_attentions=True,
        )
    A_q = _extract_attn_scores(out_q, text_positions, image_positions, target_layers)
    del out_q
    torch.cuda.empty_cache()

    # ---- Pass 2: generic description ----
    prefix_ids    = full_ids[:, :img_end + 1]
    assistant_ids = full_ids[:, seq_len - 6:]
    generic_ids   = processor.tokenizer(
        GENERIC_PROMPT,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(device)

    gen_input_ids = torch.cat([prefix_ids, generic_ids, assistant_ids], dim=1)
    gen_seq_len   = gen_input_ids.shape[1]
    gen_text_pos  = torch.arange(img_end + 1, gen_seq_len, dtype=torch.long)

    with torch.inference_mode():
        out_g = model(
            input_ids=gen_input_ids,
            pixel_values=inputs_q["pixel_values"],
            attention_mask=torch.ones_like(gen_input_ids),
            output_attentions=True,
        )
    A_g = _extract_attn_scores(out_g, gen_text_pos, image_positions, target_layers)
    del out_g
    torch.cuda.empty_cache()

    # ---- relative salience + normalize ----
    patch_scores = A_q / (A_g + 1e-8)
    s_min, s_max = patch_scores.min(), patch_scores.max()
    patch_scores = (patch_scores - s_min) / (s_max - s_min + 1e-6)

    # ---- Gaussian smoothing ----
    n    = GRID_SIZE * GRID_SIZE
    ax   = torch.arange(gaussian_ks, dtype=torch.float32) - gaussian_ks // 2
    g1d  = torch.exp(-ax ** 2 / (2 * gaussian_sigma ** 2))
    g1d /= g1d.sum()
    kernel = (g1d.unsqueeze(1) * g1d.unsqueeze(0)).view(1, 1, gaussian_ks, gaussian_ks)
    patch_scores = F.conv2d(
        patch_scores[:n].view(1, 1, GRID_SIZE, GRID_SIZE),
        kernel,
        padding=gaussian_ks // 2,
    ).view(-1)

    s_min, s_max = patch_scores.min(), patch_scores.max()
    patch_scores = (patch_scores - s_min) / (s_max - s_min + 1e-6)

    return patch_scores  # [576]


# =========================================================
# QuadTree selection
# =========================================================

def select_patches(
    patch_scores: torch.Tensor,
    split_threshold: float = 0.3,
    softmax_temperature: float = 0.2,
) -> torch.Tensor:
    """
    Run QuadTree builder + navigator to select salient patches.

    Args:
        patch_scores        : [GRID_SIZE * GRID_SIZE] CPU tensor
        split_threshold     : QVTree split_threshold
        softmax_temperature : QVTree softmax_temperature

    Returns:
        patch_ids : [M] sorted unique selected patch indices
    """
    n      = GRID_SIZE * GRID_SIZE
    qvtree = QVTree(
        D=HIDDEN_DIM,
        split_threshold=split_threshold,
        softmax_temperature=softmax_temperature,
    )

    x_dummy = torch.zeros(1, n, HIDDEN_DIM)
    ps      = patch_scores.unsqueeze(0).float()

    built  = qvtree.builder.build(x_dummy, GRID_SIZE, GRID_SIZE)
    nodes  = built["nodes"]
    sel, _ = qvtree.navigator.select_nodes(nodes=nodes, patch_scores=ps, W=GRID_SIZE)

    token_out = qvtree.navigator.nodes_to_tokens(
        nodes,
        H=GRID_SIZE,
        W=GRID_SIZE,
        selected_node_ids=sel,
        x=x_dummy,
    )
    patch_ids = token_out["selected_token_indices"][0]

    if patch_ids.numel() == 0:
        patch_ids = torch.arange(n)

    return torch.unique(patch_ids.clamp(0, n - 1))


# =========================================================
# LPD  (Layout-Preserving Downsample) -- aligned with qwen/model.py
# =========================================================

def _patch_ids_to_bboxes(patch_ids: torch.Tensor) -> list:
    bboxes = []
    for idx in patch_ids.tolist():
        r, c = int(idx) // GRID_SIZE, int(idx) % GRID_SIZE
        x0, y0 = c * PATCH_SIZE, r * PATCH_SIZE
        bboxes.append((x0, y0, x0 + PATCH_SIZE, y0 + PATCH_SIZE))
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
    """
    Recompose selected regions while preserving relative spatial layout.
    Identical logic to qwen/model.py::build_compact_image.
    """
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


def run_lpd(patch_ids: torch.Tensor, clip_image: Image.Image):
    """
    Full LPD pipeline.

    Args:
        patch_ids  : [M] selected patch indices
        clip_image : 336x336 PIL image (denormalized CLIP input)

    Returns:
        compact_image : PIL.Image
        merged_bboxes : list of (x0, y0, x1, y1)
    """
    raw_bboxes    = _patch_ids_to_bboxes(patch_ids.cpu())
    merged_bboxes = _merge_bboxes(raw_bboxes)
    compact_image = _build_compact_image(clip_image, merged_bboxes)
    return compact_image, merged_bboxes


# =========================================================
# CLIP image recovery  (for spatially-aligned visualization)
# =========================================================

def recover_clip_image(processor, image: Image.Image, question: str) -> Image.Image:
    """
    Recover the exact 336x336 image CLIP sees after processor preprocessing.
    Used so patch-level overlays are spatially aligned with the CLIP grid.
    """
    import torchvision.transforms.functional as TF

    prompt = f"USER: <image>\n{question}\nASSISTANT:"
    inputs = processor(text=prompt, images=image.convert("RGB"), return_tensors="pt")

    pv   = inputs["pixel_values"][0].cpu().float()
    mean = torch.tensor([0.48145466, 0.4578275,  0.40821073]).view(3, 1, 1)
    std  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
    return TF.to_pil_image((pv * std + mean).clamp(0, 1))


# =========================================================
# Inference
# =========================================================

def run_baseline_inference(
    model,
    processor,
    image: Image.Image,
    question: str,
    max_new_tokens: int = 32,
) -> str:
    """Standard LLaVA inference without any token selection."""
    device = next(model.parameters()).device
    prompt = f"USER: <image>\n{question}\nASSISTANT:"
    inputs = processor(
        text=prompt, images=image.convert("RGB"), return_tensors="pt"
    ).to(device)

    with torch.inference_mode():
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    return processor.batch_decode(
        out_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )[0].strip()


def run_tree_inference(
    model,
    processor,
    compact_image: Image.Image,
    question: str,
    max_new_tokens: int = 32,
) -> str:
    """LLaVA inference using the LPD compact image as visual input."""
    device = next(model.parameters()).device
    prompt = f"USER: <image>\n{question}\nASSISTANT:"
    inputs = processor(
        text=prompt, images=compact_image.convert("RGB"), return_tensors="pt"
    ).to(device)

    with torch.inference_mode():
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    return processor.batch_decode(
        out_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )[0].strip()