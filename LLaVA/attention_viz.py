import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# =========================================================
# ------------- Attention Extraction ----------------------
# =========================================================

def _forward_attn(model, processor, image, input_ids, image_positions, target_layers):
    """
    Run one forward pass and extract mean attention from text tokens → image tokens.

    Args:
        model           : LlavaForConditionalGeneration
        processor       : LlavaProcessor
        image           : PIL.Image
        input_ids       : [1, seq_len] tensor (already on device)
        image_positions : LongTensor [N_img]
        target_layers   : tuple of layer indices

    Returns:
        [N_img] tensor
    """
    img_end        = image_positions[-1].item()
    text_positions = torch.arange(img_end + 1, input_ids.shape[1], dtype=torch.long)

    inputs = processor(
        text="",   # dummy, we override input_ids directly
        images=image.convert("RGB"),
        return_tensors="pt",
    )
    pixel_values = inputs["pixel_values"].to(input_ids.device)

    with torch.inference_mode():
        out = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=torch.ones_like(input_ids),
            output_attentions=True,
        )

    scores = []
    for l in target_layers:
        attn = out.attentions[l]                  # [1, H, seq_len, seq_len]
        vp   = image_positions.to(attn.device)
        qp   = text_positions.to(attn.device)
        a    = attn[0, :, :, vp][:, qp, :]       # [H, Q, N_img]
        a    = a.mean(dim=0).mean(dim=0)          # [N_img]
        scores.append(a.cpu().float())
    return torch.stack(scores).mean(dim=0)        # [N_img]


def _get_input_ids(processor, image, question, device):
    """Tokenize prompt and return input_ids on device."""
    prompt = f"USER: <image>\n{question}\nASSISTANT:"
    inputs = processor(
        text=prompt,
        images=image.convert("RGB"),
        return_tensors="pt",
    )
    return inputs["input_ids"].to(device), inputs["pixel_values"].to(device)


def _get_attn_bidirectional(model, processor, image, question, image_positions, target_layers):
    """
    Compute attention averaged over forward and reversed question token order.
    This cancels causal bias:
      - forward:  last token (e.g. "cat") has highest causal bias → cat dominates
      - reversed: first token becomes last → bias flips
      - average:  causal bias cancels out
    """
    device = next(model.parameters()).device

    prompt = f"USER: <image>\n{question}\nASSISTANT:"
    inputs = processor(
        text=prompt,
        images=image.convert("RGB"),
        return_tensors="pt",
    ).to(device)

    full_input_ids = inputs["input_ids"]
    pixel_values   = inputs["pixel_values"]
    img_end        = image_positions[-1].item()

    # ---- forward pass ----
    text_positions_fwd = torch.arange(img_end + 1, full_input_ids.shape[1], dtype=torch.long)

    with torch.inference_mode():
        out_fwd = model(
            input_ids=full_input_ids,
            pixel_values=pixel_values,
            attention_mask=torch.ones_like(full_input_ids),
            output_attentions=True,
        )

    scores_fwd = []
    for l in target_layers:
        attn = out_fwd.attentions[l]
        vp   = image_positions.to(attn.device)
        qp   = text_positions_fwd.to(attn.device)
        a    = attn[0, :, :, vp][:, qp, :]
        a    = a.mean(dim=0).mean(dim=0)
        scores_fwd.append(a.cpu().float())
    A_fwd = torch.stack(scores_fwd).mean(dim=0)

    # ---- reversed pass ----
    prefix   = full_input_ids[:, :img_end + 1]
    suffix   = full_input_ids[:, img_end + 1:]
    reversed_input_ids = torch.cat([prefix, torch.flip(suffix, dims=[1])], dim=1)
    text_positions_rev = torch.arange(img_end + 1, reversed_input_ids.shape[1], dtype=torch.long)

    with torch.inference_mode():
        out_rev = model(
            input_ids=reversed_input_ids,
            pixel_values=pixel_values,
            attention_mask=torch.ones_like(reversed_input_ids),
            output_attentions=True,
        )

    scores_rev = []
    for l in target_layers:
        attn = out_rev.attentions[l]
        vp   = image_positions.to(attn.device)
        qp   = text_positions_rev.to(attn.device)
        a    = attn[0, :, :, vp][:, qp, :]
        a    = a.mean(dim=0).mean(dim=0)
        scores_rev.append(a.cpu().float())
    A_rev = torch.stack(scores_rev).mean(dim=0)

    # average forward + reversed → causal bias cancels
    return (A_fwd + A_rev) / 2


# =========================================================
# ------------- Attention Map Extraction ------------------
# =========================================================

def get_attention_maps(
    model,
    processor,
    image: Image.Image,
    question: str,
    target_layers: tuple = (6, 13, 20, 27),
    use_relative: bool = True,
    generic_question: str = "Write a general description of the image.",
    grid_size: int = 24,
):
    """
    Extract question-text→image relative attention from LLaVA-1.5.

    Key design:
      - Bidirectional averaging: run forward pass + reversed token order pass,
        average the two → causal bias cancels out universally without
        knowing query content.
      - Relative attention: divide by generic question attention to remove
        image-fixed attention (sink, salient regions).
      - Both specific and generic use bidirectional averaging.

    Args:
        model            : LlavaForConditionalGeneration (attn_implementation="eager")
        processor        : LlavaProcessor
        image            : PIL.Image
        question         : specific question string
        target_layers    : layers to average
        use_relative     : normalize by generic question attention
        generic_question : baseline question text
        grid_size        : 24 for LLaVA-1.5
    """
    device = next(model.parameters()).device

    prompt = f"USER: <image>\n{question}\nASSISTANT:"
    inputs = processor(
        text=prompt,
        images=image.convert("RGB"),
        return_tensors="pt",
    ).to(device)

    full_input_ids = inputs["input_ids"]

    # generate answer for display only
    with torch.inference_mode():
        gen_out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    answer = processor.batch_decode(
        gen_out[:, full_input_ids.shape[1]:],
        skip_special_tokens=True
    )[0].strip()

    # locate image token positions (id=32000)
    image_token_id  = 32000
    is_image        = (full_input_ids[0] == image_token_id)
    image_positions = is_image.nonzero(as_tuple=True)[0].cpu()

    if image_positions.numel() == 0:
        image_positions = torch.arange(576)

    print(f"[DEBUG] answer = '{answer}'")
    print(f"[DEBUG] num image tokens = {image_positions.numel()}, img_end = {image_positions[-1].item()}")

    # ===== specific question: forward + reversed =====
    print(f"[INFO] specific question: forward + reversed")
    A_q = _get_attn_bidirectional(
        model, processor, image, question, image_positions, target_layers
    )
    print(f"[DEBUG] A_q  max={A_q.max():.6f}, mean={A_q.mean():.8f}")

    if use_relative:
        # ===== generic question: forward + reversed =====
        print(f"[INFO] generic question: forward + reversed")
        A_g = _get_attn_bidirectional(
            model, processor, image, generic_question, image_positions, target_layers
        )
        print(f"[DEBUG] A_g  max={A_g.max():.6f}, mean={A_g.mean():.8f}")

        patch_scores = A_q / (A_g + 1e-8)
        print(f"[DEBUG] relative max={patch_scores.max():.4f}, mean={patch_scores.mean():.6f}")
    else:
        patch_scores = A_q

    return {
        "patch_scores" : patch_scores,
        "answer"       : answer,
        "use_relative" : use_relative,
        "grid_size"    : grid_size,
        "target_layers": target_layers,
    }


# =========================================================
# ------------- Attention Map Visualization ---------------
# =========================================================

def visualize_attention(
    image: Image.Image,
    attn_result: dict,
    alpha: float = 0.5,
    save_path: str | None = None,
):
    """
    Visualize question-text→image attention map overlaid on the image.
    Note: pass clip_image for best spatial alignment.
    """
    patch_scores  = attn_result["patch_scores"]
    answer        = attn_result.get("answer", "?")
    use_relative  = attn_result.get("use_relative", False)
    grid_size     = attn_result.get("grid_size", 24)
    target_layers = attn_result.get("target_layers", (6, 13, 20, 27))

    attn_map = patch_scores[:grid_size * grid_size].reshape(grid_size, grid_size).numpy()

    from scipy.ndimage import gaussian_filter
    attn_map = gaussian_filter(attn_map, sigma=1.0)

    vmin, vmax = attn_map.min(), attn_map.max()
    attn_map = (attn_map - vmin) / (vmax - vmin + 1e-8)

    heatmap = Image.fromarray((attn_map * 255).astype(np.uint8)).resize(
        image.size, resample=Image.BILINEAR
    )
    heatmap = np.array(heatmap)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(image)
    ax.imshow(heatmap, cmap="jet", alpha=alpha)
    ax.axis("off")

    mode = "relative" if use_relative else "raw"
    plt.suptitle(
        f"Attention Map  |  answer='{answer[:50]}'\n"
        f"mode={mode}  layers={list(target_layers)}  bidirectional",
        fontsize=10, y=1.02,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"[INFO] Saved to {save_path}")

    plt.show()