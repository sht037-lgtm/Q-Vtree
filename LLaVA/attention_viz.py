import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# =========================================================
# ------------- Single Forward Pass -----------------------
# =========================================================

def _get_answer_attention(
    model,
    processor,
    image,
    question,
    target_layers=(6, 13, 20, 27),
    text_positions=None,   # if None, auto-detect question text positions
):
    """
    Compute attention from question text tokens to image patches.

    Key design (inspired by Qwen model.py):
      - Use ALL question text tokens instead of a single token
      - Apply rater filter: keep only tokens with attention sum >= mean
        (removes noise from irrelevant tokens like '(A)', 'directly', etc.)
      - Average over filtered tokens, heads, and target layers
      - generate() called separately just to display the model's answer

    Args:
        model          : LlavaForConditionalGeneration
        processor      : LlavaProcessor
        image          : PIL.Image
        question       : question string
        target_layers  : layers to average
        text_positions : optional manual override of query token positions

    Returns:
        patch_scores    : [N_img] tensor
        answer          : str (model's generated answer, display only)
        image_positions : LongTensor of image token positions
    """
    device = next(model.parameters()).device

    prompt = f"USER: <image>\n{question}\nASSISTANT:"
    inputs = processor(
        text=prompt,
        images=image.convert("RGB"),
        return_tensors="pt",
    ).to(device)

    # generate full answer for display only
    with torch.inference_mode():
        gen_out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    answer = processor.batch_decode(
        gen_out[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )[0].strip()

    full_input_ids = inputs["input_ids"]   # [1, N]

    # precisely locate image token positions (id=32000)
    image_token_id = 32000
    is_image = (full_input_ids[0] == image_token_id)
    image_positions = is_image.nonzero(as_tuple=True)[0]   # [N_img]

    if image_positions.numel() == 0:
        image_positions = torch.arange(576, device=device)

    img_end = image_positions[-1].item()
    seq_len = full_input_ids.shape[1]

    # locate question text token positions
    # = everything after image tokens, excluding the final "ASSISTANT:" tokens
    if text_positions is None:
        text_start = img_end + 1
        text_end   = seq_len - 4  # exclude the last 2 tokens (newline + ':')
        if text_end > text_start:
            text_positions = torch.arange(text_start, text_end, dtype=torch.long)
        else:
            # fallback: just use the last input token
            text_positions = torch.tensor([seq_len - 1], dtype=torch.long)

    print(f"[DEBUG] model answer     = '{answer}'")
    print(f"[DEBUG] seq_len={seq_len}, img_end={img_end}")
    print(f"[DEBUG] text_positions: {text_positions[0].item()} ~ {text_positions[-1].item()} ({len(text_positions)} tokens)")
    print(f"[DEBUG] num image tokens = {image_positions.numel()}")

    # full forward pass
    with torch.inference_mode():
        full_outputs = model(
            input_ids=full_input_ids,
            pixel_values=inputs["pixel_values"],
            attention_mask=torch.ones_like(full_input_ids),
            output_attentions=True,
        )

    # extract attention, rater filter, average over target layers
    layer_scores = []
    for l in target_layers:
        attn = full_outputs.attentions[l]          # [1, H, seq_len, seq_len]
        vp = image_positions.to(attn.device)
        qp = text_positions.to(attn.device)

        # attention from question tokens to image tokens
        attn_to_img = attn[0, :, :, vp]            # [H, seq_len, N_img]
        attn_to_img = attn_to_img[:, qp, :]        # [H, num_query, N_img]

        # mean over all query tokens and heads → [N_img] (no rater filter)
        s = attn_to_img.mean(dim=0).mean(dim=0).cpu().float()
        layer_scores.append(s)

    patch_scores = torch.stack(layer_scores).mean(dim=0)   # [N_img]

    return patch_scores, answer, image_positions.cpu()


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
    generic_question: str = "Describe the image in detail.",
    grid_size: int = 24,
):
    """
    Extract question-text→image attention from LLaVA-1.5.

    Uses all question text tokens with rater filter (not just last input token),
    making it robust to multi-subject questions like "what is the relation between
    the cat and the dog?"

    Args:
        model            : LlavaForConditionalGeneration (attn_implementation="eager")
        processor        : LlavaProcessor
        image            : PIL.Image
        question         : specific question string
        target_layers    : layers to average
        use_relative     : normalize by generic question attention
        generic_question : baseline question for relative attention
        grid_size        : 24 for LLaVA-1.5

    Returns:
        dict with keys:
            "patch_scores" : [N_img] tensor
            "answer"       : str
            "use_relative" : bool
            "grid_size"    : int
            "target_layers": tuple
    """
    print(f"[INFO] Forward pass 1: specific question")
    A_q, answer, _ = _get_answer_attention(
        model, processor, image, question, target_layers
    )
    print(f"[DEBUG] A_q  max={A_q.max():.6f}, mean={A_q.mean():.8f}")

    if use_relative:
        print(f"[INFO] Forward pass 2: generic question")
        A_q0, _, _ = _get_answer_attention(
            model, processor, image, generic_question, target_layers
        )
        print(f"[DEBUG] A_q0 max={A_q0.max():.6f}, mean={A_q0.mean():.8f}")
        patch_scores = A_q / (A_q0 + 1e-8)
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

    Note: pass clip_image (CLIP's 336x336 view) for best spatial alignment.

    Args:
        image       : PIL.Image to overlay on (recommend using clip_image)
        attn_result : output from get_attention_maps()
        alpha       : heatmap overlay transparency
        save_path   : optional save path
    """
    patch_scores  = attn_result["patch_scores"]
    answer        = attn_result.get("answer", "?")
    use_relative  = attn_result.get("use_relative", False)
    grid_size     = attn_result.get("grid_size", 24)
    target_layers = attn_result.get("target_layers", (6, 13, 20, 27))

    # reshape to spatial grid
    attn_map = patch_scores[:grid_size * grid_size].reshape(grid_size, grid_size).numpy()

    # gaussian blur before normalization (smooth sparse hotspots)
    from scipy.ndimage import gaussian_filter
    attn_map = gaussian_filter(attn_map, sigma=1.0)

    # normalize to [0, 1]
    vmin, vmax = attn_map.min(), attn_map.max()
    attn_map = (attn_map - vmin) / (vmax - vmin + 1e-8)

    # resize to image size
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
        f"Attention Map  |  answer='{answer[:40]}'\n"
        f"mode={mode}  layers={list(target_layers)}  rater_filter",
        fontsize=10, y=1.02,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"[INFO] Saved to {save_path}")

    plt.show()