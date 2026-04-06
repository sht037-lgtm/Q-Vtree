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
):
    """
    Run one forward pass and return answer token's attention to image patches.

    Key improvements over naive approach:
      1. Use multiple layers (target_layers) and average them → more stable
      2. Precisely locate image token positions using image_token_id
         instead of assuming they occupy the first 576 positions

    Steps:
        1. generate() to get first answer token id
        2. Append answer token → full sequence (length N+1)
        3. Forward pass with output_attentions=True
        4. For each target layer:
               attn[batch=0, :heads, ans_pos, image_positions] → mean over heads → [N_img]
        5. Average across target layers → [N_img]

    Returns:
        patch_scores : [N_img] tensor, answer token's attention to each image patch
        answer_token : str
        image_positions : LongTensor of image token indices in the sequence
    """
    device = next(model.parameters()).device

    prompt = f"USER: <image>\n{question}\nASSISTANT:"
    inputs = processor(
        text=prompt,
        images=image.convert("RGB"),
        return_tensors="pt",
    ).to(device)

    # use last input token directly (no generate needed)
    full_input_ids = inputs["input_ids"]   # [1, N]

    # precisely locate image token positions
    image_token_id = 32000
    is_image = (full_input_ids[0] == image_token_id)
    image_positions = is_image.nonzero(as_tuple=True)[0]   # [N_img]

    if image_positions.numel() == 0:
        image_positions = torch.arange(576, device=device)

    # last input token (the ':' after ASSISTANT)
    ans_pos = full_input_ids.shape[1] - 1
    last_token = processor.tokenizer.decode(
        [full_input_ids[0, ans_pos].item()]
    ).strip()

    print(f"[DEBUG] last input token = '{last_token}'")
    print(f"[DEBUG] seq_len = {full_input_ids.shape[1]}, ans_pos = {ans_pos}")
    print(f"[DEBUG] num image tokens = {image_positions.numel()}")

    # full forward pass
    with torch.inference_mode():
        full_outputs = model(
            input_ids=full_input_ids,
            pixel_values=inputs["pixel_values"],
            attention_mask=torch.ones_like(full_input_ids),
            output_attentions=True,
        )

    # step 5: extract attention, average over target layers
    layer_scores = []
    for l in target_layers:
        # shape: [1, H, seq_len, seq_len]
        attn = full_outputs.attentions[l]
        # [batch=0, all heads, ans_pos, image_positions]
        ip = image_positions.to(attn.device)
        s = attn[0, :, ans_pos, ip]        # [H, N_img]
        s = s.mean(dim=0).cpu().float()    # [N_img]
        layer_scores.append(s)

    patch_scores = torch.stack(layer_scores).mean(dim=0)   # [N_img]

    return patch_scores, last_token, image_positions.cpu()


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
    generic_question: str = "Describe the image.",
    grid_size: int = 24,
):
    """
    Extract answer-token→image attention from LLaVA-1.5.

    Args:
        model            : LlavaForConditionalGeneration (attn_implementation="eager")
        processor        : LlavaProcessor
        image            : PIL.Image
        question         : specific question string (with options for MCQ)
        target_layers    : which layers to average, default (6, 13, 20, 27)
        use_relative     : whether to use relative attention normalization
        generic_question : baseline question for relative attention
        grid_size        : spatial grid size (24 for LLaVA-1.5)

    Returns:
        dict with keys:
            "patch_scores"  : [N_img] tensor (relative or raw attention)
            "answer_token"  : str
            "use_relative"  : bool
            "grid_size"     : int
    """
    # forward pass 1: specific question
    print(f"[INFO] Forward pass 1: specific question")
    A_q, answer_token, img_pos = _get_answer_attention(
        model, processor, image, question, target_layers
    )
    print(f"[DEBUG] A_q   max={A_q.max():.6f}, mean={A_q.mean():.8f}")

    if use_relative:
        # forward pass 2: generic question
        print(f"[INFO] Forward pass 2: generic question")
        A_q0, _, _ = _get_answer_attention(
            model, processor, image, generic_question, target_layers
        )
        print(f"[DEBUG] A_q0  max={A_q0.max():.6f}, mean={A_q0.mean():.8f}")

        patch_scores = A_q / (A_q0 + 1e-8)
        print(f"[DEBUG] relative max={patch_scores.max():.4f}, mean={patch_scores.mean():.6f}")
    else:
        patch_scores = A_q

    return {
        "patch_scores"  : patch_scores,
        "answer_token"  : answer_token,
        "use_relative"  : use_relative,
        "grid_size"     : grid_size,
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
    Visualize answer-token→image attention map overlaid on the original image.

    Args:
        image       : original PIL.Image
        attn_result : output from get_attention_maps()
        alpha       : heatmap overlay transparency
        save_path   : optional path to save the figure
    """
    patch_scores = attn_result["patch_scores"]   # [N_img]
    answer_token = attn_result.get("answer_token", "?")
    use_relative = attn_result.get("use_relative", False)
    grid_size    = attn_result.get("grid_size", 24)

    # reshape to spatial grid
    attn_map = patch_scores[:grid_size * grid_size].reshape(grid_size, grid_size).numpy()

    # normalize to [0, 1]
    vmin, vmax = attn_map.min(), attn_map.max()
    attn_map = (attn_map - vmin) / (vmax - vmin + 1e-8)

    # resize to original image size
    heatmap = Image.fromarray((attn_map * 255).astype(np.uint8)).resize(
        image.size, resample=Image.BILINEAR
    )
    heatmap = np.array(heatmap)

    # plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(image)
    ax.imshow(heatmap, cmap="jet", alpha=alpha)
    ax.axis("off")

    mode = "relative" if use_relative else "raw"
    plt.suptitle(
        f"Answer-to-Image Attention  |  answer='{answer_token}'  layers={mode}\n"
        f"target_layers=(6,13,20,27)  averaged",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"[INFO] Saved to {save_path}")

    plt.show()