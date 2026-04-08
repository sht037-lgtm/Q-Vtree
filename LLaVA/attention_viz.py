import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# =========================================================
# ------------- Attention Extraction ----------------------
# =========================================================

def _extract_attn_scores(full_outputs, text_positions, image_positions, target_layers):
    """
    Extract mean attention from text_positions → image_positions.
    Average over heads and target layers.

    Args:
        full_outputs    : model output with output_attentions=True
        text_positions  : LongTensor [Q]
        image_positions : LongTensor [N_img]
        target_layers   : tuple of layer indices

    Returns:
        [N_img] tensor
    """
    scores = []
    for l in target_layers:
        attn = full_outputs.attentions[l]       # [1, H, seq_len, seq_len]
        vp   = image_positions.to(attn.device)
        qp   = text_positions.to(attn.device)
        a    = attn[0, :, :, vp][:, qp, :]     # [H, Q, N_img]
        a    = a.mean(dim=0)                    # [Q, N_img] mean over heads
        # per-token normalization: remove causal bias
        # each token's attention becomes a relative distribution over image patches
        a    = a / (a.sum(dim=1, keepdim=True) + 1e-8)  # [Q, N_img]
        scores.append(a.mean(dim=0).cpu().float())       # [N_img] mean over tokens
    return torch.stack(scores).mean(dim=0)      # [N_img]


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
      - Two forward passes with IDENTICAL sequence length via padding:
          Pass 1: [prefix+image] [question tokens]                [suffix]
          Pass 2: [prefix+image] [generic tokens + PAD to same len] [suffix]
        Padding ensures both passes have exactly the same sequence length,
        so causal bias is identical and cancels out when dividing A_q / A_g.
      - Mean over all text tokens, all heads, target layers
      - generate() called once separately just to display the answer

    Args:
        model            : LlavaForConditionalGeneration (attn_implementation="eager")
        processor        : LlavaProcessor
        image            : PIL.Image
        question         : specific question string
        target_layers    : layers to average
        use_relative     : normalize by generic question attention
        generic_question : baseline question text
        grid_size        : 24 for LLaVA-1.5

    Returns:
        dict with keys:
            "patch_scores" : [N_img] tensor
            "answer"       : str
            "use_relative" : bool
            "grid_size"    : int
            "target_layers": tuple
    """
    device = next(model.parameters()).device

    prompt = f"USER: <image>\n{question}\nASSISTANT:"
    inputs = processor(
        text=prompt,
        images=image.convert("RGB"),
        return_tensors="pt",
    ).to(device)

    full_input_ids = inputs["input_ids"]   # [1, N]
    seq_len = full_input_ids.shape[1]

    # generate answer for display only (one time)
    with torch.inference_mode():
        gen_out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    answer = processor.batch_decode(
        gen_out[:, full_input_ids.shape[1]:],
        skip_special_tokens=True
    )[0].strip()

    # locate image token positions (id=32000)
    image_token_id = 32000
    is_image       = (full_input_ids[0] == image_token_id)
    image_positions = is_image.nonzero(as_tuple=True)[0].cpu()

    if image_positions.numel() == 0:
        image_positions = torch.arange(576)

    img_end = image_positions[-1].item()

    # question text positions: after image, before ASSISTANT (last 4 tokens)
    text_start = img_end + 1
    text_end   = seq_len - 4
    if text_end <= text_start:
        text_end = seq_len - 1
    text_positions = torch.arange(text_start, text_end, dtype=torch.long)
    n_question_tokens = text_end - text_start

    print(f"[DEBUG] answer = '{answer}'")
    print(f"[DEBUG] seq_len={seq_len}, img_end={img_end}")
    print(f"[DEBUG] question text: pos {text_start}~{text_end-1} ({n_question_tokens} tokens)")

    # ===== Pass 1: specific question =====
    with torch.inference_mode():
        out_q = model(
            input_ids=full_input_ids,
            pixel_values=inputs["pixel_values"],
            attention_mask=torch.ones_like(full_input_ids),
            output_attentions=True,
        )
    A_q = _extract_attn_scores(out_q, text_positions, image_positions, target_layers)
    print(f"[DEBUG] A_q  max={A_q.max():.6f}, mean={A_q.mean():.8f}")

    if use_relative:
        # ===== Pass 2: generic question padded to same length =====
        suffix_ids = full_input_ids[:, seq_len - 4:]    # last 4 tokens
        prefix_ids = full_input_ids[:, :img_end + 1]    # BOS + USER: + image

        # tokenize generic question
        generic_ids = processor.tokenizer(
            generic_question,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids.to(device)
        n_generic_tokens = generic_ids.shape[1]

        # pad generic tokens to match question token count
        pad_token_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
        if n_generic_tokens < n_question_tokens:
            # pad generic to same length as question
            padding = torch.full(
                (1, n_question_tokens - n_generic_tokens),
                pad_token_id,
                dtype=torch.long,
                device=device,
            )
            generic_padded = torch.cat([generic_ids, padding], dim=1)
        elif n_generic_tokens > n_question_tokens:
            # truncate generic to same length as question
            generic_padded = generic_ids[:, :n_question_tokens]
        else:
            generic_padded = generic_ids

        # build generic input: same structure as specific question
        generic_input_ids = torch.cat([prefix_ids, generic_padded, suffix_ids], dim=1)

        # attention mask: mask out padding tokens
        attn_mask = torch.ones_like(generic_input_ids)
        if n_generic_tokens < n_question_tokens:
            pad_start = prefix_ids.shape[1] + n_generic_tokens
            pad_end   = prefix_ids.shape[1] + n_question_tokens
            attn_mask[:, pad_start:pad_end] = 0   # mask padding

        # generic text positions (same as question text positions)
        gen_text_positions = text_positions   # same positions, same length

        print(f"[DEBUG] generic text: {n_generic_tokens} tokens + {max(0, n_question_tokens-n_generic_tokens)} pad")
        print(f"[DEBUG] generic seq_len={generic_input_ids.shape[1]} (should == {seq_len})")

        with torch.inference_mode():
            out_g = model(
                input_ids=generic_input_ids,
                pixel_values=inputs["pixel_values"],
                attention_mask=attn_mask,
                output_attentions=True,
            )
        A_g = _extract_attn_scores(out_g, gen_text_positions, image_positions, target_layers)
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

    Note: pass clip_image (CLIP's 336x336 view) for best spatial alignment.
    """
    patch_scores  = attn_result["patch_scores"]
    answer        = attn_result.get("answer", "?")
    use_relative  = attn_result.get("use_relative", False)
    grid_size     = attn_result.get("grid_size", 24)
    target_layers = attn_result.get("target_layers", (6, 13, 20, 27))

    # reshape to spatial grid
    attn_map = patch_scores[:grid_size * grid_size].reshape(grid_size, grid_size).numpy()

    # gaussian blur before normalization
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
        f"Attention Map  |  answer='{answer[:50]}'\n"
        f"mode={mode}  layers={list(target_layers)}",
        fontsize=10, y=1.02,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"[INFO] Saved to {save_path}")

    plt.show()