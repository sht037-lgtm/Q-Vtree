import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# =========================================================
# ------------- Single Forward Pass Helper ----------------
# =========================================================

def _get_answer_attention(model, processor, image, question, layers, num_image_tokens=576):
    """
    Helper: run one forward pass and return answer token's attention to image patches.

    Steps:
        1. generate() to get the first answer token id
        2. append answer token to input → length N+1
        3. full forward pass with output_attentions=True
        4. take last row of attention, slice first 576 cols (image part)

    Returns:
        answer_to_image : list of [H, 576] tensors per layer
        answer_token    : decoded answer token string
    """
    device = next(model.parameters()).device

    prompt = f"USER: <image>\n{question}\nASSISTANT:"
    inputs = processor(
        text=prompt,
        images=image.convert("RGB"),
        return_tensors="pt",
    ).to(device)

    # step 1: generate first answer token
    with torch.inference_mode():
        gen_out = model.generate(**inputs, max_new_tokens=1, do_sample=False)

    answer_token_id = gen_out[:, inputs["input_ids"].shape[1]:]   # [1, 1]
    answer_token = processor.batch_decode(
        answer_token_id, skip_special_tokens=True
    )[0].strip()

    # step 2: append answer token → length N+1
    full_input_ids = torch.cat(
        [inputs["input_ids"], answer_token_id], dim=1
    )

    # step 3: full forward pass
    with torch.inference_mode():
        full_outputs = model(
            input_ids=full_input_ids,
            pixel_values=inputs["pixel_values"],
            attention_mask=torch.ones_like(full_input_ids),
            output_attentions=True,
        )

    # step 4: extract answer token → image attention per layer
    answer_to_image = []
    for l in layers:
        attn = full_outputs.attentions[l][0]          # [H, seq_len, seq_len]
        a2i  = attn[:, -1, :num_image_tokens].clone() # [H, 576]
        a2i[:, 0] = 0.0                               # suppress sink
        answer_to_image.append(a2i.cpu().float())

    return answer_to_image, answer_token


# =========================================================
# ------------- Attention Map Extraction ------------------
# =========================================================

def get_attention_maps(
    model,
    processor,
    image: Image.Image,
    question: str,
    layers: list[int] | None = None,
    use_relative: bool = True,
    generic_question: str = "Describe the image.",
):
    """
    Extract answer-token→image attention weights from LLaVA-1.5.

    If use_relative=True (recommended), runs two forward passes:
        1. specific question  → A_q   [H, 576] per layer
        2. generic question   → A_q0  [H, 576] per layer
        relative = A_q / (A_q0 + eps)

    This cancels out fixed high-attention regions (sink, salient backgrounds)
    and leaves only question-relevant attention.

    Args:
        model            : LlavaForConditionalGeneration (attn_implementation="eager")
        processor        : LlavaProcessor
        image            : PIL.Image
        question         : specific question string (with options for MCQ)
        layers           : transformer layers to extract, default [15, 23, 31]
        use_relative     : whether to apply relative attention normalization
        generic_question : generic prompt used as baseline for relative attention

    Returns:
        dict with keys:
            "answer_to_image" : list of [576] tensors per layer (head-averaged)
            "layers"          : layer indices used
            "answer_token"    : generated answer token string
            "use_relative"    : whether relative attention was applied
    """
    if layers is None:
        layers = [15, 23, 31]

    num_image_tokens = 576

    # forward pass 1: specific question
    print("[INFO] Forward pass 1: specific question...")
    a2i_q, answer_token = _get_answer_attention(
        model, processor, image, question, layers, num_image_tokens
    )
    print(f"[DEBUG] answer token = '{answer_token}'")

    if use_relative:
        # forward pass 2: generic question as baseline
        print("[INFO] Forward pass 2: generic question (baseline)...")
        a2i_q0, generic_token = _get_answer_attention(
            model, processor, image, generic_question, layers, num_image_tokens
        )
        print(f"[DEBUG] generic token = '{generic_token}'")

    answer_to_image = []
    for i, l in enumerate(layers):
        # aggregate heads: [H, 576] → [576]
        attn_q = a2i_q[i].mean(dim=0)    # [576]

        if use_relative:
            attn_q0  = a2i_q0[i].mean(dim=0)          # [576]
            attn_map = attn_q / (attn_q0 + 1e-8)      # [576] relative attention
        else:
            attn_map = attn_q                           # [576] raw attention

        print(f"[DEBUG] Layer {l}: max={attn_map.max():.4f}, mean={attn_map.mean():.6f}")
        answer_to_image.append(attn_map)

    return {
        "answer_to_image" : answer_to_image,
        "layers"          : layers,
        "answer_token"    : answer_token,
        "use_relative"    : use_relative,
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
    Visualize answer-token→image attention maps overlaid on the original image.

    Args:
        image       : original PIL.Image
        attn_result : output from get_attention_maps()
        alpha       : heatmap overlay transparency
        save_path   : optional path to save the figure
    """
    layers          = attn_result["layers"]
    answer_to_image = attn_result["answer_to_image"]  # list of [576]
    answer_token    = attn_result.get("answer_token", "?")
    use_relative    = attn_result.get("use_relative", False)

    fig, axes = plt.subplots(1, len(layers), figsize=(5 * len(layers), 5))
    if len(layers) == 1:
        axes = [axes]

    for ax, layer_idx, attn_map in zip(axes, layers, answer_to_image):
        # attn_map: [576] already head-aggregated

        # reshape to 24x24
        heat = attn_map.reshape(24, 24).numpy()

        # normalize to [0, 1]
        vmin, vmax = heat.min(), heat.max()
        heat = (heat - vmin) / (vmax - vmin + 1e-8)

        # resize to original image size
        heatmap = Image.fromarray((heat * 255).astype(np.uint8)).resize(
            image.size, resample=Image.BILINEAR
        )

        # overlay
        ax.imshow(image)
        ax.imshow(np.array(heatmap), cmap="jet", alpha=alpha)
        ax.set_title(f"Layer {layer_idx}", fontsize=12)
        ax.axis("off")

    mode = "relative" if use_relative else "raw"
    plt.suptitle(
        f"Answer-to-Image Attention  |  answer='{answer_token}'  [{mode}]",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"[INFO] Saved to {save_path}")

    plt.show()