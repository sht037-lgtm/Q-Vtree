import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# =========================================================
# ------------- Attention Map Extraction ------------------
# =========================================================

def get_attention_maps(
    model,
    processor,
    image: Image.Image,
    question: str,
    layers: list[int] | None = None,
):
    """
    Extract answer-token→image attention weights from LLaVA-1.5.

    Strategy:
        1. generate() to get the answer token id (e.g. 'A')
        2. Append the answer token to the input sequence
        3. Full forward pass with the 640-length sequence
        4. Take the LAST row of attention (answer token's attention)
           and slice the first 576 columns (image token part)

    This correctly captures where the model looks when generating the answer.

    Args:
        model     : loaded LlavaForConditionalGeneration
                    (must be loaded with attn_implementation="eager")
        processor : corresponding LlavaProcessor
        image     : PIL.Image
        question  : question string (should include options for MCQ)
        layers    : which transformer layers to extract, e.g. [15, 23, 31]
                    defaults to [15, 23, 31]

    Returns:
        dict with keys:
            "answer_to_image" : list of [H, 576] tensors per layer
            "layers"          : layer indices used
            "answer_token"    : the generated answer token string
            "prompt"          : prompt string used
    """
    device = next(model.parameters()).device

    prompt = f"USER: <image>\n{question}\nASSISTANT:"

    inputs = processor(
        text=prompt,
        images=image.convert("RGB"),
        return_tensors="pt",
    ).to(device)

    num_image_tokens = 576  # 24×24 from CLIP ViT-L/14@336

    if layers is None:
        layers = [15, 23, 31]

    # step 1: generate the first answer token
    with torch.inference_mode():
        gen_out = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
        )

    answer_token_id = gen_out[:, inputs["input_ids"].shape[1]:]  # [1, 1]
    answer_token = processor.batch_decode(
        answer_token_id, skip_special_tokens=True
    )[0].strip()

    # step 2: append answer token to input → length 639+1=640
    full_input_ids = torch.cat(
        [inputs["input_ids"], answer_token_id], dim=1
    )  # [1, 640]

    # step 3: full forward pass with output_attentions=True
    with torch.inference_mode():
        full_outputs = model(
            input_ids=full_input_ids,
            pixel_values=inputs["pixel_values"],
            attention_mask=torch.ones_like(full_input_ids),
            output_attentions=True,
        )

    # full_outputs.attentions: tuple of 32 tensors
    # each shape: [1, 32, 640, 640]

    seq_len = full_outputs.attentions[0].shape[2]
    n_text  = seq_len - num_image_tokens
    print(f"[DEBUG] seq_len={seq_len}, num_image_tokens={num_image_tokens}, n_text_tokens={n_text}")
    print(f"[DEBUG] answer token = '{answer_token}'")

    answer_to_image = []
    for l in layers:
        # shape: [1, 32, 640, 640]
        attn = full_outputs.attentions[l][0]   # [32, 640, 640]

        # last row = answer token's attention to all previous tokens
        # first 576 columns = image tokens
        a2i = attn[:, -1, :num_image_tokens]   # [32, 576]

        # suppress attention sink at position 0
        a2i = a2i.clone()
        a2i[:, 0] = 0.0

        print(f"[DEBUG] Layer {l}: a2i shape={a2i.shape}, max={a2i.max():.4f}, mean={a2i.mean():.6f}")

        answer_to_image.append(a2i.cpu().float())

    return {
        "answer_to_image" : answer_to_image,
        "layers"          : layers,
        "answer_token"    : answer_token,
        "prompt"          : prompt,
    }


# =========================================================
# ------------- Attention Map Visualization ---------------
# =========================================================

def visualize_attention(
    image: Image.Image,
    attn_result: dict,
    head_reduce: str = "mean",
    alpha: float = 0.5,
    save_path: str | None = None,
):
    """
    Visualize answer-token→image attention maps overlaid on the original image.

    Args:
        image        : original PIL.Image
        attn_result  : output from get_attention_maps()
        head_reduce  : aggregate heads by "mean" or "max"
        alpha        : heatmap overlay transparency
        save_path    : optional path to save the figure
    """
    layers          = attn_result["layers"]
    answer_to_image = attn_result["answer_to_image"]  # list of [32, 576]
    answer_token    = attn_result.get("answer_token", "?")

    fig, axes = plt.subplots(1, len(layers), figsize=(5 * len(layers), 5))
    if len(layers) == 1:
        axes = [axes]

    for ax, layer_idx, a2i in zip(axes, layers, answer_to_image):
        # a2i: [32, 576]

        # 1. aggregate over heads
        if head_reduce == "mean":
            attn_map = a2i.mean(dim=0)          # [576]
        else:
            attn_map = a2i.max(dim=0).values    # [576]

        # 2. reshape to 24x24
        attn_map = attn_map.reshape(24, 24).numpy()

        # 3. normalize to [0, 1]
        vmin, vmax = attn_map.min(), attn_map.max()
        attn_map = (attn_map - vmin) / (vmax - vmin + 1e-8)

        # 4. resize heatmap to original image size
        heatmap = Image.fromarray((attn_map * 255).astype(np.uint8)).resize(
            image.size, resample=Image.BILINEAR
        )
        heatmap = np.array(heatmap)

        # 5. overlay
        ax.imshow(image)
        ax.imshow(heatmap, cmap="jet", alpha=alpha)
        ax.set_title(f"Layer {layer_idx}", fontsize=12)
        ax.axis("off")

    plt.suptitle(
        f"Answer-to-Image Attention  |  answer='{answer_token}'  heads={head_reduce}",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"[INFO] Saved to {save_path}")

    plt.show()