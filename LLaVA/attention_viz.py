import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
    Extract text→image attention weights from LLaVA-1.5.

    Args:
        model     : loaded LlavaForConditionalGeneration
        processor : corresponding LlavaProcessor
        image     : PIL.Image
        question  : question string (already includes options if needed)
        layers    : which transformer layers to extract, e.g. [15, 23, 31]
                    defaults to last 3 layers

    Returns:
        dict with keys:
            "attentions"  : raw attention tuple (all 32 layers)
            "text_to_image": list of [num_heads, N_text, 576] tensors per layer
            "input_ids"   : token ids
            "num_image_tokens": 576
            "prompt"      : the prompt string used
    """
    device = next(model.parameters()).device

    prompt = f"USER: <image>\n{question}\nASSISTANT:"

    inputs = processor(
        text=prompt,
        images=image.convert("RGB"),
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        outputs = model(
            **inputs,
            output_attentions=True,
        )

    # outputs.attentions: tuple of 32 tensors
    # each tensor shape: [batch, num_heads, seq_len, seq_len]
    attentions = outputs.attentions

    num_image_tokens = 576  # 24x24 patches from CLIP ViT-L/14@336

    if layers is None:
        layers = [15, 23, 31]  # early, mid, late

    text_to_image = []
    for l in layers:
        attn = attentions[l][0]              # [num_heads, seq_len, seq_len]
        t2i  = attn[:, num_image_tokens:, :num_image_tokens]  # [num_heads, N_text, 576]
        text_to_image.append(t2i.cpu().float())

    return {
        "attentions"        : attentions,
        "text_to_image"     : text_to_image,   # list indexed by layers
        "layers"            : layers,
        "input_ids"         : inputs["input_ids"],
        "num_image_tokens"  : num_image_tokens,
        "prompt"            : prompt,
    }


# =========================================================
# ------------- Attention Map Visualization ---------------
# =========================================================

def visualize_attention(
    image: Image.Image,
    attn_result: dict,
    head_reduce: str = "mean",
    token_reduce: str = "mean",
    alpha: float = 0.5,
    save_path: str | None = None,
):
    """
    Visualize text→image attention maps overlaid on the original image.

    Args:
        image        : original PIL.Image
        attn_result  : output from get_attention_maps()
        head_reduce  : how to aggregate attention heads: "mean" or "max"
        token_reduce : how to aggregate text tokens:    "mean" or "max"
        alpha        : heatmap overlay transparency
        save_path    : optional path to save the figure

    The figure shows one column per layer in attn_result["layers"].
    """
    layers         = attn_result["layers"]
    text_to_image  = attn_result["text_to_image"]   # list of [num_heads, N_text, 576]
    grid_size      = 24   # 24x24 = 576

    fig, axes = plt.subplots(1, len(layers), figsize=(5 * len(layers), 5))
    if len(layers) == 1:
        axes = [axes]

    for ax, layer_idx, t2i in zip(axes, layers, text_to_image):
        # t2i: [num_heads, N_text, 576]

        # 1. aggregate over heads
        if head_reduce == "mean":
            attn_map = t2i.mean(dim=0)   # [N_text, 576]
        else:
            attn_map = t2i.max(dim=0).values

        # 2. aggregate over text tokens
        if token_reduce == "mean":
            attn_map = attn_map.mean(dim=0)   # [576]
        else:
            attn_map = attn_map.max(dim=0).values

        # 3. reshape to 24x24 spatial grid
        attn_map = attn_map.reshape(grid_size, grid_size).numpy()

        # 4. normalize to [0, 1]
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

        # 5. resize heatmap to original image size
        heatmap = Image.fromarray((attn_map * 255).astype(np.uint8)).resize(
            image.size, resample=Image.BILINEAR
        )
        heatmap = np.array(heatmap)

        # 6. overlay on image
        ax.imshow(image)
        ax.imshow(heatmap, cmap="jet", alpha=alpha)
        ax.set_title(f"Layer {layer_idx}", fontsize=13)
        ax.axis("off")

    plt.suptitle(
        f"Text→Image Attention  |  heads={head_reduce}  tokens={token_reduce}",
        fontsize=13, y=1.02
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"[INFO] Saved to {save_path}")

    plt.show()