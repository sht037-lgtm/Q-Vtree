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
    Extract text→image attention weights from LLaVA-1.5.

    Args:
        model     : loaded LlavaForConditionalGeneration
                    (must be loaded with attn_implementation="eager")
        processor : corresponding LlavaProcessor
        image     : PIL.Image
        question  : question string
        layers    : which transformer layers to extract, e.g. [15, 23, 31]
                    defaults to [15, 23, 31]

    Returns:
        dict with keys:
            "text_to_image" : list of [num_heads, N_text, 575] tensors per layer
                              (position 0 excluded to remove attention sink)
            "layers"        : layer indices used
            "prompt"        : prompt string used
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
    # each shape: [batch, num_heads, seq_len, seq_len]
    attentions = outputs.attentions

    num_image_tokens = 576  # 24x24 from CLIP ViT-L/14@336

    if layers is None:
        layers = [15, 23, 31]

    text_to_image = []
    for l in layers:
        attn = attentions[l][0]        # [num_heads, seq_len, seq_len]

        # cut text→image block: rows=text tokens, cols=image tokens
        t2i = attn[:, num_image_tokens:, :num_image_tokens]   # [num_heads, N_text, 576]

        # remove position 0 (attention sink — first image token gets
        # disproportionate attention unrelated to visual content)
        t2i = t2i[:, :, 1:]            # [num_heads, N_text, 575]

        text_to_image.append(t2i.cpu().float())

    return {
        "text_to_image" : text_to_image,
        "layers"        : layers,
        "prompt"        : prompt,
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
        head_reduce  : aggregate heads by "mean" or "max"
        token_reduce : aggregate text tokens by "mean" or "max"
        alpha        : heatmap overlay transparency (0=invisible, 1=opaque)
        save_path    : optional path to save the figure
    """
    layers        = attn_result["layers"]
    text_to_image = attn_result["text_to_image"]  # list of [num_heads, N_text, 575]

    fig, axes = plt.subplots(1, len(layers), figsize=(5 * len(layers), 5))
    if len(layers) == 1:
        axes = [axes]

    for ax, layer_idx, t2i in zip(axes, layers, text_to_image):
        # t2i: [num_heads, N_text, 575]

        # 1. aggregate over heads
        if head_reduce == "mean":
            attn_map = t2i.mean(dim=0)        # [N_text, 575]
        else:
            attn_map = t2i.max(dim=0).values

        # 2. aggregate over text tokens
        if token_reduce == "mean":
            attn_map = attn_map.mean(dim=0)   # [575]
        else:
            attn_map = attn_map.max(dim=0).values

        # 3. pad back to 576 and reshape to 24x24
        attn_map = torch.cat([torch.zeros(1), attn_map], dim=0)  # [576]
        attn_map = attn_map.reshape(24, 24).numpy()

        # 4. normalize to [0, 1]
        vmin, vmax = attn_map.min(), attn_map.max()
        attn_map = (attn_map - vmin) / (vmax - vmin + 1e-8)

        # 5. resize heatmap to original image size
        heatmap = Image.fromarray((attn_map * 255).astype(np.uint8)).resize(
            image.size, resample=Image.BILINEAR
        )
        heatmap = np.array(heatmap)

        # 6. overlay
        ax.imshow(image)
        ax.imshow(heatmap, cmap="jet", alpha=alpha)
        ax.set_title(f"Layer {layer_idx}", fontsize=13)
        ax.axis("off")

    plt.suptitle(
        f"Text→Image Attention  |  heads={head_reduce}  tokens={token_reduce}",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"[INFO] Saved to {save_path}")

    plt.show()