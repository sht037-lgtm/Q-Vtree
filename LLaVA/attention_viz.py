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
            "text_to_image" : list of [num_heads, N_text, 576] tensors per layer
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

    attentions = outputs.attentions
    # each element shape: [batch, num_heads, seq_len, seq_len]

    num_image_tokens = 576  # 24x24 from CLIP ViT-L/14@336

    # debug: print sequence shape
    seq_len = attentions[0].shape[2]
    n_text  = seq_len - num_image_tokens
    print(f"[DEBUG] seq_len={seq_len}, num_image_tokens={num_image_tokens}, n_text_tokens={n_text}")

    if layers is None:
        layers = [15, 23, 31]

    text_to_image = []
    for l in layers:
        attn = attentions[l][0]        # [num_heads, seq_len, seq_len]

        # cut text→image block: full 576 image tokens, no special treatment
        t2i = attn[:, num_image_tokens:, :num_image_tokens]   # [num_heads, N_text, 576]

        text_to_image.append(t2i.cpu().float())

    return {
        "text_to_image" : text_to_image,
        "layers"        : layers,
        "prompt"        : prompt,
    }


# =========================================================
# ------------- Rater Filtering (from Qwen model.py) ------
# =========================================================

def apply_rater_filter(attn_map: torch.Tensor) -> torch.Tensor:
    """
    Filter text tokens by their mean visual attention (rater selection).
    Only keep tokens whose mean visual attention >= global mean.
    This removes noise from irrelevant tokens like '(A)', 'directly', etc.

    Args:
        attn_map : [N_text, N_patches] — already head-aggregated

    Returns:
        [N_patches] — patch scores averaged over rater tokens only
    """
    r = attn_map.mean(dim=1)           # [N_text]
    rater_mask = r >= r.mean()

    if not rater_mask.any():
        rater_mask = torch.ones_like(rater_mask, dtype=torch.bool)

    patch_scores = attn_map[rater_mask].mean(dim=0)   # [N_patches]
    return patch_scores


# =========================================================
# ------------- Attention Map Visualization ---------------
# =========================================================

def visualize_attention(
    image: Image.Image,
    attn_result: dict,
    head_reduce: str = "mean",
    token_reduce: str = "rater",
    token_idx: int | None = None,
    alpha: float = 0.5,
    save_path: str | None = None,
):
    """
    Visualize text→image attention maps overlaid on the original image.

    Args:
        image        : original PIL.Image
        attn_result  : output from get_attention_maps()
        head_reduce  : aggregate heads by "mean" or "max"
        token_reduce : how to aggregate text tokens:
                         "rater" — filter by visual attention first (recommended)
                         "mean"  — simple average over all text tokens
                         "max"   — max over all text tokens
        token_idx    : if set, ignore token_reduce and use this specific token index
        alpha        : heatmap overlay transparency
        save_path    : optional path to save the figure
    """
    layers        = attn_result["layers"]
    text_to_image = attn_result["text_to_image"]  # list of [num_heads, N_text, 576]

    fig, axes = plt.subplots(1, len(layers), figsize=(5 * len(layers), 5))
    if len(layers) == 1:
        axes = [axes]

    for ax, layer_idx, t2i in zip(axes, layers, text_to_image):
        # t2i: [num_heads, N_text, 576]

        # 1. aggregate over heads
        if head_reduce == "mean":
            attn_map = t2i.mean(dim=0)         # [N_text, 576]
        else:
            attn_map = t2i.max(dim=0).values

        # 2. aggregate over text tokens
        if token_idx is not None:
            attn_map = attn_map[token_idx]     # [576]
        elif token_reduce == "rater":
            attn_map = apply_rater_filter(attn_map)   # [576]
        elif token_reduce == "mean":
            attn_map = attn_map.mean(dim=0)
        else:
            attn_map = attn_map.max(dim=0).values

        # 3. reshape to 24x24
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

        mode_label = f"token={token_idx}" if token_idx is not None else token_reduce
        ax.set_title(f"Layer {layer_idx}  [{mode_label}]", fontsize=12)
        ax.axis("off")

    plt.suptitle(
        f"Text→Image Attention  |  heads={head_reduce}",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"[INFO] Saved to {save_path}")

    plt.show()