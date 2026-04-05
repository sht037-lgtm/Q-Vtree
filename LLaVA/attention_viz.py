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

    Instead of using prefill (question→image) attention, we generate
    the first answer token and use ITS attention to image tokens.
    This reflects where the model looks when making its decision.

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
            "answer_to_image" : list of [num_heads, 576] tensors per layer
                                answer token's attention to each image patch
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

    num_image_tokens = 576  # 24x24 from CLIP ViT-L/14@336

    if layers is None:
        layers = [15, 23, 31]

    # generate only 1 token (the first answer token)
    # output_attentions=True returns attentions for the generated token
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            output_attentions=True,
            return_dict_in_generate=True,
            do_sample=False,
        )

    # outputs.attentions is a tuple of length max_new_tokens (=1)
    # outputs.attentions[0] = attentions for the 1st generated token
    # each element: tuple of L tensors, each [batch, H, 1, seq_len]
    gen_attentions = outputs.attentions[0]
    # gen_attentions: tuple of 32 tensors (one per layer)
    # each tensor shape: [batch, H, 1, seq_len]
    #   - 1      = the single generated answer token
    #   - seq_len = full input sequence length (image + question tokens)

    # decode the generated answer token for display
    answer_token = processor.batch_decode(
        outputs.sequences[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )[0].strip()

    seq_len = gen_attentions[0].shape[-1]
    n_text  = seq_len - num_image_tokens
    print(f"[DEBUG] seq_len={seq_len}, num_image_tokens={num_image_tokens}, n_text_tokens={n_text}")
    print(f"[DEBUG] answer token = '{answer_token}'")

    answer_to_image = []
    for l in layers:
        # shape: [batch, H, 1, seq_len]
        attn = gen_attentions[l][0]    # [H, 1, seq_len]

        # take answer token's attention to image tokens only
        a2i = attn[:, 0, :num_image_tokens]   # [H, 576]

        # suppress attention sink at position 0
        a2i = a2i.clone()
        a2i[:, 0] = 0.0

        print(f"[DEBUG] Layer {l}: a2i shape = {a2i.shape}")

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
    answer_to_image = attn_result["answer_to_image"]  # list of [H, 576]
    answer_token    = attn_result.get("answer_token", "?")

    fig, axes = plt.subplots(1, len(layers), figsize=(5 * len(layers), 5))
    if len(layers) == 1:
        axes = [axes]

    for ax, layer_idx, a2i in zip(axes, layers, answer_to_image):
        # a2i: [H, 576]

        # 1. aggregate over heads
        if head_reduce == "mean":
            attn_map = a2i.mean(dim=0)         # [576]
        else:
            attn_map = a2i.max(dim=0).values   # [576]

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