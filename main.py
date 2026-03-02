import os, math
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from qvtree import QVTree


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

@torch.no_grad()
def get_clip_tokens_and_query_aligned(image_path, query_text, device,
                                      model_name="openai/clip-vit-large-patch14"):
    from transformers import CLIPProcessor, CLIPModel
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()

    img = Image.open(image_path).convert("RGB")
    inputs = processor(text=[query_text], images=[img], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    vision_out = model.vision_model(pixel_values=inputs["pixel_values"])
    v_hidden = vision_out[0]            # [B,1+N,Hv]
    v_patches = v_hidden[:, 1:, :]      # [B,N,Hv]
    x = model.visual_projection(v_patches)  # [B,N,Dp]

    text_out = model.text_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
    t_cls = text_out[0][:, 0, :]
    q = model.text_projection(t_cls)    # [B,Dp]

    B, N, D = x.shape
    H = int(math.isqrt(N))
    if H * H != N:
        raise ValueError(f"N={N} not square.")
    return x, q, H, H, img


# --------------------------
# A) overlay selected patches on original image
# --------------------------
def overlay_selected_patches_on_image(img: Image.Image, H: int, W: int, selected_idx: torch.Tensor,
                                      save_path: str, alpha: float = 0.35):
    """
    img: PIL image (the one fed into CLIPProcessor; already resized internally by processor,
         but we overlay on the loaded original img here. For best alignment, we resize img to 224.)
    selected_idx: 1D tensor of token indices in [0, H*W)
    """
    # CLIPProcessor uses 224x224 by default; patch size for patch14 is 14px
    # We'll resize the image to 224x224 for correct patch overlay.
    img224 = img.resize((224, 224))
    overlay = Image.new("RGBA", img224.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    patch_w = img224.size[0] / W
    patch_h = img224.size[1] / H

    idx_list = selected_idx.detach().cpu().tolist()
    for idx in idx_list:
        r = idx // W
        c = idx % W
        x0 = int(c * patch_w)
        y0 = int(r * patch_h)
        x1 = int((c + 1) * patch_w)
        y1 = int((r + 1) * patch_h)
        draw.rectangle([x0, y0, x1, y1], fill=(255, 0, 0, int(255 * alpha)), outline=(255, 0, 0, 255))

    out = Image.alpha_composite(img224.convert("RGBA"), overlay)
    out.save(save_path)


# --------------------------
# B) draw only visited subtree (readable)
# --------------------------
def build_parent_map(nodes):
    parent = {}
    children = {}
    for n in nodes:
        parent[n.node_id] = n.parent
        children[n.node_id] = list(n.children) if n.children is not None else []
    return parent, children


def subtree_layout(nodes, kept_ids):
    # layout by level only for kept ids
    by_level = {}
    for nid in kept_ids:
        lvl = nodes[nid].level
        by_level.setdefault(lvl, []).append(nid)
    pos = {}
    for lvl in sorted(by_level.keys()):
        ids = sorted(by_level[lvl])
        for i, nid in enumerate(ids):
            pos[nid] = (i, -lvl)
    return pos


def plot_subtree(nodes, visited_ids, selected_ids, save_path: str, max_label_nodes: int = 80):
    """
    visited_ids: nodes that were popped from Q (actually evaluated)
    selected_ids: final S
    """
    kept = set(visited_ids) | set(selected_ids)  # show visited + selected
    parent, children = build_parent_map(nodes)

    # only keep edges inside kept
    edges = []
    for u in kept:
        for v in children[u]:
            if v in kept:
                edges.append((u, v))

    pos = subtree_layout(nodes, kept)

    # if still too many nodes, label only top-K by area (readable)
    areas = {nid: nodes[nid].region.area for nid in kept}
    label_ids = set(sorted(kept, key=lambda k: areas[k], reverse=True)[:max_label_nodes])
    label_ids |= set(selected_ids)  # always label selected nodes

    plt.figure(figsize=(14, 8))
    # edges
    for (u, v) in edges:
        x1, y1 = pos[u]; x2, y2 = pos[v]
        plt.plot([x1, x2], [y1, y2], linewidth=0.8, alpha=0.6)

    # nodes
    sel_set = set(selected_ids)
    for nid in kept:
        x, y = pos[nid]
        is_sel = nid in sel_set
        plt.scatter([x], [y], s=160 if is_sel else 70)
        if nid in label_ids:
            a = nodes[nid].region.area
            plt.text(x, y, f"{nid}\nA={a}", fontsize=7, ha="center", va="center",
                     color="white" if is_sel else "black")

    plt.title("Visited subtree (only nodes evaluated in Algorithm 1). Selected nodes highlighted.")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    device = get_device()
    print("Using device:", device)

    IMAGE_PATH = "./demo.jpg"
    QUERY_TEXT = "how many flags are in the photo?"

    os.makedirs("output", exist_ok=True)

    x, q, H, W, img = get_clip_tokens_and_query_aligned(IMAGE_PATH, QUERY_TEXT, device)
    print(f"[Input] x={x.shape} q={q.shape} H=W={H}")

    mod = QVTree(D=x.shape[-1], Dq=q.shape[-1]).to(device)
    out = mod(x, q)

    # ---- sanity print ----
    print("num_nodes:", out["num_nodes"])
    print("selected_nodes:", [len(s) for s in out["selected_node_ids"]])
    print("selected_feats_padded:", out["selected_feats_padded"].shape)
    print("mask true counts:", out["selected_mask"].sum(dim=1).tolist())

    nodes = out["nodes"]

    # Per-sample visualize
    for b in range(len(out["selected_node_ids"])):
        sel_idx = out["selected_token_indices"][b]
        overlay_path = f"output/overlay_sample{b}.png"
        overlay_selected_patches_on_image(img, H, W, sel_idx, overlay_path)
        print("Saved overlay:", overlay_path)

        # subtree
        if "visited_node_ids" in out:
            visited = out["visited_node_ids"][b]
        else:
            # fallback: if you didn't add visited, at least show selected nodes only (still readable)
            visited = out["selected_node_ids"][b]

        subtree_path = f"output/subtree_sample{b}.png"
        plot_subtree(nodes, visited, out["selected_node_ids"][b], subtree_path)
        print("Saved subtree:", subtree_path)

        # accounting
        areas_sum = sum(int(nodes[nid].region.area) for nid in out["selected_node_ids"][b])
        uniq_tokens = int(sel_idx.numel())
        print(f"[Sample {b}] sum(area)={areas_sum}, unique_tokens={uniq_tokens}")

    print("Done.")


if __name__ == "__main__":
    main()