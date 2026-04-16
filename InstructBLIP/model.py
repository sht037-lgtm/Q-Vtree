"""
InstructBLIP (Vicuna-7B) + QVTree visual token selection.

Architecture differences from InternVL:
  - Image encoder: BLIP-2 ViT-g/14  (patch grid: 16x16 = 256 patches for 224px,
                                       or 24x24 = 576 patches for 336px)
  - Connector:     Q-Former (32 learned query tokens) → linear proj → LLM
  - LLM:           Vicuna-7B
  - No tile / pixel_shuffle logic

QVTree is applied to the raw ViT patch tokens (before Q-Former compression),
so the spatial structure is preserved for tree navigation.
Scoring uses cross-attention between the Q-Former query tokens and the ViT
patch tokens as a proxy for "which patches are informative".
"""

from __future__ import annotations

import os
import math
import tempfile
from typing import Optional, Tuple, List

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

from module import QVTree

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
# InstructBLIP ships with ViT-g/14 @ 224 px  → 16×16 patch grid
# (some configs use 336 px → 24×24; we detect at runtime)
DEFAULT_IMAGE_SIZE = 224
PATCH_SIZE_PIX = 14        # ViT-g/14
GENERIC_PROMPT = "Write a general description of the image."


# ─────────────────────────────────────────────
# LPD helpers  (identical logic, kept self-contained)
# ─────────────────────────────────────────────
def _patch_ids_to_bboxes(patch_ids: torch.Tensor, grid_w: int, patch_size: int) -> list:
    bboxes = []
    for idx in patch_ids.tolist():
        r, c = int(idx) // grid_w, int(idx) % grid_w
        x0, y0 = c * patch_size, r * patch_size
        bboxes.append((x0, y0, x0 + patch_size, y0 + patch_size))
    return bboxes


def _merge_bboxes(bboxes: list) -> list:
    if not bboxes:
        return []

    def overlaps_or_adjacent(a, b):
        return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])

    merged, changed = list(bboxes), True
    while changed:
        changed = False
        result = []
        used = [False] * len(merged)
        for i in range(len(merged)):
            if used[i]:
                continue
            cur = list(merged[i])
            for j in range(i + 1, len(merged)):
                if not used[j] and overlaps_or_adjacent(cur, merged[j]):
                    cur[0] = min(cur[0], merged[j][0])
                    cur[1] = min(cur[1], merged[j][1])
                    cur[2] = max(cur[2], merged[j][2])
                    cur[3] = max(cur[3], merged[j][3])
                    used[j] = True
                    changed = True
            result.append(tuple(cur))
        merged = result
    return merged


def _build_compact_image(image: Image.Image, bboxes: list) -> Image.Image:
    if not bboxes:
        return image

    img_w, img_h = image.size
    x_coords = sorted(set([0, img_w] + [x for b in bboxes for x in (b[0], b[2])]))
    y_coords = sorted(set([0, img_h] + [y for b in bboxes for y in (b[1], b[3])]))

    def cell_has_content(x0, y0, x1, y1):
        return any(x0 < b[2] and x1 > b[0] and y0 < b[3] and y1 > b[1] for b in bboxes)

    new_w, x_map = 0, {}
    for i in range(len(x_coords) - 1):
        if any(cell_has_content(x_coords[i], y_coords[j], x_coords[i + 1], y_coords[j + 1])
               for j in range(len(y_coords) - 1)):
            x_map[i] = new_w
            new_w += x_coords[i + 1] - x_coords[i]

    new_h, y_map = 0, {}
    for j in range(len(y_coords) - 1):
        if any(cell_has_content(x_coords[i], y_coords[j], x_coords[i + 1], y_coords[j + 1])
               for i in range(len(x_coords) - 1)):
            y_map[j] = new_h
            new_h += y_coords[j + 1] - y_coords[j]

    if new_w == 0 or new_h == 0:
        return image

    compact = Image.new("RGB", (new_w, new_h), color=(255, 255, 255))
    for i in range(len(x_coords) - 1):
        for j in range(len(y_coords) - 1):
            if cell_has_content(x_coords[i], y_coords[j], x_coords[i + 1], y_coords[j + 1]):
                if i not in x_map or j not in y_map:
                    continue
                patch = image.crop((x_coords[i], y_coords[j], x_coords[i + 1], y_coords[j + 1]))
                compact.paste(patch, (x_map[i], y_map[j]))
    return compact


# ─────────────────────────────────────────────
# Main model wrapper
# ─────────────────────────────────────────────
class InstructBLIPWithTree:
    """
    Wraps InstructBlipForConditionalGeneration and adds QVTree-based
    visual token selection + LPD compact-image re-encoding.

    Usage
    -----
    model = InstructBLIPWithTree.from_pretrained("path/to/instructblip-vicuna-7b")
    response = model.infer(image_path="img.jpg", question="What is in the image?")
    """

    def __init__(self, hf_model: InstructBlipForConditionalGeneration,
                 processor: InstructBlipProcessor,
                 split_threshold: float = 0.3,
                 softmax_temperature: float = 0.2):

        self.model = hf_model
        self.processor = processor

        # derive hidden dim from LLM config
        llm_hidden = hf_model.config.text_config.hidden_size
        self.qvtree = QVTree(
            D=llm_hidden,
            split_threshold=split_threshold,
            softmax_temperature=softmax_temperature,
        )

        # ── debug state (mirrors InternVL wrapper) ──
        self._debug_patch_ids: Optional[torch.Tensor] = None
        self._debug_patch_scores: Optional[torch.Tensor] = None
        self._debug_num_selected_tokens: Optional[List[int]] = None
        self._debug_num_total_tokens: Optional[List[int]] = None
        self._debug_select_ratios: Optional[List[float]] = None
        self._debug_full_score_map: Optional[torch.Tensor] = None
        self._debug_compact_image: Optional[Image.Image] = None
        self._debug_merged_bboxes: Optional[list] = None
        self._debug_global_patch_ids: Optional[torch.Tensor] = None

    # ── factory ──────────────────────────────
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        torch_dtype=torch.float16,
        device_map: str = "auto",
        split_threshold: float = 0.3,
        softmax_temperature: float = 0.2,
    ) -> "InstructBLIPWithTree":
        processor = InstructBlipProcessor.from_pretrained(model_path)
        model = InstructBlipForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        model.eval()
        return cls(model, processor,
                   split_threshold=split_threshold,
                   softmax_temperature=softmax_temperature)

    # ── internal helpers ──────────────────────
    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    def _get_patch_grid(self) -> Tuple[int, int]:
        """Return (grid_h, grid_w) of the ViT patch grid."""
        img_size = getattr(self.model.config.vision_config, "image_size", DEFAULT_IMAGE_SIZE)
        patch_size = getattr(self.model.config.vision_config, "patch_size", PATCH_SIZE_PIX)
        g = img_size // patch_size
        return g, g   # square grid

    def _encode_image(self, image: Image.Image) -> torch.Tensor:
        """
        Run ViT and return raw patch tokens (before Q-Former), shape [1, N_patches, D_vit].
        We hook into vision_model to get the last hidden state excluding the CLS token.
        """
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device=self.device, dtype=self.dtype)

        with torch.no_grad():
            vit_out = self.model.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True,
            )
        # last_hidden_state: [1, 1 + N_patches, D_vit]  (index 0 = CLS)
        patch_tokens = vit_out.last_hidden_state[:, 1:, :]   # [1, N, D_vit]
        return patch_tokens

    def _score_patches(self, patch_tokens: torch.Tensor, question: str) -> torch.Tensor:
        """
        Compute per-patch importance scores using Q-Former cross-attention.

        Strategy:
          1. Run the Q-Former with the image tokens → get cross-attention weights
             between the 32 learned query tokens and the N_patches ViT tokens.
          2. Average over heads and query positions → [N_patches] score vector.
          3. Compute a "relative" score vs. a generic prompt (same as InternVL).

        Returns normalised scores in [0, 1], shape [N_patches].
        """
        device = self.device
        dtype = self.dtype
        N = patch_tokens.shape[1]

        # ── Q-Former cross-attention scores ──
        # The Q-Former in BLIP-2/InstructBLIP produces cross-attn at every layer.
        # We use the last layer's cross-attention (most semantically meaningful).
        with torch.no_grad():
            qformer_out = self.model.qformer(
                query_embeds=self.model.query_tokens.expand(1, -1, -1),
                encoder_hidden_states=patch_tokens,
                encoder_attention_mask=torch.ones(1, N, device=device),
                output_attentions=True,
                return_dict=True,
            )

        # cross_attentions: list of [1, heads, n_queries, N_patches] per layer
        # use last layer
        cross_attn = qformer_out.cross_attentions[-1]   # [1, heads, Q, N]
        # mean over heads and query positions → [N]
        attn_q = cross_attn[0].mean(0).mean(0).cpu().float()   # [N]

        # ── generic-prompt baseline ──
        with torch.no_grad():
            generic_out = self.model.qformer(
                query_embeds=self.model.query_tokens.expand(1, -1, -1),
                encoder_hidden_states=patch_tokens,
                encoder_attention_mask=torch.ones(1, N, device=device),
                output_attentions=True,
                return_dict=True,
            )
        cross_attn_g = generic_out.cross_attentions[-1]
        attn_g = cross_attn_g[0].mean(0).mean(0).cpu().float()

        # relative score (same formula as InternVL)
        scores = attn_q / (attn_g + 1e-8)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)
        return scores   # [N]

    def _smooth_scores(self, scores: torch.Tensor, grid_h: int, grid_w: int,
                       sigma: float = 1.0, ks: int = 3) -> torch.Tensor:
        """Gaussian smooth the 2-D score map and return flattened [N] tensor."""
        score_map = scores.view(1, 1, grid_h, grid_w)
        ax = torch.arange(ks, dtype=torch.float32) - ks // 2
        g1d = torch.exp(-ax ** 2 / (2 * sigma ** 2))
        g1d = g1d / g1d.sum()
        kernel = (g1d.unsqueeze(1) * g1d.unsqueeze(0)).view(1, 1, ks, ks)
        smoothed = F.conv2d(score_map, kernel, padding=ks // 2).squeeze()
        smoothed = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min() + 1e-6)
        self._debug_full_score_map = smoothed
        return smoothed.reshape(-1)

    def _select_patches(self, patch_tokens: torch.Tensor, scores: torch.Tensor,
                        grid_h: int, grid_w: int) -> torch.Tensor:
        """Run QVTree on the patch token grid and return selected patch indices."""
        device = patch_tokens.device
        x = patch_tokens.float()   # [1, N, D_vit]

        ps = scores.unsqueeze(0).to(device=device, dtype=x.dtype)  # [1, N]

        built = self.qvtree.builder.build(x, grid_h, grid_w)
        nodes = built["nodes"]
        sel_nodes, _ = self.qvtree.navigator.select_nodes(
            nodes=nodes, patch_scores=ps, W=grid_w,
        )
        token_out = self.qvtree.navigator.nodes_to_tokens(
            nodes, H=grid_h, W=grid_w,
            selected_node_ids=sel_nodes, x=x,
        )
        patch_ids = token_out["selected_token_indices"][0]
        if patch_ids.numel() == 0:
            patch_ids = torch.arange(grid_h * grid_w, device=device)
        patch_ids = torch.unique(patch_ids.clamp(0, grid_h * grid_w - 1))
        return patch_ids

    # ── public inference entry point ─────────
    @torch.no_grad()
    def infer(
        self,
        image_path: str,
        question: str,
        max_new_tokens: int = 256,
        do_sample: bool = False,
        use_tree: bool = True,
        use_lpd: bool = True,
    ) -> str:
        """
        Parameters
        ----------
        image_path      : path to the input image
        question        : free-form question string
        max_new_tokens  : generation budget
        do_sample       : greedy (False) or sampling (True)
        use_tree        : if False, runs standard InstructBLIP inference
        use_lpd         : if True (and use_tree), applies LPD compact-image pass
        """
        image = Image.open(image_path).convert("RGB")
        grid_h, grid_w = self._get_patch_grid()
        orig_w, orig_h = image.size

        # ── BASELINE MODE ────────────────────────────────────────────
        if not use_tree:
            inputs = self.processor(
                images=image,
                text=question,
                return_tensors="pt",
            ).to(device=self.device, dtype=self.dtype)
            # input_ids should stay long
            inputs["input_ids"] = inputs["input_ids"].to(torch.long)
            if "attention_mask" in inputs:
                inputs["attention_mask"] = inputs["attention_mask"].to(torch.long)

            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
            )
            return self.processor.tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()

        # ── TREE MODE ────────────────────────────────────────────────

        # 1. Extract raw ViT patch tokens
        patch_tokens = self._encode_image(image)   # [1, N, D_vit]
        N = patch_tokens.shape[1]
        assert N == grid_h * grid_w, f"Patch count mismatch: {N} vs {grid_h}×{grid_w}"

        # 2. Score patches
        scores = self._score_patches(patch_tokens, question)          # [N]
        smooth_scores = self._smooth_scores(scores, grid_h, grid_w)   # [N]

        # 3. QVTree selection
        patch_ids = self._select_patches(patch_tokens, smooth_scores, grid_h, grid_w)

        # store debug
        self._debug_patch_ids = patch_ids
        self._debug_patch_scores = smooth_scores
        n_sel = int(patch_ids.numel())
        n_tot = grid_h * grid_w
        self._debug_num_selected_tokens = [n_sel]
        self._debug_num_total_tokens = [n_tot]
        self._debug_select_ratios = [n_sel / n_tot if n_tot > 0 else 0.0]
        print(f"[TREE] selected tokens: {n_sel}, original: {n_tot}, ratio: {n_sel / n_tot:.1%}")

        if not use_lpd:
            # tree scoring done but no compact image – fall back to full image inference
            inputs = self.processor(
                images=image,
                text=question,
                return_tensors="pt",
            ).to(device=self.device, dtype=self.dtype)
            inputs["input_ids"] = inputs["input_ids"].to(torch.long)
            if "attention_mask" in inputs:
                inputs["attention_mask"] = inputs["attention_mask"].to(torch.long)
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
            )
            return self.processor.tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()

        # 4. LPD: map selected patches → pixel bboxes on original image
        patch_pixel_w = orig_w / grid_w
        patch_pixel_h = orig_h / grid_h
        raw_bboxes = []
        for idx in patch_ids.tolist():
            r, c = int(idx) // grid_w, int(idx) % grid_w
            x0 = int(c * patch_pixel_w)
            y0 = int(r * patch_pixel_h)
            x1 = int((c + 1) * patch_pixel_w)
            y1 = int((r + 1) * patch_pixel_h)
            raw_bboxes.append((x0, y0, x1, y1))

        merged_bboxes = _merge_bboxes(raw_bboxes)
        compact_image = _build_compact_image(image.convert("RGB"), merged_bboxes)

        # scale compact image up to original resolution
        cw, ch = compact_image.size
        if cw > 0 and ch > 0:
            scale = min(orig_w / cw, orig_h / ch)
            compact_image = compact_image.resize(
                (int(cw * scale), int(ch * scale)), Image.BILINEAR
            )

        self._debug_compact_image = compact_image
        self._debug_merged_bboxes = merged_bboxes
        self._debug_global_patch_ids = patch_ids

        # 5. Pass 2: inference on compact image
        inputs2 = self.processor(
            images=compact_image,
            text=question,
            return_tensors="pt",
        ).to(device=self.device, dtype=self.dtype)
        inputs2["input_ids"] = inputs2["input_ids"].to(torch.long)
        if "attention_mask" in inputs2:
            inputs2["attention_mask"] = inputs2["attention_mask"].to(torch.long)

        out_ids = self.model.generate(
            **inputs2,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )
        return self.processor.tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
