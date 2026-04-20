import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image

from transformers import AutoTokenizer, GenerationConfig

from ..src.modeling_internvl_chat import InternVLChatModel
from ..src.configuration_internvl_chat import InternVLChatConfig
from ..src.conversation import get_conv_template
from module import QVTree

# =========================================================
# Image loading utilities
# =========================================================
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
TILE_SIZE = 448  # InternVL tile size in pixels
PATCH_SIZE = 28  # effective patch size after pixel_shuffle (448/16)
GRID_SIZE = 16  # patches per tile side (448/28)
GENERIC_PROMPT = "Write a general description of the image."


def build_transform(input_size=448):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    best_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * best_ratio[0]
    target_height = image_size * best_ratio[1]
    blocks = best_ratio[0] * best_ratio[1]

    resized = image.resize((target_width, target_height))
    tiles = []
    for i in range(blocks):
        col = i % best_ratio[0]
        row = i // best_ratio[0]
        box = (
            col * image_size, row * image_size,
            (col + 1) * image_size, (row + 1) * image_size,
        )
        tiles.append(resized.crop(box))

    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))

    return tiles, best_ratio


def load_image(image_path, input_size=448, max_num=6):
    image = Image.open(image_path).convert('RGB')
    transform = build_transform(input_size)
    tiles, best_ratio = dynamic_preprocess(
        image, min_num=1, max_num=max_num,
        image_size=input_size, use_thumbnail=True,
    )
    pixel_values = torch.stack([transform(t) for t in tiles])
    return pixel_values, best_ratio


# =========================================================
# LPD helpers
# =========================================================
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


def run_lpd_on_tile(tile_pil: Image.Image, patch_ids: torch.Tensor,
                    grid_h: int = GRID_SIZE, grid_w: int = GRID_SIZE,
                    patch_size: int = PATCH_SIZE) -> Tuple[Image.Image, list]:
    """Apply LPD to a single tile PIL image."""
    # ensure tile is at the right size
    tile = tile_pil.convert("RGB").resize((grid_w * patch_size, grid_h * patch_size), Image.BILINEAR)
    raw_bboxes = _patch_ids_to_bboxes(patch_ids.cpu(), grid_w, patch_size)
    merged_bboxes = _merge_bboxes(raw_bboxes)
    compact = _build_compact_image(tile, merged_bboxes)
    return compact, merged_bboxes


# =========================================================
# Model
# =========================================================
class InternVLChatModelWithTree(InternVLChatModel):

    def __init__(self, config: InternVLChatConfig, vision_model=None, language_model=None, use_flash_attn=False):
        super().__init__(config, vision_model=vision_model, language_model=language_model,
                         use_flash_attn=use_flash_attn)

        self.qvtree = QVTree(D=config.llm_config.hidden_size)
        self._generic_token_ids = None

        # debug states
        self._debug_selected_idx = None
        self._debug_patch_ids = None
        self._debug_num_selected_tokens = None
        self._debug_num_total_tokens = None
        self._debug_select_ratios = None
        self._debug_patch_scores = None
        self._debug_best_ratio = None
        self._debug_full_score_map = None
        self._debug_compact_tiles = None  # list of compact PIL images per tile
        self._debug_merged_bboxes = None  # list of merged bboxes per tile
        self._debug_layer_score_maps = {}  # {layer_idx: full_smooth_map} for ablation
        self._debug_thumbnail_score_map = None  # [grid_h, grid_w] thumbnail attention
        self.all_tied_weights_keys = {}

    # ------------------------------------------------------------------
    # extract_feature with grid shape
    # ------------------------------------------------------------------
    def extract_feature_with_grid(self, pixel_values):
        if self.select_layer == -1:
            raw = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True,
            ).last_hidden_state
        else:
            raw = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True,
            ).hidden_states[self.select_layer]

        raw = raw[:, 1:, :]  # drop CLS token
        h = w = int(raw.shape[1] ** 0.5)
        x = raw.reshape(raw.shape[0], h, w, -1)
        x = self.pixel_shuffle(x, scale_factor=self.downsample_ratio)
        grid_h = x.shape[1]
        grid_w = x.shape[2]
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        x = self.mlp1(x)
        return x, grid_h, grid_w

    # ------------------------------------------------------------------
    # LLM attention scorer
    # ------------------------------------------------------------------
    # ablation layers to probe
    ABLATION_LAYERS = [3, 7, 11, 15, 19, 23, 27]

    def _run_lm_forward(self, input_embeds, attention_mask):
        """Single LLM forward pass with all attentions; result is cached per call."""
        with torch.no_grad():
            out = self.language_model(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                output_attentions=True,
                output_hidden_states=False,
                return_dict=True,
                use_cache=False,
            )
        return out

    def _extract_layer_attn(self, lm_out, layer_idx, img_positions, text_positions):
        """Extract mean text→vision attention for a single layer. Returns [N_img]."""
        layer_attn = lm_out.attentions[layer_idx]   # [1, heads, seq, seq]
        vp = img_positions.to(layer_attn.device)
        qp = text_positions.to(layer_attn.device)
        attn = layer_attn[0, :, :, vp][:, qp, :]    # [heads, Q, N_img]
        return attn.mean(0).mean(0).cpu().float()    # [N_img]

    def _get_attn_scores(self, input_embeds, attention_mask, img_positions, text_positions):
        """Original scoring path: mean of layers [20, 27]. Kept for non-ablation use."""
        out = self._run_lm_forward(input_embeds, attention_mask)
        scores_list = [
            self._extract_layer_attn(out, i, img_positions, text_positions)
            for i in [20, 27]
        ]
        return torch.stack(scores_list).mean(dim=0)

    # ------------------------------------------------------------------
    # Scoring → QVTree selection → per-tile patch_ids
    # ------------------------------------------------------------------
    def _score_and_select(self, vit_embeds, input_embeds_full, attention_mask,
                          input_ids, grid_h, grid_w, best_ratio):
        B_tiles, N_per_tile, D = vit_embeds.shape
        ratio_w, ratio_h = best_ratio
        num_content_tiles = ratio_w * ratio_h
        has_thumbnail = B_tiles > num_content_tiles
        device = vit_embeds.device
        dtype = vit_embeds.dtype

        # init debug
        self._debug_patch_ids = []
        self._debug_num_selected_tokens = []
        self._debug_num_total_tokens = []
        self._debug_select_ratios = []
        self._debug_best_ratio = best_ratio

        # ---- build shared inputs for question and generic prompts ----
        ids_flat = input_ids.reshape(-1)
        sel_mask = (ids_flat == self.img_context_token_id)
        vis_positions_all = sel_mask.nonzero(as_tuple=True)[0].cpu()

        n_content_tokens = num_content_tiles * N_per_tile
        vis_positions = vis_positions_all[:n_content_tokens]

        img_end = vis_positions_all[-1].item()
        seq_len = input_ids.shape[1]
        text_positions = torch.arange(img_end + 1, seq_len, dtype=torch.long)

        # question-prompt LLM forward (single pass, reused across all layers)
        lm_out_q = self._run_lm_forward(input_embeds_full, attention_mask)

        # generic-prompt LLM forward (single pass, reused across all layers)
        if self._generic_token_ids is None:
            _tok = AutoTokenizer.from_pretrained(self.config._name_or_path, trust_remote_code=True)
            self._generic_token_ids = _tok(
                GENERIC_PROMPT, return_tensors="pt", add_special_tokens=False,
            ).input_ids

        generic_ids = self._generic_token_ids.to(input_ids.device)
        img_end_all = vis_positions_all[-1].item()
        new_ids = torch.cat([input_ids[:, :img_end_all + 1], generic_ids, input_ids[:, -3:]], dim=1)

        gen_emb = self.language_model.get_input_embeddings()(new_ids)
        gflat = gen_emb.reshape(-1, D)
        gen_sel = (new_ids.reshape(-1) == self.img_context_token_id)
        gflat[gen_sel] = vit_embeds.reshape(-1, D).to(device=gflat.device, dtype=gflat.dtype)
        gen_emb = torch.nan_to_num(gflat.reshape(1, -1, D))
        gen_mask = torch.ones(1, new_ids.shape[1], dtype=attention_mask.dtype, device=device)
        generic_text_positions = torch.arange(
            img_end_all + 1, img_end_all + 1 + generic_ids.shape[1], dtype=torch.long
        )
        lm_out_g = self._run_lm_forward(gen_emb, gen_mask)

        # ---- gaussian kernel (shared) ----
        sigma, ks = 1.0, 3
        ax = torch.arange(ks, dtype=torch.float32) - ks // 2
        g1d = torch.exp(-ax ** 2 / (2 * sigma ** 2))
        g1d = g1d / g1d.sum()
        kernel = (g1d.unsqueeze(1) * g1d.unsqueeze(0)).view(1, 1, ks, ks)

        def _stitch_and_smooth(scores_flat):
            """Stitch per-tile scores into global map, gaussian-smooth, return (smooth_map, smoothed_tiles)."""
            scores_tiled = scores_flat.view(num_content_tiles, grid_h, grid_w)
            rows = []
            for row in range(ratio_h):
                cols = [scores_tiled[row * ratio_w + col] for col in range(ratio_w)]
                rows.append(torch.cat(cols, dim=1))
            full_map = torch.cat(rows, dim=0).float()

            smooth = F.conv2d(full_map.unsqueeze(0).unsqueeze(0), kernel, padding=ks // 2).squeeze()
            smooth = (smooth - smooth.min()) / (smooth.max() - smooth.min() + 1e-6)

            per_tile = []
            for row in range(ratio_h):
                for col in range(ratio_w):
                    per_tile.append(
                        smooth[row * grid_h:(row + 1) * grid_h,
                               col * grid_w:(col + 1) * grid_w].reshape(-1)
                    )
            if has_thumbnail:
                thumb_raw = scores_tiled[-1].float().unsqueeze(0).unsqueeze(0)
                thumb_s = F.conv2d(thumb_raw, kernel, padding=ks // 2).view(-1)
                thumb_s = (thumb_s - thumb_s.min()) / (thumb_s.max() - thumb_s.min() + 1e-6)
                per_tile.append(thumb_s)
            return smooth, per_tile

        # ---- per-layer relative attention + gaussian smoothing ----
        # _debug_layer_score_maps: dict {layer_idx: full_smooth_map [H_global, W_global]}
        self._debug_layer_score_maps = {}

        layer_smoothed_tiles = {}   # layer_idx -> per-tile smoothed scores
        for layer_idx in self.ABLATION_LAYERS:
            A_q_l      = self._extract_layer_attn(lm_out_q, layer_idx, vis_positions, text_positions)
            A_gen_l    = self._extract_layer_attn(lm_out_g, layer_idx, vis_positions, generic_text_positions)
            scores_rel = A_q_l / (A_gen_l + 1e-8)
            scores_rel = (scores_rel - scores_rel.min()) / (scores_rel.max() - scores_rel.min() + 1e-6)

            smooth_map, per_tile = _stitch_and_smooth(scores_rel)
            self._debug_layer_score_maps[layer_idx] = smooth_map
            layer_smoothed_tiles[layer_idx] = per_tile

        # ---- final scoring: mean of [20, 27] layers (original behaviour) ----
        mean_scores = (
            (lm_out_q.attentions[20].shape[0] * 0 +   # just a device-safe way to start
             self._extract_layer_attn(lm_out_q, 20, vis_positions, text_positions) /
             (self._extract_layer_attn(lm_out_g, 20, vis_positions, generic_text_positions) + 1e-8)
             + self._extract_layer_attn(lm_out_q, 27, vis_positions, text_positions) /
             (self._extract_layer_attn(lm_out_g, 27, vis_positions, generic_text_positions) + 1e-8)
             ) / 2
        )
        mean_scores = (mean_scores - mean_scores.min()) / (mean_scores.max() - mean_scores.min() + 1e-6)
        full_map_smooth, smoothed = _stitch_and_smooth(mean_scores)
        self._debug_full_score_map = full_map_smooth
        self._debug_patch_scores = smoothed

        # ---- thumbnail relative attention scoring (for visualization only) ----
        if has_thumbnail:
            thumb_vis_positions = vis_positions_all[num_content_tiles * N_per_tile:
                                                    (num_content_tiles + 1) * N_per_tile]
            if thumb_vis_positions.numel() > 0:
                A_q_thumb = (
                    self._extract_layer_attn(lm_out_q, 19, thumb_vis_positions, text_positions) +
                    self._extract_layer_attn(lm_out_q, 27, thumb_vis_positions, text_positions)
                ) / 2
                A_g_thumb = (
                    self._extract_layer_attn(lm_out_g, 19, thumb_vis_positions, generic_text_positions) +
                    self._extract_layer_attn(lm_out_g, 27, thumb_vis_positions, generic_text_positions)
                ) / 2
                thumb_scores = A_q_thumb / (A_g_thumb + 1e-8)
                thumb_scores = (thumb_scores - thumb_scores.min()) / (thumb_scores.max() - thumb_scores.min() + 1e-6)
                # smooth
                thumb_map = thumb_scores.reshape(grid_h, grid_w).float()
                thumb_smooth = F.conv2d(thumb_map.unsqueeze(0).unsqueeze(0), kernel, padding=ks // 2).squeeze()
                thumb_smooth = (thumb_smooth - thumb_smooth.min()) / (thumb_smooth.max() - thumb_smooth.min() + 1e-6)
                self._debug_thumbnail_score_map = thumb_smooth
            else:
                self._debug_thumbnail_score_map = None

        # ---- QVTree selection per content tile only (ignore thumbnail) ----
        patch_ids_per_tile = []
        for i in range(num_content_tiles):
            tokens = vit_embeds[i]
            x = tokens.unsqueeze(0)
            patch_scores = smoothed[i].unsqueeze(0).to(device=device, dtype=dtype)

            built = self.qvtree.builder.build(x, grid_h, grid_w)
            nodes = built["nodes"]
            sel_nodes_list, _ = self.qvtree.navigator.select_nodes(
                nodes=nodes, patch_scores=patch_scores, W=grid_w,
            )
            token_out = self.qvtree.navigator.nodes_to_tokens(
                nodes, H=grid_h, W=grid_w,
                selected_node_ids=sel_nodes_list, x=x,
            )
            patch_ids = token_out["selected_token_indices"][0]
            if patch_ids.numel() == 0:
                patch_ids = torch.arange(tokens.size(0), device=device)
            patch_ids = torch.unique(patch_ids.clamp(0, tokens.size(0) - 1))

            patch_ids_per_tile.append(patch_ids)
            self._debug_patch_ids.append(patch_ids)
            self._debug_num_selected_tokens.append(int(patch_ids.numel()))
            self._debug_num_total_tokens.append(int(tokens.size(0)))
            self._debug_select_ratios.append(patch_ids.numel() / tokens.size(0))

        # thumbnail: keep all patches (no selection)
        if has_thumbnail:
            thumb_patch_ids = torch.arange(vit_embeds[-1].size(0), device=device)
            patch_ids_per_tile.append(thumb_patch_ids)

        return patch_ids_per_tile

    # ------------------------------------------------------------------
    # Override generate: scoring + selection only, no LPD here
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            best_ratio: Optional[Tuple[int, int]] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:

        assert self.img_context_token_id is not None

        if pixel_values is not None:
            if visual_features is not None:
                return super().generate(
                    pixel_values=None, input_ids=input_ids,
                    attention_mask=attention_mask, visual_features=visual_features,
                    generation_config=generation_config,
                    output_hidden_states=output_hidden_states, **generate_kwargs,
                )

            # 1. ViT features
            vit_embeds, grid_h, grid_w = self.extract_feature_with_grid(pixel_values)
            B_tiles = vit_embeds.shape[0]

            if best_ratio is None:
                n = B_tiles - 1
                sq = int(math.isqrt(n))
                best_ratio = (sq, sq) if sq * sq == n else (n, 1)

            # 2. Build full input_embeds for scoring
            emb = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = emb.shape
            flat = emb.reshape(B * N, C)
            ids_flat = input_ids.reshape(B * N)
            sel = (ids_flat == self.img_context_token_id)
            flat[sel] = vit_embeds.reshape(-1, C).to(flat.device)
            input_embeds_full = torch.nan_to_num(flat.reshape(B, N, C))

            # 3. Score + QVTree selection (stores debug info)
            self._score_and_select(
                vit_embeds=vit_embeds,
                input_embeds_full=input_embeds_full,
                attention_mask=attention_mask,
                input_ids=input_ids,
                grid_h=grid_h,
                grid_w=grid_w,
                best_ratio=best_ratio,
            )

            # 4. Use original embeddings for generation (LPD handled in infer)
            input_embeds_final = input_embeds_full

        else:
            input_embeds_final = self.language_model.get_input_embeddings()(input_ids)

        return self.language_model.generate(
            inputs_embeds=input_embeds_final,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            use_cache=True,
            **generate_kwargs,
        )

    # ------------------------------------------------------------------
    # Convenience inference entry point
    # ------------------------------------------------------------------
    def infer(
            self,
            tokenizer,
            image_path: str,
            question: str,
            max_new_tokens: int = 256,
            do_sample: bool = False,
            input_size: int = 448,
            max_num: int = 6,
            use_lpd: bool = True,
            use_tree: bool = True,
            IMG_START_TOKEN: str = '<img>',
            IMG_END_TOKEN: str = '</img>',
            IMG_CONTEXT_TOKEN: str = '<IMG_CONTEXT>',
    ) -> str:
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        self.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

        image = Image.open(image_path).convert('RGB')
        transform = build_transform(input_size)

        tiles, best_ratio = dynamic_preprocess(
            image,
            min_num=1,
            max_num=max_num,
            image_size=input_size,
            use_thumbnail=True,
        )

        pixel_values = torch.stack([transform(t) for t in tiles]).to(
            dtype=dtype,
            device=device,
        )

        ratio_w, ratio_h = best_ratio
        num_content_tiles = ratio_w * ratio_h

        # =========================================================
        # BASELINE MODE: bypass tree / compact pipeline completely
        # =========================================================
        if not use_tree:
            generation_config = dict(
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
            )

            return self.chat(
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                question=question,
                generation_config=generation_config,
            )

        # =========================================================
        # TREE MODE: original pipeline below
        # =========================================================

        # ===== 原 tree pipeline 从这里继续 =====

        def _build_query(n_patches):
            template = get_conv_template(self.template)
            template.system_message = self.system_message
            template.append_message(template.roles[0], '<image>\n' + question)
            template.append_message(template.roles[1], None)
            q = template.get_prompt()
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * n_patches + IMG_END_TOKEN
            q = q.replace('<image>', image_tokens, 1)
            eos = tokenizer.convert_tokens_to_ids(template.sep.strip())
            return q, eos

        # ── Pass 1: scoring forward on original image ──
        query1, eos_token_id = _build_query(len(tiles))
        inputs1 = tokenizer(query1, return_tensors='pt')
        input_ids1 = inputs1['input_ids'].to(device)
        attention_mask1 = inputs1['attention_mask'].to(device)

        self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids1,
            attention_mask=attention_mask1,
            best_ratio=best_ratio,
            max_new_tokens=1,  # just need scoring, not real output
            do_sample=False,
            eos_token_id=eos_token_id,
        )

        if not use_lpd or self._debug_patch_ids is None:
            # base inference: re-run with original image
            outputs = self.language_model.generate(
                inputs_embeds=None, input_ids=input_ids1,
                attention_mask=attention_mask1,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                eos_token_id=eos_token_id,
                use_cache=True,
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            return response.split(tokenizer.decode([eos_token_id]).strip())[0].strip()

        # ── LPD: map global patch_ids back to original image ──
        # stitch content tile patch_ids into global grid
        global_h = ratio_h * GRID_SIZE
        global_w = ratio_w * GRID_SIZE
        hidden_dim = self.config.llm_config.hidden_size

        # build global patch scores from debug info
        global_scores = torch.zeros(global_h * global_w)
        for tile_idx in range(num_content_tiles):
            tile_col = tile_idx % ratio_w
            tile_row = tile_idx // ratio_w
            scores_i = self._debug_patch_scores[tile_idx]  # [GRID_SIZE*GRID_SIZE]
            for local_idx in range(GRID_SIZE * GRID_SIZE):
                local_r, local_c = local_idx // GRID_SIZE, local_idx % GRID_SIZE
                global_r = tile_row * GRID_SIZE + local_r
                global_c = tile_col * GRID_SIZE + local_c
                global_scores[global_r * global_w + global_c] = scores_i[local_idx]

        # QuadTree selection on global grid
        x_dummy = torch.zeros(1, global_h * global_w, hidden_dim)
        ps = global_scores.unsqueeze(0).float()
        built = self.qvtree.builder.build(x_dummy, global_h, global_w)
        nodes = built["nodes"]
        sel_nodes, _ = self.qvtree.navigator.select_nodes(nodes=nodes, patch_scores=ps, W=global_w)
        token_out = self.qvtree.navigator.nodes_to_tokens(
            nodes, H=global_h, W=global_w, selected_node_ids=sel_nodes, x=x_dummy,
        )
        global_patch_ids = token_out["selected_token_indices"][0]
        if global_patch_ids.numel() == 0:
            global_patch_ids = torch.arange(global_h * global_w)
        global_patch_ids = torch.unique(global_patch_ids.clamp(0, global_h * global_w - 1))

        # map global patch_ids to original image pixel coords
        orig_w, orig_h = image.size
        patch_pixel_w = orig_w / global_w
        patch_pixel_h = orig_h / global_h
        raw_bboxes = []
        for idx in global_patch_ids.tolist():
            r, c = int(idx) // global_w, int(idx) % global_w
            x0 = int(c * patch_pixel_w)
            y0 = int(r * patch_pixel_h)
            x1 = int((c + 1) * patch_pixel_w)
            y1 = int((r + 1) * patch_pixel_h)
            raw_bboxes.append((x0, y0, x1, y1))

        merged_bboxes = _merge_bboxes(raw_bboxes)
        compact_image = _build_compact_image(image.convert("RGB"), merged_bboxes)

        # scale up compact image
        cw, ch = compact_image.size
        if cw > 0 and ch > 0:
            scale = min(orig_w / cw, orig_h / ch)
            compact_image = compact_image.resize((int(cw * scale), int(ch * scale)), Image.BILINEAR)

        # store debug
        self._debug_compact_tiles = [compact_image]
        self._debug_merged_bboxes = merged_bboxes
        self._debug_global_patch_ids = global_patch_ids

        n_sel = int(global_patch_ids.numel())
        n_tot = global_h * global_w
        print(f"[TREE] selected tokens: {n_sel}, original: {n_tot}, ratio: {n_sel / n_tot:.1%}")

        # update debug stats to reflect global selection
        self._debug_num_selected_tokens = [n_sel]
        self._debug_num_total_tokens    = [n_tot]
        self._debug_select_ratios       = [n_sel / n_tot if n_tot > 0 else 0.0]

        # ── Pass 2: inference with compact image ──
        import tempfile, os as _os
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        compact_image.save(tmp.name);
        tmp.close()

        compact_tiles, compact_ratio = dynamic_preprocess(
            compact_image, min_num=1, max_num=max_num,
            image_size=input_size, use_thumbnail=True,
        )
        compact_pv = torch.stack([transform(t) for t in compact_tiles]).to(dtype=dtype, device=device)

        query2, _ = _build_query(len(compact_tiles))
        inputs2 = tokenizer(query2, return_tensors='pt')
        input_ids2 = inputs2['input_ids'].to(device)
        attention_mask2 = inputs2['attention_mask'].to(device)

        _os.unlink(tmp.name)

        # use parent generate (no scoring) for compact image
        outputs = super().generate(
            pixel_values=compact_pv,
            input_ids=input_ids2,
            attention_mask=attention_mask2,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            eos_token_id=eos_token_id,
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        response = response.split(tokenizer.decode([eos_token_id]).strip())[0].strip()
        return response