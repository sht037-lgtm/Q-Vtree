import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image

from transformers import AutoTokenizer, GenerationConfig

from .src.modeling_internvl_chat import InternVLChatModel
from .src.configuration_internvl_chat import InternVLChatConfig
from .src.conversation import get_conv_template
from .module import QVTree

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
        self._debug_patch_ids = None
        self._debug_num_selected_tokens = None
        self._debug_num_total_tokens = None
        self._debug_select_ratios = None
        self._debug_patch_scores = None
        self._debug_best_ratio = None
        self._debug_full_score_map = None
        self._debug_selected_map = None
        self._debug_compact_tiles = None
        self._debug_merged_bboxes = None
        self._debug_baseline_gpu_time = None
        self._debug_baseline_peak_memory = None
        self._debug_tree_gpu_time = None
        self._debug_tree_peak_memory = None
        self._debug_pass1_gpu_time = None
        self._debug_pass1_peak_memory = None
        self._debug_pass2_gpu_time = None
        self._debug_pass2_peak_memory = None
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

        raw = raw[:, 1:, :]
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
    SCORE_LAYERS = [11, 23]

    def _run_lm_forward(self, input_embeds, attention_mask):
        with torch.no_grad():
            return self.language_model(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                output_attentions=True,
                output_hidden_states=False,
                return_dict=True,
                use_cache=False,
            )

    def _extract_layer_attn(self, lm_out, layer_idx, img_positions, text_positions):
        layer_attn = lm_out.attentions[layer_idx]
        vp = img_positions.to(layer_attn.device)
        qp = text_positions.to(layer_attn.device)
        return layer_attn[0, :, :, vp][:, qp, :].mean(0).mean(0).cpu().float()

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

        self._debug_patch_ids = []
        self._debug_num_selected_tokens = []
        self._debug_num_total_tokens = []
        self._debug_select_ratios = []
        self._debug_best_ratio = best_ratio

        ids_flat = input_ids.reshape(-1)
        sel_mask = (ids_flat == self.img_context_token_id)
        vis_positions_all = sel_mask.nonzero(as_tuple=True)[0].cpu()
        n_content_tokens = num_content_tiles * N_per_tile

        if has_thumbnail:
            vis_positions = vis_positions_all[n_content_tokens: n_content_tokens + N_per_tile]
        else:
            vis_positions = vis_positions_all[:n_content_tokens]

        img_end = vis_positions_all[-1].item()
        text_positions = torch.arange(img_end + 1, input_ids.shape[1], dtype=torch.long)

        if self._generic_token_ids is None:
            _tok = AutoTokenizer.from_pretrained(self.config._name_or_path, trust_remote_code=True)
            self._generic_token_ids = _tok(
                GENERIC_PROMPT, return_tensors="pt", add_special_tokens=False,
            ).input_ids

        generic_ids = self._generic_token_ids.to(input_ids.device)
        new_ids = torch.cat([input_ids[:, :img_end + 1], generic_ids, input_ids[:, -3:]], dim=1)
        gen_emb = self.language_model.get_input_embeddings()(new_ids)
        gflat = gen_emb.reshape(-1, D)
        gflat[(new_ids.reshape(-1) == self.img_context_token_id)] = \
            vit_embeds.reshape(-1, D).to(device=gflat.device, dtype=gflat.dtype)
        gen_emb = torch.nan_to_num(gflat.reshape(1, -1, D))
        gen_mask = torch.ones(1, new_ids.shape[1], dtype=attention_mask.dtype, device=device)
        generic_text_positions = torch.arange(
            img_end + 1, img_end + 1 + generic_ids.shape[1], dtype=torch.long
        )

        lm_out_q = self._run_lm_forward(input_embeds_full, attention_mask)
        lm_out_g = self._run_lm_forward(gen_emb, gen_mask)

        scores = sum(
            self._extract_layer_attn(lm_out_q, li, vis_positions, text_positions) /
            (self._extract_layer_attn(lm_out_g, li, vis_positions, generic_text_positions) + 1e-8)
            for li in self.SCORE_LAYERS
        ) / len(self.SCORE_LAYERS)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)

        sigma, ks = 1.0, 3
        ax = torch.arange(ks, dtype=torch.float32) - ks // 2
        g1d = torch.exp(-ax ** 2 / (2 * sigma ** 2))
        g1d = g1d / g1d.sum()
        kernel = (g1d.unsqueeze(1) * g1d.unsqueeze(0)).view(1, 1, ks, ks)
        score_map = scores.reshape(grid_h, grid_w).float()
        score_map = F.conv2d(score_map.unsqueeze(0).unsqueeze(0), kernel, padding=ks // 2).squeeze()
        score_map = (score_map - score_map.min()) / (score_map.max() - score_map.min() + 1e-6)
        self._debug_full_score_map = score_map
        self._debug_patch_scores = [score_map.reshape(-1)]

        thumb_tokens = vit_embeds[-1] if has_thumbnail else vit_embeds[0]
        x_thumb = thumb_tokens.unsqueeze(0)
        ps_thumb = score_map.reshape(1, -1).to(device=device, dtype=dtype)
        built = self.qvtree.builder.build(x_thumb, grid_h, grid_w)
        nodes = built["nodes"]
        sel_nodes_list, _ = self.qvtree.navigator.select_nodes(
            nodes=nodes, patch_scores=ps_thumb, W=grid_w,
        )
        token_out = self.qvtree.navigator.nodes_to_tokens(
            nodes, H=grid_h, W=grid_w, selected_node_ids=sel_nodes_list, x=x_thumb,
        )
        thumb_patch_ids = token_out["selected_token_indices"][0]
        if thumb_patch_ids.numel() == 0:
            thumb_patch_ids = torch.arange(grid_h * grid_w, device=device)
        thumb_patch_ids = torch.unique(thumb_patch_ids.clamp(0, grid_h * grid_w - 1))

        selected_map = torch.zeros(grid_h, grid_w)
        for pid in thumb_patch_ids.tolist():
            selected_map[int(pid) // grid_w, int(pid) % grid_w] = 1.0
        self._debug_selected_map = selected_map

        self._debug_patch_ids = [thumb_patch_ids]
        n_sel = int(thumb_patch_ids.numel())
        n_tot = grid_h * grid_w
        self._debug_num_selected_tokens = [n_sel]
        self._debug_num_total_tokens = [n_tot]
        self._debug_select_ratios = [n_sel / n_tot]

        patch_ids_per_tile = [torch.arange(vit_embeds[i].size(0), device=device)
                              for i in range(B_tiles)]
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

            vit_embeds, grid_h, grid_w = self.extract_feature_with_grid(pixel_values)
            B_tiles = vit_embeds.shape[0]

            if best_ratio is None:
                n = B_tiles - 1
                sq = int(math.isqrt(n))
                best_ratio = (sq, sq) if sq * sq == n else (n, 1)

            emb = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = emb.shape
            flat = emb.reshape(B * N, C)
            ids_flat = input_ids.reshape(B * N)
            sel = (ids_flat == self.img_context_token_id)
            flat[sel] = vit_embeds.reshape(-1, C).to(flat.device)
            input_embeds_full = torch.nan_to_num(flat.reshape(B, N, C))

            self._score_and_select(
                vit_embeds=vit_embeds,
                input_embeds_full=input_embeds_full,
                attention_mask=attention_mask,
                input_ids=input_ids,
                grid_h=grid_h,
                grid_w=grid_w,
                best_ratio=best_ratio,
            )

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
        self._debug_baseline_gpu_time = None
        self._debug_baseline_peak_memory = None
        self._debug_tree_gpu_time = None
        self._debug_tree_peak_memory = None
        self._debug_pass1_gpu_time = None
        self._debug_pass1_peak_memory = None
        self._debug_pass2_gpu_time = None
        self._debug_pass2_peak_memory = None

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
            torch.cuda.reset_peak_memory_stats()
            _s0 = torch.cuda.Event(enable_timing=True)
            _e0 = torch.cuda.Event(enable_timing=True)
            _s0.record()
            response = super(InternVLChatModelWithTree, self).chat(
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                question=question,
                generation_config=generation_config,
            )
            _e0.record()
            torch.cuda.synchronize()
            self._debug_baseline_gpu_time = _s0.elapsed_time(_e0) / 1000.0
            self._debug_baseline_peak_memory = torch.cuda.max_memory_allocated() / 1024 ** 3
            return response

        # =========================================================
        # TREE MODE: original pipeline below
        # =========================================================
        torch.cuda.reset_peak_memory_stats()
        _st = torch.cuda.Event(enable_timing=True)
        _et = torch.cuda.Event(enable_timing=True)
        _st.record()

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

        # ── Pass 1: scoring on thumbnail ──
        thumb_image = image.resize((input_size, input_size))
        thumb_pv = transform(thumb_image).unsqueeze(0).to(dtype=dtype, device=device)
        thumb_ratio = (1, 1)

        query1, eos_token_id = _build_query(1)
        inputs1 = tokenizer(query1, return_tensors='pt')
        input_ids1 = inputs1['input_ids'].to(device)
        attention_mask1 = inputs1['attention_mask'].to(device)

        import time as _time
        torch.cuda.reset_peak_memory_stats()
        _s1 = torch.cuda.Event(enable_timing=True)
        _e1 = torch.cuda.Event(enable_timing=True)
        _s1.record()

        self.generate(
            pixel_values=thumb_pv,
            input_ids=input_ids1,
            attention_mask=attention_mask1,
            best_ratio=thumb_ratio,
            max_new_tokens=1,
            do_sample=False,
            eos_token_id=eos_token_id,
        )

        _e1.record()
        torch.cuda.synchronize()
        self._debug_pass1_gpu_time = _s1.elapsed_time(_e1) / 1000.0
        self._debug_pass1_peak_memory = torch.cuda.max_memory_allocated() / 1024 ** 3

        if not use_lpd or self._debug_patch_ids is None:
            outputs = self.language_model.generate(
                inputs_embeds=None, input_ids=input_ids1,
                attention_mask=attention_mask1,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                eos_token_id=eos_token_id,
                use_cache=True,
            )
            _et.record()
            torch.cuda.synchronize()
            self._debug_tree_gpu_time = _st.elapsed_time(_et) / 1000.0
            self._debug_tree_peak_memory = torch.cuda.max_memory_allocated() / 1024 ** 3
            response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            return response.split(tokenizer.decode([eos_token_id]).strip())[0].strip()

        # ── LPD ──
        orig_w, orig_h = image.size
        thumb_patch_ids = self._debug_patch_ids[0]
        patch_pixel_w = orig_w / GRID_SIZE
        patch_pixel_h = orig_h / GRID_SIZE
        raw_bboxes = []
        for idx in thumb_patch_ids.tolist():
            r, c = int(idx) // GRID_SIZE, int(idx) % GRID_SIZE
            x0 = int(c * patch_pixel_w)
            y0 = int(r * patch_pixel_h)
            x1 = int((c + 1) * patch_pixel_w)
            y1 = int((r + 1) * patch_pixel_h)
            raw_bboxes.append((x0, y0, x1, y1))

        merged_bboxes = _merge_bboxes(raw_bboxes)
        compact_image = _build_compact_image(image.convert("RGB"), merged_bboxes)

        self._debug_compact_tiles = [compact_image]
        self._debug_merged_bboxes = merged_bboxes

        n_sel = int(thumb_patch_ids.numel())
        n_tot = GRID_SIZE * GRID_SIZE
        print(f"[TREE] selected tokens: {n_sel}, original: {n_tot}, ratio: {n_sel / n_tot:.1%}")

        self._debug_num_selected_tokens = [n_sel]
        self._debug_num_total_tokens    = [n_tot]
        self._debug_select_ratios       = [n_sel / n_tot if n_tot > 0 else 0.0]

        # ── Pass 2: inference with compact image only ──
        import tempfile, os as _os
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        compact_image.save(tmp.name)
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

        torch.cuda.reset_peak_memory_stats()
        _s2 = torch.cuda.Event(enable_timing=True)
        _e2 = torch.cuda.Event(enable_timing=True)
        _s2.record()

        outputs = super().generate(
            pixel_values=compact_pv,
            input_ids=input_ids2,
            attention_mask=attention_mask2,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            eos_token_id=eos_token_id,
        )

        _e2.record()
        torch.cuda.synchronize()
        self._debug_pass2_gpu_time = _s2.elapsed_time(_e2) / 1000.0
        self._debug_pass2_peak_memory = torch.cuda.max_memory_allocated() / 1024 ** 3
        self._debug_tree_gpu_time = _st.elapsed_time(_e2) / 1000.0
        self._debug_tree_peak_memory = torch.cuda.max_memory_allocated() / 1024 ** 3

        response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        response = response.split(tokenizer.decode([eos_token_id]).strip())[0].strip()
        return response
