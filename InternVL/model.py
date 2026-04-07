import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image

from transformers import AutoTokenizer, GenerationConfig

from .modeling_internvl_chat import InternVLChatModel
from .configuration_internvl_chat import InternVLChatConfig
from .conversation import get_conv_template
from module import QVTree


# =========================================================
# Image loading utilities
# =========================================================
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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
            col * image_size,
            row * image_size,
            (col + 1) * image_size,
            (row + 1) * image_size,
        )
        tiles.append(resized.crop(box))

    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))

    return tiles, best_ratio


def load_image(image_path, input_size=448, max_num=6):
    """
    Load an image with dynamic tiling.

    Returns:
        pixel_values : [num_tiles, 3, H, W]  (last tile is thumbnail)
        best_ratio   : (ratio_w, ratio_h)
    """
    image = Image.open(image_path).convert('RGB')
    transform = build_transform(input_size)
    tiles, best_ratio = dynamic_preprocess(
        image, min_num=1, max_num=max_num,
        image_size=input_size, use_thumbnail=False,  # 改这里
    )
    pixel_values = torch.stack([transform(t) for t in tiles])
    return pixel_values, best_ratio


# =========================================================
# Model
# =========================================================
class InternVLChatModelWithTree(InternVLChatModel):

    def __init__(self, config: InternVLChatConfig, vision_model=None, language_model=None, use_flash_attn=True):
        super().__init__(config, vision_model=vision_model, language_model=language_model, use_flash_attn=use_flash_attn)

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
        self._debug_full_score_map = None  # [ratio_h*grid_h, ratio_w*grid_w]

    # ------------------------------------------------------------------
    # extract_feature with grid shape
    # ------------------------------------------------------------------
    def extract_feature_with_grid(self, pixel_values):
        """
        Returns:
            vit_embeds     : [B_tiles, N_per_tile, D]
            grid_h, grid_w : int  (per-tile spatial grid after pixel_shuffle)
        """
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
    def _get_attn_scores(self, input_embeds, attention_mask, img_positions):
        with torch.no_grad():
            out = self.language_model(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                output_attentions=True,
                output_hidden_states=False,
                return_dict=True,
                use_cache=False,
            )
        ans_pos = input_embeds.shape[1] - 2
        scores_list = []
        # choose transformer layers
        for i in [20]:
            layer_attn = out.attentions[i]
            vp = img_positions.to(layer_attn.device)
            scores_list.append(layer_attn[0, :, ans_pos, vp].mean(dim=0).cpu())
        return torch.stack(scores_list).mean(dim=0)

    # ------------------------------------------------------------------
    # Global scoring → stitch tiles → smooth → per-tile QVTree → modulate
    # ------------------------------------------------------------------
    def _score_and_modulate(self, vit_embeds, input_embeds_full, attention_mask, input_ids, grid_h, grid_w, best_ratio):
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

        # ---- relative attention scoring ----
        ids_flat = input_ids.reshape(-1)
        sel_mask = (ids_flat == self.img_context_token_id)
        vis_positions = sel_mask.nonzero(as_tuple=True)[0].cpu()

        A_q = self._get_attn_scores(input_embeds_full, attention_mask, vis_positions)

        if self._generic_token_ids is None:
            _tok = AutoTokenizer.from_pretrained(self.config._name_or_path, trust_remote_code=True)
            self._generic_token_ids = _tok(
                "Write a general description of the image.",
                return_tensors="pt", add_special_tokens=False,
            ).input_ids

        generic_ids = self._generic_token_ids.to(input_ids.device)
        img_end = vis_positions[-1].item()
        new_ids = torch.cat([input_ids[:, :img_end + 1], generic_ids, input_ids[:, -3:]], dim=1)

        gen_emb = self.language_model.get_input_embeddings()(new_ids)
        gflat = gen_emb.reshape(-1, D)
        gen_sel = (new_ids.reshape(-1) == self.img_context_token_id)
        gflat[gen_sel] = vit_embeds.reshape(-1, D).to(device=gflat.device, dtype=gflat.dtype)
        gen_emb = torch.nan_to_num(gflat.reshape(1, -1, D))
        gen_mask = torch.ones(1, new_ids.shape[1], dtype=attention_mask.dtype, device=device)

        A_generic = self._get_attn_scores(gen_emb, gen_mask, vis_positions)

        # global relative scores: [B_tiles * N_per_tile]
        scores_all = A_q / (A_generic + 1e-8)
        scores_all = (scores_all - scores_all.min()) / (scores_all.max() - scores_all.min() + 1e-6)

        # ---- stitch content tiles into full spatial map ----
        scores_tiled = scores_all.view(B_tiles, grid_h, grid_w)
        rows = []
        for row in range(ratio_h):
            cols = [scores_tiled[row * ratio_w + col] for col in range(ratio_w)]
            rows.append(torch.cat(cols, dim=1))
        full_map = torch.cat(rows, dim=0).float()  # [ratio_h*grid_h, ratio_w*grid_w]

        # ---- 2-D Gaussian smoothing on full map ----
        sigma, ks = 1.0, 3
        ax = torch.arange(ks, dtype=torch.float32) - ks // 2
        g1d = torch.exp(-ax ** 2 / (2 * sigma ** 2))
        g1d = g1d / g1d.sum()
        kernel = (g1d.unsqueeze(1) * g1d.unsqueeze(0)).view(1, 1, ks, ks)

        full_map_smooth = F.conv2d(full_map.unsqueeze(0).unsqueeze(0), kernel, padding=ks // 2).squeeze()
        full_map_smooth = (full_map_smooth - full_map_smooth.min()) / (full_map_smooth.max() - full_map_smooth.min() + 1e-6)

        self._debug_full_score_map = full_map_smooth

        # ---- slice back per content tile ----
        smoothed = []
        for row in range(ratio_h):
            for col in range(ratio_w):
                tile_score = full_map_smooth[
                    row * grid_h: (row + 1) * grid_h,
                    col * grid_w: (col + 1) * grid_w,
                ].reshape(-1)
                smoothed.append(tile_score)

        # thumbnail: smooth independently
        if has_thumbnail:
            thumb = scores_tiled[-1].float().unsqueeze(0).unsqueeze(0)
            thumb = F.conv2d(thumb, kernel, padding=ks // 2).view(-1)
            thumb = (thumb - thumb.min()) / (thumb.max() - thumb.min() + 1e-6)
            smoothed.append(thumb)

        self._debug_patch_scores = smoothed

        # ---- QVTree navigation + alpha-beta masking ----
        alpha, beta = 5.0, 0.5
        selected_idx_per_tile = []
        new_embeds_list = []

        for i in range(B_tiles):
            tokens = vit_embeds[i]
            x = tokens.unsqueeze(0)
            patch_scores = smoothed[i].unsqueeze(0).to(device=device, dtype=dtype)

            built = self.qvtree.builder.build(x, grid_h, grid_w)
            nodes = built["nodes"]

            sel_nodes_list, _ = self.qvtree.navigator.select_nodes(
                nodes=nodes, patch_scores=patch_scores, W=grid_w,
            )
            selected_idx_per_tile.append(sel_nodes_list[0])

            token_out = self.qvtree.navigator.nodes_to_tokens(
                nodes, H=grid_h, W=grid_w,
                selected_node_ids=sel_nodes_list, x=x,
            )
            patch_ids = token_out["selected_token_indices"][0]

            if patch_ids.numel() == 0:
                patch_ids = torch.arange(tokens.size(0), device=device)
            patch_ids = torch.unique(patch_ids.clamp(0, tokens.size(0) - 1))

            self._debug_patch_ids.append(patch_ids)
            self._debug_num_selected_tokens.append(int(patch_ids.numel()))
            self._debug_num_total_tokens.append(int(tokens.size(0)))
            self._debug_select_ratios.append(patch_ids.numel() / tokens.size(0))

            keep = torch.full((tokens.size(0),), beta, device=device, dtype=dtype)
            keep[patch_ids] = alpha
            new_embeds_list.append(torch.nan_to_num(tokens * keep.unsqueeze(-1)))

        self._debug_selected_idx = selected_idx_per_tile
        return torch.cat(new_embeds_list, dim=0)  # [B_tiles * N_per_tile, D]

    # ------------------------------------------------------------------
    # Override generate
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

            # infer best_ratio if not provided
            if best_ratio is None:
                n = B_tiles - 1  # exclude thumbnail
                sq = int(math.isqrt(n))
                best_ratio = (sq, sq) if sq * sq == n else (n, 1)

            # 2. Build full input_embeds
            emb = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = emb.shape
            flat = emb.reshape(B * N, C)
            ids_flat = input_ids.reshape(B * N)
            sel = (ids_flat == self.img_context_token_id)
            flat[sel] = vit_embeds.reshape(-1, C).to(flat.device)
            input_embeds_full = torch.nan_to_num(flat.reshape(B, N, C))

            # 3. Score + modulate
            new_vit = self._score_and_modulate(
                vit_embeds=vit_embeds,
                input_embeds_full=input_embeds_full,
                attention_mask=attention_mask,
                input_ids=input_ids,
                grid_h=grid_h,
                grid_w=grid_w,
                best_ratio=best_ratio,
            )

            # 4. Place modulated tokens back
            flat2 = emb.reshape(B * N, C).clone()
            flat2[sel] = new_vit.reshape(-1, C).to(flat2.device)
            input_embeds_final = torch.nan_to_num(flat2.reshape(B, N, C))

        else:
            input_embeds_final = self.language_model.get_input_embeddings()(input_ids)

        # 5. LLM generate
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
        IMG_START_TOKEN: str = '<img>',
        IMG_END_TOKEN: str = '</img>',
        IMG_CONTEXT_TOKEN: str = '<IMG_CONTEXT>',
    ) -> str:
        """Single-image inference. Returns response string."""
        pixel_values, best_ratio = load_image(image_path, input_size=input_size, max_num=max_num)
        pixel_values = pixel_values.to(
            dtype=next(self.parameters()).dtype,
            device=next(self.parameters()).device,
        )

        self.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        num_patches = pixel_values.shape[0]

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        template.append_message(template.roles[0], '<image>\n' + question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
        query = query.replace('<image>', image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].to(next(self.parameters()).device)
        attention_mask = model_inputs['attention_mask'].to(next(self.parameters()).device)
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

        outputs = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            best_ratio=best_ratio,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            eos_token_id=eos_token_id,
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        response = response.split(template.sep.strip())[0].strip()
        return response