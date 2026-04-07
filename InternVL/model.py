import torch
from typing import List, Optional, Tuple, Union

from transformers import GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from .modeling_internvl_chat import InternVLChatModel
from .configuration_internvl_chat import InternVLChatConfig
from .conversation import get_conv_template
from module import QVTree


class InternVLChatModelWithTree(InternVLChatModel):

    def __init__(self, config: InternVLChatConfig, vision_model=None, language_model=None, use_flash_attn=True):
        super().__init__(config, vision_model=vision_model, language_model=language_model, use_flash_attn=use_flash_attn)

        llm_hidden_size = config.llm_config.hidden_size
        self.qvtree = QVTree(D=llm_hidden_size)

        # cache generic token ids (lazy init)
        self._generic_token_ids = None

        # debug states
        self._debug_selected_idx = None
        self._debug_patch_ids = None
        self._debug_num_selected_tokens = None
        self._debug_num_total_tokens = None
        self._debug_select_ratios = None
        self._debug_patch_scores = None

    # ------------------------------------------------------------------
    # Internal helper: run one LLM forward and extract attention scores
    # from the last token (answer start) to all image token positions.
    # ------------------------------------------------------------------
    def _get_attn_scores(
        self,
        input_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        img_positions: torch.Tensor,
        target_layer: int = 27,
    ) -> torch.Tensor:
        """
        Run a no-grad LLM forward and return per-patch attention scores.

        Args:
            input_embeds  : [1, L, D]
            attention_mask: [1, L]
            img_positions : 1-D tensor of image token indices in the sequence
            target_layer  : which decoder layer to read attention from

        Returns:
            scores: [N_img]  (float32, CPU)
        """
        with torch.no_grad():
            out = self.language_model(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                output_attentions=True,
                output_hidden_states=False,
                return_dict=True,
                use_cache=False,
            )

        # last token attends to image positions
        ans_pos = input_embeds.shape[1] - 1
        layer_attn = out.attentions[target_layer]          # [1, heads, L, L]
        ld = layer_attn.device
        vp = img_positions.to(ld)
        scores = layer_attn[0, :, ans_pos, vp].mean(dim=0).cpu()  # [N_img]
        return scores

    # ------------------------------------------------------------------
    # Override extract_feature to also return grid shape (H, W)
    # ------------------------------------------------------------------
    def extract_feature_with_grid(self, pixel_values):
        """
        Same as extract_feature but also returns (H, W) of the patch grid
        after pixel_shuffle downsampling.

        Returns:
            vit_embeds : [B, N_down, D]   (after mlp1)
            grid_h     : int
            grid_w     : int
        """
        if self.select_layer == -1:
            vit_embeds_raw = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True,
            ).last_hidden_state
        else:
            vit_embeds_raw = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True,
            ).hidden_states[self.select_layer]

        vit_embeds_raw = vit_embeds_raw[:, 1:, :]  # drop CLS token

        h_raw = w_raw = int(vit_embeds_raw.shape[1] ** 0.5)
        vit_embeds_2d = vit_embeds_raw.reshape(vit_embeds_raw.shape[0], h_raw, w_raw, -1)
        vit_embeds_2d = self.pixel_shuffle(vit_embeds_2d, scale_factor=self.downsample_ratio)

        grid_h = vit_embeds_2d.shape[1]
        grid_w = vit_embeds_2d.shape[2]

        vit_embeds = vit_embeds_2d.reshape(vit_embeds_2d.shape[0], -1, vit_embeds_2d.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds, grid_h, grid_w

    # ------------------------------------------------------------------
    # Override generate — this is where token selection happens
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
            **generate_kwargs,
    ) -> torch.LongTensor:

        assert self.img_context_token_id is not None

        if pixel_values is not None:

            # ===================================================
            # 1. Extract vision features + grid shape
            # ===================================================
            if visual_features is not None:
                # visual_features provided externally: no grid info, fall back to base
                return super().generate(
                    pixel_values=None,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    visual_features=visual_features,
                    generation_config=generation_config,
                    output_hidden_states=output_hidden_states,
                    **generate_kwargs,
                )

            # vit_embeds: [B_img, N_down, D]
            vit_embeds, grid_h, grid_w = self.extract_feature_with_grid(pixel_values)

            # init debug containers
            self._debug_patch_ids = []
            self._debug_num_selected_tokens = []
            self._debug_num_total_tokens = []
            self._debug_select_ratios = []

            # ===================================================
            # 2. Build full input_embeds (all image tokens placed)
            # ===================================================
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds_flat = input_embeds.reshape(B * N, C)
            input_ids_flat = input_ids.reshape(B * N)

            selected_mask = (input_ids_flat == self.img_context_token_id)
            input_embeds_flat[selected_mask] = vit_embeds.reshape(-1, C).to(input_embeds_flat.device)
            input_embeds_full = input_embeds_flat.reshape(B, N, C)
            input_embeds_full = torch.nan_to_num(input_embeds_full)

            # ===================================================
            # 3. Relative attention scoring (ViCrop style)
            # All tiles are concatenated → one global attention pass
            # ===================================================
            vis_positions = selected_mask.nonzero(as_tuple=True)[0].cpu()  # all img token positions

            # --- Question-specific pass ---
            A_q = self._get_attn_scores(input_embeds_full, attention_mask, vis_positions)

            # --- Generic description pass ---
            if self._generic_token_ids is None:
                from transformers import AutoTokenizer
                _tok = AutoTokenizer.from_pretrained(self.config._name_or_path)
                generic_text = "Write a general description of the image."
                self._generic_token_ids = _tok(
                    generic_text, return_tensors="pt", add_special_tokens=False
                ).input_ids

            generic_ids = self._generic_token_ids.to(input_ids.device)

            img_end = vis_positions[-1].item()
            prefix_ids = input_ids[:, :img_end + 1]
            suffix_ids = input_ids[:, -3:]
            new_input_ids = torch.cat([prefix_ids, generic_ids, suffix_ids], dim=1)

            generic_embeds = self.language_model.get_input_embeddings()(new_input_ids)
            gB, gN, gC = generic_embeds.shape
            generic_embeds_flat = generic_embeds.reshape(gB * gN, gC)
            new_input_ids_flat = new_input_ids.reshape(gB * gN)
            gen_selected = (new_input_ids_flat == self.img_context_token_id)
            generic_embeds_flat[gen_selected] = vit_embeds.reshape(-1, gC).to(
                device=generic_embeds_flat.device, dtype=generic_embeds_flat.dtype
            )
            generic_embeds = generic_embeds_flat.reshape(gB, gN, gC)
            generic_embeds = torch.nan_to_num(generic_embeds)

            generic_attn_mask = torch.ones(
                1, new_input_ids.shape[1],
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )

            A_generic = self._get_attn_scores(generic_embeds, generic_attn_mask, vis_positions)

            # Relative score + global normalize
            # A_q and A_generic have length = num_tiles * N_per_tile
            patch_scores_all = A_q / (A_generic + 1e-8)
            s_min = patch_scores_all.min()
            s_max = patch_scores_all.max()
            patch_scores_all = (patch_scores_all - s_min) / (s_max - s_min + 1e-6)

            # Gaussian smoothing per tile
            import torch.nn.functional as F
            B_img = vit_embeds.shape[0]
            N_per_tile = grid_h * grid_w
            sigma, ks = 1.0, 3
            ax = torch.arange(ks, dtype=torch.float32) - ks // 2
            gauss_1d = torch.exp(-ax ** 2 / (2 * sigma ** 2))
            gauss_1d = gauss_1d / gauss_1d.sum()
            kernel = (gauss_1d.unsqueeze(1) * gauss_1d.unsqueeze(0)).view(1, 1, ks, ks)

            # patch_scores_all: [B_img * N_per_tile] → smooth each tile independently
            patch_scores_per_tile = patch_scores_all.view(B_img, N_per_tile)  # [B_img, N_per_tile]
            smoothed_tiles = []
            for i in range(B_img):
                score_map = patch_scores_per_tile[i].float().view(1, 1, grid_h, grid_w)
                score_map = F.conv2d(score_map, kernel, padding=ks // 2)
                score_tile = score_map.view(-1)
                s_min_t = score_tile.min()
                s_max_t = score_tile.max()
                score_tile = (score_tile - s_min_t) / (s_max_t - s_min_t + 1e-6)
                smoothed_tiles.append(score_tile)

            self._debug_patch_scores = smoothed_tiles

            # ===================================================
            # 4. QuadTree navigation — one pass per tile,
            #    using that tile's own smoothed score slice
            # ===================================================
            selected_idx_per_image = []
            new_vit_embeds_list = []

            for i in range(B_img):
                tokens = vit_embeds[i]      # [N_per_tile, D]
                x = tokens.unsqueeze(0)     # [1, N_per_tile, D]

                if N_per_tile != tokens.size(0):
                    raise ValueError(
                        f"Grid mismatch at tile {i}: grid_h={grid_h}, grid_w={grid_w}, "
                        f"H*W={N_per_tile}, tokens={tokens.size(0)}"
                    )

                # use this tile's smoothed scores
                patch_scores = smoothed_tiles[i].unsqueeze(0).to(
                    tokens.device, tokens.dtype
                )  # [1, N_per_tile]

                built = self.qvtree.builder.build(x, grid_h, grid_w)
                nodes = built["nodes"]

                sel_nodes_list, _ = self.qvtree.navigator.select_nodes(
                    nodes=nodes,
                    patch_scores=patch_scores,
                    W=grid_w,
                )
                selected_idx_per_image.append(sel_nodes_list[0])

                token_out = self.qvtree.navigator.nodes_to_tokens(
                    nodes,
                    H=grid_h,
                    W=grid_w,
                    selected_node_ids=sel_nodes_list,
                    x=x,
                )

                patch_ids = token_out["selected_token_indices"][0]

                # Fallback: empty selection → keep all
                if patch_ids.numel() == 0:
                    patch_ids = torch.arange(tokens.size(0), device=tokens.device)

                patch_ids = patch_ids.clamp(min=0, max=tokens.size(0) - 1)
                patch_ids = torch.unique(patch_ids)

                # Debug
                self._debug_patch_ids.append(patch_ids)
                num_selected = int(patch_ids.numel())
                num_total = int(tokens.size(0))
                self._debug_num_selected_tokens.append(num_selected)
                self._debug_num_total_tokens.append(num_total)
                self._debug_select_ratios.append(num_selected / num_total if num_total > 0 else 0.0)

                # ===================================================
                # 5. Alpha-beta soft masking
                # ===================================================
                alpha = 5.0
                beta  = 0.5

                keep = torch.full(
                    (tokens.size(0),), beta,
                    device=tokens.device, dtype=tokens.dtype,
                )
                keep[patch_ids] = alpha

                tokens_modulated = tokens * keep.unsqueeze(-1)
                tokens_modulated = torch.nan_to_num(tokens_modulated)
                new_vit_embeds_list.append(tokens_modulated)

            self._debug_selected_idx = selected_idx_per_image

            # ===================================================
            # 6. Place modulated tokens back into input_embeds
            # ===================================================
            new_vit_embeds = torch.cat(new_vit_embeds_list, dim=0)  # [sum_tiles * N_down, D] after reshape

            input_embeds2 = self.language_model.get_input_embeddings()(input_ids)
            B2, N2, C2 = input_embeds2.shape
            input_embeds2_flat = input_embeds2.reshape(B2 * N2, C2)
            input_ids_flat2 = input_ids.reshape(B2 * N2)
            selected2 = (input_ids_flat2 == self.img_context_token_id)
            input_embeds2_flat[selected2] = new_vit_embeds.reshape(-1, C2).to(input_embeds2_flat.device)
            input_embeds_final = input_embeds2_flat.reshape(B2, N2, C2)
            input_embeds_final = torch.nan_to_num(input_embeds_final)

        else:
            input_embeds_final = self.language_model.get_input_embeddings()(input_ids)

        # ===================================================
        # 7. LLM generate
        # ===================================================
        outputs = self.language_model.generate(
            inputs_embeds=input_embeds_final,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs