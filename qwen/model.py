import torch
from module import QVTree
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLModel,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLModelOutputWithPast,
)


class Qwen2_5_VLModelWithTree(Qwen2_5_VLModel):
    def __init__(self, config):
        super().__init__(config)

        self.qvtree = QVTree(D=config.text_config.hidden_size)

        # attention extraction is needed for the query-conditioned score matrix
        self.config.text_config._attn_implementation = "eager"
        self.language_model.config._attn_implementation = "eager"

        # use an intermediate layer for more stable grounding signals
        self.tree_attention_layer = 8

        # debug states
        self._debug_selected_idx = None
        self._debug_patch_ids = None
        self._debug_num_selected_tokens = None
        self._debug_num_total_tokens = None
        self._debug_select_ratios = None
        self._debug_score_matrix = None
        self._debug_patch_scores = None
        self._debug_text_scores = None
        self._debug_rater_mask = None

    @staticmethod
    def _extract_score_matrix_from_attn(attn_layer, text_mask, vision_mask):
        """
        attn_layer: [H, T, T] or [1, H, T, T]
        text_mask:  [T] bool
        vision_mask:[T] bool
        return:     [Lt, Lv]
        """
        if attn_layer.dim() == 4:
            attn_layer = attn_layer[0]
        if attn_layer.dim() != 3:
            raise ValueError(f"Expected attention layer [H,T,T], got {tuple(attn_layer.shape)}")

        attn_mean = attn_layer.mean(dim=0)  # [T, T]
        P = attn_mean[text_mask][:, vision_mask]  # [Lt, Lv]
        return torch.nan_to_num(P)

    @staticmethod
    def _aggregate_patch_scores(score_matrix):
        """
        score_matrix: [Lt, Lv]
        returns:
          patch_scores: [Lv]
          text_scores:  [Lt]
          rater_mask:   [Lt]
        """
        if score_matrix.numel() == 0:
            raise ValueError("Empty score_matrix encountered.")

        text_scores = score_matrix.mean(dim=-1)  # [Lt]
        threshold = text_scores.mean()
        rater_mask = text_scores >= threshold
        if not rater_mask.any():
            rater_mask = torch.ones_like(rater_mask, dtype=torch.bool)

        patch_scores = score_matrix[rater_mask].mean(dim=0)  # [Lv]
        min_val = patch_scores.min()
        max_val = patch_scores.max()
        patch_scores = (patch_scores - min_val) / (max_val - min_val + 1e-6)
        return torch.nan_to_num(patch_scores), text_scores, rater_mask

    def _build_prefill_inputs_with_images(
        self,
        *,
        input_ids,
        inputs_embeds,
        pixel_values,
        image_grid_thw,
    ):
        image_tokens_list = self.get_image_features(
            pixel_values,
            image_grid_thw,
            return_dict=True,
        ).pooler_output

        image_embeds = torch.cat(image_tokens_list, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        image_embeds = torch.nan_to_num(image_embeds)
        inputs_embeds = torch.nan_to_num(inputs_embeds)

        image_mask, _ = self.get_placeholder_mask(
            input_ids,
            inputs_embeds=inputs_embeds,
            image_features=image_embeds,
        )
        merged_inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        merged_inputs_embeds = torch.nan_to_num(merged_inputs_embeds)
        return merged_inputs_embeds, image_tokens_list

    def _compute_tree_scores_from_attention(
        self,
        *,
        merged_inputs_embeds,
        image_tokens_list,
        input_ids,
        attention_mask,
        position_ids,
        past_key_values,
        mm_token_type_ids,
        cache_position,
        output_hidden_states,
        use_cache,
        **kwargs,
    ):
        with torch.no_grad():
            pre_outputs = self.language_model(
                input_ids=None,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=merged_inputs_embeds,
                use_cache=False,
                output_attentions=True,
                output_hidden_states=output_hidden_states,
                return_dict=True,
                cache_position=cache_position,
                **kwargs,
            )

        attentions = pre_outputs.attentions
        if attentions is None:
            raise RuntimeError("Failed to extract attentions. Set attn implementation to eager/sdpa.")

        layer_id = min(self.tree_attention_layer, len(attentions) - 1)

        patch_scores_per_image = []
        score_matrices = []
        text_scores_list = []
        rater_masks = []

        B = merged_inputs_embeds.shape[0]
        for i in range(B):
            if mm_token_type_ids is not None:
                text_mask_i = (mm_token_type_ids[i] == 0)
                vision_mask_i = (mm_token_type_ids[i] == 1)
            else:
                raise ValueError("mm_token_type_ids is required to extract text->vision score matrices.")

            score_matrix = self._extract_score_matrix_from_attn(
                attentions[layer_id][i],
                text_mask=text_mask_i,
                vision_mask=vision_mask_i,
            )

            patch_scores, text_scores, rater_mask = self._aggregate_patch_scores(score_matrix)
            expected_lv = image_tokens_list[i].shape[0]
            if patch_scores.numel() != expected_lv:
                raise ValueError(
                    f"Visual token length mismatch for sample {i}: "
                    f"score matrix gives Lv={patch_scores.numel()}, image tokens have {expected_lv}."
                )

            patch_scores_per_image.append(patch_scores)
            score_matrices.append(score_matrix)
            text_scores_list.append(text_scores)
            rater_masks.append(rater_mask)

        return patch_scores_per_image, score_matrices, text_scores_list, rater_masks

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        rope_deltas=None,
        mm_token_type_ids=None,
        cache_position=None,
        second_per_grid_ts=None,
        **kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # -------------------------------------------------
        # Image part: build merged inputs once, extract query-conditioned
        # score matrix from decoder attention, then run quadtree on the
        # resulting patch scores. Visual tokens themselves stay unchanged.
        # -------------------------------------------------
        if pixel_values is not None:
            self._debug_patch_ids = []
            self._debug_num_selected_tokens = []
            self._debug_num_total_tokens = []
            self._debug_select_ratios = []
            self._debug_score_matrix = []
            self._debug_patch_scores = []
            self._debug_text_scores = []
            self._debug_rater_mask = []

            merged_inputs_embeds, image_tokens_list = self._build_prefill_inputs_with_images(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )

            if position_ids is None:
                position_ids = self.compute_3d_position_ids(
                    input_ids=input_ids,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    inputs_embeds=merged_inputs_embeds,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    mm_token_type_ids=mm_token_type_ids,
                )

            patch_scores_per_image, score_matrices, text_scores_list, rater_masks = self._compute_tree_scores_from_attention(
                merged_inputs_embeds=merged_inputs_embeds,
                image_tokens_list=image_tokens_list,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                mm_token_type_ids=mm_token_type_ids,
                cache_position=cache_position,
                output_hidden_states=output_hidden_states,
                use_cache=use_cache,
                **kwargs,
            )

            selected_idx_per_image = []
            for i, tokens in enumerate(image_tokens_list):
                x = tokens.unsqueeze(0)  # [1, Ni, D]

                grid_t, grid_h_raw, grid_w_raw = image_grid_thw[i].tolist()
                grid_h = grid_h_raw // 2
                grid_w = grid_w_raw // 2
                if grid_h * grid_w != tokens.size(0):
                    raise ValueError(
                        f"Downsampled grid mismatch: raw=({grid_h_raw}, {grid_w_raw}), "
                        f"down=({grid_h}, {grid_w}), H*W={grid_h * grid_w}, tokens={tokens.size(0)}"
                    )

                patch_scores_i = patch_scores_per_image[i].unsqueeze(0).to(tokens.device, tokens.dtype)
                out = self.qvtree(
                    x,
                    H=grid_h,
                    W=grid_w,
                    patch_scores=patch_scores_i,
                    score_matrix=score_matrices[i].unsqueeze(0).to(tokens.device, tokens.dtype),
                )

                sel_nodes = out["selected_node_ids"][0]
                selected_idx_per_image.append(sel_nodes)
                patch_ids = out["selected_token_indices"][0]

                if patch_ids.numel() == 0:
                    patch_ids = torch.arange(tokens.size(0), device=tokens.device)
                patch_ids = patch_ids.clamp(min=0, max=tokens.size(0) - 1)
                patch_ids = torch.unique(patch_ids)

                self._debug_patch_ids.append(patch_ids)
                self._debug_score_matrix.append(score_matrices[i])
                self._debug_patch_scores.append(patch_scores_per_image[i])
                self._debug_text_scores.append(text_scores_list[i])
                self._debug_rater_mask.append(rater_masks[i])

                num_selected = int(patch_ids.numel())
                num_total = int(tokens.size(0))
                ratio = num_selected / num_total if num_total > 0 else 0.0
                self._debug_num_selected_tokens.append(num_selected)
                self._debug_num_total_tokens.append(num_total)
                self._debug_select_ratios.append(ratio)

            self._debug_selected_idx = selected_idx_per_image

            # keep original visual tokens unchanged; no alpha/beta scaling
            inputs_embeds = merged_inputs_embeds

        # -------------------------------------------------
        # Video part (unchanged)
        # -------------------------------------------------
        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(
                pixel_values_videos,
                video_grid_thw,
                return_dict=True,
            ).pooler_output

            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                video_features=video_embeds,
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if position_ids is None:
            position_ids = self.compute_3d_position_ids(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                mm_token_type_ids=mm_token_type_ids,
            )

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        output = Qwen2_5_VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )
        return output if return_dict else output.to_tuple()


class Qwen2_5_VLForConditionalGenerationWithTree(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)

        old_state_dict = self.model.state_dict()
        self.model = Qwen2_5_VLModelWithTree(config)
        self.model.load_state_dict(old_state_dict, strict=False)
