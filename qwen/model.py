import torch
from module import QVTree
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLModel,  # override
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLModelOutputWithPast
)

# Qwen2.5-VL image token id
IMAGE_TOKEN_ID = 151655

# which LLM layer to extract attention from (0-indexed)
ATTN_LAYER_IDX = 16


class Qwen2_5_VLModelWithTree(Qwen2_5_VLModel):
    def __init__(self, config):
        super().__init__(config)

        self.qvtree = QVTree(D=config.text_config.hidden_size)

        # debug states
        self._debug_selected_idx = None
        self._debug_patch_ids = None
        self._debug_patch_scores = None
        self._debug_num_selected_tokens = None
        self._debug_num_total_tokens = None
        self._debug_select_ratios = None

    def _extract_patch_scores_from_attention(
        self,
        inputs_embeds,
        attention_mask,
        position_ids,
        input_ids,
    ):
        """
        First forward pass (no grad, output_attentions=True).
        Returns:
          patch_scores: [N] relevance score per visual token
          visual_positions: [N] sequence positions of visual tokens
        """
        with torch.no_grad():
            first_out = self.language_model(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=True,
                output_hidden_states=False,
                return_dict=True,
                use_cache=False,
            )

        if first_out.attentions is None:
            raise RuntimeError(
                "output_attentions=True returned None. "
                "Load the model with attn_implementation='eager'."
            )

        # locate image and question token positions (batch item 0)
        is_image = (input_ids[0] == IMAGE_TOKEN_ID)  # [L]
        if not is_image.any():
            return None, None

        img_positions = is_image.nonzero(as_tuple=True)[0]
        img_end = img_positions[-1].item()

        attn_device = first_out.attentions[0].device
        visual_positions = img_positions.to(attn_device)
        question_positions = torch.arange(
            img_end + 1, input_ids.shape[1], device=attn_device
        )                                                          # [Lq]

        if question_positions.numel() == 0:
            return None, None

        # multi-layer average to reduce attention sink effect
        layers = [8, 12, 16, 20, 24]
        attn_q2v_list = []
        for l in layers:
            layer_attn = first_out.attentions[l]  # [B, heads, L, L]
            attn_q2v_list.append(
                layer_attn[
                    0,                              # batch item 0
                    :,                              # all heads
                    question_positions[:, None],    # [Lq, 1]
                    visual_positions[None, :],      # [1,  N]
                ].mean(dim=[0, 1])                  # [N]
            )

        # average over layers -> [N]
        patch_scores = torch.stack(attn_q2v_list).mean(dim=0)

        # min-max normalise to [0, 1]
        s_min = patch_scores.min()
        s_max = patch_scores.max()
        patch_scores = (patch_scores - s_min) / (s_max - s_min + 1e-6)

        return patch_scores, visual_positions

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

        # =============================
        # Vision Part
        # =============================
        if pixel_values is not None:

            # init debug containers
            self._debug_patch_ids = []
            self._debug_patch_scores = []
            self._debug_num_selected_tokens = []
            self._debug_num_total_tokens = []
            self._debug_select_ratios = []

            # 1. Get patch tokens (List[Tensor(Ni, D)]) after downsample
            image_tokens_list = self.get_image_features(
                pixel_values,
                image_grid_thw,
                return_dict=True
            ).pooler_output

            # 2. Build full inputs_embeds with all visual tokens for
            #    the first forward pass
            image_embeds_full = torch.cat(image_tokens_list, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            image_embeds_full = torch.nan_to_num(image_embeds_full)
            inputs_embeds_full = torch.nan_to_num(inputs_embeds.clone())

            image_mask_full, _ = self.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds_full,
                image_features=image_embeds_full,
            )
            inputs_embeds_full = inputs_embeds_full.masked_scatter(
                image_mask_full, image_embeds_full
            )
            inputs_embeds_full = torch.nan_to_num(inputs_embeds_full)

            # 3. Compute position_ids once based on full sequence
            if position_ids is None:
                position_ids = self.compute_3d_position_ids(
                    input_ids=input_ids,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    inputs_embeds=inputs_embeds_full,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    mm_token_type_ids=mm_token_type_ids,
                )

            # 4. First forward pass: extract attention-based patch scores
            patch_scores_global, visual_positions = \
                self._extract_patch_scores_from_attention(
                    inputs_embeds=inputs_embeds_full,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    input_ids=input_ids,
                )

            # 5. Per-image quadtree navigation using attention scores
            selected_idx_per_image = []
            new_image_tokens_list = []

            for i, tokens in enumerate(image_tokens_list):
                x = tokens.unsqueeze(0)  # [1, Ni, D]

                grid_t, grid_h_raw, grid_w_raw = image_grid_thw[i].tolist()
                grid_h = grid_h_raw // 2
                grid_w = grid_w_raw // 2

                if grid_h * grid_w != tokens.size(0):
                    raise ValueError(
                        f"Downsampled grid mismatch: "
                        f"raw=({grid_h_raw},{grid_w_raw}), "
                        f"down=({grid_h},{grid_w}), "
                        f"H*W={grid_h * grid_w}, tokens={tokens.size(0)}"
                    )

                patch_offset = tokens.size(0) - grid_h * grid_w

                # use attention-derived scores (only implemented for i==0 /
                # single-image; fallback to uniform for additional images)
                if patch_scores_global is not None and i == 0:
                    grid_scores = patch_scores_global.unsqueeze(0)  # [1, N]
                else:
                    grid_scores = torch.ones(
                        1, grid_h * grid_w,
                        device=tokens.device, dtype=tokens.dtype
                    )

                # build quadtree and navigate
                built = self.qvtree.builder.build(x, grid_h, grid_w)
                nodes = built["nodes"]

                selected_node_ids, _ = self.qvtree.navigator.select_nodes(
                    nodes=nodes,
                    patch_scores=grid_scores.to(tokens.device),
                    W=grid_w,
                )
                selected_idx_per_image.append(selected_node_ids[0])

                token_out = self.qvtree.navigator.nodes_to_tokens(
                    nodes=nodes,
                    H=grid_h,
                    W=grid_w,
                    selected_node_ids=selected_node_ids,
                    x=x,
                )

                patch_ids = token_out["selected_token_indices"][0]

                # fallback: keep all if empty
                if patch_ids.numel() == 0:
                    patch_ids = torch.arange(tokens.size(0), device=tokens.device)

                patch_ids = patch_ids.clamp(0, tokens.size(0) - 1)
                patch_ids = torch.unique(patch_ids)
                patch_ids = patch_ids + patch_offset

                # debug
                self._debug_patch_ids.append(patch_ids)
                self._debug_patch_scores.append(
                    grid_scores.squeeze(0) if patch_scores_global is not None else None
                )
                num_selected = int(patch_ids.numel())
                num_total = int(tokens.size(0))
                self._debug_num_selected_tokens.append(num_selected)
                self._debug_num_total_tokens.append(num_total)
                self._debug_select_ratios.append(
                    num_selected / num_total if num_total > 0 else 0.0
                )

                # tokens pass through unchanged; selection enforced via attn mask
                new_image_tokens_list.append(tokens)

            self._debug_selected_idx = selected_idx_per_image

            # 6. Build final inputs_embeds
            image_embeds = torch.cat(new_image_tokens_list, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            image_embeds = torch.nan_to_num(image_embeds)
            inputs_embeds = torch.nan_to_num(inputs_embeds)

            image_mask, _ = self.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                image_features=image_embeds,
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            inputs_embeds = torch.nan_to_num(inputs_embeds)

            # 7. Suppress non-selected visual tokens via attention mask
            if visual_positions is not None and self._debug_patch_ids:
                modified_mask = attention_mask.clone()  # [B, L]

                for b in range(inputs_embeds.shape[0]):
                    if b >= len(self._debug_patch_ids):
                        continue

                    patch_ids = self._debug_patch_ids[b]
                    grid_t, grid_h_raw, grid_w_raw = image_grid_thw[b].tolist()
                    grid_h = grid_h_raw // 2
                    grid_w = grid_w_raw // 2
                    patch_offset = image_tokens_list[b].size(0) - grid_h * grid_w

                    relative_selected = (patch_ids - patch_offset).clamp(
                        0, len(visual_positions) - 1
                    )

                    non_selected = torch.ones(
                        len(visual_positions), dtype=torch.bool,
                        device=modified_mask.device
                    )
                    non_selected[relative_selected] = False
                    non_selected_positions = visual_positions[non_selected]
                    modified_mask[b, non_selected_positions] = 0

                attention_mask = modified_mask

        # =============================
        # Video part (no change)
        # =============================
        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(
                pixel_values_videos,
                video_grid_thw,
                return_dict=True
            ).pooler_output

            video_embeds = torch.cat(video_embeds, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )

            _, video_mask = self.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        # =============================
        # position_ids (only if not already computed above)
        # =============================
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

        # =============================
        # Second (main) LLM forward
        # =============================
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

        # Save original backbone weights
        old_state_dict = self.model.state_dict()

        # Replace backbone
        self.model = Qwen2_5_VLModelWithTree(config)

        # Load the original weight
        self.model.load_state_dict(old_state_dict, strict=False)