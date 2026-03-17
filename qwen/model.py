import torch
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLModel,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLModelOutputWithPast,
)


class Qwen2_5_VLModelWithTree(Qwen2_5_VLModel):
    def __init__(self, config):
        super().__init__(config)

        self.qvtree = QVTree(D=config.text_config.hidden_size)

        # debug states
        self._debug_selected_idx = None
        self._debug_patch_ids = None
        self._debug_num_selected_tokens = None
        self._debug_num_total_tokens = None
        self._debug_select_ratios = None

        # observed from your processor output
        self.vision_start_id = 151652
        self.vision_token_id = 151655
        self.vision_end_id = 151653

    def _get_vision_token_positions(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Return positions of <vision_token> in the sequence.
        Assumes batch size = 1.
        """
        assert input_ids is not None and input_ids.size(0) == 1, "Current implementation assumes batch size = 1."
        return (input_ids[0] == self.vision_token_id).nonzero(as_tuple=False).squeeze(-1)

    def _slice_position_ids(self, position_ids: torch.Tensor, keep_indices: torch.Tensor) -> torch.Tensor:
        """
        Slice position_ids on sequence dim while preserving original layout.
        Handles both [3, B, L] and [B, 3, L].
        """
        if position_ids.dim() != 3:
            raise ValueError(f"Unexpected position_ids shape: {tuple(position_ids.shape)}")

        if position_ids.size(0) == 3:
            # [3, B, L]
            return position_ids[:, :, keep_indices]
        elif position_ids.size(1) == 3:
            # [B, 3, L]
            return position_ids[:, :, keep_indices]
        else:
            raise ValueError(f"Cannot parse position_ids shape: {tuple(position_ids.shape)}")

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

        # save originals for full position-id computation
        original_input_ids = input_ids
        original_inputs_embeds = inputs_embeds
        original_attention_mask = attention_mask
        original_mm_token_type_ids = mm_token_type_ids

        # =============================
        # Vision Part
        # =============================
        if pixel_values is not None:
            assert input_ids is not None and input_ids.size(0) == 1, "Current implementation assumes batch size = 1."

            # init debug containers
            self._debug_patch_ids = []
            self._debug_num_selected_tokens = []
            self._debug_num_total_tokens = []
            self._debug_select_ratios = []

            # 1) full position ids on ORIGINAL full sequence, before any pruning
            if position_ids is None:
                full_position_ids = self.compute_3d_position_ids(
                    input_ids=original_input_ids,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    inputs_embeds=original_inputs_embeds,
                    attention_mask=original_attention_mask,
                    past_key_values=past_key_values,
                    mm_token_type_ids=original_mm_token_type_ids,
                )
            else:
                full_position_ids = position_ids

            # 2) get all vision tokens from vision tower
            image_tokens_list = self.get_image_features(
                pixel_values,
                image_grid_thw,
                return_dict=True
            ).pooler_output

            # 3) build text tokens from ORIGINAL sequence, excluding only <vision_token>
            with torch.no_grad():
                text_embed = original_inputs_embeds  # [1, L, D]
                text_mask = (original_input_ids != self.vision_token_id)
                text_tokens = text_embed[text_mask].view(1, -1, text_embed.size(-1))
                text_tokens = text_tokens.to(original_inputs_embeds.device, original_inputs_embeds.dtype)

            selected_idx_per_image = []
            new_image_tokens_list = []
            selected_placeholder_positions_all = []

            # all original <vision_token> positions in sequence
            vision_positions_full = self._get_vision_token_positions(original_input_ids)

            # we need to split these positions by image, following image_tokens_list order
            cursor = 0

            for i, tokens in enumerate(image_tokens_list):
                # tokens: [Ni, D]
                x = tokens.unsqueeze(0)  # [1, Ni, D]
                ti = text_tokens.to(tokens.device, tokens.dtype)

                grid_t, grid_h_raw, grid_w_raw = image_grid_thw[i].tolist()
                grid_h = grid_h_raw // 2
                grid_w = grid_w_raw // 2

                if grid_h * grid_w != tokens.size(0):
                    raise ValueError(
                        f"Downsampled grid mismatch: "
                        f"raw=({grid_h_raw}, {grid_w_raw}), "
                        f"down=({grid_h}, {grid_w}), "
                        f"H*W={grid_h * grid_w}, "
                        f"tokens={tokens.size(0)}"
                    )

                # placeholder positions for this image in ORIGINAL full sequence
                num_tokens_i = tokens.size(0)
                positions_i = vision_positions_full[cursor: cursor + num_tokens_i]
                cursor += num_tokens_i

                if positions_i.numel() != num_tokens_i:
                    raise ValueError(
                        f"Placeholder/token mismatch for image {i}: "
                        f"placeholder count={positions_i.numel()}, token count={num_tokens_i}"
                    )

                # run tree
                out = self.qvtree(x, ti, H=grid_h, W=grid_w)

                sel_nodes = out["selected_node_ids"][0]
                selected_idx_per_image.append(sel_nodes)

                nodes = out["nodes"]
                grid_h = out["H"]
                grid_w = out["W"]

                token_out = self.qvtree.navigator.nodes_to_tokens(
                    nodes,
                    H=grid_h,
                    W=grid_w,
                    selected_node_ids=[sel_nodes],
                    x=x,
                )

                patch_ids = token_out["selected_token_indices"][0]

                # fallback 1: if empty patch, keep all
                if patch_ids.numel() == 0:
                    patch_ids = torch.arange(tokens.size(0), device=tokens.device)

                # fallback 2: boundary inspection
                patch_ids = patch_ids.clamp(min=0, max=tokens.size(0) - 1)
                patch_ids = torch.unique(patch_ids)

                patch_offset = tokens.size(0) - grid_h * grid_w
                patch_ids = patch_ids + patch_offset
                patch_ids = patch_ids.clamp(min=0, max=tokens.size(0) - 1)
                patch_ids = torch.unique(patch_ids)

                # debug
                self._debug_patch_ids.append(patch_ids)

                num_selected = int(patch_ids.numel())
                num_total = int(tokens.size(0))
                select_ratio = num_selected / num_total if num_total > 0 else 0.0

                self._debug_num_selected_tokens.append(num_selected)
                self._debug_num_total_tokens.append(num_total)
                self._debug_select_ratios.append(select_ratio)

                # selected visual tokens
                selected_tokens = tokens[patch_ids]
                new_image_tokens_list.append(selected_tokens)

                # map selected patch ids -> ORIGINAL placeholder positions
                selected_placeholder_positions = positions_i[patch_ids]
                selected_placeholder_positions_all.append(selected_placeholder_positions)

            self._debug_selected_idx = selected_idx_per_image

            if cursor != vision_positions_full.numel():
                raise ValueError(
                    f"Not all vision placeholders were consumed: consumed={cursor}, total={vision_positions_full.numel()}"
                )

            # concat selected image tokens
            image_embeds = torch.cat(new_image_tokens_list, dim=0).to(
                original_inputs_embeds.device,
                original_inputs_embeds.dtype,
            )
            image_embeds = torch.nan_to_num(image_embeds)

            # concat selected placeholder positions in ORIGINAL full sequence
            selected_placeholder_positions_all = torch.cat(selected_placeholder_positions_all, dim=0)
            selected_placeholder_positions_all = torch.unique(selected_placeholder_positions_all, sorted=True)

            # 4) build keep mask on ORIGINAL full sequence
            seq_len = original_input_ids.size(1)
            keep_mask = torch.ones(seq_len, dtype=torch.bool, device=original_input_ids.device)

            # remove all <vision_token> first
            keep_mask[vision_positions_full] = False
            # add back selected <vision_token> positions
            keep_mask[selected_placeholder_positions_all] = True

            keep_indices = keep_mask.nonzero(as_tuple=False).squeeze(-1)

            # 5) prune sequence tensors
            input_ids = original_input_ids[:, keep_indices]
            inputs_embeds = original_inputs_embeds[:, keep_indices]

            if original_attention_mask is not None:
                attention_mask = original_attention_mask[:, keep_indices]
            else:
                attention_mask = None

            if original_mm_token_type_ids is not None:
                mm_token_type_ids = original_mm_token_type_ids[:, keep_indices]

            # 6) prune position_ids using SAME keep_indices
            position_ids = self._slice_position_ids(full_position_ids, keep_indices)

            # 7) scatter selected image_embeds into reduced placeholders
            image_mask, _ = self.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                image_features=image_embeds,
            )

            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            inputs_embeds = torch.nan_to_num(inputs_embeds)

            # keep cache_position consistent with reduced sequence during prefill
            if cache_position is not None and cache_position.numel() != inputs_embeds.size(1):
                cache_position = torch.arange(
                    inputs_embeds.size(1),
                    device=inputs_embeds.device,
                    dtype=cache_position.dtype,
                )

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
                inputs_embeds.device,
                inputs_embeds.dtype
            )

            _, video_mask = self.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                video_features=video_embeds
            )

            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        # =============================
        # position ids
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
        # LLM
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

        old_state_dict = self.model.state_dict()
        self.model = Qwen2_5_VLModelWithTree(config)
        self.model.load_state_dict(old_state_dict, strict=False)