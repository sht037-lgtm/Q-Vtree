import torch
from module import QVTree
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLModel,  # override
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLModelOutputWithPast
)


def prune_kv_cache(past_key_values, keep_positions):
    if past_key_values is None:
        return None

    new_past = []
    for k, v in past_key_values:
        # k/v: [B, heads, seq_len, dim]
        k = k[:, :, keep_positions, :]
        v = v[:, :, keep_positions, :]
        new_past.append((k, v))

    return new_past


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

        # 保存 full position ids
        full_position_ids = position_ids

        # =============================
        # Vision Part
        # =============================
        if pixel_values is not None:

            self._debug_patch_ids = []
            self._debug_num_selected_tokens = []
            self._debug_num_total_tokens = []
            self._debug_select_ratios = []

            image_tokens_list = self.get_image_features(
                pixel_values,
                image_grid_thw,
                return_dict=True
            ).pooler_output

            with torch.no_grad():
                text_embed = inputs_embeds

                if mm_token_type_ids is not None:
                    text_mask = (mm_token_type_ids == 0)
                else:
                    text_mask = torch.ones(
                        (text_embed.shape[0], text_embed.shape[1]),
                        dtype=torch.bool,
                        device=text_embed.device,
                    )

                text_tokens = text_embed[text_mask].view(1, -1, text_embed.size(-1))
                text_tokens = text_tokens.to(inputs_embeds.device, inputs_embeds.dtype)

            selected_idx_per_image = []
            new_image_tokens_list = []

            for i, tokens in enumerate(image_tokens_list):

                x = tokens.unsqueeze(0)
                ti = text_tokens.to(tokens.device, tokens.dtype)

                grid_t, grid_h_raw, grid_w_raw = image_grid_thw[i].tolist()
                grid_h = grid_h_raw // 2
                grid_w = grid_w_raw // 2

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

                if patch_ids.numel() == 0:
                    patch_ids = torch.arange(tokens.size(0), device=tokens.device)

                patch_ids = patch_ids.clamp(min=0, max=tokens.size(0) - 1)
                patch_ids = torch.unique(patch_ids)

                patch_offset = tokens.size(0) - grid_h * grid_w
                patch_ids = patch_ids + patch_offset
                patch_ids = patch_ids.clamp(min=0, max=tokens.size(0) - 1)

                self._debug_patch_ids.append(patch_ids)

                num_selected = int(patch_ids.numel())
                num_total = int(tokens.size(0))
                select_ratio = num_selected / num_total

                self._debug_num_selected_tokens.append(num_selected)
                self._debug_num_total_tokens.append(num_total)
                self._debug_select_ratios.append(select_ratio)

                # 关键修改：只保留 selected tokens
                selected_tokens = tokens[patch_ids]
                selected_tokens = torch.nan_to_num(selected_tokens)

                new_image_tokens_list.append(selected_tokens)

            self._debug_selected_idx = selected_idx_per_image

            selected_image_embeds = torch.cat(new_image_tokens_list, dim=0).to(
                inputs_embeds.device,
                inputs_embeds.dtype,
            )

            # -------------------------------
            # placeholder reduction
            # -------------------------------

            VISION_TOKEN_ID = 151655

            vision_mask = (input_ids == VISION_TOKEN_ID)

            vision_positions = vision_mask.nonzero(as_tuple=False)[:, 1]
            text_positions = (~vision_mask).nonzero(as_tuple=False)[:, 1]

            selected_patch_ids = torch.cat(self._debug_patch_ids)

            selected_vision_positions = vision_positions[selected_patch_ids]

            keep_positions = torch.cat([
                text_positions,
                selected_vision_positions
            ])

            keep_positions = torch.sort(keep_positions).values

            # -------------------------------
            # prune sequence
            # -------------------------------

            input_ids = input_ids[:, keep_positions]

            if attention_mask is not None:
                attention_mask = attention_mask[:, keep_positions]

            inputs_embeds = inputs_embeds[:, keep_positions]

            if mm_token_type_ids is not None:
                mm_token_type_ids = mm_token_type_ids[:, keep_positions]

            if full_position_ids is not None:
                position_ids = full_position_ids[:, :, keep_positions]

            if cache_position is not None:
                cache_position = torch.arange(
                    keep_positions.numel(),
                    device=keep_positions.device,
                    dtype=cache_position.dtype,
                )

            # -------------------------------
            # scatter selected vision tokens
            # -------------------------------

            image_mask, _ = self.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                image_features=selected_image_embeds,
            )

            inputs_embeds = inputs_embeds.masked_scatter(image_mask, selected_image_embeds)
            inputs_embeds = torch.nan_to_num(inputs_embeds)

        # =============================
        # Video part
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
            raise RuntimeError("position_ids should not be None after pruning")

        print("input_ids:", None if input_ids is None else input_ids.shape)
        print("inputs_embeds:", None if inputs_embeds is None else inputs_embeds.shape)
        print("attention_mask:", None if attention_mask is None else attention_mask.shape)
        print("position_ids:", None if position_ids is None else position_ids.shape)
        print("cache_position:", None if cache_position is None else cache_position.shape)

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

        # Save original backbone weights
        old_state_dict = self.model.state_dict()

        # Replace backbone
        self.model = Qwen2_5_VLModelWithTree(config)

        # Load the original weight
        self.model.load_state_dict(old_state_dict, strict=False)