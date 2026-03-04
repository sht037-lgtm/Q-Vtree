import torch
import torch.nn as nn
from qvtree import QVTree
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLModel,  # override
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLModelOutputWithPast
)


class Qwen2_5_VLModelWithTree(Qwen2_5_VLModel):
    def __init__(self, config):
        super().__init__(config)

        self.qvtree = QVTree(D=config.text_config.hidden_size)
        self._debug_selected_idx = None
        self._debug_patch_ids = None

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

            # init debug container
            self._debug_patch_ids = []

            # 1. Get patch tokens（List[Tensor(Ni, D)]）
            image_tokens_list = self.get_image_features(
                pixel_values,
                image_grid_thw,
                return_dict=True
            ).pooler_output

            # 2. Compute TEXT semantic query q using LLM hidden states. q: [B, D]
            with torch.no_grad():
                # Run a text-only LM pass to get contextualized representations
                # (We don't inject image tokens here; inputs_embeds is still text embeddings.)
                text_outputs = self.language_model(
                    input_ids=None,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    position_ids=None,  # text-only
                    use_cache=False,
                    return_dict=True,
                    **kwargs,
                )
                hidden = text_outputs.last_hidden_state  # [B, L, D]
                B, L, D = hidden.shape

                # build text_mask on the SAME device as hidden
                if mm_token_type_ids is not None:
                    text_mask = (mm_token_type_ids == 0).to(hidden.device)  # [B, L]
                else:
                    text_mask = torch.ones((B, L), dtype=torch.bool, device=hidden.device)

                # compute average text embedding
                mask = text_mask.unsqueeze(-1)  # [B,L,1]

                q = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)

            selected_idx_per_image = []
            new_image_tokens_list = []

            # run tree for every image
            for i, tokens in enumerate(image_tokens_list):

                # tokens: [Ni, D]
                x = tokens.unsqueeze(0)  # [1, Ni, D]
                """
                to be fixed: only support single image input now.
                """
                qi = q[i:i+1].to(tokens.device, tokens.dtype)  # get i-th averaged query qi. [1, D]

                # selected nodes ids
                out = self.qvtree(x, qi)
                sel_nodes = out["selected_node_ids"][0]
                selected_idx_per_image.append(sel_nodes)
                # tree nodes
                nodes = out["nodes"]
                grid_h = out["H"]
                grid_w = out["W"]

                # node ids -> patch ids
                patch_ids = []
                for nid in sel_nodes:
                    region = nodes[nid].region

                    for r in range(region.r0, region.r1):
                        for c in range(region.c0, region.c1):
                            patch_ids.append(r * grid_w + c)
                patch_ids = sorted(set(patch_ids))

                # debug store
                self._debug_patch_ids.append(patch_ids)

                keep = torch.zeros(tokens.size(0), device=tokens.device, dtype=tokens.dtype)
                keep[patch_ids] = 1.0

                tokens_masked = tokens * keep.unsqueeze(-1)

                # tokens_masked is finally selected tokens with shape [Ni, D]
                new_image_tokens_list.append(tokens_masked)

            # debug save
            self._debug_selected_idx = selected_idx_per_image

            # 5. Concat back. List[Tensor(Ni, D)] -> Tensor(sum_i Ni, D)
            image_embeds = torch.cat(new_image_tokens_list, dim=0).to(
                inputs_embeds.device,
                inputs_embeds.dtype
            )

            # no change
            image_mask, _ = self.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                image_features=image_embeds
            )

            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

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

        # Save original backbone weights
        old_state_dict = self.model.state_dict()

        # Replace backbone
        self.model = Qwen2_5_VLModelWithTree(config)

        # Load the original weight
        self.model.load_state_dict(old_state_dict, strict=False)