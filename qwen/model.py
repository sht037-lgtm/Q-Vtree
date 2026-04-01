import torch
from module import QVTree
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLModel,  # override
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLModelOutputWithPast
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
            self._debug_num_selected_tokens = []
            self._debug_num_total_tokens = []
            self._debug_select_ratios = []

            # 1. Get patch tokens（List[Tensor(Ni, D)]）(After downsample)
            image_tokens_list = self.get_image_features(
                pixel_values,
                image_grid_thw,
                return_dict=True
            ).pooler_output

            # 2. Compute TEXT tokens using embedding (NO LLM)
            with torch.no_grad():
                text_embed = inputs_embeds  # [B, L, D]

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

            print(f"[DEBUG] total seq len: {inputs_embeds.shape[1]}")  # 应该是1509左右
            print(f"[DEBUG] visual tokens: {(mm_token_type_ids[0] == 1).sum().item()}")  # 740
            print(f"[DEBUG] text tokens: {(mm_token_type_ids[0] == 0).sum().item()}")  # 769
            print(f"[DEBUG] img_end position: {(mm_token_type_ids[0] == 1).nonzero(as_tuple=True)[0][-1].item()}")
            print(f"[DEBUG] question tokens after img_end: {inputs_embeds.shape[1] - (mm_token_type_ids[0] == 1).nonzero(as_tuple=True)[0][-1].item() - 1}")

            selected_idx_per_image = []
            new_image_tokens_list = []

            # run tree for every image
            for i, tokens in enumerate(image_tokens_list):
                # vision tokens: [Ni, D]
                x = tokens.unsqueeze(0)  # [1, Ni, D]

                # text tokens
                ti = text_tokens.to(tokens.device, tokens.dtype)  # [1, Lt, D]
                print(f"[DEBUG] x shape: {x.shape}")
                print(f"[DEBUG] ti shape: {ti.shape}")

                # infer downsampled grid
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

                # run tree
                out = self.qvtree(x, ti, H=grid_h, W=grid_w)

                sel_nodes = out["selected_node_ids"][0]
                selected_idx_per_image.append(sel_nodes)

                nodes = out["nodes"]
                grid_h = out["H"]
                grid_w = out["W"]

                # node ids -> patch ids
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

                # debug store
                self._debug_patch_ids.append(patch_ids)

                num_selected = int(patch_ids.numel())
                num_total = int(tokens.size(0))
                select_ratio = num_selected / num_total if num_total > 0 else 0.0

                self._debug_num_selected_tokens.append(num_selected)
                self._debug_num_total_tokens.append(num_total)
                self._debug_select_ratios.append(select_ratio)

                # =============================
                # alpha-beta soft masking
                # =============================
                alpha = 5.0  # selected
                beta = 0.5  # unselected

                keep = torch.full(
                    (tokens.size(0),),
                    beta,
                    device=tokens.device,
                    dtype=tokens.dtype,
                )

                keep[patch_ids] = alpha

                tokens_modulated = tokens * keep.unsqueeze(-1)
                tokens_modulated = torch.nan_to_num(tokens_modulated)

                new_image_tokens_list.append(tokens_modulated)

            # debug save
            self._debug_selected_idx = selected_idx_per_image

            # 5. Concat back. List[Tensor(Ni, D)] -> Tensor(sum_i Ni, D)
            image_embeds = torch.cat(new_image_tokens_list, dim=0).to(
                inputs_embeds.device,
                inputs_embeds.dtype,
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