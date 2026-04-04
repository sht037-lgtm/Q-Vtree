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


    def _get_raw_qk_scores(self, inputs_embeds, attention_mask, position_ids, input_ids):
        """
        Hook into LLM attention layers to get pre-softmax QK scores.
        Average over layers [8, 16, 24] using last instruction token as query.
        """
        image_token_id = 151655
        is_image = (input_ids[0] == image_token_id)
        vis_positions = is_image.nonzero(as_tuple=True)[0]
        last_que_pos = input_ids.shape[1] - 1  # last token in sequence

        target_layers = [8, 16, 24]
        layer_outputs = {}

        def make_hook(layer_idx):
            def hook(module, args, kwargs, output):
                hidden = args[0]
                with torch.no_grad():
                    q = module.q_proj(hidden)
                    k = module.k_proj(hidden)
                layer_outputs[layer_idx] = (q.detach().cpu(), k.detach().cpu())
            return hook

        hooks = []
        for layer_idx in target_layers:
            layer = self.language_model.model.layers[layer_idx].self_attn
            h = layer.register_forward_hook(make_hook(layer_idx), with_kwargs=True)
            hooks.append(h)

        with torch.no_grad():
            self.language_model(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
                use_cache=False,
            )

        for h in hooks:
            h.remove()

        qk_scores = []
        for layer_idx in target_layers:
            q, k = layer_outputs[layer_idx]  # [B, L, HD] on cpu
            B, L, HD = q.shape
            attn_module = self.language_model.model.layers[layer_idx].self_attn
            num_heads = attn_module.num_heads
            num_kv_heads = attn_module.num_key_value_heads
            head_dim = HD // num_heads

            q = q.view(B, L, num_heads, head_dim).transpose(1, 2)
            kd = k.shape[-1] // num_kv_heads
            k = k.view(B, L, num_kv_heads, kd).transpose(1, 2)

            n_rep = num_heads // num_kv_heads
            if n_rep > 1:
                k = k.repeat_interleave(n_rep, dim=1)

            vp = vis_positions.cpu()
            q_last = q[0, :, last_que_pos, :]   # [heads, head_dim]
            k_vis  = k[0, :, vp, :]             # [heads, N, head_dim]

            scores = torch.einsum('hd,hnd->hn', q_last, k_vis) / (head_dim ** 0.5)
            scores = scores.mean(dim=0)          # [N]
            qk_scores.append(scores)

        patch_scores = torch.stack(qk_scores).mean(dim=0)  # [N]
        s_min = patch_scores.min()
        s_max = patch_scores.max()
        patch_scores = (patch_scores - s_min) / (s_max - s_min + 1e-6)
        return patch_scores, vis_positions

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

            # 2. Build full inputs_embeds for first forward pass
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

            # 3. Compute position_ids based on full sequence
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

            # 4. Extract raw QK scores from layers [8, 16, 24]
            patch_scores_global, vis_positions = self._get_raw_qk_scores(
                inputs_embeds=inputs_embeds_full,
                attention_mask=attention_mask,
                position_ids=position_ids,
                input_ids=input_ids,
            )
            self._debug_patch_scores = [patch_scores_global]


            selected_idx_per_image = []
            new_image_tokens_list = []

            # run tree for every image
            for i, tokens in enumerate(image_tokens_list):
                # vision tokens: [Ni, D]
                x = tokens.unsqueeze(0)  # [1, Ni, D]

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

                # use self-attention patch scores directly
                patch_scores = patch_scores_global.unsqueeze(0).to(
                    tokens.device, tokens.dtype
                )  # [1, N]

                # build quadtree and navigate
                built = self.qvtree.builder.build(x, grid_h, grid_w)
                nodes = built["nodes"]

                sel_nodes_list, _ = self.qvtree.navigator.select_nodes(
                    nodes=nodes,
                    patch_scores=patch_scores,
                    W=grid_w,
                )
                sel_nodes = sel_nodes_list[0]
                selected_idx_per_image.append(sel_nodes)

                # node ids -> patch ids
                token_out = self.qvtree.navigator.nodes_to_tokens(
                    nodes,
                    H=grid_h,
                    W=grid_w,
                    selected_node_ids=sel_nodes_list,
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