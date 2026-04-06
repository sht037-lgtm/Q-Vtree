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

        # pre-encode generic description for relative attention
        # hardcoded token ids for "Write a general description of the image."
        # using qwen2.5 tokenizer token ids
        self._generic_token_ids = None  # will be lazily initialized in forward

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

            # 4. Relative attention scoring (ViCrop style)
            # Two forward passes: question-specific and generic description
            # Use starting answer token (last token) attention to image tokens
            image_token_id = 151655
            is_image = (input_ids[0] == image_token_id)
            vis_positions = is_image.nonzero(as_tuple=True)[0].cpu()

            def get_attn_scores(inputs_embeds_in, attention_mask_in, position_ids_in):
                with torch.no_grad():
                    out = self.language_model(
                        input_ids=None,
                        inputs_embeds=inputs_embeds_in,
                        attention_mask=attention_mask_in,
                        position_ids=position_ids_in,
                        output_attentions=True,
                        output_hidden_states=False,
                        return_dict=True,
                        use_cache=False,
                    )
                # starting answer token = last token in sequence
                ans_pos = inputs_embeds_in.shape[1] - 1
                # average over 4 evenly-spaced layers: [6, 13, 20, 27]
                target_layers = [27]
                scores = []
                for i, layer_attn in enumerate(out.attentions):
                    if i not in target_layers:
                        continue
                    ld = layer_attn.device
                    vp = vis_positions.to(ld)
                    s = layer_attn[0, :, ans_pos, vp].mean(dim=0).cpu()
                    scores.append(s)
                return torch.stack(scores).mean(dim=0)  # [N]

            # First pass: question-specific attention
            A_q = get_attn_scores(inputs_embeds_full, attention_mask, position_ids)

            # Second pass: generic description attention
            # Build generic inputs_embeds by replacing question tokens with generic description
            # Use the processor to build generic input_ids
            generic_text = "Write a general description of the image."
            # locate img_end to replace question tokens
            img_end = vis_positions[-1].item()

            # encode generic text (lazy init using language model's tokenizer vocab)
            if self._generic_token_ids is None:
                from transformers import AutoTokenizer
                _tok = AutoTokenizer.from_pretrained(self.config._name_or_path)
                _generic_text = "Write a general description of the image."
                self._generic_token_ids = _tok(
                    _generic_text, return_tensors="pt", add_special_tokens=False
                ).input_ids
            generic_ids = self._generic_token_ids.to(input_ids.device)

            # build new input_ids: keep prefix+image, replace question with generic
            prefix_ids = input_ids[:, :img_end + 1]
            # get suffix (assistant start tokens) from original
            # find where question ends and assistant starts
            # heuristic: keep last 3 tokens (im_end + im_start + assistant)
            suffix_ids = input_ids[:, -3:]
            new_input_ids = torch.cat([prefix_ids, generic_ids, suffix_ids], dim=1)

            # rebuild inputs_embeds for generic
            generic_embeds = self.get_input_embeddings()(new_input_ids)
            image_mask_generic, _ = self.get_placeholder_mask(
                new_input_ids,
                inputs_embeds=generic_embeds,
                image_features=image_embeds_full,
            )
            generic_embeds = generic_embeds.masked_scatter(image_mask_generic, image_embeds_full)
            generic_embeds = torch.nan_to_num(generic_embeds)

            # build attention mask and position_ids for generic
            generic_attn_mask = torch.ones(
                1, new_input_ids.shape[1],
                dtype=attention_mask.dtype,
                device=attention_mask.device
            )
            generic_pos_ids = self.compute_3d_position_ids(
                input_ids=new_input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                inputs_embeds=generic_embeds,
                attention_mask=generic_attn_mask,
                past_key_values=None,
            )

            A_generic = get_attn_scores(generic_embeds, generic_attn_mask, generic_pos_ids)

            # relative attention: element-wise division
            patch_scores_global = A_q / (A_generic + 1e-8)
            s_min = patch_scores_global.min()
            s_max = patch_scores_global.max()
            patch_scores_global = (patch_scores_global - s_min) / (s_max - s_min + 1e-6)

            # Gaussian smoothing: reshape to 2D spatial grid, smooth, reshape back
            # sigma and ks control spread; ks must be odd (1,3,5,7,...)
            grid_t0, grid_h0_raw, grid_w0_raw = image_grid_thw[0].tolist()
            grid_h0 = grid_h0_raw // 2
            grid_w0 = grid_w0_raw // 2
            if patch_scores_global.shape[0] == grid_h0 * grid_w0:
                import torch.nn.functional as F
                sigma, ks = 1.0, 3  # tune: larger sigma/ks = more spread
                ax = torch.arange(ks, dtype=torch.float32) - ks // 2
                gauss_1d = torch.exp(-ax ** 2 / (2 * sigma ** 2))
                gauss_1d = gauss_1d / gauss_1d.sum()
                kernel = (gauss_1d.unsqueeze(1) * gauss_1d.unsqueeze(0)).view(1, 1, ks, ks)
                score_map = patch_scores_global.float().view(1, 1, grid_h0, grid_w0)
                score_map = F.conv2d(score_map, kernel, padding=ks // 2)
                patch_scores_global = score_map.view(-1)
                s_min = patch_scores_global.min()
                s_max = patch_scores_global.max()
                patch_scores_global = (patch_scores_global - s_min) / (s_max - s_min + 1e-6)

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