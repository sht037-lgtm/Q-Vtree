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
        self._debug_selected_idx = None
        self._debug_patch_ids = None

    def _expand_multimodal_sequence_for_selected_tokens(
            self,
            inputs_embeds,
            attention_mask,
            mm_token_type_ids,
            image_token_lengths,
            num_selected_tokens_per_image,
    ):
        """
        Assume batch size = 1.

        Original sequence layout:
            text + image_tokens + text + image_tokens + ...

        We expand each image span from N -> N+K by appending K zero slots
        right after the original image span. Later these zero slots will be
        filled by concatenated image_embeds through a manual scatter-by-span.
        """
        device = inputs_embeds.device
        dtype = inputs_embeds.dtype

        B, L, D = inputs_embeds.shape
        assert B == 1, "Current implementation only supports batch size = 1."

        if mm_token_type_ids is None:
            raise ValueError("mm_token_type_ids is required for selected-token concat mode.")

        old_embeds = inputs_embeds[0]  # [L, D]
        old_mask = attention_mask[0] if attention_mask is not None else torch.ones(L, device=device, dtype=torch.long)
        old_mmtype = mm_token_type_ids[0]  # [L]

        # find all image-token positions
        image_pos = torch.nonzero(old_mmtype == 1, as_tuple=False).squeeze(-1)
        if image_pos.numel() == 0:
            return inputs_embeds, attention_mask, mm_token_type_ids, []

        # split image_pos into contiguous spans
        spans = []
        start = image_pos[0].item()
        prev = image_pos[0].item()
        for p in image_pos[1:].tolist():
            if p == prev + 1:
                prev = p
            else:
                spans.append((start, prev + 1))  # [start, end)
                start = p
                prev = p
        spans.append((start, prev + 1))

        if len(spans) != len(image_token_lengths):
            raise ValueError(
                f"Found {len(spans)} image spans in mm_token_type_ids, "
                f"but got {len(image_token_lengths)} images from vision encoder."
            )

        new_embed_chunks = []
        new_mask_chunks = []
        new_mmtype_chunks = []
        image_aug_spans = []

        cursor = 0
        for i, ((s, e), n_img, n_sel) in enumerate(zip(spans, image_token_lengths, num_selected_tokens_per_image)):
            # text chunk before image span
            if cursor < s:
                new_embed_chunks.append(old_embeds[cursor:s])
                new_mask_chunks.append(old_mask[cursor:s])
                new_mmtype_chunks.append(old_mmtype[cursor:s])

            # original image span length check
            old_span_len = e - s
            if old_span_len != n_img:
                raise ValueError(
                    f"Image span length mismatch at image {i}: "
                    f"sequence has {old_span_len}, vision encoder has {n_img}"
                )

            # original image tokens
            img_chunk = old_embeds[s:e]  # [N, D]

            # append K zero slots for selected tokens
            if n_sel > 0:
                sel_zeros = torch.zeros((n_sel, D), device=device, dtype=dtype)
                img_aug = torch.cat([img_chunk, sel_zeros], dim=0)  # [N+K, D]

                mask_aug = torch.cat([
                    old_mask[s:e],
                    torch.ones(n_sel, device=device, dtype=old_mask.dtype)
                ], dim=0)

                mmtype_aug = torch.cat([
                    old_mmtype[s:e],
                    torch.ones(n_sel, device=device, dtype=old_mmtype.dtype)
                ], dim=0)
            else:
                img_aug = img_chunk
                mask_aug = old_mask[s:e]
                mmtype_aug = old_mmtype[s:e]

            aug_start = sum(x.size(0) for x in new_embed_chunks)
            aug_end = aug_start + img_aug.size(0)
            image_aug_spans.append((aug_start, aug_end))

            new_embed_chunks.append(img_aug)
            new_mask_chunks.append(mask_aug)
            new_mmtype_chunks.append(mmtype_aug)

            cursor = e

        # tail text chunk
        if cursor < L:
            new_embed_chunks.append(old_embeds[cursor:])
            new_mask_chunks.append(old_mask[cursor:])
            new_mmtype_chunks.append(old_mmtype[cursor:])

        new_inputs_embeds = torch.cat(new_embed_chunks, dim=0).unsqueeze(0)  # [1, L_new, D]
        new_attention_mask = torch.cat(new_mask_chunks, dim=0).unsqueeze(0)
        new_mm_token_type_ids = torch.cat(new_mmtype_chunks, dim=0).unsqueeze(0)

        return new_inputs_embeds, new_attention_mask, new_mm_token_type_ids, image_aug_spans

    def _fill_image_embeds_by_spans(
            self,
            inputs_embeds,
            image_embeds_list,
            image_aug_spans,
    ):
        """
        Fill each expanded image span with its corresponding augmented image embeds.
        """
        x = inputs_embeds.clone()
        assert x.size(0) == 1, "Current implementation only supports batch size = 1."

        for embeds_i, (s, e) in zip(image_embeds_list, image_aug_spans):
            if embeds_i.size(0) != (e - s):
                raise ValueError(
                    f"Augmented image embed length mismatch: "
                    f"span len = {e - s}, embeds len = {embeds_i.size(0)}"
                )
            x[0, s:e, :] = embeds_i.to(x.device, x.dtype)

        x = torch.nan_to_num(x)
        return x

    def _build_position_ids_with_selected_tokens(
            self,
            input_ids,
            image_grid_thw,
            video_grid_thw,
            second_per_grid_ts,
            inputs_embeds_before_expand,
            attention_mask_before_expand,
            mm_token_type_ids_before_expand,
            attention_mask_after_expand,
            mm_token_type_ids_after_expand,
            past_key_values,
            selected_patch_ids_per_image,
            image_token_lengths,
            num_selected_tokens_per_image,
    ):
        """
        Build expanded position_ids by reusing original image position ids
        for the appended selected tokens.
        Assume batch size = 1.
        """
        # 1) original position ids on the original sequence length
        base_position_ids = self.compute_3d_position_ids(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            inputs_embeds=inputs_embeds_before_expand,
            attention_mask=attention_mask_before_expand,
            past_key_values=past_key_values,
            mm_token_type_ids=mm_token_type_ids_before_expand,
        )  # expected [3, B, L] or [B, 3, L]

        # normalize to [B, 3, L]
        if base_position_ids.dim() != 3:
            raise ValueError(f"Unexpected position_ids shape: {base_position_ids.shape}")

        if base_position_ids.size(0) == 3:
            base_position_ids = base_position_ids.permute(1, 0, 2).contiguous()  # [B,3,L]
        elif base_position_ids.size(1) == 3:
            pass
        else:
            raise ValueError(f"Cannot parse position_ids shape: {base_position_ids.shape}")

        assert base_position_ids.size(0) == 1, "Current implementation only supports batch size = 1."

        old_pos = base_position_ids[0]  # [3, L_old]
        old_mm = mm_token_type_ids_before_expand[0]  # [L_old]
        new_mm = mm_token_type_ids_after_expand[0]  # [L_new]

        image_pos = torch.nonzero(old_mm == 1, as_tuple=False).squeeze(-1)
        if image_pos.numel() == 0:
            return base_position_ids

        # find contiguous image spans in old sequence
        old_spans = []
        start = image_pos[0].item()
        prev = image_pos[0].item()
        for p in image_pos[1:].tolist():
            if p == prev + 1:
                prev = p
            else:
                old_spans.append((start, prev + 1))
                start = p
                prev = p
        old_spans.append((start, prev + 1))

        if len(old_spans) != len(image_token_lengths):
            raise ValueError(
                f"Found {len(old_spans)} old image spans, but got {len(image_token_lengths)} image lengths."
            )

        # build new pos by chunks
        new_pos_chunks = []
        cursor = 0
        for i, ((s, e), n_img, n_sel, patch_ids) in enumerate(
                zip(old_spans, image_token_lengths, num_selected_tokens_per_image, selected_patch_ids_per_image)
        ):
            # text/video chunk before image span
            if cursor < s:
                new_pos_chunks.append(old_pos[:, cursor:s])

            old_span = old_pos[:, s:e]  # [3, N]
            if old_span.size(1) != n_img:
                raise ValueError(
                    f"Position image span mismatch at image {i}: "
                    f"pos span len = {old_span.size(1)}, expected = {n_img}"
                )

            if n_sel > 0:
                selected_pos = old_span[:, patch_ids]  # [3, K]
                new_span = torch.cat([old_span, selected_pos], dim=1)  # [3, N+K]
            else:
                new_span = old_span

            new_pos_chunks.append(new_span)
            cursor = e

        # tail
        if cursor < old_pos.size(1):
            new_pos_chunks.append(old_pos[:, cursor:])

        new_pos = torch.cat(new_pos_chunks, dim=1)  # [3, L_new]

        if new_pos.size(1) != new_mm.size(0):
            raise ValueError(
                f"Expanded position_ids length mismatch: pos len = {new_pos.size(1)}, "
                f"new sequence len = {new_mm.size(0)}"
            )

        return new_pos.unsqueeze(0)  # [1,3,L_new]

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

            # save original sequence info for later position-id construction
            inputs_embeds_before_expand = inputs_embeds
            attention_mask_before_expand = attention_mask
            mm_token_type_ids_before_expand = mm_token_type_ids

            self._debug_patch_ids = []
            self._debug_num_selected_tokens = []
            self._debug_image_token_lengths = []
            self._debug_augmented_lengths = []

            # 1. Get patch tokens（List[Tensor(Ni, D)]）
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
                        device=text_embed.device
                    )

                text_tokens = text_embed[text_mask].view(1, -1, text_embed.size(-1))
                text_tokens = text_tokens.to(inputs_embeds.device, inputs_embeds.dtype)

            selected_idx_per_image = []
            new_image_tokens_list = []

            # run tree for every image
            for i, tokens in enumerate(image_tokens_list):
                # tokens: [Ni, D]
                x = tokens.unsqueeze(0)  # [1, Ni, D]
                ti = text_tokens.to(tokens.device, tokens.dtype)  # [1, Lt, D]

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
                    x=x
                )

                patch_ids = token_out["selected_token_indices"][0]

                # fallback 1
                if patch_ids.numel() == 0:
                    patch_ids = torch.arange(tokens.size(0), device=tokens.device)

                # fallback 2
                patch_ids = patch_ids.clamp(min=0, max=tokens.size(0) - 1)
                patch_ids = torch.unique(patch_ids)

                patch_offset = tokens.size(0) - grid_h * grid_w
                patch_ids = patch_ids + patch_offset
                patch_ids = patch_ids.clamp(min=0, max=tokens.size(0) - 1)
                patch_ids = torch.unique(patch_ids)

                # concat selected tokens after original image tokens
                selected_tokens = tokens[patch_ids]                       # [K, D]
                tokens_augmented = torch.cat([tokens, selected_tokens], dim=0)  # [N+K, D]

                self._debug_patch_ids.append(patch_ids)
                self._debug_num_selected_tokens.append(selected_tokens.size(0))
                self._debug_image_token_lengths.append(tokens.size(0))
                self._debug_augmented_lengths.append(tokens_augmented.size(0))

                new_image_tokens_list.append(tokens_augmented)

            self._debug_selected_idx = selected_idx_per_image

            # 3. Expand sequence to create extra slots for selected tokens
            inputs_embeds, attention_mask, mm_token_type_ids, image_aug_spans = \
                self._expand_multimodal_sequence_for_selected_tokens(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    mm_token_type_ids=mm_token_type_ids,
                    image_token_lengths=self._debug_image_token_lengths,
                    num_selected_tokens_per_image=self._debug_num_selected_tokens,
                )

            # 4. Fill augmented image embeddings directly by spans
            inputs_embeds = self._fill_image_embeds_by_spans(
                inputs_embeds=inputs_embeds,
                image_embeds_list=new_image_tokens_list,
                image_aug_spans=image_aug_spans,
            )
            
            # 5. Debug
            print("image_token_lengths:", self._debug_image_token_lengths)
            print("num_selected_tokens:", self._debug_num_selected_tokens)
            print("augmented_lengths:", self._debug_augmented_lengths)
            print("inputs_embeds.shape after expand:", inputs_embeds.shape)
            print("attention_mask.shape after expand:", attention_mask.shape)
            print("mm_token_type_ids.shape after expand:", mm_token_type_ids.shape)

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
            if pixel_values is not None:
                position_ids = self._build_position_ids_with_selected_tokens(
                    input_ids=input_ids,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    inputs_embeds_before_expand=inputs_embeds_before_expand,
                    attention_mask_before_expand=attention_mask_before_expand,
                    mm_token_type_ids_before_expand=mm_token_type_ids_before_expand,
                    attention_mask_after_expand=attention_mask,
                    mm_token_type_ids_after_expand=mm_token_type_ids,
                    past_key_values=past_key_values,
                    selected_patch_ids_per_image=self._debug_patch_ids,
                    image_token_lengths=self._debug_image_token_lengths,
                    num_selected_tokens_per_image=self._debug_num_selected_tokens,
                )
            else:
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