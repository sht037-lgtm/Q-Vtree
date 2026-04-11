import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from module import QVTree
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLModel,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLModelOutputWithPast
)


class Qwen2_5_VLModelWithTree(Qwen2_5_VLModel):
    def __init__(self, config):
        super().__init__(config)
        self.qvtree = QVTree(D=config.text_config.hidden_size)
        self._generic_token_ids = None

        # debug states
        self._debug_selected_idx = None
        self._debug_patch_ids = None
        self._debug_num_selected_tokens = None
        self._debug_num_total_tokens = None
        self._debug_select_ratios = None
        self._debug_patch_scores = None

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

        if pixel_values is not None:
            self._debug_patch_ids = []
            self._debug_num_selected_tokens = []
            self._debug_num_total_tokens = []
            self._debug_select_ratios = []

            # encode image tokens
            image_tokens_list = self.get_image_features(
                pixel_values, image_grid_thw, return_dict=True
            ).pooler_output

            # build full inputs_embeds for attention scoring
            image_embeds_full = torch.nan_to_num(
                torch.cat(image_tokens_list, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            )
            inputs_embeds_full = torch.nan_to_num(inputs_embeds.clone())
            image_mask_full, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds_full, image_features=image_embeds_full,
            )
            inputs_embeds_full = torch.nan_to_num(
                inputs_embeds_full.masked_scatter(image_mask_full, image_embeds_full)
            )

            if position_ids is None:
                position_ids = self.compute_3d_position_ids(
                    input_ids=input_ids,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    inputs_embeds=inputs_embeds_full,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                )

            vis_positions = (input_ids[0] == 151655).nonzero(as_tuple=True)[0].cpu()
            img_end = vis_positions[-1].item()
            seq_len = input_ids.shape[1]
            text_start, text_end = img_end + 1, seq_len - 3

            text_positions = (
                torch.arange(text_start, text_end, dtype=torch.long)
                if text_end > text_start
                else torch.tensor([seq_len - 1], dtype=torch.long)
            )

            def get_attn_scores(embeds_in, mask_in, pos_in, query_positions):
                """Mean attention from query tokens to image patches.
                Uses hooks on layer 20 and averages results.
                """
                target_layer_ids = [20]
                lm = self.language_model
                lm_layers = getattr(lm, "layers", None) or getattr(lm, "model", lm).layers

                captured = {}
                handles = []

                for lid in target_layer_ids:
                    def make_hook(layer_id):
                        def hook_fn(module, input, output):
                            if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                                captured[layer_id] = output[1].detach().cpu()
                        return hook_fn
                    handles.append(
                        lm_layers[lid].self_attn.register_forward_hook(make_hook(lid))
                    )

                try:
                    with torch.no_grad():
                        self.language_model(
                            input_ids=None, inputs_embeds=embeds_in, attention_mask=mask_in,
                            position_ids=pos_in, output_attentions=True,
                            output_hidden_states=False, return_dict=True, use_cache=False,
                        )
                finally:
                    for h in handles:
                        h.remove()

                if not captured:
                    raise RuntimeError("Hooks did not capture attention weights. "
                                       "Ensure model is loaded with attn_implementation='eager'.")

                vp = vis_positions
                qp = query_positions
                scores = []
                for lid in target_layer_ids:
                    if lid not in captured:
                        continue
                    layer_attn = captured[lid]  # [B, heads, seq_len, seq_len]
                    attn = layer_attn[0, :, :, vp][:, qp, :]  # [heads, Q, N_img]
                    scores.append(attn.mean(dim=0).mean(dim=0).float())
                result = torch.stack(scores).mean(dim=0)  # [N_img]
                del captured
                torch.cuda.empty_cache()
                return result

            # question-specific attention
            A_q = get_attn_scores(inputs_embeds_full, attention_mask, position_ids, text_positions)

            # generic attention (baseline for relative normalization)
            if self._generic_token_ids is None:
                from transformers import AutoTokenizer
                tok = AutoTokenizer.from_pretrained(self.config._name_or_path)
                self._generic_token_ids = tok(
                    "Write a general description of the image.",
                    return_tensors="pt", add_special_tokens=False
                ).input_ids
            generic_ids = self._generic_token_ids.to(input_ids.device)
            new_input_ids = torch.cat(
                [input_ids[:, :img_end + 1], generic_ids, input_ids[:, -3:]], dim=1
            )
            generic_embeds = self.get_input_embeddings()(new_input_ids)
            image_mask_g, _ = self.get_placeholder_mask(
                new_input_ids, inputs_embeds=generic_embeds, image_features=image_embeds_full,
            )
            generic_embeds = torch.nan_to_num(
                generic_embeds.masked_scatter(image_mask_g, image_embeds_full)
            )
            generic_attn_mask = torch.ones(
                1, new_input_ids.shape[1], dtype=attention_mask.dtype, device=attention_mask.device
            )
            generic_pos_ids = self.compute_3d_position_ids(
                input_ids=new_input_ids, image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw, second_per_grid_ts=second_per_grid_ts,
                inputs_embeds=generic_embeds, attention_mask=generic_attn_mask, past_key_values=None,
            )
            generic_text_positions = torch.arange(
                img_end + 1, img_end + 1 + generic_ids.shape[1], dtype=torch.long
            )
            A_generic = get_attn_scores(generic_embeds, generic_attn_mask, generic_pos_ids, generic_text_positions)

            # relative attention + normalize
            patch_scores_global = A_q / (A_generic + 1e-8)
            s_min, s_max = patch_scores_global.min(), patch_scores_global.max()
            patch_scores_global = (patch_scores_global - s_min) / (s_max - s_min + 1e-6)

            # Gaussian smoothing (sigma=1.0, ks=3)
            _, grid_h0_raw, grid_w0_raw = image_grid_thw[0].tolist()
            grid_h0, grid_w0 = grid_h0_raw // 2, grid_w0_raw // 2
            if patch_scores_global.shape[0] == grid_h0 * grid_w0:
                sigma, ks = 1.0, 3
                ax = torch.arange(ks, dtype=torch.float32) - ks // 2
                g1d = torch.exp(-ax ** 2 / (2 * sigma ** 2))
                g1d /= g1d.sum()
                kernel = (g1d.unsqueeze(1) * g1d.unsqueeze(0)).view(1, 1, ks, ks)
                score_map = F.conv2d(patch_scores_global.float().view(1, 1, grid_h0, grid_w0), kernel, padding=ks // 2)
                patch_scores_global = score_map.view(-1)
                s_min, s_max = patch_scores_global.min(), patch_scores_global.max()
                patch_scores_global = (patch_scores_global - s_min) / (s_max - s_min + 1e-6)

            self._debug_patch_scores = [patch_scores_global]

            # QuadTree selection: directly index already-encoded tokens
            selected_tokens_list, selected_idx_per_image = [], []

            for i, tokens in enumerate(image_tokens_list):
                _, grid_h_raw, grid_w_raw = image_grid_thw[i].tolist()
                grid_h, grid_w = grid_h_raw // 2, grid_w_raw // 2

                if grid_h * grid_w != tokens.size(0):
                    raise ValueError(
                        f"Downsampled grid mismatch: raw=({grid_h_raw},{grid_w_raw}), "
                        f"down=({grid_h},{grid_w}), H*W={grid_h*grid_w}, tokens={tokens.size(0)}"
                    )

                x = tokens.unsqueeze(0)
                patch_scores = patch_scores_global.unsqueeze(0).to(tokens.device, tokens.dtype)

                built = self.qvtree.builder.build(x, grid_h, grid_w)
                nodes = built["nodes"]
                sel_nodes_list, _ = self.qvtree.navigator.select_nodes(
                    nodes=nodes, patch_scores=patch_scores, W=grid_w,
                )
                selected_idx_per_image.append(sel_nodes_list[0])

                token_out = self.qvtree.navigator.nodes_to_tokens(
                    nodes, H=grid_h, W=grid_w, selected_node_ids=sel_nodes_list, x=x,
                )
                patch_ids = token_out["selected_token_indices"][0]
                if patch_ids.numel() == 0:
                    patch_ids = torch.arange(tokens.size(0), device=tokens.device)
                patch_ids = torch.unique(patch_ids.clamp(min=0, max=tokens.size(0) - 1))

                self._debug_patch_ids.append(patch_ids)
                num_selected, num_total = int(patch_ids.numel()), int(tokens.size(0))
                self._debug_num_selected_tokens.append(num_selected)
                self._debug_num_total_tokens.append(num_total)
                self._debug_select_ratios.append(num_selected / num_total if num_total > 0 else 0.0)

                # directly select tokens from already-encoded features
                selected_tokens_list.append(tokens[patch_ids])

            self._debug_selected_idx = selected_idx_per_image

            # concat selected tokens across all images
            image_embeds_selected = torch.cat(selected_tokens_list, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            n_selected = int(image_embeds_selected.shape[0])
            print(f"[TREE] selected tokens: {n_selected}, original: {sum(t.shape[0] for t in image_tokens_list)}")

            # rebuild input_ids with updated placeholder count
            original_ids = input_ids[0].tolist()
            try:
                vs_pos = original_ids.index(151652)  # vision_start
                ve_pos = original_ids.index(151653)  # vision_end
            except ValueError:
                vs_pos = None

            if vs_pos is not None:
                new_ids = original_ids[:vs_pos + 1] + [151655] * n_selected + original_ids[ve_pos:]
                new_input_ids = torch.tensor([new_ids], dtype=input_ids.dtype, device=input_ids.device)
                new_attention_mask = torch.ones(
                    1, len(new_ids), dtype=attention_mask.dtype, device=attention_mask.device
                )
                new_inputs_embeds = torch.nan_to_num(self.get_input_embeddings()(new_input_ids))
                image_mask_selected, _ = self.get_placeholder_mask(
                    new_input_ids, inputs_embeds=new_inputs_embeds, image_features=image_embeds_selected,
                )
                new_inputs_embeds = torch.nan_to_num(
                    new_inputs_embeds.masked_scatter(image_mask_selected, image_embeds_selected)
                )
                position_ids = self.compute_3d_position_ids(
                    input_ids=new_input_ids, image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw, second_per_grid_ts=second_per_grid_ts,
                    inputs_embeds=new_inputs_embeds, attention_mask=new_attention_mask,
                    past_key_values=past_key_values,
                )
                inputs_embeds = new_inputs_embeds
                attention_mask = new_attention_mask
            else:
                inputs_embeds = inputs_embeds_full

        if pixel_values_videos is not None:
            video_embeds = torch.cat(
                self.get_video_features(pixel_values_videos, video_grid_thw, return_dict=True).pooler_output,
                dim=0
            ).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if position_ids is None:
            position_ids = self.compute_3d_position_ids(
                input_ids=input_ids, image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw, second_per_grid_ts=second_per_grid_ts,
                inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                past_key_values=past_key_values,
            )

        # recompute cache_position after token selection (sequence length may have changed)
        seq_len = inputs_embeds.shape[1]
        past_len = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(past_len, past_len + seq_len, device=inputs_embeds.device)

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