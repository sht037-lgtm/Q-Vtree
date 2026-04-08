import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from module import QVTree
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLModel,  # override
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLModelOutputWithPast
)


# =============================
# LPD: Layout-Preserving Decoupling
# =============================

def patch_ids_to_bboxes(patch_ids: torch.Tensor, grid_w: int, patch_size: int = 28):
    """
    Convert flat patch indices to pixel bounding boxes.

    Args:
        patch_ids : 1D tensor of selected patch indices (already offset-corrected)
        grid_w    : width of the downsampled patch grid
        patch_size: pixel size of each downsampled patch (default 28)

    Returns:
        List of (x0, y0, x1, y1) tuples in pixel coordinates
    """
    bboxes = []
    for idx in patch_ids.tolist():
        r = int(idx) // grid_w
        c = int(idx) % grid_w
        x0 = c * patch_size
        y0 = r * patch_size
        x1 = x0 + patch_size
        y1 = y0 + patch_size
        bboxes.append((x0, y0, x1, y1))
    return bboxes


def merge_bboxes(bboxes):
    """
    Merge overlapping or adjacent bounding boxes via union.
    Uses iterative merging until stable.

    Args:
        bboxes: List of (x0, y0, x1, y1)

    Returns:
        List of merged (x0, y0, x1, y1)
    """
    if not bboxes:
        return []

    def overlaps_or_adjacent(a, b):
        # treat touching edges as adjacent (<=)
        return not (a[2] <= b[0] or b[2] <= a[0] or
                    a[3] <= b[1] or b[3] <= a[1])

    merged = list(bboxes)
    changed = True
    while changed:
        changed = False
        result = []
        used = [False] * len(merged)
        for i in range(len(merged)):
            if used[i]:
                continue
            cur = list(merged[i])
            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                if overlaps_or_adjacent(cur, merged[j]):
                    cur[0] = min(cur[0], merged[j][0])
                    cur[1] = min(cur[1], merged[j][1])
                    cur[2] = max(cur[2], merged[j][2])
                    cur[3] = max(cur[3], merged[j][3])
                    used[j] = True
                    changed = True
            result.append(tuple(cur))
        merged = result
    return merged


def build_compact_image(image: Image.Image, bboxes) -> Image.Image:
    """
    Layout-Preserving Decoupling (LPD):
    Crop selected regions and recompose into a compact image that
    preserves their relative spatial layout, discarding background gaps.

    This follows the grid-based reconstruction in HiDe Algorithm 1.

    Args:
        image : PIL Image resized to (grid_w*28, grid_h*28)
        bboxes: List of merged (x0, y0, x1, y1) in pixel coordinates

    Returns:
        compact_image: PIL Image with background strips removed
    """
    if not bboxes:
        return image

    img_w, img_h = image.size

    # ── Step 1: build canonical grid lines from bbox coordinates ──
    x_coords = sorted(set([0, img_w] + [x for b in bboxes for x in (b[0], b[2])]))
    y_coords = sorted(set([0, img_h] + [y for b in bboxes for y in (b[1], b[3])]))

    # ── Step 2: identify which columns/rows contain content ──
    def cell_has_content(x0, y0, x1, y1):
        for b in bboxes:
            # check intersection
            if x0 < b[2] and x1 > b[0] and y0 < b[3] and y1 > b[1]:
                return True
        return False

    # build x-mapping: old x-grid-index -> new x pixel position
    new_w = 0
    x_map = {0: 0}
    for i in range(len(x_coords) - 1):
        col_w = x_coords[i + 1] - x_coords[i]
        # check if any row in this column strip has content
        col_has = any(
            cell_has_content(x_coords[i], y_coords[j], x_coords[i + 1], y_coords[j + 1])
            for j in range(len(y_coords) - 1)
        )
        if col_has:
            new_w += col_w
        x_map[i + 1] = new_w

    # build y-mapping: old y-grid-index -> new y pixel position
    new_h = 0
    y_map = {0: 0}
    for j in range(len(y_coords) - 1):
        row_h = y_coords[j + 1] - y_coords[j]
        # check if any column in this row strip has content
        row_has = any(
            cell_has_content(x_coords[i], y_coords[j], x_coords[i + 1], y_coords[j + 1])
            for i in range(len(x_coords) - 1)
        )
        if row_has:
            new_h += row_h
        y_map[j + 1] = new_h

    if new_w == 0 or new_h == 0:
        return image

    # ── Step 3: paste content cells into compact canvas ──
    compact = Image.new("RGB", (new_w, new_h), color=(0, 0, 0))

    for i in range(len(x_coords) - 1):
        for j in range(len(y_coords) - 1):
            if not cell_has_content(x_coords[i], y_coords[j],
                                    x_coords[i + 1], y_coords[j + 1]):
                continue
            patch = image.crop((x_coords[i], y_coords[j],
                                 x_coords[i + 1], y_coords[j + 1]))
            paste_x = x_map[i]
            paste_y = y_map[j]
            compact.paste(patch, (paste_x, paste_y))

    return compact


# =============================
# Model
# =============================

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
        self._debug_patch_scores = None

        # LPD debug: stores compact PIL Images for each image in the batch
        # access in notebook: tree_model.model._debug_compact_images[0]
        self._debug_compact_images = None
        # LPD debug: stores merged bboxes per image
        self._debug_bboxes = None

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
            self._debug_compact_images = []
            self._debug_bboxes = []

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
                ans_pos = inputs_embeds_in.shape[1] - 1
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
            generic_text = "Write a general description of the image."
            img_end = vis_positions[-1].item()

            if self._generic_token_ids is None:
                from transformers import AutoTokenizer
                _tok = AutoTokenizer.from_pretrained(self.config._name_or_path)
                _generic_text = "Write a general description of the image."
                self._generic_token_ids = _tok(
                    _generic_text, return_tensors="pt", add_special_tokens=False
                ).input_ids
            generic_ids = self._generic_token_ids.to(input_ids.device)

            prefix_ids = input_ids[:, :img_end + 1]
            suffix_ids = input_ids[:, -3:]
            new_input_ids = torch.cat([prefix_ids, generic_ids, suffix_ids], dim=1)

            generic_embeds = self.get_input_embeddings()(new_input_ids)
            image_mask_generic, _ = self.get_placeholder_mask(
                new_input_ids,
                inputs_embeds=generic_embeds,
                image_features=image_embeds_full,
            )
            generic_embeds = generic_embeds.masked_scatter(image_mask_generic, image_embeds_full)
            generic_embeds = torch.nan_to_num(generic_embeds)

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

            # relative attention
            patch_scores_global = A_q / (A_generic + 1e-8)
            s_min = patch_scores_global.min()
            s_max = patch_scores_global.max()
            patch_scores_global = (patch_scores_global - s_min) / (s_max - s_min + 1e-6)

            # Gaussian smoothing
            grid_t0, grid_h0_raw, grid_w0_raw = image_grid_thw[0].tolist()
            grid_h0 = grid_h0_raw // 2
            grid_w0 = grid_w0_raw // 2
            if patch_scores_global.shape[0] == grid_h0 * grid_w0:
                sigma, ks = 1.0, 3
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

            # =============================
            # 5. QuadTree selection + LPD per image
            # =============================
            new_pixel_values_list = []  # compact pixel_values to re-encode
            new_grid_thw_list = []      # updated grid_thw after re-encoding

            for i, tokens in enumerate(image_tokens_list):
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

                x = tokens.unsqueeze(0)  # [1, Ni, D]

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

                token_out = self.qvtree.navigator.nodes_to_tokens(
                    nodes,
                    H=grid_h,
                    W=grid_w,
                    selected_node_ids=sel_nodes_list,
                    x=x,
                )

                patch_ids = token_out["selected_token_indices"][0]

                # fallback: if empty, keep all
                if patch_ids.numel() == 0:
                    patch_ids = torch.arange(tokens.size(0), device=tokens.device)

                patch_ids = patch_ids.clamp(min=0, max=tokens.size(0) - 1)
                patch_ids = torch.unique(patch_ids)

                # debug store
                self._debug_patch_ids.append(patch_ids)

                num_selected = int(patch_ids.numel())
                num_total = int(tokens.size(0))
                self._debug_num_selected_tokens.append(num_selected)
                self._debug_num_total_tokens.append(num_total)
                self._debug_select_ratios.append(
                    num_selected / num_total if num_total > 0 else 0.0
                )

                # ── LPD: pixel-level compact image ──────────────────────────
                patch_size = 28
                image_w_px = grid_w * patch_size
                image_h_px = grid_h * patch_size

                # reconstruct the image at the resolution the vision encoder saw
                # pixel_values is a flat tensor of patches; recover via processor
                # We use the stored pixel_values directly via Qwen's visual encoder
                # by reconstructing a PIL image from the patch grid.
                # The processor resizes to (grid_h_raw * 14, grid_w_raw * 14) before
                # patch extraction, so we resize the original to match that, then
                # downsample to (grid_h*28, grid_w*28) for LPD pixel operations.

                # Extract the raw pixel patches for image i from pixel_values.
                # pixel_values: [total_patches, C, 14, 14] (before merge)
                # After 2x2 merge the vision encoder sees grid_h*grid_w patches.
                # We reconstruct a (grid_h*28, grid_w*28) image from pixel_values.
                n_patches_raw = grid_h_raw * grid_w_raw  # before merge
                # offset into pixel_values for image i
                patch_offset_raw = sum(
                    image_grid_thw[k][1].item() * image_grid_thw[k][2].item()
                    for k in range(i)
                )
                # shape: [n_patches_raw, C, 14, 14]
                img_patches = pixel_values[
                    patch_offset_raw: patch_offset_raw + n_patches_raw
                ]  # [H_raw*W_raw, C, 14, 14]

                # reshape to image: [C, H_raw*14, W_raw*14]
                C = img_patches.shape[1]
                img_tensor = img_patches.view(
                    grid_h_raw, grid_w_raw, C, 14, 14
                ).permute(2, 0, 3, 1, 4).reshape(C, grid_h_raw * 14, grid_w_raw * 14)

                # denormalize (Qwen uses mean/std from SigLIP)
                mean = torch.tensor([0.5, 0.5, 0.5],
                                    device=img_tensor.device,
                                    dtype=img_tensor.dtype).view(3, 1, 1)
                std  = torch.tensor([0.5, 0.5, 0.5],
                                    device=img_tensor.device,
                                    dtype=img_tensor.dtype).view(3, 1, 1)
                img_tensor = (img_tensor * std + mean).clamp(0, 1)

                # to PIL at (grid_w_raw*14, grid_h_raw*14), then resize to (grid_w*28, grid_h*28)
                img_np = (img_tensor.cpu().float().numpy() * 255).astype(np.uint8)
                img_np = np.transpose(img_np, (1, 2, 0))  # [H, W, C]
                pil_img = Image.fromarray(img_np).resize(
                    (image_w_px, image_h_px), Image.BILINEAR
                )

                # patch_ids → bboxes → merge → compact image
                raw_bboxes = patch_ids_to_bboxes(patch_ids.cpu(), grid_w, patch_size)
                merged_bboxes = merge_bboxes(raw_bboxes)
                compact_img = build_compact_image(pil_img, merged_bboxes)

                # store debug
                self._debug_compact_images.append(compact_img)
                self._debug_bboxes.append(merged_bboxes)

                # re-encode compact image via vision encoder
                # normalize back to tensor for pixel_values format
                compact_np = np.array(compact_img).astype(np.float32) / 255.0
                compact_np = np.transpose(compact_np, (2, 0, 1))  # [C, H, W]
                compact_tensor = torch.tensor(compact_np,
                                              device=pixel_values.device,
                                              dtype=pixel_values.dtype)
                compact_tensor = (compact_tensor - mean.squeeze(-1).squeeze(-1)[:, None, None]) \
                                 / std.squeeze(-1).squeeze(-1)[:, None, None]

                # split into 14×14 patches
                cH, cW = compact_tensor.shape[1], compact_tensor.shape[2]
                # pad to multiple of 14
                pH = (14 - cH % 14) % 14
                pW = (14 - cW % 14) % 14
                if pH > 0 or pW > 0:
                    compact_tensor = F.pad(compact_tensor, (0, pW, 0, pH))
                cH_pad = compact_tensor.shape[1]
                cW_pad = compact_tensor.shape[2]

                nH = cH_pad // 14
                nW = cW_pad // 14
                # [nH*nW, C, 14, 14]
                compact_patches = compact_tensor.reshape(
                    C, nH, 14, nW, 14
                ).permute(1, 3, 0, 2, 4).reshape(nH * nW, C, 14, 14)

                new_pixel_values_list.append(compact_patches)
                # grid_thw: (t=1, h=nH, w=nW)
                new_grid_thw_list.append(
                    torch.tensor([1, nH, nW],
                                 dtype=image_grid_thw.dtype,
                                 device=image_grid_thw.device)
                )

            # ── Re-encode all compact images through vision encoder ──────────
            compact_pixel_values = torch.cat(new_pixel_values_list, dim=0)
            compact_grid_thw = torch.stack(new_grid_thw_list, dim=0)

            image_tokens_compact = self.get_image_features(
                compact_pixel_values,
                compact_grid_thw,
                return_dict=True
            ).pooler_output

            # ── Rebuild inputs_embeds with compact image tokens ──────────────
            # The number of image placeholder tokens in input_ids must match
            # the total compact tokens. Since compact images have fewer patches,
            # we need to update input_ids to have the right number of placeholders.
            # We rebuild inputs_embeds from scratch with updated placeholders.

            image_embeds_compact = torch.cat(image_tokens_compact, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            image_embeds_compact = torch.nan_to_num(image_embeds_compact)

            # Rebuild input_ids with updated image placeholder counts
            image_token_id = 151655
            vision_start_id = 151652
            vision_end_id   = 151653

            original_ids = input_ids[0].tolist()

            # find vision_start and vision_end positions
            try:
                vs_pos = original_ids.index(vision_start_id)
                ve_pos = original_ids.index(vision_end_id)
            except ValueError:
                # fallback: no vision tokens found, skip LPD
                image_embeds_compact = image_embeds_full
                compact_grid_thw = image_grid_thw
                vs_pos = None

            if vs_pos is not None:
                n_compact_tokens = int(image_embeds_compact.shape[0])
                new_ids = (
                    original_ids[:vs_pos + 1]           # prefix including vision_start
                    + [image_token_id] * n_compact_tokens
                    + original_ids[ve_pos:]              # vision_end onwards
                )
                new_input_ids = torch.tensor(
                    [new_ids],
                    dtype=input_ids.dtype,
                    device=input_ids.device
                )
                new_attention_mask = torch.ones(
                    1, len(new_ids),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )
                new_inputs_embeds = self.get_input_embeddings()(new_input_ids)
                new_inputs_embeds = torch.nan_to_num(new_inputs_embeds)

                image_mask_compact, _ = self.get_placeholder_mask(
                    new_input_ids,
                    inputs_embeds=new_inputs_embeds,
                    image_features=image_embeds_compact,
                )
                new_inputs_embeds = new_inputs_embeds.masked_scatter(
                    image_mask_compact, image_embeds_compact
                )
                new_inputs_embeds = torch.nan_to_num(new_inputs_embeds)

                # recompute position_ids for new sequence
                position_ids = self.compute_3d_position_ids(
                    input_ids=new_input_ids,
                    image_grid_thw=compact_grid_thw,
                    video_grid_thw=video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    inputs_embeds=new_inputs_embeds,
                    attention_mask=new_attention_mask,
                    past_key_values=past_key_values,
                    mm_token_type_ids=mm_token_type_ids,
                )
                inputs_embeds = new_inputs_embeds
                attention_mask = new_attention_mask
            else:
                # fallback: use full embeddings
                inputs_embeds = inputs_embeds_full

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
        # position ids (text-only or video-only path)
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