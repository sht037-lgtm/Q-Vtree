import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (AutoModel, GenerationConfig, LlamaForCausalLM,
                          Qwen2ForCausalLM)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging

from .configuration_internvl_chat import InternVLChatConfig
from .conversation import get_conv_template
from .modeling_intern_vit import InternVisionModel, has_flash_attn
from module import QVTree

logger = logging.get_logger(__name__)


def version_cmp(v1, v2, op='eq'):
    import operator
    from packaging import version
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))


class InternVLChatModelWithTree(PreTrainedModel):
    config_class = InternVLChatConfig
    main_input_name = 'pixel_values'
    base_model_prefix = 'language_model'
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True
    _no_split_modules = ['InternVisionModel', 'LlamaDecoderLayer', 'Qwen2DecoderLayer']

    def __init__(self, config: InternVLChatConfig, vision_model=None, language_model=None, use_flash_attn=True):
        super().__init__(config)

        assert version_cmp(transformers.__version__, '4.37.0', 'ge')
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio ** 2))
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        use_flash_attn = use_flash_attn if has_flash_attn else False
        config.vision_config.use_flash_attn = True if use_flash_attn else False
        config.llm_config._attn_implementation = 'flash_attention_2' if use_flash_attn else 'eager'

        logger.info(f'num_image_token: {self.num_image_token}')
        logger.info(f'ps_version: {self.ps_version}')

        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = InternVisionModel(config.vision_config)

        if language_model is not None:
            self.language_model = language_model
        else:
            if config.llm_config.architectures[0] == 'LlamaForCausalLM':
                self.language_model = LlamaForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'Qwen2ForCausalLM':
                self.language_model = Qwen2ForCausalLM(config.llm_config)
            else:
                raise NotImplementedError(f'{config.llm_config.architectures[0]} is not implemented.')

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )

        self.img_context_token_id = None
        self.conv_template = get_conv_template(self.template)
        self.system_message = self.conv_template.system_message

        # QVTree module
        self.qvtree = QVTree(D=llm_hidden_size)

        # generic description token ids (lazy init)
        self._generic_token_ids = None

        # debug states
        self._debug_selected_idx = None
        self._debug_patch_ids = None
        self._debug_num_selected_tokens = None
        self._debug_num_total_tokens = None
        self._debug_select_ratios = None

    # =========================================================
    # Vision feature extraction (unchanged from original)
    # =========================================================
    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                          'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds  # [num_patches, N_per_patch, D]

    # =========================================================
    # QVTree: relative attention scoring + tree navigation
    # =========================================================
    def _apply_qvtree(
        self,
        vit_embeds: torch.Tensor,       # [num_patches, N_per_patch, D]
        input_embeds: torch.Tensor,     # [B*N, D]  (flat, before reshape)
        input_ids_flat: torch.Tensor,   # [B*N]     (flat)
        attention_mask: torch.Tensor,   # [B, N]
    ) -> torch.Tensor:
        """
        Apply relative attention scoring (ViCrop style) + QVTree navigation.
        Returns modulated vit_embeds [total_vis_tokens, D] ready to be
        scattered back into input_embeds.
        """

        # ---- init debug containers ----
        self._debug_patch_ids = []
        self._debug_num_selected_tokens = []
        self._debug_num_total_tokens = []
        self._debug_select_ratios = []

        device = input_embeds.device
        dtype = input_embeds.dtype
        B_N = input_embeds.shape[0]
        D = input_embeds.shape[1]

        # B can be inferred from attention_mask shape
        B = attention_mask.shape[0]
        N = B_N // B

        # ---- rebuild [B, N, D] input_embeds for LLM forward ----
        input_embeds_2d = input_embeds.reshape(B, N, D)

        # ---- grid size: each patch has N_per_patch tokens after pixel_shuffle ----
        # num_image_token = H_grid * W_grid
        N_per_patch = self.num_image_token
        grid_hw = int(N_per_patch ** 0.5)  # square grid assumption
        grid_h = grid_hw
        grid_w = grid_hw

        # ---- helper: one LLM forward, return attention at last token -> vis tokens ----
        # locate all image token positions in the flat sequence
        selected_mask = (input_ids_flat == self.img_context_token_id)  # [B*N]
        # reshape to [B, N]
        selected_mask_2d = selected_mask.reshape(B, N)

        def get_attn_scores(embeds_in, attn_mask_in):
            """embeds_in: [B, N, D], attn_mask_in: [B, N]"""
            with torch.no_grad():
                out = self.language_model(
                    inputs_embeds=embeds_in,
                    attention_mask=attn_mask_in,
                    output_attentions=True,
                    output_hidden_states=False,
                    return_dict=True,
                    use_cache=False,
                )
            # last token attends to image tokens
            # use last layer attention
            target_layer = len(out.attentions) - 1
            layer_attn = out.attentions[target_layer]  # [B, heads, seq, seq]
            ans_pos = embeds_in.shape[1] - 1

            # for each sample in batch, average over heads, pick last token -> vis positions
            scores_batch = []
            for b in range(B):
                vis_pos = selected_mask_2d[b].nonzero(as_tuple=True)[0].to(layer_attn.device)
                s = layer_attn[b, :, ans_pos, vis_pos].mean(dim=0).cpu()
                scores_batch.append(s)
            # NOTE: all samples in a batch share the same image here (single-image inference)
            # take the first sample's scores (same as Qwen version)
            return scores_batch[0]  # [num_vis_tokens]

        # ---- first pass: question-specific attention ----
        A_q = get_attn_scores(input_embeds_2d, attention_mask)

        # ---- second pass: generic description attention ----
        # replace question tokens with generic description tokens
        # locate where image tokens end in flat ids
        vis_positions = selected_mask_2d[0].nonzero(as_tuple=True)[0]
        img_end = vis_positions[-1].item()

        if self._generic_token_ids is None:
            from transformers import AutoTokenizer
            _tok = AutoTokenizer.from_pretrained(
                self.config._name_or_path, trust_remote_code=True
            )
            _generic_text = "Write a general description of the image."
            self._generic_token_ids = _tok(
                _generic_text, return_tensors="pt", add_special_tokens=False
            ).input_ids

        input_ids_2d = input_ids_flat.reshape(B, N)
        generic_ids = self._generic_token_ids.to(device)

        prefix_ids = input_ids_2d[:, :img_end + 1]      # image prefix
        suffix_ids = input_ids_2d[:, -3:]                # assistant start tokens
        new_input_ids = torch.cat([prefix_ids, generic_ids.expand(B, -1), suffix_ids], dim=1)

        generic_embeds = self.language_model.get_input_embeddings()(new_input_ids)
        # fill in vision tokens
        new_selected = (new_input_ids == self.img_context_token_id)
        new_B, new_N, _ = generic_embeds.shape
        generic_embeds_flat = generic_embeds.reshape(new_B * new_N, D)
        new_selected_flat = new_selected.reshape(new_B * new_N)
        generic_embeds_flat[new_selected_flat] = vit_embeds.reshape(-1, D).to(device=generic_embeds_flat.device,
                                                                              dtype=dtype)
        generic_embeds = generic_embeds_flat.reshape(new_B, new_N, D)
        generic_embeds = torch.nan_to_num(generic_embeds)

        generic_attn_mask = torch.ones(
            new_B, new_N, dtype=attention_mask.dtype, device=device
        )

        A_generic = get_attn_scores(generic_embeds, generic_attn_mask)

        # ---- relative attention: element-wise division + normalize ----
        patch_scores_global = A_q / (A_generic + 1e-8)
        s_min = patch_scores_global.min()
        s_max = patch_scores_global.max()
        patch_scores_global = (patch_scores_global - s_min) / (s_max - s_min + 1e-6)

        # ---- Gaussian smoothing ----
        # vit_embeds shape: [num_patches, N_per_patch, D]
        # patch_scores_global has num_patches * N_per_patch entries (one per vis token)
        # reshape to [num_patches, grid_h, grid_w] then smooth
        num_patches = vit_embeds.shape[0]
        if patch_scores_global.shape[0] == num_patches * grid_h * grid_w:
            sigma, ks = 1.0, 3
            ax = torch.arange(ks, dtype=torch.float32) - ks // 2
            gauss_1d = torch.exp(-ax ** 2 / (2 * sigma ** 2))
            gauss_1d = gauss_1d / gauss_1d.sum()
            kernel = (gauss_1d.unsqueeze(1) * gauss_1d.unsqueeze(0)).view(1, 1, ks, ks)

            score_map = patch_scores_global.float().view(1, 1, num_patches * grid_h, grid_w)
            # better: treat each patch independently
            score_map = patch_scores_global.float().view(num_patches, 1, grid_h, grid_w)
            score_map = F.conv2d(score_map, kernel, padding=ks // 2)
            patch_scores_global = score_map.view(-1)
            s_min = patch_scores_global.min()
            s_max = patch_scores_global.max()
            patch_scores_global = (patch_scores_global - s_min) / (s_max - s_min + 1e-6)

        self._debug_patch_scores = [patch_scores_global]

        # ---- QVTree navigation per patch tile ----
        # vit_embeds: [num_patches, N_per_patch, D]
        # each patch tile has grid_h * grid_w tokens
        selected_idx_per_image = []
        new_vit_embeds_list = []

        for i in range(num_patches):
            tokens = vit_embeds[i]  # [N_per_patch, D]
            x = tokens.unsqueeze(0)  # [1, N_per_patch, D]

            # patch scores for this tile
            start = i * grid_h * grid_w
            end = start + grid_h * grid_w
            patch_scores_tile = patch_scores_global[start:end].unsqueeze(0).to(
                tokens.device, tokens.dtype
            )  # [1, N_per_patch]

            # build tree and navigate
            built = self.qvtree.builder.build(x, grid_h, grid_w)
            nodes = built["nodes"]

            sel_nodes_list, _ = self.qvtree.navigator.select_nodes(
                nodes=nodes,
                patch_scores=patch_scores_tile,
                W=grid_w,
            )
            sel_nodes = sel_nodes_list[0]
            selected_idx_per_image.append(sel_nodes)

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

            # debug
            self._debug_patch_ids.append(patch_ids)
            num_selected = int(patch_ids.numel())
            num_total = int(tokens.size(0))
            select_ratio = num_selected / num_total if num_total > 0 else 0.0
            self._debug_num_selected_tokens.append(num_selected)
            self._debug_num_total_tokens.append(num_total)
            self._debug_select_ratios.append(select_ratio)

            # ---- alpha-beta soft masking ----
            alpha = 5.0   # selected
            beta = 0.5    # unselected

            keep = torch.full(
                (tokens.size(0),),
                beta,
                device=tokens.device,
                dtype=tokens.dtype,
            )
            keep[patch_ids] = alpha

            tokens_modulated = tokens * keep.unsqueeze(-1)
            tokens_modulated = torch.nan_to_num(tokens_modulated)
            new_vit_embeds_list.append(tokens_modulated)

        self._debug_selected_idx = selected_idx_per_image

        # [num_patches, N_per_patch, D] -> [total_vis_tokens, D]
        new_vit_embeds = torch.stack(new_vit_embeds_list, dim=0).reshape(-1, D)
        return new_vit_embeds

    # =========================================================
    # forward
    # =========================================================
    def forward(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_flags = image_flags.squeeze(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

        vit_embeds = self.extract_feature(pixel_values)
        vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)
        input_ids_flat = input_ids.reshape(B * N)

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            print(f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')

        selected = (input_ids_flat == self.img_context_token_id)

        # ---- apply QVTree modulation ----
        vit_embeds_modulated = self._apply_qvtree(
            vit_embeds=vit_embeds.reshape(-1, self.num_image_token, C),
            input_embeds=input_embeds,
            input_ids_flat=input_ids_flat,
            attention_mask=attention_mask,
        )

        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds_modulated.reshape(-1, C)
        except Exception as e:
            vit_flat = vit_embeds_modulated.reshape(-1, C)
            print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                  f'vit_embeds.shape={vit_flat.shape}')
            n_token = min(selected.sum(), vit_flat.size(0))
            input_embeds[selected][:n_token] = input_embeds[selected][:n_token] * 0.0 + vit_flat[:n_token]

        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # =========================================================
    # generate (with QVTree)
    # =========================================================
    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:

        assert self.img_context_token_id is not None

        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.extract_feature(pixel_values)

            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds_flat = input_embeds.reshape(B * N, C)
            input_ids_flat = input_ids.reshape(B * N)

            selected = (input_ids_flat == self.img_context_token_id)
            assert selected.sum() != 0

            # ---- apply QVTree modulation ----
            vit_embeds_modulated = self._apply_qvtree(
                vit_embeds=vit_embeds.reshape(-1, self.num_image_token, C),
                input_embeds=input_embeds_flat,
                input_ids_flat=input_ids_flat,
                attention_mask=attention_mask,
            )

            input_embeds_flat[selected] = vit_embeds_modulated.reshape(-1, C).to(input_embeds_flat.device)
            input_embeds = input_embeds_flat.reshape(B, N, C)

        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs

    # =========================================================
    # chat / batch_chat (unchanged from original)
    # =========================================================
    def batch_chat(self, tokenizer, pixel_values, questions, generation_config, num_patches_list=None,
                   history=None, return_history=False, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>',
                   IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', verbose=False, image_counts=None):
        if history is not None or return_history:
            print('Now multi-turn chat is not supported in batch_chat.')
            raise NotImplementedError

        if image_counts is not None:
            num_patches_list = image_counts
            print('Warning: `image_counts` is deprecated. Please use `num_patches_list` instead.')

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question
            template = get_conv_template(self.template)
            template.system_message = self.system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
            queries.append(query)

        tokenizer.padding_side = 'left'
        model_inputs = tokenizer(queries, return_tensors='pt', padding=True)
        input_ids = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        responses = [response.split(template.sep.strip())[0].strip() for response in responses]
        return responses

    def chat(self, tokenizer, pixel_values, question, generation_config, history=None, return_history=False,
             num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
             verbose=False):

        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(template.sep.strip())[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            if verbose:
                print(query_to_print, response)
            return response

    @property
    def lm_head(self):
        return self.language_model.get_output_embeddings()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()