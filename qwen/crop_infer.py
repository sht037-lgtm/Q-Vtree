# crop_infer.py

import torch
from qwen_vl_utils import process_vision_info
from PIL import Image


class CropInferenceWrapper:
    def __init__(self, tree_model, base_model, processor):
        """
        Args:
            tree_model: 用来做 selection（QVTree）
            base_model: 原版 Qwen（推理）
            processor: AutoProcessor
        """
        self.tree_model = tree_model
        self.base_model = base_model
        self.processor = processor

        self.device = next(base_model.parameters()).device

    @torch.inference_mode()
    def __call__(self, image_path, question, max_new_tokens=16):
        """
        输入：image + text
        输出：模型回答
        """

        # =====================================
        # Step 1: 用 tree_model 跑一遍拿 patch_ids
        # =====================================
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question},
            ],
        }]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
        )

        inputs = {
            k: (v.to(self.device) if torch.is_tensor(v) else v)
            for k, v in inputs.items()
        }

        # forward（只为了拿 selection）
        _ = self.tree_model.generate(
            **inputs,
            max_new_tokens=1,
        )

        # =====================================
        # Step 2: 取 patch_ids
        # =====================================
        patch_ids = self.tree_model.model._debug_patch_ids[0]

        grid_h = inputs["image_grid_thw"][0][1].item() // 2
        grid_w = inputs["image_grid_thw"][0][2].item() // 2

        # =====================================
        # Step 3: crop
        # =====================================
        img, crop = get_crop_images(
            image_path,
            patch_ids,
            grid_h,
            grid_w,
        )

        # =====================================
        # Step 4: 构造 multi-image message
        # =====================================
        messages = build_crop_messages(
            img,
            crop,
            question
        )

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
        )

        inputs = {
            k: (v.to(self.device) if torch.is_tensor(v) else v)
            for k, v in inputs.items()
        }

        # =====================================
        # Step 5: 用 base_model 推理
        # =====================================
        outputs = self.base_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
        )

        pred = self.processor.batch_decode(
            outputs[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )[0]

        return pred.strip()


# =========================
# patch_ids → bounding box
# =========================
def patches_to_box(patch_ids, grid_h, grid_w, patch_size=28, pad=1):
    """
    Args:
        patch_ids: Tensor or list
        grid_h, grid_w: downsampled grid
        patch_size: pixel size per patch
        pad: expand box by N patches

    Returns:
        (x0, y0, x1, y1)
    """

    if hasattr(patch_ids, "cpu"):
        patch_ids = patch_ids.cpu().tolist()

    rows = [p // grid_w for p in patch_ids]
    cols = [p % grid_w for p in patch_ids]

    r_min = max(min(rows) - pad, 0)
    r_max = min(max(rows) + pad, grid_h - 1)
    c_min = max(min(cols) - pad, 0)
    c_max = min(max(cols) + pad, grid_w - 1)

    x0 = c_min * patch_size
    y0 = r_min * patch_size
    x1 = (c_max + 1) * patch_size
    y1 = (r_max + 1) * patch_size

    return x0, y0, x1, y1


# =========================
# 主函数：返回原图 + crop
# =========================
def get_crop_images(
    img_path,
    patch_ids,
    grid_h,
    grid_w,
    patch_size=28,
    crop_size=336,
    pad=1,
):
    """
    Args:
        img_path: 原图路径
        patch_ids: selected patches
        grid_h, grid_w: downsampled grid
        crop_size: crop resize size

    Returns:
        global_img (PIL)
        crop_img (PIL)
    """

    # load
    img = Image.open(img_path).convert("RGB")

    # resize to patch-aligned resolution
    img_resized = img.resize((grid_w * patch_size, grid_h * patch_size))

    # get box
    x0, y0, x1, y1 = patches_to_box(
        patch_ids, grid_h, grid_w, patch_size, pad
    )

    crop = img_resized.crop((x0, y0, x1, y1))

    # resize crop（关键！）
    crop = crop.resize((crop_size, crop_size))

    return img, crop


# =========================
# 构造Qwen messages
# =========================
def build_crop_messages(img, crop, question):
    """
    Return Qwen multi-image messages
    """

    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "image", "image": crop},
                {
                    "type": "text",
                    "text": question + "\nFocus on the second image.",
                },
            ],
        }
    ]