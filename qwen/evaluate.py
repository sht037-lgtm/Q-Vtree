import os
import io
import re
import json
import base64
import pandas as pd
import torch

from PIL import Image
from tqdm import tqdm
from qwen_vl_utils import process_vision_info


# =========================================================
# -------------------- Common Utils -----------------------
# =========================================================
def extract_option_letter(text: str) -> str:
    if text is None:
        return ""

    text = text.strip().upper()

    m = re.search(r"\b([A-D])\b", text)
    if m:
        return m.group(1)

    m = re.search(r"^\(?([A-D])\)?[.:)]?", text)
    if m:
        return m.group(1)

    return ""


def decode_base64_image(img_str: str) -> Image.Image:
    img_bytes = base64.b64decode(img_str)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def evaluate_mcq_predictions(pred_file: str) -> float:
    total = 0
    correct = 0

    with open(pred_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            pred = str(item["prediction_option"]).strip().upper()
            label = str(item["label"]).strip().upper()

            total += 1
            if pred == label:
                correct += 1

    acc = correct / total if total > 0 else 0.0
    print(f"Accuracy: {acc:.4f} ({correct}/{total})")
    return acc


# =========================================================
# ---------------------- V-Star ---------------------------
# =========================================================
import os
import json
from tqdm import tqdm
from PIL import Image
import torch
from qwen_vl_utils import process_vision_info


def run_vstar_inference(
    model,
    processor,
    dataset_dir: str = "datasets/vstar_bench",
    anno_file: str = "test_questions.jsonl",
    output_file: str | None = None,
    max_samples: int | None = None,
    max_new_tokens: int = 16,
    model_type: str = "base_qwen",   # "base_qwen" or "tree_qwen"
    run_name: str | None = None,     # optional custom tag for output naming
):
    """
    Run inference on V-Star.

    Args:
        model: loaded model
        processor: corresponding processor
        dataset_dir: dataset root
        anno_file: annotation jsonl
        output_file: optional explicit output filename
        max_samples: optional subset size
        max_new_tokens: generation length
        model_type: type tag for experiment tracking
                    supported: "base_qwen", "tree_qwen"
        run_name: optional custom run tag, e.g. "tree_alpha05_beta02"

    Returns:
        output_path: saved jsonl prediction path
    """
    print("[DEBUG] model class =", model.__class__.__name__)

    if model_type not in ["base_qwen", "tree_qwen"]:
        raise ValueError(f"Unsupported model_type: {model_type}")

    anno_path = os.path.join(dataset_dir, anno_file)

    if output_file is None:
        tag = run_name if run_name is not None else model_type
        output_file = f"vstar_predictions_{tag}.jsonl"

    output_path = os.path.join(dataset_dir, output_file)

    with open(anno_path, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]

    if max_samples is not None:
        samples = samples[:max_samples]

    device = next(model.parameters()).device

    with open(output_path, "w", encoding="utf-8") as fout:
        for sample in tqdm(samples, desc=f"Running V-Star inference [{model_type}]"):
            img_path = os.path.join(dataset_dir, sample["image"])
            question = sample["text"]

            try:
                image = Image.open(img_path).convert("RGB")

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": question},
                        ],
                    }
                ]

                text = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

                image_inputs, video_inputs = process_vision_info(messages)

                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )

                inputs = {
                    k: (v.to(device) if torch.is_tensor(v) else v)
                    for k, v in inputs.items()
                }

                with torch.inference_mode():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                    )

                pred_text = processor.batch_decode(
                    outputs[:, inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )[0].strip()

                pred_option = extract_option_letter(pred_text)

            except Exception as e:
                pred_text = ""
                pred_option = ""
                print(f"[ERROR][{model_type}] question_id={sample.get('question_id')}: {e}")

            result = {
                "question_id": sample["question_id"],
                "image": sample["image"],
                "category": sample["category"],
                "question": sample["text"],
                "label": sample["label"],
                "prediction_text": pred_text,
                "prediction_option": pred_option,
                "model_type": model_type,
                "run_name": run_name if run_name is not None else model_type,
            }

            fout.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"[INFO] Saved predictions to: {output_path}")
    return output_path


def evaluate_vstar_predictions(pred_file: str) -> float:
    return evaluate_mcq_predictions(pred_file)


# =========================================================
# --------------------- HR-Bench --------------------------
# =========================================================
import os
import json
import pandas as pd
from tqdm import tqdm
import torch
from qwen_vl_utils import process_vision_info


def run_hrbench_inference(
    model,
    processor,
    split: str = "4k",
    dataset_dir: str = "datasets/hr_bench",
    output_file: str | None = None,
    max_samples: int | None = None,
    max_new_tokens: int = 16,
    model_type: str = "base_qwen",   # "base_qwen" or "tree_qwen"
    run_name: str | None = None,     # optional custom tag
):
    """
    Run inference on HR-Bench.

    Args:
        model: loaded model
        processor: corresponding processor
        split: "4k" or "8k"
        dataset_dir: dataset root
        output_file: optional explicit output filename
        max_samples: optional subset size
        max_new_tokens: generation length
        model_type: type tag for experiment tracking
                    supported: "base_qwen", "tree_qwen"
        run_name: optional custom run tag, e.g. "tree_alpha05_beta02"

    Returns:
        output_path: saved jsonl prediction path
    """
    print("[DEBUG] model class =", model.__class__.__name__)

    if split not in ["4k", "8k"]:
        raise ValueError("split must be either '4k' or '8k'")

    if model_type not in ["base_qwen", "tree_qwen"]:
        raise ValueError(f"Unsupported model_type: {model_type}")

    tsv_path = os.path.join(dataset_dir, f"hr_bench_{split}.tsv")

    if output_file is None:
        tag = run_name if run_name is not None else model_type
        output_file = f"hr_bench_{split}_predictions_{tag}.jsonl"

    output_path = os.path.join(dataset_dir, output_file)

    df = pd.read_csv(tsv_path, sep="\t")
    if max_samples is not None:
        df = df.iloc[:max_samples]

    device = next(model.parameters()).device

    with open(output_path, "w", encoding="utf-8") as fout:
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Running HR-Bench {split} [{model_type}]"):
            try:
                image = decode_base64_image(row["image"])

                question_text = (
                    f"{row['question']}\n"
                    f"(A) {row['A']}\n"
                    f"(B) {row['B']}\n"
                    f"(C) {row['C']}\n"
                    f"(D) {row['D']}\n"
                    f"Answer with the option's letter from the given choices directly."
                )

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": question_text},
                        ],
                    }
                ]

                text = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

                image_inputs, video_inputs = process_vision_info(messages)

                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )

                inputs = {
                    k: (v.to(device) if torch.is_tensor(v) else v)
                    for k, v in inputs.items()
                }

                with torch.inference_mode():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                    )

                pred_text = processor.batch_decode(
                    outputs[:, inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )[0].strip()

                pred_option = extract_option_letter(pred_text)

            except Exception as e:
                pred_text = ""
                pred_option = ""
                print(f"[ERROR][{model_type}] index={row.get('index', 'unknown')}: {e}")

            result = {
                "index": int(row["index"]),
                "split": split,
                "question": row["question"],
                "A": row["A"],
                "B": row["B"],
                "C": row["C"],
                "D": row["D"],
                "category": row["category"],
                "cycle_category": row["cycle_category"],
                "label": row["answer"],
                "prediction_text": pred_text,
                "prediction_option": pred_option,
                "model_type": model_type,
                "run_name": run_name if run_name is not None else model_type,
            }

            fout.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"[INFO] Saved predictions to: {output_path}")
    return output_path


def evaluate_hrbench_predictions(pred_file: str) -> float:
    return evaluate_mcq_predictions(pred_file)