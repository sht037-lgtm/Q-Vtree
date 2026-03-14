import os
import json
import re
from PIL import Image
from tqdm import tqdm
import torch
from qwen_vl_utils import process_vision_info


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


def run_vstar_inference(
    model,
    processor,
    dataset_dir="datasets/vstar_bench",
    anno_file="test_questions.jsonl",
    output_file="vstar_predictions.jsonl",
    max_samples=None,
    max_new_tokens=16,
):
    anno_path = os.path.join(dataset_dir, anno_file)
    output_path = os.path.join(dataset_dir, output_file)

    with open(anno_path, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]

    if max_samples is not None:
        samples = samples[:max_samples]

    device = next(model.parameters()).device

    with open(output_path, "w", encoding="utf-8") as fout:
        for sample in tqdm(samples, desc="Running VStar inference"):
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
                print(f"[ERROR] question_id={sample.get('question_id')}: {e}")

            result = {
                "question_id": sample["question_id"],
                "image": sample["image"],
                "category": sample["category"],
                "question": sample["text"],
                "label": sample["label"],
                "prediction_text": pred_text,
                "prediction_option": pred_option,
            }

            fout.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"[INFO] Saved predictions to: {output_path}")


def evaluate_vstar_predictions(pred_file):
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