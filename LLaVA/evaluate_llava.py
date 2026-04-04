import os
import re
import json
import torch
from PIL import Image
from tqdm import tqdm


# =========================================================
# -------------------- Common Utils -----------------------
# =========================================================

def extract_option_letter(text: str) -> str:
    """Extract A/B/C/D from model output."""
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


def evaluate_vstar_predictions(pred_file: str) -> dict:
    """
    Compute overall and per-category accuracy from prediction jsonl.
    Prints results and returns a dict.
    """
    total, correct = 0, 0
    category_stats = {}

    with open(pred_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            pred  = str(item["prediction_option"]).strip().upper()
            label = str(item["label"]).strip().upper()
            cat   = item.get("category", "unknown")

            total += 1
            hit = int(pred == label)
            correct += hit

            if cat not in category_stats:
                category_stats[cat] = {"correct": 0, "total": 0}
            category_stats[cat]["total"]   += 1
            category_stats[cat]["correct"] += hit

    overall = correct / total if total > 0 else 0.0

    print(f"\n{'='*45}")
    print(f"  Overall Accuracy : {overall*100:.2f}%  ({correct}/{total})")
    print(f"{'='*45}")
    for cat, s in sorted(category_stats.items()):
        acc = s["correct"] / s["total"]
        print(f"  {cat:30s}: {acc*100:.2f}%  ({s['correct']}/{s['total']})")
    print(f"{'='*45}\n")

    return {"overall": overall, "per_category": category_stats}


# =========================================================
# --------------------- LLaVA 1.5 ------------------------
# =========================================================

def run_vstar_inference_llava(
    model,
    processor,
    dataset,
    dataset_dir: str = "vstar_bench",
    output_file: str = "vstar_predictions_llava15.jsonl",
    max_samples: int | None = None,
    max_new_tokens: int = 16,
    run_name: str = "llava15_7b",
):
    """
    Run LLaVA-1.5-7B inference on V-Star benchmark.

    Args:
        model         : loaded LlavaForConditionalGeneration
        processor     : corresponding LlavaProcessor
        dataset       : HuggingFace Dataset object, e.g. ds["test"]
        dataset_dir   : root dir of cloned vstar_bench repo
                        (images live at dataset_dir/direct_attributes/xxx.jpg)
        output_file   : path to save prediction jsonl
        max_samples   : optional subset for quick testing
        max_new_tokens: tokens to generate
        run_name      : tag written into each prediction record

    Returns:
        output_file: path of saved jsonl
    """
    samples = dataset.select(range(max_samples)) if max_samples is not None else dataset
    device  = next(model.parameters()).device

    with open(output_file, "w", encoding="utf-8") as fout:
        for sample in tqdm(samples, desc="[LLaVA-1.5] V-Star"):
            try:
                # join dataset root dir with the relative path stored in dataset
                img_path = os.path.join(dataset_dir, sample["image"])
                image    = Image.open(img_path).convert("RGB")

                # LLaVA-1.5 official prompt format
                prompt = f"USER: <image>\n{sample['text']}\nASSISTANT:"

                inputs = processor(
                    text=prompt,
                    images=image,
                    return_tensors="pt",
                ).to(device)

                with torch.inference_mode():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                    )

                pred_text = processor.batch_decode(
                    output_ids[:, inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )[0].strip()

                pred_option = extract_option_letter(pred_text)

            except Exception as e:
                pred_text   = ""
                pred_option = ""
                print(f"[ERROR] question_id={sample.get('question_id')}: {e}")

            result = {
                "question_id"      : sample["question_id"],
                "category"         : sample["category"],
                "question"         : sample["text"],
                "label"            : sample["label"],
                "prediction_text"  : pred_text,
                "prediction_option": pred_option,
                "run_name"         : run_name,
            }

            fout.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"[INFO] Predictions saved to: {output_file}")
    return output_file