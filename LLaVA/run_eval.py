"""
run_eval.py

Evaluates LLaVA-1.5-7B on V-Star benchmark.
Runs baseline and/or tree inference, reports Attr / Spatial / Overall accuracy.

Usage:
    python run_eval.py --mode baseline
    python run_eval.py --mode tree
    python run_eval.py --mode both
    python run_eval.py --mode report
    python run_eval.py --mode both --max_samples 50

Output:
    datasets/vstar_bench/results_baseline.jsonl
    datasets/vstar_bench/results_tree.jsonl
"""

import os
import sys
import json
import argparse
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE         = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_LLAVA_DIR    = os.path.join(_PROJECT_ROOT, "LLaVA")

for p in [_PROJECT_ROOT, _LLAVA_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID    = os.path.join(_PROJECT_ROOT, "checkpoints", "llava-1.5-7b-hf")
DATASET_DIR = os.path.join(_PROJECT_ROOT, "datasets", "vstar_bench")
ANNO_FILE   = os.path.join(DATASET_DIR, "test_questions.jsonl")

OUT_BASELINE = os.path.join(DATASET_DIR, "results_baseline.jsonl")
OUT_TREE     = os.path.join(DATASET_DIR, "results_tree.jsonl")

SPLIT_THRESHOLD     = 0.3
SOFTMAX_TEMPERATURE = 0.2
TARGET_LAYERS       = (14, 15, 16, 17)
MAX_NEW_TOKENS      = 16


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def extract_option_letter(text):
    import re
    if not text:
        return ""
    text = text.strip().upper()
    m = re.search(r"\b([A-D])\b", text)
    if m:
        return m.group(1)
    m = re.search(r"^\(?([A-D])\)?[.:)]?", text)
    if m:
        return m.group(1)
    return ""


def load_samples(max_samples=None):
    with open(ANNO_FILE) as f:
        samples = [json.loads(l) for l in f]
    if max_samples is not None:
        samples = samples[:max_samples]
    return samples


def load_results_from_file(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def print_results(results, run_name):
    category_stats = {}
    total, correct = 0, 0

    for r in results:
        pred  = str(r["prediction_option"]).strip().upper()
        label = str(r["label"]).strip().upper()
        cat   = r["category"]
        hit   = int(pred == label)
        total   += 1
        correct += hit
        if cat not in category_stats:
            category_stats[cat] = {"correct": 0, "total": 0}
        category_stats[cat]["correct"] += hit
        category_stats[cat]["total"]   += 1

    overall = correct / total * 100 if total > 0 else 0.0

    print("\n" + "=" * 52)
    print("  " + run_name)
    print("=" * 52)
    print("  {:<22s}: {:.2f}%  ({}/{})".format("Overall", overall, correct, total))
    print("-" * 52)
    for cat in sorted(category_stats.keys()):
        s   = category_stats[cat]
        acc = s["correct"] / s["total"] * 100
        print("  {:<22s}: {:.2f}%  ({}/{})".format(cat, acc, s["correct"], s["total"]))
    print("=" * 52)

    return {"overall": overall, "per_category": category_stats}


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------
def run_baseline(model, processor, samples, output_path):
    from llava_with_tree import run_baseline_inference
    from PIL import Image

    print("\nRunning BASELINE on {} samples ...".format(len(samples)))

    with open(output_path, "w") as fout:
        for sample in tqdm(samples, desc="Baseline"):
            try:
                image       = Image.open(os.path.join(DATASET_DIR, sample["image"])).convert("RGB")
                pred_text   = run_baseline_inference(
                    model, processor, image, sample["text"],
                    max_new_tokens=MAX_NEW_TOKENS,
                )
                pred_option = extract_option_letter(pred_text)
            except Exception as e:
                pred_text   = ""
                pred_option = ""
                print("[ERROR] id={}: {}".format(sample.get("question_id"), e))

            fout.write(json.dumps({
                "question_id"      : sample["question_id"],
                "category"         : sample["category"],
                "label"            : sample["label"],
                "prediction_text"  : pred_text,
                "prediction_option": pred_option,
            }) + "\n")

    print("Saved: " + output_path)
    return load_results_from_file(output_path)


# ---------------------------------------------------------------------------
# Tree
# ---------------------------------------------------------------------------
def run_tree(model, processor, samples, output_path):
    from llava_with_tree import (
        run_baseline_inference,
        recover_clip_image,
        compute_patch_scores,
        select_patches,
        run_lpd,
    )
    from PIL import Image

    print("\nRunning TREE on {} samples ...".format(len(samples)))

    all_ratios = []

    with open(output_path, "w") as fout:
        for sample in tqdm(samples, desc="Tree"):
            try:
                image    = Image.open(os.path.join(DATASET_DIR, sample["image"])).convert("RGB")
                question = sample["text"]

                # Step 1: relative attention -> patch_scores
                patch_scores = compute_patch_scores(
                    model=model,
                    processor=processor,
                    image=image,
                    question=question,
                    target_layers=TARGET_LAYERS,
                )

                # Step 2: QuadTree selection
                patch_ids    = select_patches(
                    patch_scores,
                    split_threshold=SPLIT_THRESHOLD,
                    softmax_temperature=SOFTMAX_TEMPERATURE,
                )
                num_selected = int(patch_ids.numel())
                num_total    = 576
                select_ratio = num_selected / num_total
                all_ratios.append(select_ratio)

                # Step 3: LPD -> compact image
                clip_image        = recover_clip_image(processor, image, question)
                compact_image, _  = run_lpd(patch_ids, clip_image)

                # Step 4: second-pass inference
                pred_text   = run_baseline_inference(
                    model, processor, compact_image, question,
                    max_new_tokens=MAX_NEW_TOKENS,
                )
                pred_option = extract_option_letter(pred_text)

            except Exception as e:
                pred_text    = ""
                pred_option  = ""
                num_selected = 0
                num_total    = 576
                select_ratio = 0.0
                print("[ERROR] id={}: {}".format(sample.get("question_id"), e))

            fout.write(json.dumps({
                "question_id"      : sample["question_id"],
                "category"         : sample["category"],
                "label"            : sample["label"],
                "prediction_text"  : pred_text,
                "prediction_option": pred_option,
                "num_selected"     : num_selected,
                "num_total"        : num_total,
                "select_ratio"     : select_ratio,
            }) + "\n")

    if all_ratios:
        print("Mean token select ratio: {:.1f}%".format(
            sum(all_ratios) / len(all_ratios) * 100
        ))

    print("Saved: " + output_path)
    return load_results_from_file(output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "tree", "both", "report"],
                        default="both")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    if args.mode == "report":
        for path, name in [(OUT_BASELINE, "Baseline"), (OUT_TREE, "Tree")]:
            if os.path.exists(path):
                print_results(load_results_from_file(path), name)
            else:
                print("File not found: " + path)
        return

    import torch
    from transformers import LlavaForConditionalGeneration, AutoProcessor

    print("Loading model: " + MODEL_ID)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()
    print("Model ready. Device: {}".format(next(model.parameters()).device))

    samples = load_samples(args.max_samples)
    print("Loaded {} samples".format(len(samples)))

    baseline_results = None
    tree_results     = None

    if args.mode in ("baseline", "both"):
        baseline_results = run_baseline(model, processor, samples, OUT_BASELINE)
        print_results(baseline_results, "Baseline")

    if args.mode in ("tree", "both"):
        tree_results = run_tree(model, processor, samples, OUT_TREE)
        print_results(tree_results, "Tree")

    if baseline_results and tree_results:
        b = print_results(baseline_results, "Baseline")
        t = print_results(tree_results,     "Tree")

        print("\n" + "=" * 52)
        print("  Delta (Tree - Baseline)")
        print("=" * 52)
        all_cats = sorted(set(
            list(b["per_category"].keys()) + list(t["per_category"].keys())
        ))
        for cat in all_cats:
            b_acc = b["per_category"][cat]["correct"] / b["per_category"][cat]["total"] * 100 \
                    if cat in b["per_category"] else 0.0
            t_acc = t["per_category"][cat]["correct"] / t["per_category"][cat]["total"] * 100 \
                    if cat in t["per_category"] else 0.0
            d    = t_acc - b_acc
            sign = "+" if d >= 0 else ""
            print("  {:<22s}: {}{}".format(cat, sign, "{:.2f}%".format(d)))
        d_overall = t["overall"] - b["overall"]
        sign      = "+" if d_overall >= 0 else ""
        print("  {:<22s}: {}{}".format("Overall", sign, "{:.2f}%".format(d_overall)))
        print("=" * 52)


if __name__ == "__main__":
    main()