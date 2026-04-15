"""
run_eval.py

Evaluates LLaVA-1.5-7B on V-Star or HR-Bench benchmark.

Usage:
    # V-Star (default)
    python run_eval.py --mode baseline
    python run_eval.py --mode tree
    python run_eval.py --mode both
    python run_eval.py --mode report

    # HR-Bench
    python run_eval.py --dataset hrbench --split 4k --mode baseline
    python run_eval.py --dataset hrbench --split 4k --mode tree
    python run_eval.py --dataset hrbench --split 8k --mode both

    # Quick test
    python run_eval.py --mode both --max_samples 50
    python run_eval.py --dataset hrbench --split 4k --mode both --max_samples 50
"""

import os
import sys
import re
import io
import json
import base64
import argparse
import pandas as pd
from tqdm import tqdm
from PIL import Image

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
VSTAR_DIR   = os.path.join(_PROJECT_ROOT, "datasets", "vstar_bench")
HRBENCH_DIR = os.path.join(_PROJECT_ROOT, "datasets", "hr_bench")

SPLIT_THRESHOLD     = 0.3
SOFTMAX_TEMPERATURE = 0.2
TARGET_LAYERS       = (14, 15, 16, 17)
MAX_NEW_TOKENS      = 16
BASELINE_TOKENS     = 576


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_option_letter(text: str) -> str:
    if not text:
        return ""
    text = text.strip().upper()
    m = re.search(r"\b([A-D])\b", text) or re.search(r"^\(?([A-D])\)?[.:)]?", text)
    return m.group(1) if m else ""


def load_results(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def decode_base64_image(img_str: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(img_str))).convert("RGB")


# ---------------------------------------------------------------------------
# Result printing
# ---------------------------------------------------------------------------

def print_vstar_results(results, run_name) -> dict:
    """Reports: direct_attributes / relative_position / Overall."""
    stats = {}
    total = correct = 0
    for r in results:
        pred = str(r["prediction_option"]).strip().upper()
        hit  = int(pred == str(r["label"]).strip().upper())
        cat  = r["category"]
        total += 1;  correct += hit
        if cat not in stats:
            stats[cat] = {"correct": 0, "total": 0}
        stats[cat]["correct"] += hit
        stats[cat]["total"]   += 1

    overall = correct / total * 100 if total else 0.0
    print("\n" + "=" * 52)
    print("  " + run_name)
    print("=" * 52)
    print("  {:<22s}: {:.2f}%  ({}/{})".format("Overall", overall, correct, total))
    print("-" * 52)
    for cat in sorted(stats):
        s = stats[cat]
        print("  {:<22s}: {:.2f}%  ({}/{})".format(
            cat, s["correct"] / s["total"] * 100, s["correct"], s["total"]))
    print("=" * 52)
    return {"overall": overall, "per_category": stats}


def print_hrbench_results(results, run_name) -> dict:
    """Reports: FSP (single) / FCP (cross) / Overall."""
    stats = {}
    total = correct = 0
    for r in results:
        pred = str(r["prediction_option"]).strip().upper()
        hit  = int(pred == str(r["label"]).strip().upper())
        cat  = r.get("category", "unknown")
        total += 1;  correct += hit
        if cat not in stats:
            stats[cat] = {"correct": 0, "total": 0}
        stats[cat]["correct"] += hit
        stats[cat]["total"]   += 1

    overall = correct / total * 100 if total else 0.0

    label_map = {"single": "FSP", "cross": "FCP"}

    print("\n" + "=" * 52)
    print("  " + run_name)
    print("=" * 52)
    print("  %-22s: %.2f%%  (%d/%d)" % ("Overall", overall, correct, total))
    print("-" * 52)
    for cat in sorted(stats):
        st   = stats[cat]
        acc  = st["correct"] / st["total"] * 100
        name = label_map.get(cat, cat)
        print("  %-22s: %.2f%%  (%d/%d)" % (name, acc, st["correct"], st["total"]))
    print("=" * 52)
    return {"overall": overall, "per_category": stats}


def print_select_ratio_stats(select_ratios):
    if not select_ratios:
        return
    mean_sel = sum(select_ratios) / len(select_ratios)
    print("\n" + "-" * 52)
    print("  QuadTree token selection ratio")
    print("-" * 52)
    print("  mean selected patches: {:.1f} / {}  ({:.1f}%)".format(
        mean_sel * BASELINE_TOKENS, BASELINE_TOKENS, mean_sel * 100))
    print("-" * 52)


def print_delta(b, t):
    print("\n" + "=" * 52)
    print("  Delta (Tree - Baseline)")
    print("=" * 52)
    for cat in sorted(set(list(b["per_category"]) + list(t["per_category"]))):
        b_acc = b["per_category"][cat]["correct"] / b["per_category"][cat]["total"] * 100 \
                if cat in b["per_category"] else 0.0
        t_acc = t["per_category"][cat]["correct"] / t["per_category"][cat]["total"] * 100 \
                if cat in t["per_category"] else 0.0
        print("  {:<22s}: {:+.2f}%".format(cat, t_acc - b_acc))
    print("  {:<22s}: {:+.2f}%".format("Overall", t["overall"] - b["overall"]))
    print("=" * 52)


# ---------------------------------------------------------------------------
# Tree inference (shared between datasets)
# ---------------------------------------------------------------------------

def _tree_inference(model, processor, image, question):
    from llava_with_tree import (
        compute_patch_scores, select_patches,
        pad_resize_with_meta, run_lpd_on_original, run_tree_inference,
    )
    patch_scores     = compute_patch_scores(
        model, processor, image, question, target_layers=TARGET_LAYERS)
    patch_ids        = select_patches(patch_scores, SPLIT_THRESHOLD, SOFTMAX_TEMPERATURE)
    num_selected     = int(patch_ids.numel())
    select_ratio     = num_selected / BASELINE_TOKENS
    _, meta          = pad_resize_with_meta(image)
    compact_image, _ = run_lpd_on_original(patch_ids, image, meta)
    pred_text        = run_tree_inference(model, processor, compact_image, question,
                                         max_new_tokens=MAX_NEW_TOKENS)
    return pred_text, num_selected, select_ratio


# ---------------------------------------------------------------------------
# V-Star
# ---------------------------------------------------------------------------

def run_vstar(model, processor, mode, max_samples, out_baseline, out_tree):
    from llava_with_tree import run_baseline_inference

    anno = os.path.join(VSTAR_DIR, "test_questions.jsonl")
    with open(anno) as f:
        samples = [json.loads(l) for l in f]
    if max_samples:
        samples = samples[:max_samples]

    baseline_results = tree_results = None

    if mode in ("baseline", "both"):
        print("\nRunning V-Star BASELINE on {} samples ...".format(len(samples)))
        with open(out_baseline, "w") as fout:
            for s in tqdm(samples, desc="V-Star Baseline"):
                try:
                    image    = Image.open(os.path.join(VSTAR_DIR, s["image"])).convert("RGB")
                    pred_text = run_baseline_inference(model, processor, image, s["text"],
                                                       max_new_tokens=MAX_NEW_TOKENS)
                    pred_opt  = extract_option_letter(pred_text)
                except Exception as e:
                    pred_text, pred_opt = "", ""
                    print("[ERROR] id={}: {}".format(s.get("question_id"), e))
                fout.write(json.dumps({
                    "question_id": s["question_id"], "category": s["category"],
                    "label": s["label"], "prediction_text": pred_text,
                    "prediction_option": pred_opt,
                }) + "\n")
        print("Saved: " + out_baseline)
        baseline_results = load_results(out_baseline)

    if mode in ("tree", "both"):
        print("\nRunning V-Star TREE on {} samples ...".format(len(samples)))
        select_ratios = []
        with open(out_tree, "w") as fout:
            for s in tqdm(samples, desc="V-Star Tree"):
                try:
                    image    = Image.open(os.path.join(VSTAR_DIR, s["image"])).convert("RGB")
                    pred_text, num_sel, sel_ratio = _tree_inference(
                        model, processor, image, s["text"])
                    pred_opt = extract_option_letter(pred_text)
                    select_ratios.append(sel_ratio)
                except Exception as e:
                    pred_text, pred_opt = "", ""
                    num_sel, sel_ratio = 0, 0.0
                    print("[ERROR] id={}: {}".format(s.get("question_id"), e))
                fout.write(json.dumps({
                    "question_id": s["question_id"], "category": s["category"],
                    "label": s["label"], "prediction_text": pred_text,
                    "prediction_option": pred_opt,
                    "num_selected": num_sel, "num_total": BASELINE_TOKENS,
                    "select_ratio": sel_ratio,
                }) + "\n")
        print_select_ratio_stats(select_ratios)
        print("Saved: " + out_tree)
        tree_results = load_results(out_tree)

    return baseline_results, tree_results


# ---------------------------------------------------------------------------
# HR-Bench
# ---------------------------------------------------------------------------

def _hrbench_question(row) -> str:
    return (
        "{}\n(A) {}\n(B) {}\n(C) {}\n(D) {}\n"
        "Answer with the option's letter from the given choices directly."
    ).format(row["question"], row["A"], row["B"], row["C"], row["D"])


def run_hrbench(model, processor, split, mode, max_samples, out_baseline, out_tree):
    from llava_with_tree import run_baseline_inference

    tsv_path = os.path.join(HRBENCH_DIR, "hr_bench_{}.tsv".format(split))
    df = pd.read_csv(tsv_path, sep="\t")
    if max_samples:
        df = df.iloc[:max_samples]

    baseline_results = tree_results = None

    if mode in ("baseline", "both"):
        print("\nRunning HR-Bench {} BASELINE on {} samples ...".format(split, len(df)))
        with open(out_baseline, "w") as fout:
            for _, row in tqdm(df.iterrows(), total=len(df), desc="HR-Bench Baseline"):
                try:
                    image     = decode_base64_image(row["image"])
                    question  = _hrbench_question(row)
                    pred_text = run_baseline_inference(model, processor, image, question,
                                                       max_new_tokens=MAX_NEW_TOKENS)
                    pred_opt  = extract_option_letter(pred_text)
                except Exception as e:
                    pred_text, pred_opt = "", ""
                    print("[ERROR] index={}: {}".format(row.get("index", "?"), e))
                fout.write(json.dumps({
                    "index": int(row["index"]), "split": split,
                    "category": row["category"], "cycle_category": row["cycle_category"],
                    "label": row["answer"], "prediction_text": pred_text,
                    "prediction_option": pred_opt,
                }) + "\n")
        print("Saved: " + out_baseline)
        baseline_results = load_results(out_baseline)

    if mode in ("tree", "both"):
        print("\nRunning HR-Bench {} TREE on {} samples ...".format(split, len(df)))
        select_ratios = []
        with open(out_tree, "w") as fout:
            for _, row in tqdm(df.iterrows(), total=len(df), desc="HR-Bench Tree"):
                try:
                    image    = decode_base64_image(row["image"])
                    question = _hrbench_question(row)
                    pred_text, num_sel, sel_ratio = _tree_inference(
                        model, processor, image, question)
                    pred_opt = extract_option_letter(pred_text)
                    select_ratios.append(sel_ratio)
                except Exception as e:
                    pred_text, pred_opt = "", ""
                    num_sel, sel_ratio = 0, 0.0
                    print("[ERROR] index={}: {}".format(row.get("index", "?"), e))
                fout.write(json.dumps({
                    "index": int(row["index"]), "split": split,
                    "category": row["category"], "cycle_category": row["cycle_category"],
                    "label": row["answer"], "prediction_text": pred_text,
                    "prediction_option": pred_opt,
                    "num_selected": num_sel, "num_total": BASELINE_TOKENS,
                    "select_ratio": sel_ratio,
                }) + "\n")
        print_select_ratio_stats(select_ratios)
        print("Saved: " + out_tree)
        tree_results = load_results(out_tree)

    return baseline_results, tree_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",    choices=["baseline", "tree", "both", "report"],
                        default="both")
    parser.add_argument("--dataset", choices=["vstar", "hrbench"], default="vstar")
    parser.add_argument("--split",   choices=["4k", "8k"], default="4k",
                        help="HR-Bench split, ignored for vstar")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    is_hrbench = (args.dataset == "hrbench")
    print_fn   = print_hrbench_results if is_hrbench else print_vstar_results

    if is_hrbench:
        out_baseline = os.path.join(HRBENCH_DIR,
                                    "results_{}_baseline.jsonl".format(args.split))
        out_tree     = os.path.join(HRBENCH_DIR,
                                    "results_{}_tree.jsonl".format(args.split))
    else:
        out_baseline = os.path.join(VSTAR_DIR, "results_baseline.jsonl")
        out_tree     = os.path.join(VSTAR_DIR, "results_tree.jsonl")

    if args.mode == "report":
        for path, name in [(out_baseline, "Baseline"), (out_tree, "Tree")]:
            if os.path.exists(path):
                print_fn(load_results(path), name)
            else:
                print("File not found: " + path)
        return

    import torch
    from transformers import LlavaForConditionalGeneration, AutoProcessor

    print("Loading model: " + MODEL_ID)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16,
        device_map="auto", attn_implementation="eager",
    )
    model.eval()
    print("Model ready. Device: {}".format(next(model.parameters()).device))

    if is_hrbench:
        baseline_results, tree_results = run_hrbench(
            model, processor, args.split, args.mode,
            args.max_samples, out_baseline, out_tree)
    else:
        baseline_results, tree_results = run_vstar(
            model, processor, args.mode,
            args.max_samples, out_baseline, out_tree)

    if baseline_results:
        print_fn(baseline_results, "Baseline")
    if tree_results:
        print_fn(tree_results, "Tree")
    if baseline_results and tree_results:
        b = print_fn(baseline_results, "Baseline")
        t = print_fn(tree_results,     "Tree")
        print_delta(b, t)


if __name__ == "__main__":
    main()