"""
run_eval_onevision.py

Evaluates LLaVA-OneVision-Qwen2-7B on V-Star, HR-Bench, POPE, TextVQA, or DocVQA.

Usage:
    python run_eval_onevision.py --mode baseline
    python run_eval_onevision.py --mode tree
    python run_eval_onevision.py --mode both
    python run_eval_onevision.py --mode report

    python run_eval_onevision.py --dataset hrbench --split 4k --mode both
    python run_eval_onevision.py --dataset pope --split adversarial --mode both
    python run_eval_onevision.py --dataset textvqa --mode both
    python run_eval_onevision.py --dataset textvqa --mode both --max_samples 100
    python run_eval_onevision.py --dataset docvqa  --mode both

    # dual mode: original (anyres) + compact (anyres)
    python run_eval_onevision.py --mode both --dual

    # local-only mode (default): compact image only
    python run_eval_onevision.py --mode both
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

for p in [_PROJECT_ROOT, _HERE]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID    = os.path.join(_PROJECT_ROOT, "checkpoints", "llava-onevision-qwen2-7b-ov")
VSTAR_DIR   = os.path.join(_PROJECT_ROOT, "datasets", "vstar_bench")
HRBENCH_DIR = os.path.join(_PROJECT_ROOT, "datasets", "hr_bench")
POPE_DIR    = os.path.join(_PROJECT_ROOT, "datasets", "pope")
TEXTVQA_DIR = os.path.join(_PROJECT_ROOT, "datasets", "textvqa")
DOCVQA_DIR  = os.path.join(_PROJECT_ROOT, "datasets", "docvqa")

SPLIT_THRESHOLD     = 0.25
SOFTMAX_TEMPERATURE = 0.2
TARGET_LAYERS       = (14, 15, 16, 17)
MAX_NEW_TOKENS      = 16
BASELINE_TOKENS     = 729   # 27x27 for SigLIP 384/patch14 – used for ratio display


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_option_letter(text):
    if not text:
        return ""
    text = text.strip().upper()
    m = re.search(r"\b([A-D])\b", text) or re.search(r"^\(?([A-D])\)?[.:)]?", text)
    return m.group(1) if m else ""


def extract_yes_no(text):
    if not text:
        return ""
    text = text.strip().upper()
    if text.startswith("YES"):  return "YES"
    if text.startswith("NO"):   return "NO"
    if "YES" in text:           return "YES"
    if "NO"  in text:           return "NO"
    return ""


def load_results(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def decode_base64_image(img_str):
    return Image.open(io.BytesIO(base64.b64decode(img_str))).convert("RGB")


def load_image_from_row(raw):
    if isinstance(raw, dict) and "bytes" in raw:
        return Image.open(io.BytesIO(raw["bytes"])).convert("RGB")
    elif isinstance(raw, Image.Image):
        return raw.convert("RGB")
    else:
        return Image.fromarray(raw).convert("RGB")


def _normalize_answer(text):
    import string
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def _vqa_accuracy(prediction, answers):
    pred    = _normalize_answer(prediction)
    matches = sum(1 for a in answers if _normalize_answer(a) == pred)
    return min(matches / 3.0, 1.0)


def _docvqa_accuracy(prediction, answers):
    pred = _normalize_answer(prediction)
    for a in answers:
        norm_a = _normalize_answer(a)
        if pred == norm_a:
            return 1.0
        if norm_a and pred and (norm_a in pred or pred in norm_a):
            return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# Result printing
# ---------------------------------------------------------------------------

def print_vstar_results(results, run_name):
    stats = {}
    total = correct = 0
    for r in results:
        pred = str(r["prediction_option"]).strip().upper()
        hit  = int(pred == str(r["label"]).strip().upper())
        cat  = r["category"]
        total += 1; correct += hit
        if cat not in stats:
            stats[cat] = {"correct": 0, "total": 0}
        stats[cat]["correct"] += hit
        stats[cat]["total"]   += 1
    overall = correct / total * 100 if total else 0.0
    print("\n" + "=" * 52)
    print("  " + run_name)
    print("=" * 52)
    print("  %-22s: %.2f%%  (%d/%d)" % ("Overall", overall, correct, total))
    print("-" * 52)
    for cat in sorted(stats):
        st = stats[cat]
        print("  %-22s: %.2f%%  (%d/%d)" % (
            cat, st["correct"] / st["total"] * 100, st["correct"], st["total"]))
    print("=" * 52)
    return {"overall": overall, "per_category": stats}


def print_hrbench_results(results, run_name):
    stats = {}
    total = correct = 0
    for r in results:
        pred = str(r["prediction_option"]).strip().upper()
        hit  = int(pred == str(r["label"]).strip().upper())
        cat  = r.get("category", "unknown")
        total += 1; correct += hit
        if cat not in stats:
            stats[cat] = {"correct": 0, "total": 0}
        stats[cat]["correct"] += hit
        stats[cat]["total"]   += 1
    overall   = correct / total * 100 if total else 0.0
    label_map = {"single": "FSP", "cross": "FCP"}
    print("\n" + "=" * 52)
    print("  " + run_name)
    print("=" * 52)
    print("  %-22s: %.2f%%  (%d/%d)" % ("Overall", overall, correct, total))
    print("-" * 52)
    for cat in sorted(stats):
        st   = stats[cat]
        name = label_map.get(cat, cat)
        print("  %-22s: %.2f%%  (%d/%d)" % (
            name, st["correct"] / st["total"] * 100, st["correct"], st["total"]))
    print("=" * 52)
    return {"overall": overall, "per_category": stats}


def print_pope_results(results, run_name):
    total = correct = 0
    for r in results:
        pred  = str(r["prediction"]).strip().upper()
        label = str(r["label"]).strip().upper()
        total += 1; correct += int(pred == label)
    overall = correct / total * 100 if total else 0.0
    print("\n" + "=" * 52)
    print("  " + run_name)
    print("=" * 52)
    print("  %-22s: %.2f%%  (%d/%d)" % ("Accuracy", overall, correct, total))
    print("=" * 52)
    return {"overall": overall}


def print_textvqa_results(results, run_name):
    total = 0; acc_sum = 0.0
    for r in results:
        total   += 1
        acc_sum += r.get("vqa_acc", 0.0)
    overall = acc_sum / total * 100 if total else 0.0
    print("\n" + "=" * 52)
    print("  " + run_name)
    print("=" * 52)
    print("  %-22s: %.2f%%  (%d samples)" % ("VQA Accuracy", overall, total))
    print("=" * 52)
    return {"overall": overall}


def print_select_ratio_stats(select_ratios, baseline_tokens):
    if not select_ratios:
        return
    mean_sel = sum(select_ratios) / len(select_ratios)
    print("\n" + "-" * 52)
    print("  QuadTree token selection ratio")
    print("-" * 52)
    print("  mean selected patches: %.1f / %d  (%.1f%%)" % (
        mean_sel * baseline_tokens, baseline_tokens, mean_sel * 100))
    print("-" * 52)


def print_delta(b, t):
    print("\n" + "=" * 52)
    print("  Delta (Tree - Baseline)")
    print("=" * 52)
    if b.get("per_category") and t.get("per_category"):
        for cat in sorted(set(list(b["per_category"]) + list(t["per_category"]))):
            b_acc = b["per_category"][cat]["correct"] / b["per_category"][cat]["total"] * 100 \
                    if cat in b["per_category"] else 0.0
            t_acc = t["per_category"][cat]["correct"] / t["per_category"][cat]["total"] * 100 \
                    if cat in t["per_category"] else 0.0
            print("  %-22s: %+.2f%%" % (cat, t_acc - b_acc))
    print("  %-22s: %+.2f%%" % ("Overall", t["overall"] - b["overall"]))
    print("=" * 52)


# ---------------------------------------------------------------------------
# Tree inference (shared)
# ---------------------------------------------------------------------------

def _tree_inference(model, processor, image, question, dual=False):
    from onevision_with_tree import (
        compute_patch_scores, select_patches,
        pad_resize_with_meta, run_lpd_on_original, run_tree_inference,
        _get_patch_size,
    )
    patch_scores, grid_size, score_size = compute_patch_scores(
        model, processor, image, question, target_layers=TARGET_LAYERS)

    patch_ids    = select_patches(patch_scores, grid_size,
                                  SPLIT_THRESHOLD, SOFTMAX_TEMPERATURE)
    num_selected = int(patch_ids.numel())
    total_tokens = grid_size * grid_size
    select_ratio = num_selected / total_tokens

    patch_size       = _get_patch_size(model)
    _, meta          = pad_resize_with_meta(image, score_size)
    compact_image, _ = run_lpd_on_original(patch_ids, image, meta, grid_size, patch_size)

    pred_text = run_tree_inference(
        model, processor, compact_image, question,
        original_image=image if dual else None,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    return pred_text, num_selected, select_ratio, total_tokens


# ---------------------------------------------------------------------------
# V-Star
# ---------------------------------------------------------------------------

def run_vstar(model, processor, mode, max_samples, out_baseline, out_tree, dual=False):
    from onevision_with_tree import run_baseline_inference

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
                    image     = Image.open(os.path.join(VSTAR_DIR, s["image"])).convert("RGB")
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
                    pred_text, num_sel, sel_ratio, tot = _tree_inference(
                        model, processor, image, s["text"], dual=dual)
                    pred_opt = extract_option_letter(pred_text)
                    select_ratios.append(sel_ratio)
                except Exception as e:
                    pred_text, pred_opt = "", ""
                    num_sel, sel_ratio, tot = 0, 0.0, BASELINE_TOKENS
                    print("[ERROR] id={}: {}".format(s.get("question_id"), e))
                fout.write(json.dumps({
                    "question_id": s["question_id"], "category": s["category"],
                    "label": s["label"], "prediction_text": pred_text,
                    "prediction_option": pred_opt,
                    "num_selected": num_sel, "num_total": tot,
                    "select_ratio": sel_ratio,
                }) + "\n")
        print_select_ratio_stats(select_ratios, BASELINE_TOKENS)
        print("Saved: " + out_tree)
        tree_results = load_results(out_tree)

    return baseline_results, tree_results


# ---------------------------------------------------------------------------
# HR-Bench
# ---------------------------------------------------------------------------

def _hrbench_question(row):
    return (
        "{}\n(A) {}\n(B) {}\n(C) {}\n(D) {}\n"
        "Answer with the option's letter from the given choices directly."
    ).format(row["question"], row["A"], row["B"], row["C"], row["D"])


def run_hrbench(model, processor, split, mode, max_samples, out_baseline, out_tree, dual=False):
    from onevision_with_tree import run_baseline_inference

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
                    pred_text, num_sel, sel_ratio, tot = _tree_inference(
                        model, processor, image, question, dual=dual)
                    pred_opt = extract_option_letter(pred_text)
                    select_ratios.append(sel_ratio)
                except Exception as e:
                    pred_text, pred_opt = "", ""
                    num_sel, sel_ratio, tot = 0, 0.0, BASELINE_TOKENS
                    print("[ERROR] index={}: {}".format(row.get("index", "?"), e))
                fout.write(json.dumps({
                    "index": int(row["index"]), "split": split,
                    "category": row["category"], "cycle_category": row["cycle_category"],
                    "label": row["answer"], "prediction_text": pred_text,
                    "prediction_option": pred_opt,
                    "num_selected": num_sel, "num_total": tot,
                    "select_ratio": sel_ratio,
                }) + "\n")
        print_select_ratio_stats(select_ratios, BASELINE_TOKENS)
        print("Saved: " + out_tree)
        tree_results = load_results(out_tree)

    return baseline_results, tree_results


# ---------------------------------------------------------------------------
# POPE
# ---------------------------------------------------------------------------

def _load_pope_split(split):
    import glob
    if split in ("adversarial", "popular", "random"):
        pattern = os.path.join(POPE_DIR, "Full", "{}-*.parquet".format(split))
    else:
        pattern = os.path.join(POPE_DIR, "data", "test-*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            "No parquet files found for split '{}' at {}".format(split, pattern))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def _pope_question(row):
    return "{} Please answer yes or no.".format(row["question"])


def run_pope(model, processor, split, mode, max_samples, out_baseline, out_tree, dual=False):
    from onevision_with_tree import run_baseline_inference

    df = _load_pope_split(split)
    if max_samples:
        df = df.iloc[:max_samples]
    print("POPE split: {}  samples: {}".format(split, len(df)))

    baseline_results = tree_results = None

    if mode in ("baseline", "both"):
        print("\nRunning POPE {} BASELINE on {} samples ...".format(split, len(df)))
        with open(out_baseline, "w") as fout:
            for _, row in tqdm(df.iterrows(), total=len(df), desc="POPE Baseline"):
                try:
                    img        = load_image_from_row(row["image"])
                    question   = _pope_question(row)
                    pred_text  = run_baseline_inference(model, processor, img, question,
                                                        max_new_tokens=MAX_NEW_TOKENS)
                    prediction = extract_yes_no(pred_text)
                except Exception as e:
                    pred_text, prediction = "", ""
                    print("[ERROR] id={}: {}".format(row.get("question_id", "?"), e))
                fout.write(json.dumps({
                    "question_id": str(row.get("question_id", "")), "split": split,
                    "label": str(row["answer"]).strip().upper(),
                    "prediction_text": pred_text, "prediction": prediction,
                }) + "\n")
        print("Saved: " + out_baseline)
        baseline_results = load_results(out_baseline)

    if mode in ("tree", "both"):
        print("\nRunning POPE {} TREE on {} samples ...".format(split, len(df)))
        select_ratios = []
        with open(out_tree, "w") as fout:
            for _, row in tqdm(df.iterrows(), total=len(df), desc="POPE Tree"):
                try:
                    img       = load_image_from_row(row["image"])
                    question  = _pope_question(row)
                    pred_text, num_sel, sel_ratio, tot = _tree_inference(
                        model, processor, img, question, dual=dual)
                    prediction = extract_yes_no(pred_text)
                    select_ratios.append(sel_ratio)
                except Exception as e:
                    pred_text, prediction = "", ""
                    num_sel, sel_ratio, tot = 0, 0.0, BASELINE_TOKENS
                    print("[ERROR] id={}: {}".format(row.get("question_id", "?"), e))
                fout.write(json.dumps({
                    "question_id": str(row.get("question_id", "")), "split": split,
                    "label": str(row["answer"]).strip().upper(),
                    "prediction_text": pred_text, "prediction": prediction,
                    "num_selected": num_sel, "num_total": tot,
                    "select_ratio": sel_ratio,
                }) + "\n")
        print_select_ratio_stats(select_ratios, BASELINE_TOKENS)
        print("Saved: " + out_tree)
        tree_results = load_results(out_tree)

    return baseline_results, tree_results


# ---------------------------------------------------------------------------
# TextVQA
# ---------------------------------------------------------------------------

def _load_textvqa(split="validation"):
    import glob
    pattern = os.path.join(TEXTVQA_DIR, "data", "{}-*.parquet".format(split))
    files   = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError("No parquet files found at: " + pattern)
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def run_textvqa(model, processor, mode, max_samples, out_baseline, out_tree, dual=False):
    from onevision_with_tree import run_baseline_inference

    df = _load_textvqa(split="validation")
    if max_samples:
        df = df.iloc[:max_samples]
    print("TextVQA validation samples: {}".format(len(df)))

    baseline_results = tree_results = None

    if mode in ("baseline", "both"):
        print("\nRunning TextVQA BASELINE on {} samples ...".format(len(df)))
        with open(out_baseline, "w") as fout:
            for _, row in tqdm(df.iterrows(), total=len(df), desc="TextVQA Baseline"):
                try:
                    img       = load_image_from_row(row["image"])
                    question  = str(row["question"]) + "\nAnswer the question using a single word or phrase."
                    pred_text = run_baseline_inference(model, processor, img, question,
                                                       max_new_tokens=32)
                    answers   = list(row["answers"])
                    vqa_acc   = _vqa_accuracy(pred_text, answers)
                except Exception as e:
                    pred_text, vqa_acc = "", 0.0
                    print("[ERROR] qid={}: {}".format(row.get("question_id", "?"), e))
                fout.write(json.dumps({
                    "question_id"    : int(row["question_id"]),
                    "prediction_text": pred_text,
                    "answers"        : list(row["answers"]),
                    "vqa_acc"        : vqa_acc,
                }) + "\n")
        print("Saved: " + out_baseline)
        baseline_results = load_results(out_baseline)

    if mode in ("tree", "both"):
        print("\nRunning TextVQA TREE on {} samples ...".format(len(df)))
        select_ratios = []
        with open(out_tree, "w") as fout:
            for _, row in tqdm(df.iterrows(), total=len(df), desc="TextVQA Tree"):
                try:
                    img       = load_image_from_row(row["image"])
                    question  = str(row["question"]) + "\nAnswer the question using a single word or phrase."
                    pred_text, num_sel, sel_ratio, tot = _tree_inference(
                        model, processor, img, question, dual=dual)
                    answers  = list(row["answers"])
                    vqa_acc  = _vqa_accuracy(pred_text, answers)
                    select_ratios.append(sel_ratio)
                except Exception as e:
                    pred_text, vqa_acc = "", 0.0
                    num_sel, sel_ratio, tot = 0, 0.0, BASELINE_TOKENS
                    print("[ERROR] qid={}: {}".format(row.get("question_id", "?"), e))
                fout.write(json.dumps({
                    "question_id"    : int(row["question_id"]),
                    "prediction_text": pred_text,
                    "answers"        : list(row["answers"]),
                    "vqa_acc"        : vqa_acc,
                    "num_selected"   : num_sel,
                    "num_total"      : tot,
                    "select_ratio"   : sel_ratio,
                }) + "\n")
        print_select_ratio_stats(select_ratios, BASELINE_TOKENS)
        print("Saved: " + out_tree)
        tree_results = load_results(out_tree)

    return baseline_results, tree_results


# ---------------------------------------------------------------------------
# DocVQA
# ---------------------------------------------------------------------------

def _load_docvqa(split="validation"):
    import glob
    pattern = os.path.join(DOCVQA_DIR, "DocVQA", "{}-*.parquet".format(split))
    files   = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError("No parquet files found at: " + pattern)
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def run_docvqa(model, processor, mode, max_samples, out_baseline, out_tree, dual=False):
    from onevision_with_tree import run_baseline_inference

    df = _load_docvqa(split="validation")
    if max_samples:
        df = df.iloc[:max_samples]
    print("DocVQA validation samples: {}".format(len(df)))

    baseline_results = tree_results = None

    if mode in ("baseline", "both"):
        print("\nRunning DocVQA BASELINE on {} samples ...".format(len(df)))
        with open(out_baseline, "w") as fout:
            for _, row in tqdm(df.iterrows(), total=len(df), desc="DocVQA Baseline"):
                try:
                    img      = load_image_from_row(row["image"])
                    question = str(row["question"]) + "\nAnswer the question using a single word or phrase."
                    pred_text = run_baseline_inference(model, processor, img, question,
                                                       max_new_tokens=32)
                    answers  = list(row["answers"])
                    vqa_acc  = _docvqa_accuracy(pred_text, answers)
                except Exception as e:
                    pred_text, vqa_acc = "", 0.0
                    print("[ERROR] qid={}: {}".format(row.get("questionId", "?"), e))
                fout.write(json.dumps({
                    "question_id"    : str(row["questionId"]),
                    "prediction_text": pred_text,
                    "answers"        : list(row["answers"]),
                    "vqa_acc"        : vqa_acc,
                }) + "\n")
        print("Saved: " + out_baseline)
        baseline_results = load_results(out_baseline)

    if mode in ("tree", "both"):
        print("\nRunning DocVQA TREE on {} samples ...".format(len(df)))
        select_ratios = []
        with open(out_tree, "w") as fout:
            for _, row in tqdm(df.iterrows(), total=len(df), desc="DocVQA Tree"):
                try:
                    img      = load_image_from_row(row["image"])
                    question = str(row["question"]) + "\nAnswer the question using a single word or phrase."
                    pred_text, num_sel, sel_ratio, tot = _tree_inference(
                        model, processor, img, question, dual=dual)
                    answers  = list(row["answers"])
                    vqa_acc  = _docvqa_accuracy(pred_text, answers)
                    select_ratios.append(sel_ratio)
                except Exception as e:
                    pred_text, vqa_acc = "", 0.0
                    num_sel, sel_ratio, tot = 0, 0.0, BASELINE_TOKENS
                    print("[ERROR] qid={}: {}".format(row.get("questionId", "?"), e))
                fout.write(json.dumps({
                    "question_id"    : str(row["questionId"]),
                    "prediction_text": pred_text,
                    "answers"        : list(row["answers"]),
                    "vqa_acc"        : vqa_acc,
                    "num_selected"   : num_sel,
                    "num_total"      : tot,
                    "select_ratio"   : sel_ratio,
                }) + "\n")
        print_select_ratio_stats(select_ratios, BASELINE_TOKENS)
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
    parser.add_argument("--dataset", choices=["vstar", "hrbench", "pope", "textvqa", "docvqa"],
                        default="vstar")
    parser.add_argument("--split",   default="4k",
                        help="hrbench: 4k/8k  |  pope: adversarial/popular/random")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--dual",    action="store_true", default=False,
                        help="Tree mode: dual-image (original anyres + compact anyres). "
                             "Default: local-only (compact image only).")
    args = parser.parse_args()

    if args.dataset == "vstar":
        out_baseline = os.path.join(VSTAR_DIR,   "results_ov_baseline.jsonl")
        out_tree     = os.path.join(VSTAR_DIR,   "results_ov_tree.jsonl")
        print_fn     = print_vstar_results
    elif args.dataset == "hrbench":
        out_baseline = os.path.join(HRBENCH_DIR, "results_ov_{}_baseline.jsonl".format(args.split))
        out_tree     = os.path.join(HRBENCH_DIR, "results_ov_{}_tree.jsonl".format(args.split))
        print_fn     = print_hrbench_results
    elif args.dataset == "pope":
        out_baseline = os.path.join(POPE_DIR,    "results_ov_{}_baseline.jsonl".format(args.split))
        out_tree     = os.path.join(POPE_DIR,    "results_ov_{}_tree.jsonl".format(args.split))
        print_fn     = print_pope_results
    elif args.dataset == "textvqa":
        out_baseline = os.path.join(TEXTVQA_DIR, "results_ov_baseline.jsonl")
        out_tree     = os.path.join(TEXTVQA_DIR, "results_ov_tree.jsonl")
        print_fn     = print_textvqa_results
    else:  # docvqa
        out_baseline = os.path.join(DOCVQA_DIR,  "results_ov_baseline.jsonl")
        out_tree     = os.path.join(DOCVQA_DIR,  "results_ov_tree.jsonl")
        print_fn     = print_textvqa_results

    if args.mode == "report":
        for path, name in [(out_baseline, "Baseline"), (out_tree, "Tree")]:
            if os.path.exists(path):
                print_fn(load_results(path), name)
            else:
                print("File not found: " + path)
        return

    import torch
    from transformers import LlavaOnevisionForConditionalGeneration, AutoProcessor

    print("Loading model: " + MODEL_ID)
    print("Tree mode: {}".format(
        "dual-image (original anyres + compact anyres)" if args.dual
        else "local-only (compact image anyres)"))

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager",
        ignore_mismatched_sizes=True,
    )
    model.eval()
    print("Model ready. Device: {}".format(next(model.parameters()).device))

    from onevision_with_tree import get_score_size, _get_patch_size, pad_resize_with_meta, run_lpd_on_original, run_tree_inference, compute_patch_scores, select_patches, run_baseline_inference
    score_size = get_score_size(processor)
    patch_size = _get_patch_size(model)
    grid_size  = score_size // patch_size
    print("Score grid: {}x{}={} tokens  (score_size={}, patch_size={})".format(
        grid_size, grid_size, grid_size * grid_size, score_size, patch_size))

    if args.dataset == "vstar":
        baseline_results, tree_results = run_vstar(
            model, processor, args.mode, args.max_samples, out_baseline, out_tree,
            dual=args.dual)
    elif args.dataset == "hrbench":
        baseline_results, tree_results = run_hrbench(
            model, processor, args.split, args.mode, args.max_samples,
            out_baseline, out_tree, dual=args.dual)
    elif args.dataset == "pope":
        baseline_results, tree_results = run_pope(
            model, processor, args.split, args.mode, args.max_samples,
            out_baseline, out_tree, dual=args.dual)
    elif args.dataset == "textvqa":
        baseline_results, tree_results = run_textvqa(
            model, processor, args.mode, args.max_samples, out_baseline, out_tree,
            dual=args.dual)
    else:
        baseline_results, tree_results = run_docvqa(
            model, processor, args.mode, args.max_samples, out_baseline, out_tree,
            dual=args.dual)

    b = print_fn(baseline_results, "Baseline") if baseline_results else None
    t = print_fn(tree_results,     "Tree")     if tree_results     else None
    if b and t:
        print_delta(b, t)


if __name__ == "__main__":
    main()