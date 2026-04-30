"""
run_eval.py

Evaluates LLaVA-1.5-7B on V-Star, HR-Bench, POPE, TextVQA, DocVQA,
MME-RealWorld-Perception, or MME-RealWorld-Reasoning benchmarks.

Usage:
    python run_eval.py --mode baseline
    python run_eval.py --mode tree
    python run_eval.py --mode both
    python run_eval.py --mode report

    python run_eval.py --dataset hrbench --split 4k --mode both
    python run_eval.py --dataset pope --split adversarial --mode both
    python run_eval.py --dataset textvqa --mode both
    python run_eval.py --dataset textvqa --mode both --max_samples 100

    # MME-RealWorld  (100 samples per l2-category, --dual for compact+global)
    python run_eval.py --dataset mme-perception --mode baseline
    python run_eval.py --dataset mme-perception --mode tree
    python run_eval.py --dataset mme-perception --mode tree --dual
    python run_eval.py --dataset mme-reasoning  --mode both
    python run_eval.py --dataset mme-reasoning  --mode both --dual
    python run_eval.py --dataset mme-perception --mode report
"""

import os
import sys
import re
import io
import ast
import json
import base64
import argparse
import pandas as pd
from tqdm import tqdm
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_LLAVA_DIR = os.path.join(_PROJECT_ROOT, "LLaVA")

for p in [_PROJECT_ROOT, _LLAVA_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID = os.path.join(_PROJECT_ROOT, "checkpoints", "llava-1.5-7b-hf")
VSTAR_DIR = os.path.join(_PROJECT_ROOT, "datasets", "vstar_bench")
HRBENCH_DIR = os.path.join(_PROJECT_ROOT, "datasets", "hr_bench")
POPE_DIR = os.path.join(_PROJECT_ROOT, "datasets", "pope")
TEXTVQA_DIR = os.path.join(_PROJECT_ROOT, "datasets", "textvqa")
DOCVQA_DIR = os.path.join(_PROJECT_ROOT, "datasets", "docvqa")
MME_DIR = os.path.join(_PROJECT_ROOT, "datasets", "mme_realworld")

SPLIT_THRESHOLD = 0.3
SOFTMAX_TEMPERATURE = 0.2
TARGET_LAYERS = (14, 15, 16, 17)
MAX_NEW_TOKENS = 16
BASELINE_TOKENS = 576

# Number of samples per l2-category for MME-RealWorld evaluation
MME_SAMPLES_PER_CAT = 100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_option_letter(text: str) -> str:
    if not text:
        return ""
    text = text.strip().upper()
    m = re.search(r"\b([A-D])\b", text) or re.search(r"^\(?([A-D])\)?[.:)]?", text)
    return m.group(1) if m else ""


def extract_option_letter_abcde(text: str) -> str:
    """Like extract_option_letter but also handles option E (MME-RealWorld)."""
    if not text:
        return ""
    text = text.strip().upper()
    m = re.search(r"\b([A-E])\b", text) or re.search(r"^\(?([A-E])\)?[.:)]?", text)
    return m.group(1) if m else ""


def extract_yes_no(text: str) -> str:
    if not text:
        return ""
    text = text.strip().upper()
    if text.startswith("YES"):   return "YES"
    if text.startswith("NO"):    return "NO"
    if "YES" in text:            return "YES"
    if "NO" in text:            return "NO"
    return ""


def load_results(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def decode_base64_image(img_str: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(img_str))).convert("RGB")


def load_image_from_row(raw) -> Image.Image:
    """
    Handle multiple image formats stored in parquet:
      - dict with 'bytes' key  (lmms-lab format)
      - PIL Image
      - numpy array
    """
    if isinstance(raw, dict) and "bytes" in raw:
        return Image.open(io.BytesIO(raw["bytes"])).convert("RGB")
    elif isinstance(raw, Image.Image):
        return raw.convert("RGB")
    else:
        return Image.fromarray(raw).convert("RGB")


# ---------------------------------------------------------------------------
# Result printing  (original datasets — unchanged)
# ---------------------------------------------------------------------------

def print_vstar_results(results, run_name) -> dict:
    stats = {}
    total = correct = 0
    for r in results:
        pred = str(r["prediction_option"]).strip().upper()
        hit = int(pred == str(r["label"]).strip().upper())
        cat = r["category"]
        total += 1;
        correct += hit
        if cat not in stats:
            stats[cat] = {"correct": 0, "total": 0}
        stats[cat]["correct"] += hit
        stats[cat]["total"] += 1

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


def print_hrbench_results(results, run_name) -> dict:
    stats = {}
    total = correct = 0
    for r in results:
        pred = str(r["prediction_option"]).strip().upper()
        hit = int(pred == str(r["label"]).strip().upper())
        cat = r.get("category", "unknown")
        total += 1;
        correct += hit
        if cat not in stats:
            stats[cat] = {"correct": 0, "total": 0}
        stats[cat]["correct"] += hit
        stats[cat]["total"] += 1

    overall = correct / total * 100 if total else 0.0
    label_map = {"single": "FSP", "cross": "FCP"}
    print("\n" + "=" * 52)
    print("  " + run_name)
    print("=" * 52)
    print("  %-22s: %.2f%%  (%d/%d)" % ("Overall", overall, correct, total))
    print("-" * 52)
    for cat in sorted(stats):
        st = stats[cat]
        name = label_map.get(cat, cat)
        print("  %-22s: %.2f%%  (%d/%d)" % (
            name, st["correct"] / st["total"] * 100, st["correct"], st["total"]))
    print("=" * 52)
    return {"overall": overall, "per_category": stats}


def print_pope_results(results, run_name) -> dict:
    total = correct = 0
    for r in results:
        pred = str(r["prediction"]).strip().upper()
        label = str(r["label"]).strip().upper()
        total += 1;
        correct += int(pred == label)

    overall = correct / total * 100 if total else 0.0
    print("\n" + "=" * 52)
    print("  " + run_name)
    print("=" * 52)
    print("  %-22s: %.2f%%  (%d/%d)" % ("Accuracy", overall, correct, total))
    print("=" * 52)
    return {"overall": overall}


def _normalize_answer(text: str) -> str:
    import string
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def _vqa_accuracy(prediction: str, answers: list) -> float:
    pred = _normalize_answer(prediction)
    matches = sum(1 for a in answers if _normalize_answer(a) == pred)
    return min(matches / 3.0, 1.0)


def print_textvqa_results(results, run_name) -> dict:
    total = 0
    acc_sum = 0.0
    for r in results:
        total += 1
        acc_sum += r.get("vqa_acc", 0.0)

    overall = acc_sum / total * 100 if total else 0.0
    print("\n" + "=" * 52)
    print("  " + run_name)
    print("=" * 52)
    print("  %-22s: %.2f%%  (%d samples)" % ("VQA Accuracy", overall, total))
    print("=" * 52)
    return {"overall": overall}


def print_select_ratio_stats(select_ratios):
    if not select_ratios:
        return
    mean_sel = sum(select_ratios) / len(select_ratios)
    print("\n" + "-" * 52)
    print("  QuadTree token selection ratio")
    print("-" * 52)
    print("  mean selected patches: %.1f / %d  (%.1f%%)" % (
        mean_sel * BASELINE_TOKENS, BASELINE_TOKENS, mean_sel * 100))
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

def _tree_inference(model, processor, image, question, dual=True):
    from llava_with_tree import (
        compute_patch_scores, select_patches,
        pad_resize_with_meta, run_lpd_on_original, run_tree_inference,
    )
    patch_scores = compute_patch_scores(
        model, processor, image, question, target_layers=TARGET_LAYERS)
    patch_ids = select_patches(patch_scores, SPLIT_THRESHOLD, SOFTMAX_TEMPERATURE)
    num_selected = int(patch_ids.numel())
    select_ratio = num_selected / BASELINE_TOKENS
    _, meta = pad_resize_with_meta(image)
    compact_image, _ = run_lpd_on_original(patch_ids, image, meta)
    # dual=True : original (576 tokens) + compact (576 tokens) + question
    # dual=False: compact only (576 tokens) + question
    pred_text = run_tree_inference(model, processor, compact_image, question,
                                   max_new_tokens=MAX_NEW_TOKENS,
                                   original_image=image if dual else None)
    return pred_text, num_selected, select_ratio


# ---------------------------------------------------------------------------
# V-Star
# ---------------------------------------------------------------------------

def run_vstar(model, processor, mode, max_samples, out_baseline, out_tree, dual=True):
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
                    image = Image.open(os.path.join(VSTAR_DIR, s["image"])).convert("RGB")
                    pred_text = run_baseline_inference(model, processor, image, s["text"],
                                                       max_new_tokens=MAX_NEW_TOKENS)
                    pred_opt = extract_option_letter(pred_text)
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
                    image = Image.open(os.path.join(VSTAR_DIR, s["image"])).convert("RGB")
                    pred_text, num_sel, sel_ratio = _tree_inference(
                        model, processor, image, s["text"], dual=dual)
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


def run_hrbench(model, processor, split, mode, max_samples, out_baseline, out_tree, dual=True):
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
                    image = decode_base64_image(row["image"])
                    question = _hrbench_question(row)
                    pred_text = run_baseline_inference(model, processor, image, question,
                                                       max_new_tokens=MAX_NEW_TOKENS)
                    pred_opt = extract_option_letter(pred_text)
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
                    image = decode_base64_image(row["image"])
                    question = _hrbench_question(row)
                    pred_text, num_sel, sel_ratio = _tree_inference(
                        model, processor, image, question, dual=dual)
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
        raise FileNotFoundError("No parquet files found for split '{}' at {}".format(
            split, pattern))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def _pope_question(row) -> str:
    return "{} Please answer yes or no.".format(row["question"])


def run_pope(model, processor, split, mode, max_samples, out_baseline, out_tree, dual=True):
    from llava_with_tree import run_baseline_inference

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
                    img = load_image_from_row(row["image"])
                    question = _pope_question(row)
                    pred_text = run_baseline_inference(model, processor, img, question,
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
                    img = load_image_from_row(row["image"])
                    question = _pope_question(row)
                    pred_text, num_sel, sel_ratio = _tree_inference(
                        model, processor, img, question, dual=dual)
                    prediction = extract_yes_no(pred_text)
                    select_ratios.append(sel_ratio)
                except Exception as e:
                    pred_text, prediction = "", ""
                    num_sel, sel_ratio = 0, 0.0
                    print("[ERROR] id={}: {}".format(row.get("question_id", "?"), e))
                fout.write(json.dumps({
                    "question_id": str(row.get("question_id", "")), "split": split,
                    "label": str(row["answer"]).strip().upper(),
                    "prediction_text": pred_text, "prediction": prediction,
                    "num_selected": num_sel, "num_total": BASELINE_TOKENS,
                    "select_ratio": sel_ratio,
                }) + "\n")
        print_select_ratio_stats(select_ratios)
        print("Saved: " + out_tree)
        tree_results = load_results(out_tree)

    return baseline_results, tree_results


# ---------------------------------------------------------------------------
# TextVQA
# ---------------------------------------------------------------------------

def _load_textvqa(split="validation"):
    import glob
    pattern = os.path.join(TEXTVQA_DIR, "data", "{}-*.parquet".format(split))
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError("No parquet files found at: " + pattern)
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def run_textvqa(model, processor, mode, max_samples, out_baseline, out_tree, dual=True):
    from llava_with_tree import run_baseline_inference

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
                    img = load_image_from_row(row["image"])
                    question = str(row["question"]) + "\nAnswer the question using a single word or phrase."
                    pred_text = run_baseline_inference(model, processor, img, question,
                                                       max_new_tokens=32)
                    answers = list(row["answers"])
                    vqa_acc = _vqa_accuracy(pred_text, answers)
                except Exception as e:
                    pred_text, vqa_acc = "", 0.0
                    print("[ERROR] qid={}: {}".format(row.get("question_id", "?"), e))
                fout.write(json.dumps({
                    "question_id": int(row["question_id"]),
                    "prediction_text": pred_text,
                    "answers": list(row["answers"]),
                    "vqa_acc": vqa_acc,
                }) + "\n")
        print("Saved: " + out_baseline)
        baseline_results = load_results(out_baseline)

    if mode in ("tree", "both"):
        print("\nRunning TextVQA TREE on {} samples ...".format(len(df)))
        select_ratios = []
        with open(out_tree, "w") as fout:
            for _, row in tqdm(df.iterrows(), total=len(df), desc="TextVQA Tree"):
                try:
                    img = load_image_from_row(row["image"])
                    question = str(row["question"]) + "\nAnswer the question using a single word or phrase."
                    pred_text, num_sel, sel_ratio = _tree_inference(
                        model, processor, img, question, dual=dual)
                    answers = list(row["answers"])
                    vqa_acc = _vqa_accuracy(pred_text, answers)
                    select_ratios.append(sel_ratio)
                except Exception as e:
                    pred_text, vqa_acc = "", 0.0
                    num_sel, sel_ratio = 0, 0.0
                    print("[ERROR] qid={}: {}".format(row.get("question_id", "?"), e))
                fout.write(json.dumps({
                    "question_id": int(row["question_id"]),
                    "prediction_text": pred_text,
                    "answers": list(row["answers"]),
                    "vqa_acc": vqa_acc,
                    "num_selected": num_sel,
                    "num_total": BASELINE_TOKENS,
                    "select_ratio": sel_ratio,
                }) + "\n")
        print_select_ratio_stats(select_ratios)
        print("Saved: " + out_tree)
        tree_results = load_results(out_tree)

    return baseline_results, tree_results


# ---------------------------------------------------------------------------
# DocVQA
# ---------------------------------------------------------------------------

def _load_docvqa(split="validation"):
    import glob
    pattern = os.path.join(DOCVQA_DIR, "DocVQA", "{}-*.parquet".format(split))
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError("No parquet files found at: " + pattern)
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def _docvqa_accuracy(prediction: str, answers: list) -> float:
    pred = _normalize_answer(prediction)
    for a in answers:
        norm_a = _normalize_answer(a)
        if pred == norm_a:
            return 1.0
        # fallback: answer contained in prediction or prediction contained in answer
        if norm_a and pred and (norm_a in pred or pred in norm_a):
            return 1.0
    return 0.0


def run_docvqa(model, processor, mode, max_samples, out_baseline, out_tree, dual=True):
    from llava_with_tree import run_baseline_inference
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
                    img = load_image_from_row(row["image"])
                    question = str(row["question"]) + "\nAnswer the question using a single word or phrase."
                    pred_text = run_baseline_inference(model, processor, img, question,
                                                       max_new_tokens=32)
                    answers = list(row["answers"])
                    vqa_acc = _docvqa_accuracy(pred_text, answers)
                except Exception as e:
                    pred_text, vqa_acc = "", 0.0
                    print("[ERROR] qid={}: {}".format(row.get("questionId", "?"), e))
                fout.write(json.dumps({
                    "question_id": str(row["questionId"]),
                    "prediction_text": pred_text,
                    "answers": list(row["answers"]),
                    "vqa_acc": vqa_acc,
                }) + "\n")
        print("Saved: " + out_baseline)
        baseline_results = load_results(out_baseline)
    if mode in ("tree", "both"):
        print("\nRunning DocVQA TREE on {} samples ...".format(len(df)))
        select_ratios = []
        with open(out_tree, "w") as fout:
            for _, row in tqdm(df.iterrows(), total=len(df), desc="DocVQA Tree"):
                try:
                    img = load_image_from_row(row["image"])
                    question = str(row["question"]) + "\nAnswer the question using a single word or phrase."
                    pred_text, num_sel, sel_ratio = _tree_inference(
                        model, processor, img, question, dual=dual)
                    answers = list(row["answers"])
                    vqa_acc = _docvqa_accuracy(pred_text, answers)
                    select_ratios.append(sel_ratio)
                except Exception as e:
                    pred_text, vqa_acc = "", 0.0
                    num_sel, sel_ratio = 0, 0.0
                    print("[ERROR] qid={}: {}".format(row.get("questionId", "?"), e))
                fout.write(json.dumps({
                    "question_id": str(row["questionId"]),
                    "prediction_text": pred_text,
                    "answers": list(row["answers"]),
                    "vqa_acc": vqa_acc,
                    "num_selected": num_sel,
                    "num_total": BASELINE_TOKENS,
                    "select_ratio": sel_ratio,
                }) + "\n")
        print_select_ratio_stats(select_ratios)
        print("Saved: " + out_tree)
        tree_results = load_results(out_tree)
    return baseline_results, tree_results


# ---------------------------------------------------------------------------
# MME-RealWorld  (NEW)
# ---------------------------------------------------------------------------

# Map category column prefix -> which dataset split it belongs to
_MME_PERCEPTION_PREFIX = "Perception"
_MME_REASONING_PREFIX = "Reasoning"

# Short display names for the subcategories (strip the Perception/Reasoning prefix)
_MME_CAT_DISPLAY = {
    "Perception/OCR with Complex Context": "OCR",
    "Perception/Diagram and Table": "Diagram and Table",
    "Perception/Remote Sensing": "Remote Sensing",
    "Perception/Autonomous_Driving": "Autonomous_Driving",
    "Perception/Monitoring": "Monitoring",
    "Reasoning/Autonomous_Driving": "Autonomous_Driving",
    "Reasoning/Diagram and Table": "Diagram and Table",
    "Reasoning/OCR with Complex Context": "OCR",
    "Reasoning/Monitoring": "Monitoring",
}


def _mme_build_question(row) -> str:
    """Combine question + multi-choice options into a single MCQ prompt."""
    q = str(row["question"]).strip()
    opts_raw = str(row["multi-choice options"]).strip()
    try:
        opts = ast.literal_eval(opts_raw)
        opts_str = "\n".join(str(o) for o in opts)
    except Exception:
        opts_str = opts_raw
    return "{}\n{}\nAnswer with the option's letter from the given choices directly.".format(
        q, opts_str
    )


def _load_mme_df(top_prefix: str) -> pd.DataFrame:
    tsv_path = os.path.join(MME_DIR, "mme_realworld_lite.tsv")
    df = pd.read_csv(tsv_path, sep="\t")
    df = df[df["category"].str.startswith(top_prefix)].reset_index(drop=True)

    sampled_frames = []
    for cat, grp in df.groupby("category"):
        n = min(MME_SAMPLES_PER_CAT, len(grp))
        sampled_frames.append(grp.sample(n=n, random_state=42))
    df = pd.concat(sampled_frames, ignore_index=True)

    return df


def print_mme_results(results, run_name, top_prefix) -> dict:
    """
    Print results in the same table format as the classmate's InternVL table:

    MME-Perception:  columns = subcategories + Total
    MME-Reasoning:   columns = subcategories + Total

    Each subcategory score = correct/total*100  (denominator = MME_SAMPLES_PER_CAT = 100)
    Total = mean of per-subcategory accuracies  (matches the classmate's averaging method)
    """
    # Collect per-subcategory stats
    # key: display name (e.g. "OCR", "Diagram and Table")
    cat_stats = {}
    for r in results:
        full_cat = r.get("category", "")
        display = _MME_CAT_DISPLAY.get(full_cat, full_cat.split("/")[-1])
        pred = str(r["prediction_option"]).strip().upper()
        label = str(r["label"]).strip().upper()
        hit = int(pred == label)
        if display not in cat_stats:
            cat_stats[display] = {"correct": 0, "total": 0}
        cat_stats[display]["correct"] += hit
        cat_stats[display]["total"] += 1

    # Ordered subcategory columns (match classmate's table order)
    if top_prefix == _MME_PERCEPTION_PREFIX:
        col_order = ["Autonomous_Driving", "Diagram and Table", "Monitoring", "OCR", "Remote Sensing"]
        title = "MME-Perception"
    else:
        col_order = ["Autonomous_Driving", "Diagram and Table", "Monitoring", "OCR"]
        title = "MME-Reasoning"

    # Only keep columns that actually appear in results
    cols = [c for c in col_order if c in cat_stats]

    # Per-subcategory accuracy (out of 100)
    col_accs = {}
    for c in cols:
        st = cat_stats[c]
        col_accs[c] = st["correct"] / st["total"] * 100 if st["total"] else 0.0

    # Total = mean of per-subcategory accuracies (same as classmate)
    total_acc = sum(col_accs[c] for c in cols) / len(cols) if cols else 0.0

    # ---- Print table ----
    col_w = 18
    sep = "=" * (12 + col_w * (len(cols) + 1))

    print("\n" + sep)
    print("  {}:  {}".format(title, run_name))
    print(sep)

    # Header row
    header = "  {:<10}".format("Model")
    for c in cols:
        header += " | {:<{w}}".format(c, w=col_w - 3)
    header += " | {:<{w}}".format("Total", w=col_w - 3)
    print(header)
    print("-" * len(header))

    # Data row
    row_str = "  {:<10}".format("LLaVA")
    for c in cols:
        st = cat_stats[c]
        acc_str = "{:.2f}  ({}/{})".format(col_accs[c], st["correct"], st["total"])
        row_str += " | {:<{w}}".format(acc_str, w=col_w - 3)
    row_str += " | {:<{w}}".format("{:.2f}".format(total_acc), w=col_w - 3)
    print(row_str)
    print(sep)

    # Also print a compact per-subcategory breakdown for easy reading
    print("\n  Per-subcategory breakdown:")
    print("  " + "-" * 40)
    for c in cols:
        st = cat_stats[c]
        print("  {:<28}: {:.2f}%  ({}/{})".format(
            c, col_accs[c], st["correct"], st["total"]))
    print("  {:<28}: {:.2f}%".format("Total (mean)", total_acc))
    print("  " + "-" * 40)

    return {
        "overall": total_acc,
        "per_category": {
            c: {"correct": cat_stats[c]["correct"], "total": cat_stats[c]["total"]}
            for c in cols
        },
    }


def print_mme_delta(b, t):
    """Print delta table between baseline and tree for MME."""
    print("\n" + "=" * 52)
    print("  Delta (Tree - Baseline)")
    print("=" * 52)
    all_cats = sorted(set(list(b.get("per_category", {})) + list(t.get("per_category", {}))))
    for cat in all_cats:
        b_acc = (b["per_category"][cat]["correct"] / b["per_category"][cat]["total"] * 100
                 if cat in b.get("per_category", {}) else 0.0)
        t_acc = (t["per_category"][cat]["correct"] / t["per_category"][cat]["total"] * 100
                 if cat in t.get("per_category", {}) else 0.0)
        delta = t_acc - b_acc
        sign = "+" if delta >= 0 else ""
        print("  {:<28}: {}{:.2f}%".format(cat, sign, delta))
    delta_total = t["overall"] - b["overall"]
    sign = "+" if delta_total >= 0 else ""
    print("  {:<28}: {}{:.2f}%".format("Total", sign, delta_total))
    print("=" * 52)


def run_mme(model, processor, top_prefix, mode, out_baseline, out_tree, dual=False):
    """
    Run MME-RealWorld evaluation.

    mode      : "baseline" | "tree" | "both" | "report"
    top_prefix: "Perception" or "Reasoning"
    dual      : False -> compact only (--mode tree)
                True  -> compact + global (--mode tree --dual)
    """
    from llava_with_tree import run_baseline_inference

    df = _load_mme_df(top_prefix)
    label_name = "MME-{}".format(top_prefix)
    print("\n{} samples loaded: {}".format(label_name, len(df)))

    baseline_results = tree_results = None

    # ---- Baseline ----
    if mode in ("baseline", "both"):
        print("\nRunning {} BASELINE on {} samples ...".format(label_name, len(df)))
        with open(out_baseline, "w") as fout:
            for _, row in tqdm(df.iterrows(), total=len(df),
                               desc="{} Baseline".format(label_name)):
                try:
                    image = decode_base64_image(str(row["image"]))
                    question = _mme_build_question(row)
                    pred_text = run_baseline_inference(model, processor, image, question,
                                                       max_new_tokens=MAX_NEW_TOKENS)
                    pred_opt = extract_option_letter_abcde(pred_text)
                except Exception as e:
                    pred_text, pred_opt = "", ""
                    print("[ERROR] index={}: {}".format(row.get("index", "?"), e))
                fout.write(json.dumps({
                    "index": int(row["index"]),
                    "category": str(row["category"]),
                    "l2_category": str(row["l2-category"]),
                    "label": str(row["answer"]).strip().upper(),
                    "prediction_text": pred_text,
                    "prediction_option": pred_opt,
                }) + "\n")
        print("Saved: " + out_baseline)
        baseline_results = load_results(out_baseline)

    # ---- Tree ----
    if mode in ("tree", "both"):
        mode_label = "{} TREE-DUAL".format(label_name) if dual else "{} TREE".format(label_name)
        print("\nRunning {} on {} samples ...".format(mode_label, len(df)))
        select_ratios = []
        with open(out_tree, "w") as fout:
            for _, row in tqdm(df.iterrows(), total=len(df), desc=mode_label):
                try:
                    image = decode_base64_image(str(row["image"]))
                    question = _mme_build_question(row)
                    pred_text, num_sel, sel_ratio = _tree_inference(
                        model, processor, image, question, dual=dual)
                    pred_opt = extract_option_letter_abcde(pred_text)
                    select_ratios.append(sel_ratio)
                except Exception as e:
                    pred_text, pred_opt = "", ""
                    num_sel, sel_ratio = 0, 0.0
                    print("[ERROR] index={}: {}".format(row.get("index", "?"), e))
                fout.write(json.dumps({
                    "index": int(row["index"]),
                    "category": str(row["category"]),
                    "l2_category": str(row["l2-category"]),
                    "label": str(row["answer"]).strip().upper(),
                    "prediction_text": pred_text,
                    "prediction_option": pred_opt,
                    "num_selected": num_sel,
                    "num_total": BASELINE_TOKENS,
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
    parser.add_argument("--mode", choices=["baseline", "tree", "both", "report"],
                        default="both")
    parser.add_argument("--dataset",
                        choices=["vstar", "hrbench", "pope", "textvqa", "docvqa",
                                 "mme-perception", "mme-reasoning"],
                        default="vstar")
    parser.add_argument("--split", default="4k",
                        help="hrbench: 4k/8k  |  pope: adversarial/popular/random")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--dual", action="store_true", default=False,
                        help="Tree mode: use dual-image inference (original + compact). "
                             "Default: compact image only.")
    args = parser.parse_args()

    # ---- Route output paths and print functions ----
    is_mme = args.dataset in ("mme-perception", "mme-reasoning")

    if args.dataset == "vstar":
        out_baseline = os.path.join(VSTAR_DIR, "results_baseline.jsonl")
        out_tree = os.path.join(VSTAR_DIR, "results_tree.jsonl")
        print_fn = print_vstar_results
    elif args.dataset == "hrbench":
        out_baseline = os.path.join(HRBENCH_DIR, "results_{}_baseline.jsonl".format(args.split))
        out_tree = os.path.join(HRBENCH_DIR, "results_{}_tree.jsonl".format(args.split))
        print_fn = print_hrbench_results
    elif args.dataset == "pope":
        out_baseline = os.path.join(POPE_DIR, "results_{}_baseline.jsonl".format(args.split))
        out_tree = os.path.join(POPE_DIR, "results_{}_tree.jsonl".format(args.split))
        print_fn = print_pope_results
    elif args.dataset == "textvqa":
        out_baseline = os.path.join(TEXTVQA_DIR, "results_baseline.jsonl")
        out_tree = os.path.join(TEXTVQA_DIR, "results_tree.jsonl")
        print_fn = print_textvqa_results
    elif args.dataset == "docvqa":
        out_baseline = os.path.join(DOCVQA_DIR, "results_baseline.jsonl")
        out_tree = os.path.join(DOCVQA_DIR, "results_tree.jsonl")
        print_fn = print_textvqa_results  # same VQA accuracy metric
    elif args.dataset == "mme-perception":
        tree_suffix = "tree_dual" if args.dual else "tree"
        out_baseline = os.path.join(MME_DIR, "results_perception_baseline.jsonl")
        out_tree = os.path.join(MME_DIR, "results_perception_{}.jsonl".format(tree_suffix))
        mme_prefix = _MME_PERCEPTION_PREFIX
    else:  # mme-reasoning
        tree_suffix = "tree_dual" if args.dual else "tree"
        out_baseline = os.path.join(MME_DIR, "results_reasoning_baseline.jsonl")
        out_tree = os.path.join(MME_DIR, "results_reasoning_{}.jsonl".format(tree_suffix))
        mme_prefix = _MME_REASONING_PREFIX

    # ---- Report-only mode ----
    if args.mode == "report":
        if is_mme:
            for path, name in [(out_baseline, "Baseline"), (out_tree, "Tree")]:
                if os.path.exists(path):
                    print_mme_results(load_results(path), name, mme_prefix)
                else:
                    print("File not found: " + path)
        else:
            for path, name in [(out_baseline, "Baseline"), (out_tree, "Tree")]:
                if os.path.exists(path):
                    print_fn(load_results(path), name)
                else:
                    print("File not found: " + path)
        return

    # ---- Load model ----
    import torch
    from transformers import LlavaForConditionalGeneration, AutoProcessor

    print("Loading model: " + MODEL_ID)
    if is_mme:
        tree_desc = ("dual-image (compact + global)" if args.dual
                     else "single-image (compact only)")
        print("Tree mode: {}".format(tree_desc))
    else:
        print("Tree mode: {}".format(
            "dual-image (original + compact)" if args.dual else "single-image (compact only)"))

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16,
        device_map="auto", attn_implementation="eager",
    )
    model.eval()
    print("Model ready. Device: {}".format(next(model.parameters()).device))

    # ---- Run evaluation ----
    if args.dataset == "vstar":
        baseline_results, tree_results = run_vstar(
            model, processor, args.mode, args.max_samples, out_baseline, out_tree,
            dual=args.dual)
        b = print_fn(baseline_results, "Baseline") if baseline_results else None
        t = print_fn(tree_results, "Tree") if tree_results else None
        if b and t:
            print_delta(b, t)

    elif args.dataset == "hrbench":
        baseline_results, tree_results = run_hrbench(
            model, processor, args.split, args.mode, args.max_samples, out_baseline, out_tree,
            dual=args.dual)
        b = print_fn(baseline_results, "Baseline") if baseline_results else None
        t = print_fn(tree_results, "Tree") if tree_results else None
        if b and t:
            print_delta(b, t)

    elif args.dataset == "pope":
        baseline_results, tree_results = run_pope(
            model, processor, args.split, args.mode, args.max_samples, out_baseline, out_tree,
            dual=args.dual)
        b = print_fn(baseline_results, "Baseline") if baseline_results else None
        t = print_fn(tree_results, "Tree") if tree_results else None
        if b and t:
            print_delta(b, t)

    elif args.dataset == "textvqa":
        baseline_results, tree_results = run_textvqa(
            model, processor, args.mode, args.max_samples, out_baseline, out_tree,
            dual=args.dual)
        b = print_fn(baseline_results, "Baseline") if baseline_results else None
        t = print_fn(tree_results, "Tree") if tree_results else None
        if b and t:
            print_delta(b, t)

    elif args.dataset == "docvqa":
        baseline_results, tree_results = run_docvqa(
            model, processor, args.mode, args.max_samples, out_baseline, out_tree,
            dual=args.dual)
        b = print_fn(baseline_results, "Baseline") if baseline_results else None
        t = print_fn(tree_results, "Tree") if tree_results else None
        if b and t:
            print_delta(b, t)

    else:
        # MME-Perception or MME-Reasoning
        baseline_results, tree_results = run_mme(
            model, processor, mme_prefix, args.mode, out_baseline, out_tree,
            dual=args.dual)
        b = (print_mme_results(baseline_results, "Baseline", mme_prefix)
             if baseline_results else None)
        t = (print_mme_results(tree_results,
                               "Tree-Dual (compact+global)" if args.dual else "Tree (compact only)",
                               mme_prefix)
             if tree_results else None)
        if b and t:
            print_mme_delta(b, t)


if __name__ == "__main__":
    main()