import os
import io
import re
import json
import base64
import tempfile
import pandas as pd
import torch

from PIL import Image
from tqdm import tqdm


# =========================================================
# Common Utils
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


def pil_to_tempfile(image: Image.Image, suffix=".jpg") -> str:
    """Save PIL image to a temp file and return the path."""
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    image.save(tmp.name)
    tmp.close()
    return tmp.name


def infer_from_pil(model, tokenizer, image: Image.Image, question: str,
                   max_new_tokens: int = 16) -> str:
    """Run InternVL inference from a PIL image."""
    tmp_path = pil_to_tempfile(image)
    try:
        response = model.infer(
            tokenizer=tokenizer,
            image_path=tmp_path,
            question=question,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    finally:
        os.unlink(tmp_path)
    return response


# =========================================================
# V-Star
# =========================================================
def run_vstar_inference_internvl(
    model,
    tokenizer,
    dataset_dir: str = "datasets/vstar_bench",
    anno_file: str = "test_questions.jsonl",
    output_file: str | None = None,
    max_samples: int | None = None,
    max_new_tokens: int = 16,
    model_type: str = "base_internvl",
    run_name: str | None = None,
    warmup: int = 5,
):
    import time
    import torch

    if model_type not in ["base_internvl", "tree_internvl"]:
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

    all_select_ratios = []
    all_num_selected = []
    all_num_total = []
    category_select_ratios = {}
    # unified total time
    total_gpu_times = []
    total_peak_memories = []

    with open(output_path, "w", encoding="utf-8") as fout:
        for i, sample in enumerate(tqdm(samples, desc=f"Running V-Star [{model_type}]", miniters=10)):
            img_path = os.path.join(dataset_dir, sample["image"])
            question = sample["text"]

            sample_num_selected = None
            sample_num_total = None
            sample_select_ratio = None

            try:
                if model_type == "base_internvl":
                    model._debug_baseline_gpu_time = None
                    model._debug_baseline_peak_memory = None
                elif model_type == "tree_internvl":
                    model._debug_tree_gpu_time = None
                    model._debug_tree_peak_memory = None

                pred_text = model.infer(
                    tokenizer=tokenizer,
                    image_path=img_path,
                    question=question,
                    max_new_tokens=max_new_tokens,
                    use_tree=(model_type == "tree_internvl"),
                )

                if model_type == "base_internvl":
                    if i >= warmup:
                        if getattr(model, '_debug_baseline_gpu_time', None) is not None:
                            total_gpu_times.append(model._debug_baseline_gpu_time)
                            total_peak_memories.append(model._debug_baseline_peak_memory)

                elif model_type == "tree_internvl":
                    if i >= warmup:
                        if getattr(model, '_debug_tree_gpu_time', None) is not None:
                            total_gpu_times.append(model._debug_tree_gpu_time)
                            total_peak_memories.append(model._debug_tree_peak_memory)

                    if hasattr(model, '_debug_num_selected_tokens') and model._debug_num_selected_tokens:
                        sample_num_selected = sum(model._debug_num_selected_tokens)
                        sample_num_total = sum(model._debug_num_total_tokens)
                        sample_select_ratio = (
                            sample_num_selected / sample_num_total
                            if sample_num_total > 0 else 0.0
                        )
                        all_num_selected.append(sample_num_selected)
                        all_num_total.append(sample_num_total)
                        all_select_ratios.append(sample_select_ratio)
                        cat = sample["category"]
                        category_select_ratios.setdefault(cat, []).append(sample_select_ratio)

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
                "num_selected_tokens": sample_num_selected,
                "num_total_tokens": sample_num_total,
                "select_ratio": sample_select_ratio,
            }

            fout.write(json.dumps(result, ensure_ascii=False) + "\n")

    if total_gpu_times:
        print(f"[TIME]   mean GPU time    = {sum(total_gpu_times)/len(total_gpu_times):.3f}s  (n={len(total_gpu_times)})")
        print(f"[MEMORY] mean peak memory = {sum(total_peak_memories)/len(total_peak_memories):.2f}GB")

    if all_select_ratios:
        print(f"[STATS] mean selected tokens = {sum(all_num_selected)/len(all_num_selected):.2f}")
        print(f"[STATS] mean total tokens    = {sum(all_num_total)/len(all_num_total):.2f}")
        print(f"[STATS] mean select ratio    = {sum(all_select_ratios)/len(all_select_ratios):.4f}")

        print("[STATS] select ratio by category:")
        for cat, vals in sorted(category_select_ratios.items()):
            print(f"  - {cat}: {sum(vals)/len(vals):.4f} (n={len(vals)})")

    print(f"[INFO] Saved predictions to: {output_path}")
    return output_path


def evaluate_vstar_predictions(pred_file: str) -> float:
    total, correct = 0, 0
    category_stats = {}

    with open(pred_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            pred  = str(item["prediction_option"]).strip().upper()
            label = str(item["label"]).strip().upper()
            cat   = item.get("category", "unknown")

            total += 1
            if pred == label:
                correct += 1

            if cat not in category_stats:
                category_stats[cat] = {"total": 0, "correct": 0}
            category_stats[cat]["total"] += 1
            if pred == label:
                category_stats[cat]["correct"] += 1

    acc = correct / total if total > 0 else 0.0
    print(f"Accuracy: {acc:.4f} ({correct}/{total})")
    for cat, s in sorted(category_stats.items()):
        cat_acc = s["correct"] / s["total"] if s["total"] > 0 else 0.0
        print(f"  - {cat}: {cat_acc:.4f} ({s['correct']}/{s['total']})")
    return acc


# =========================================================
# HR-Bench
# =========================================================
def run_hrbench_inference_internvl(
    model,
    tokenizer,
    split: str = "4k",
    dataset_dir: str = "datasets/hr_bench",
    output_file: str | None = None,
    max_samples: int | None = None,
    max_new_tokens: int = 16,
    model_type: str = "base_internvl",
    run_name: str | None = None,
):
    if split not in ["4k", "8k"]:
        raise ValueError("split must be '4k' or '8k'")
    if model_type not in ["base_internvl", "tree_internvl"]:
        raise ValueError(f"Unsupported model_type: {model_type}")

    tsv_path = os.path.join(dataset_dir, f"hr_bench_{split}.tsv")

    if output_file is None:
        tag = run_name if run_name is not None else model_type
        output_file = f"hr_bench_{split}_predictions_{tag}.jsonl"

    output_path = os.path.join(dataset_dir, output_file)

    df = pd.read_csv(tsv_path, sep="\t")
    if max_samples is not None:
        df = df.iloc[:max_samples]

    all_select_ratios = []
    all_num_selected = []
    all_num_total = []
    category_select_ratios = {}
    cycle_category_select_ratios = {}

    with open(output_path, "w", encoding="utf-8") as fout:
        for _, row in tqdm(df.iterrows(), total=len(df),
                           desc=f"Running HR-Bench {split} [{model_type}]", miniters=10):

            sample_num_selected = None
            sample_num_total = None
            sample_select_ratio = None

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

                tmp_path = pil_to_tempfile(image)
                try:
                    pred_text = model.infer(
                        tokenizer=tokenizer,
                        image_path=tmp_path,
                        question=question_text,
                        max_new_tokens=max_new_tokens,
                        use_tree=(model_type == "tree_internvl"),
                    )
                finally:
                    os.unlink(tmp_path)

                if model_type == "tree_internvl":
                    if hasattr(model, '_debug_num_selected_tokens') and model._debug_num_selected_tokens:
                        sample_num_selected = sum(model._debug_num_selected_tokens)
                        sample_num_total = sum(model._debug_num_total_tokens)
                        sample_select_ratio = (
                            sample_num_selected / sample_num_total
                            if sample_num_total > 0 else 0.0
                        )

                        all_num_selected.append(sample_num_selected)
                        all_num_total.append(sample_num_total)
                        all_select_ratios.append(sample_select_ratio)

                        category_select_ratios.setdefault(
                            row["category"], []
                        ).append(sample_select_ratio)

                        cycle_category_select_ratios.setdefault(
                            row["cycle_category"], []
                        ).append(sample_select_ratio)

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
                "num_selected_tokens": sample_num_selected,
                "num_total_tokens": sample_num_total,
                "select_ratio": sample_select_ratio,
            }

            fout.write(json.dumps(result, ensure_ascii=False) + "\n")

    if all_select_ratios:
        print(f"[STATS] mean selected tokens = {sum(all_num_selected)/len(all_num_selected):.2f}")
        print(f"[STATS] mean total tokens    = {sum(all_num_total)/len(all_num_total):.2f}")
        print(f"[STATS] mean select ratio    = {sum(all_select_ratios)/len(all_select_ratios):.4f}")

        print("[STATS] select ratio by category:")
        for cat, vals in sorted(category_select_ratios.items()):
            print(f"  - {cat}: {sum(vals)/len(vals):.4f} (n={len(vals)})")

        print("[STATS] select ratio by cycle_category:")
        for cyc, vals in sorted(cycle_category_select_ratios.items()):
            print(f"  - {cyc}: {sum(vals)/len(vals):.4f} (n={len(vals)})")

    print(f"[INFO] Saved predictions to: {output_path}")
    return output_path

def evaluate_hrbench_predictions(pred_file: str) -> float:
    total, correct = 0, 0
    with open(pred_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            pred  = str(item["prediction_option"]).strip().upper()
            label = str(item["label"]).strip().upper()
            total += 1
            if pred == label:
                correct += 1
    acc = correct / total if total > 0 else 0.0
    print(f"Accuracy: {acc:.4f} ({correct}/{total})")
    return acc

# =========================================================
# MMBench
# =========================================================
def run_mmbench_inference_internvl(
    model,
    tokenizer,
    dataset_dir: str = "datasets/mmbench",
    output_file: str | None = None,
    max_samples: int | None = None,
    max_new_tokens: int = 16,
    model_type: str = "base_internvl",
    run_name: str | None = None,
    resume: bool = True,
):
    import glob
    import pyarrow.parquet as pq

    if model_type not in ["base_internvl", "tree_internvl"]:
        raise ValueError(f"Unsupported model_type: {model_type}")

    parquet_files = glob.glob(os.path.join(dataset_dir, "**", "*.parquet"), recursive=True)
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {dataset_dir}")

    table = pq.read_table(parquet_files[0])
    df = table.to_pandas()

    if max_samples is not None:
        df = df.iloc[:max_samples]

    if output_file is None:
        tag = run_name if run_name is not None else model_type
        output_file = f"mmbench_predictions_{tag}.jsonl"
    output_path = os.path.join(dataset_dir, output_file)

    # ── 读取已完成的 index ──
    completed_indices = set()
    if resume and os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    completed_indices.add(json.loads(line)["index"])
                except:
                    pass
        print(f"[INFO] Resuming: {len(completed_indices)} samples already done, skipping.")

    IDX2LETTER = {0: "A", 1: "B", 2: "C", 3: "D"}
    all_select_ratios, all_num_selected, all_num_total = [], [], []

    with open(output_path, "a" if completed_indices else "w", encoding="utf-8") as fout:
        for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df),
                                          desc=f"Running MMBench [{model_type}]", miniters=10)):
            if i in completed_indices:
                continue

            sample_num_selected = sample_num_total = sample_select_ratio = None
            try:
                img_bytes = row["image"]["bytes"] if isinstance(row["image"], dict) else row["image"]
                image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

                options_str = f"(A) {row['A']}\n(B) {row['B']}"
                if row["C"] and str(row["C"]) != "None":
                    options_str += f"\n(C) {row['C']}"
                if row["D"] and str(row["D"]) != "None":
                    options_str += f"\n(D) {row['D']}"

                hint_str = f"Hint: {row['hint']}\n" if row["hint"] and str(row["hint"]) != "None" else ""

                question_text = (
                    f"{hint_str}"
                    f"{row['question']}\n"
                    f"{options_str}\n"
                    f"Answer with the option's letter from the given choices directly."
                )

                tmp_path = pil_to_tempfile(image)
                try:
                    pred_text = model.infer(
                        tokenizer=tokenizer,
                        image_path=tmp_path,
                        question=question_text,
                        max_new_tokens=max_new_tokens,
                        use_tree=(model_type == "tree_internvl"),
                    )
                finally:
                    os.unlink(tmp_path)

                if model_type == "tree_internvl":
                    if hasattr(model, '_debug_num_selected_tokens') and model._debug_num_selected_tokens:
                        sample_num_selected = sum(model._debug_num_selected_tokens)
                        sample_num_total = sum(model._debug_num_total_tokens)
                        sample_select_ratio = (
                            sample_num_selected / sample_num_total
                            if sample_num_total > 0 else 0.0
                        )
                        all_num_selected.append(sample_num_selected)
                        all_num_total.append(sample_num_total)
                        all_select_ratios.append(sample_select_ratio)

                pred_option = extract_option_letter(pred_text)

            except Exception as e:
                pred_text = ""
                pred_option = ""
                print(f"[ERROR][{model_type}] index={i}: {e}")

            label_letter = IDX2LETTER.get(int(row["label"]), "A")

            result = {
                "index": i,
                "question": row["question"],
                "A": row["A"],
                "B": row["B"],
                "C": row["C"],
                "D": row["D"],
                "label": label_letter,
                "prediction_text": pred_text,
                "prediction_option": pred_option,
                "model_type": model_type,
                "run_name": run_name if run_name is not None else model_type,
                "num_selected_tokens": sample_num_selected,
                "num_total_tokens": sample_num_total,
                "select_ratio": sample_select_ratio,
            }
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            fout.flush()  # 每条立即落盘，防止再次断掉丢数据

    if all_select_ratios:
        print(f"[STATS] mean selected = {sum(all_num_selected)/len(all_num_selected):.2f}, "
              f"total = {sum(all_num_total)/len(all_num_total):.2f}, "
              f"ratio = {sum(all_select_ratios)/len(all_select_ratios):.4f}")

    print(f"[INFO] Saved predictions to: {output_path}")
    return output_path


def evaluate_mmbench_predictions(pred_file: str) -> dict:
    import pandas as pd

    records = []
    with open(pred_file, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    df = pd.DataFrame(records)

    df["correct"] = df.apply(
        lambda r: str(r["prediction_option"]).strip().upper() == str(r["label"]).strip().upper(),
        axis=1
    )
    total   = len(df)
    correct = df["correct"].sum()
    acc     = correct / total if total > 0 else 0.0
    print(f"\nMMBench Overall Accuracy: {acc:.4f} ({correct}/{total})")

    results = {"overall": acc}

    for cat_col in ["category", "l2-category"]:
        if cat_col not in df.columns:
            continue
        print(f"\n── by {cat_col} ──")
        cat_stats = (
            df.groupby(cat_col)["correct"]
            .agg(total="count", correct="sum")
            .assign(acc=lambda x: x["correct"] / x["total"])
            .sort_values("acc", ascending=False)
        )
        for cat, row in cat_stats.iterrows():
            print(f"  {cat:<40s}  {row['acc']:.4f}  ({int(row['correct'])}/{int(row['total'])})")
        results[cat_col] = cat_stats["acc"].to_dict()

    return results
