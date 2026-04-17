import os
from huggingface_hub import snapshot_download


def download_textvqa_dataset(
        save_dir: str = "./textvqa",
        repo_id: str = "lmms-lab/textvqa",
) -> str:
    if os.path.exists(save_dir) and len(os.listdir(save_dir)) > 0:
        print(f"[INFO] TextVQA dataset already exists at {save_dir}")
        return save_dir

    print(f"[INFO] Downloading TextVQA from: {repo_id}")

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=save_dir,
        resume_download=True,
        max_workers=1,
    )

    print("[INFO] TextVQA download complete.")
    return save_dir


def download_docvqa_dataset(
        save_dir: str = "./docvqa",
        repo_id: str = "lmms-lab/DocVQA",
) -> str:
    if os.path.exists(save_dir) and len(os.listdir(save_dir)) > 0:
        print(f"[INFO] DocVQA dataset already exists at {save_dir}")
        return save_dir

    print(f"[INFO] Downloading DocVQA from: {repo_id}")

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=save_dir,
        resume_download=True,
        max_workers=1,
    )

    print("[INFO] DocVQA download complete.")
    return save_dir


def download_vstar_dataset(
        save_dir: str = "./vstar_bench",
        repo_id: str = "craigwu/vstar_bench",
) -> str:
    """
    Download VStar Bench from Hugging Face.
    """
    if os.path.exists(save_dir) and len(os.listdir(save_dir)) > 0:
        print(f"[INFO] VStar dataset already exists at {save_dir}")
        return save_dir

    print(f"[INFO] Downloading VStar dataset from: {repo_id}")

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=save_dir,
        resume_download=True,
        max_workers=1,
    )

    print("[INFO] VStar download complete.")
    return save_dir


def download_pope_dataset(
        save_dir: str = "./pope",
        repo_id: str = "lmms-lab/POPE",
) -> str:
    if os.path.exists(save_dir) and len(os.listdir(save_dir)) > 0:
        print(f"[INFO] POPE dataset already exists at {save_dir}")
        return save_dir

    print(f"[INFO] Downloading POPE from: {repo_id}")

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=save_dir,
        resume_download=True,
        max_workers=1,
    )

    print("[INFO] POPE download complete.")
    return save_dir


def download_hrbench_dataset(
        save_dir: str = "./hr_bench",
        repo_id: str = "DreamMr/HR-Bench",
) -> str:
    if os.path.exists(save_dir) and len(os.listdir(save_dir)) > 0:
        print(f"[INFO] HR-Bench already exists at {save_dir}")
        return save_dir

    print(f"[INFO] Downloading HR-Bench from: {repo_id}")

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=save_dir,
        resume_download=True,
        max_workers=1,
    )

    print("[INFO] HR-Bench download complete.")
    return save_dir


def download_aokvqa_dataset(
        save_dir: str = "./aokvqa",
        repo_id: str = "HuggingFaceM4/A-OKVQA",
) -> str:
    if os.path.exists(save_dir) and len(os.listdir(save_dir)) > 0:
        print(f"[INFO] AOKVQA dataset already exists at {save_dir}")
        return save_dir

    print(f"[INFO] Downloading AOKVQA from: {repo_id}")

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=save_dir,
        resume_download=True,
        max_workers=1,
    )

    print("[INFO] AOKVQA download complete.")
    return save_dir


def download_gqa_dataset(
        save_dir: str = "./gqa",
        repo_id: str = "lmms-lab/GQA",
) -> str:
    if os.path.exists(save_dir) and len(os.listdir(save_dir)) > 0:
        print(f"[INFO] GQA dataset already exists at {save_dir}")
        return save_dir

    print(f"[INFO] Downloading GQA from: {repo_id}")

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=save_dir,
        resume_download=True,
        max_workers=1,
    )

    print("[INFO] GQA download complete.")
    return save_dir


def download_vqav2_dataset(
        save_dir: str = "./vqav2",
        repo_id: str = "lmms-lab/VQAv2",
) -> str:
    if os.path.exists(save_dir) and len(os.listdir(save_dir)) > 0:
        print(f"[INFO] VQAv2 dataset already exists at {save_dir}")
        return save_dir

    print(f"[INFO] Downloading VQAv2 from: {repo_id}")

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=save_dir,
        resume_download=True,
        max_workers=1,
    )

    print("[INFO] VQAv2 download complete.")
    return save_dir


if __name__ == '__main__':
    download_aokvqa_dataset()
    download_gqa_dataset()
    download_vqav2_dataset()
