import os
from huggingface_hub import snapshot_download


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
    )

    print("[INFO] VStar download complete.")
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
    )

    print("[INFO] HR-Bench download complete.")
    return save_dir

if __name__ == '__main__':
    download_vstar_dataset()
    download_hrbench_dataset()