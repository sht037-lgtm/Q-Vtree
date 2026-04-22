import os
from huggingface_hub import snapshot_download


def download_vstar_dataset(
    save_dir: str = "./vstar_bench",
    repo_id: str = "craigwu/vstar_bench",
) -> str:
    """
    Download VStar Bench from Hugging Face.
    """

    print(f"[INFO] Downloading VStar dataset from: {repo_id}")

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=save_dir,
        resume_download=True,
        # max_workers=4,
    )

    print("[INFO] VStar download complete.")
    return save_dir

def download_hrbench_dataset(
    save_dir: str = "./hr_bench",
    repo_id: str = "DreamMr/HR-Bench",
) -> str:

    print(f"[INFO] Downloading HR-Bench from: {repo_id}")

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=save_dir,
        resume_download=True,
        # max_workers=4,
    )

    print("[INFO] HR-Bench download complete.")
    return save_dir

def download_mmbench_dataset(
    save_dir: str = "./mmbench",
    repo_id: str = "HuggingFaceM4/MMBench_dev",
) -> str:
    print(f"[INFO] Downloading MMBench (dev) from: {repo_id}")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=save_dir,
        resume_download=True,
    )
    print("[INFO] MMBench download complete.")
    return save_dir

def download_mme_realworld_dataset(
    save_dir: str = "./mme_realworld",
    repo_id: str = "yifanzhang114/MME-RealWorld-Base64",
    use_lite: bool = False,
) -> str:
    if use_lite:
        repo_id = "yifanzhang114/MME-RealWorld-Lite"

    print(f"[INFO] Downloading MME-RealWorld ({'Lite' if use_lite else 'Full'}) from: {repo_id}")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=save_dir,
        ignore_patterns=["*CN*", "*cn*"],
        resume_download=True,
    )
    print("[INFO] MME-RealWorld download complete.")
    return save_dir

if __name__ == '__main__':
    download_mme_realworld_dataset()