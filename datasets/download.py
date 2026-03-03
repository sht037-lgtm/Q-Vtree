import os
from huggingface_hub import snapshot_download
import shutil
import zipfile


"""
ATTENTION: Please log into your huggingface first.
"""

def download_hf_dataset(
    repo_id: str,
    save_dir: str,
) -> str:
    if os.path.exists(save_dir) and len(os.listdir(save_dir)) > 0:
        print(f"[INFO] Dataset already exists at {save_dir}")
        return save_dir

    print(f"[INFO] Downloading dataset: {repo_id}")

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=save_dir,
        resume_download=True,
    )

    print("[INFO] Download complete.")
    return save_dir

def download_zoom_eye_data(
    raw_dir: str = "./zoom_eye_data_raw",
    extract_to: str = ".",
    cleanup: bool = True,
):
    """
    Download and extract ZoomEye dataset.
    """

    repo_id = "omlab/zoom_eye_data"

    download_hf_dataset(
        repo_id=repo_id,
        save_dir=raw_dir,
    )

    zip_path = os.path.join(raw_dir, "zoom_eye_data.zip")

    if not os.path.exists(zip_path):
        print("[INFO] zip file not found. Possibly already extracted.")
        return

    print("[INFO] Extracting ZoomEye dataset...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)

    print("[INFO] Extraction complete.")

    if cleanup:
        print("[INFO] Cleaning up raw folder...")
        shutil.rmtree(raw_dir, ignore_errors=True)
        print("[INFO] Cleanup complete.")

if __name__ == '__main__':
    download_zoom_eye_data()