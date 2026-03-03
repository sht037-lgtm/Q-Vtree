import os
from huggingface_hub import snapshot_download


def download_hf_model(
    repo_id: str,
    save_root: str = ".",
    repo_type: str = "model",
    use_symlinks: bool = False,
) -> str:

    model_name = repo_id.split("/")[-1]
    local_dir = os.path.join(save_root, model_name)

    if os.path.exists(local_dir) and len(os.listdir(local_dir)) > 0:
        print(f"[INFO] Model already exists at {local_dir}")
        return local_dir

    print(f"[INFO] Downloading {repo_id} ...")

    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=local_dir,
        local_dir_use_symlinks=use_symlinks,
    )

    print(f"[INFO] Download complete: {local_dir}")
    return local_dir

def download_qwen():

    model_path = download_hf_model(
        repo_id="Qwen/Qwen2.5-VL-3B-Instruct",
        save_root="./models"
    )

    print("Model path:", model_path)

if __name__ == '__main__':
    download_qwen()