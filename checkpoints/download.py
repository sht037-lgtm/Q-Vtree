import os
from huggingface_hub import snapshot_download


def download_hf_model(
    repo_id: str,
    save_root: str = None,
    repo_type: str = "model",
) -> str:

    if save_root is None:
        save_root = os.path.dirname(os.path.abspath(__file__))

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
        resume_download=True,
    )

    print(f"[INFO] Download complete: {local_dir}")
    return local_dir


def download_qwen():
    model_path = download_hf_model(
        repo_id="Qwen/Qwen2.5-VL-3B-Instruct",
    )
    print("Model path:", model_path)
    return model_path


def download_internvl3():
    model_path = download_hf_model(
        repo_id="OpenGVLab/InternVL3-8B",
    )
    print("Model path:", model_path)
    return model_path

def download_instructblip_vicuna7b():
    model_path = snapshot_download(
        repo_id="Salesforce/instructblip-vicuna-7b",
        allow_patterns=[
            "*.json",
            "*.model",
            "*.txt",
            "*.safetensors",
            "*.py",
        ],
    )
    print("Model path:", model_path)
    return model_path

def download_llava_onevision_7b():
    model_path = download_hf_model(
        repo_id="lmms-lab/llava-onevision-qwen2-7b-ov",
    )
    print("Model path:", model_path)
    return model_path

if __name__ == '__main__':
    download_llava_onevision_7b()