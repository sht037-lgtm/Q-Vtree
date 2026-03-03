import os
import shutil
import zipfile
from huggingface_hub import login, snapshot_download

# ======================
# Config
# ======================
HF_TOKEN = "to be filled"
SAVE_DIR = "./zoom_eye_data_raw"
UNZIP_DIR = "."

if HF_TOKEN is None:
    raise ValueError("Please set HF_TOKEN environment variable.")

# ======================
# Login
# ======================
print("Logging into HuggingFace...")
login(HF_TOKEN)

# ======================
# Download
# ======================
print("Downloading dataset...")
snapshot_download(
    repo_id="omlab/zoom_eye_data",
    repo_type="dataset",
    local_dir=SAVE_DIR,
    local_dir_use_symlinks=False,
)

print("Download complete.")

# ======================
# Unzip
# ======================
zip_path = os.path.join(SAVE_DIR, "zoom_eye_data.zip")

if os.path.exists(zip_path):
    print("Unzipping dataset...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(UNZIP_DIR)
    print("Unzip complete.")
else:
    print("zip file not found. Maybe already extracted?")

# ======================
# Cleanup
# ======================
print("Cleaning up raw download folder...")
shutil.rmtree(SAVE_DIR, ignore_errors=True)
print("Cleanup complete.")