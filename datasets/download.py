import os
import shutil
import zipfile
from huggingface_hub import login, snapshot_download

"""
ATTENTION: Please log into your huggingface first.
"""

# ======================
# Config
# ======================
SAVE_DIR = "./zoom_eye_data_raw"
UNZIP_DIR = "."

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