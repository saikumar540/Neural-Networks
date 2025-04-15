import os
import zipfile
import subprocess

# Set Kaggle dataset and target directory
dataset = "paultimothymooney/chest-xray-pneumonia"
download_dir = "downloads"
extract_dir = "dataset"

# Ensure Kaggle API key is properly configured
if not os.path.exists(os.path.expanduser("kaggle.json")):
    raise FileNotFoundError("Kaggle API key not found. Place kaggle.json in ~/.kaggle/")

# Create download folder
os.makedirs(download_dir, exist_ok=True)

# Download dataset using kaggle CLI
print("📦 Downloading dataset...")
subprocess.run([
    "kaggle", "datasets", "download",
    "-d", dataset,
    "-p", download_dir,
    "--unzip"
], check=True)

# Rename extracted folder if needed
if os.path.exists(f"{download_dir}/chest_xray"):
    os.rename(f"{download_dir}/chest_xray", extract_dir)

print("✅ Dataset ready at:", extract_dir)
