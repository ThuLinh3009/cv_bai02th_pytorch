"""
Downloads the pizza, steak, sushi image dataset (10% of Food101) into data/pizza_steak_sushi.
Skips the download if the data already exists.

Usage: python get_data.py
"""
import os
import zipfile
from pathlib import Path

import requests

DATA_URL = "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip"

# Setup path to data folder
data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"

# If the image folder doesn't exist, download it and prepare it...
if image_path.is_dir():
    print(f"[INFO] {image_path} directory exists, skipping download.")
else:
    print(f"[INFO] Did not find {image_path} directory, creating one...")
    image_path.mkdir(parents=True, exist_ok=True)

    # Download pizza, steak, sushi data
    zip_path = data_path / "pizza_steak_sushi.zip"
    with open(zip_path, "wb") as f:
        print(f"[INFO] Downloading pizza, steak, sushi data from {DATA_URL}...")
        request = requests.get(DATA_URL)
        request.raise_for_status()
        f.write(request.content)

    # Unzip pizza, steak, sushi data
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        print("[INFO] Unzipping pizza, steak, sushi data...")
        zip_ref.extractall(image_path)

    # Remove zip file
    os.remove(zip_path)

print(f"[INFO] Train images: {len(list((image_path / 'train').glob('*/*.jpg')))} | "
      f"Test images: {len(list((image_path / 'test').glob('*/*.jpg')))}")
