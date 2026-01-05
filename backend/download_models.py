"""
Download AI models for segmentation
Run this once: python download_models.py
"""

import os
import urllib.request
from pathlib import Path

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

def download_with_progress(url, filepath):
    """Download file with progress bar."""
    def reporthook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        print(f"\rDownloading: {percent}%", end='')
    
    urllib.request.urlretrieve(url, filepath, reporthook)
    print()  # New line after download

print("=" * 60)
print("AI Model Downloader for Rug Visualizer")
print("=" * 60)

# SAM2 Model (Best quality, 1.2GB)
sam_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
sam_path = MODELS_DIR / "sam_vit_h.pth"

if sam_path.exists():
    print(f"✓ SAM model already exists at {sam_path}")
else:
    print(f"\nDownloading SAM model (1.2GB)...")
    print(f"From: {sam_url}")
    print(f"To: {sam_path}")
    try:
        download_with_progress(sam_url, sam_path)
        print(f"✓ SAM model downloaded successfully!")
    except Exception as e:
        print(f"✗ Failed to download SAM model: {e}")
        print("You can download manually from:")
        print("https://github.com/facebookresearch/segment-anything#model-checkpoints")

print("\n" + "=" * 60)
print("Setup complete!")
print("=" * 60)
print("\nNOTE: AI features will work without models using heuristic fallbacks.")
print("For best results, ensure models are downloaded.")