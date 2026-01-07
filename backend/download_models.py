"""
Download required AI models
Run once: python download_models.py
"""
import urllib.request
from pathlib import Path
import sys

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

def download_with_progress(url, filepath):
    """Download file with progress"""
    def reporthook(count, block_size, total_size):
        if total_size > 0:
            percent = int(count * block_size * 100 / total_size)
            sys.stdout.write(f"\rDownloading: {percent}%")
            sys.stdout.flush()
    
    urllib.request.urlretrieve(url, filepath, reporthook)
    print()

print("=" * 60)
print("AI Model Downloader - Rug Visualizer v2")
print("=" * 60)

# SAM Model (2.4GB - ViT-H)
sam_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
sam_path = MODELS_DIR / "sam_vit_h_4b8939.pth"

if sam_path.exists():
    print(f"✓ SAM model already exists: {sam_path}")
    print(f"  Size: {sam_path.stat().st_size / 1024 / 1024:.1f} MB")
else:
    print(f"\n📥 Downloading SAM ViT-H model (2.4GB)...")
    print(f"From: {sam_url}")
    print(f"To: {sam_path}")
    print("This will take several minutes...")
    
    try:
        download_with_progress(sam_url, sam_path)
        print(f"✓ SAM model downloaded successfully!")
        print(f"  Size: {sam_path.stat().st_size / 1024 / 1024:.1f} MB")
    except Exception as e:
        print(f"✗ Failed to download SAM model: {e}")
        print("\nManual download:")
        print("1. Go to: https://github.com/facebookresearch/segment-anything#model-checkpoints")
        print("2. Download 'ViT-H SAM model' (2.4GB)")
        print(f"3. Save as: {sam_path}")

print("\n" + "=" * 60)
print("MiDaS Note:")
print("=" * 60)
print("MiDaS models download automatically via torch.hub on first use.")
print("No manual download needed.")

print("\n" + "=" * 60)
print("Setup Complete!")
print("=" * 60)
print("\nYou can now run: python main.py")