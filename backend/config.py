"""
Configuration for Rug Visualizer Backend
"""
from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Directories
    BASE_DIR: Path = Path(__file__).parent
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    OUTPUT_DIR: Path = BASE_DIR / "outputs"
    CACHE_DIR: Path = BASE_DIR / "cache"
    MODELS_DIR: Path = BASE_DIR / "models"
    
    # Model Settings
    MIDAS_MODEL: str = "DPT_Large"  # Best quality
    SAM_MODEL: str = "vit_h"
    SAM_CHECKPOINT: str = "sam_vit_h_4b8939.pth"
    
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    CORS_ORIGINS: list = ["*"]
    
    # Processing Settings
    MAX_IMAGE_SIZE: int = 2048  # Max dimension
    CACHE_ENABLED: bool = True
    
    # Device
    DEVICE: str = "cuda"  # Will fallback to CPU if CUDA unavailable
    
    class Config:
        env_file = ".env"

settings = Settings()

# Create directories
for directory in [settings.UPLOAD_DIR, settings.OUTPUT_DIR, 
                  settings.CACHE_DIR, settings.MODELS_DIR]:
    directory.mkdir(exist_ok=True, parents=True)