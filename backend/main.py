"""
Production Rug Visualizer API
"""
import io
import uuid
import base64
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import settings
from ai.depth_estimator import DepthEstimator
from ai.sam_segmenter import SAMSegmenter, FloorDetector
from core.compositor import RugCompositor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="Rug Visualizer API v2", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models (lazy loaded)
_depth_estimator: Optional[DepthEstimator] = None
_sam_segmenter: Optional[SAMSegmenter] = None
_floor_detector: Optional[FloorDetector] = None
_compositor: Optional[RugCompositor] = None

def get_depth_estimator():
    global _depth_estimator
    if _depth_estimator is None:
        logger.info("Initializing depth estimator...")
        _depth_estimator = DepthEstimator(
            model_type=settings.MIDAS_MODEL,
            device=settings.DEVICE
        )
    return _depth_estimator

def get_sam_segmenter():
    global _sam_segmenter
    if _sam_segmenter is None:
        checkpoint = settings.MODELS_DIR / settings.SAM_CHECKPOINT
        logger.info(f"Initializing SAM from {checkpoint}...")
        _sam_segmenter = SAMSegmenter(
            checkpoint_path=str(checkpoint),
            model_type=settings.SAM_MODEL,
            device=settings.DEVICE
        )
    return _sam_segmenter

def get_floor_detector():
    global _floor_detector
    if _floor_detector is None:
        sam = get_sam_segmenter()
        _floor_detector = FloorDetector(sam)
    return _floor_detector

def get_compositor():
    global _compositor
    if _compositor is None:
        _compositor = RugCompositor()
    return _compositor

# Pydantic models
class AnalysisResult(BaseModel):
    success: bool
    floor_mask_base64: Optional[str] = None
    depth_map_base64: Optional[str] = None
    floor_confidence: float = 0.0
    furniture_count: int = 0
    message: str = ""

class CompositeRequest(BaseModel):
    room_image_id: str
    rug_data_url: str
    position_x: float
    position_y: float
    base_scale: float = 1.0
    rotation: float = 0.0
    use_depth: bool = True
    use_furniture_occlusion: bool = True

class CompositeResponse(BaseModel):
    success: bool
    image_base64: str = ""
    message: str = ""

# Utilities
def array_to_base64(arr: np.ndarray) -> str:
    """Convert numpy array to base64 PNG"""
    if arr.dtype == bool:
        arr = (arr * 255).astype(np.uint8)
    elif arr.dtype == np.float32 or arr.dtype == np.float64:
        arr = (arr * 255).astype(np.uint8)
    
    img = Image.fromarray(arr)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

def image_to_base64(img: Image.Image) -> str:
    """Convert PIL Image to base64"""
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

def base64_to_image(b64_str: str) -> Image.Image:
    """Convert base64 to PIL Image"""
    if b64_str.startswith('data:'):
        b64_str = b64_str.split(',')[1]
    img_data = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(img_data))

def save_upload(file: UploadFile) -> tuple[str, Path]:
    """Save uploaded file"""
    file_id = str(uuid.uuid4())
    ext = Path(file.filename).suffix or ".png"
    file_path = settings.UPLOAD_DIR / f"{file_id}{ext}"
    
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    
    return file_id, file_path

def find_image(image_id: str) -> Optional[Path]:
    """Find uploaded image"""
    for ext in [".png", ".jpg", ".jpeg", ".webp"]:
        path = settings.UPLOAD_DIR / f"{image_id}{ext}"
        if path.exists():
            return path
    return None

def get_cache_path(image_id: str) -> Path:
    """Get cache file path"""
    return settings.CACHE_DIR / f"{image_id}_analysis.npz"

def load_cache(image_id: str) -> Optional[dict]:
    """Load cached analysis"""
    cache_file = get_cache_path(image_id)
    if cache_file.exists():
        try:
            data = np.load(cache_file, allow_pickle=True)
            return {
                "floor_mask": data["floor_mask"],
                "depth_map": data["depth_map"],
                "furniture_masks": list(data["furniture_masks"]),
                "floor_confidence": float(data["floor_confidence"])
            }
        except:
            return None
    return None

def save_cache(image_id: str, data: dict):
    """Save analysis to cache"""
    cache_file = get_cache_path(image_id)
    np.savez_compressed(
        cache_file,
        floor_mask=data["floor_mask"],
        depth_map=data["depth_map"],
        furniture_masks=np.array(data["furniture_masks"], dtype=object),
        floor_confidence=data["floor_confidence"]
    )

# API Endpoints
@app.get("/")
async def root():
    return {
        "status": "ready",
        "service": "Rug Visualizer API v2",
        "device": settings.DEVICE,
        "models": {
            "depth": settings.MIDAS_MODEL,
            "segmentation": settings.SAM_MODEL
        }
    }

@app.post("/upload/room")
async def upload_room(file: UploadFile = File(...)):
    """Upload room image"""
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")
    
    file_id, file_path = save_upload(file)
    
    with Image.open(file_path) as img:
        width, height = img.size
    
    logger.info(f"Uploaded: {file_id} ({width}x{height})")
    
    return {
        "image_id": file_id,
        "width": width,
        "height": height,
        "message": "Upload successful"
    }

@app.post("/analyze/complete", response_model=AnalysisResult)
async def analyze_complete(image_id: str = Form(...)):
    """Complete AI analysis"""
    
    # Check cache
    cached = load_cache(image_id)
    if cached:
        logger.info(f"Using cached analysis: {image_id}")
        return AnalysisResult(
            success=True,
            floor_mask_base64=array_to_base64(cached["floor_mask"]),
            depth_map_base64=array_to_base64(cached["depth_map"]),
            floor_confidence=cached["floor_confidence"],
            furniture_count=len(cached["furniture_masks"]),
            message="Analysis complete (cached)"
        )
    
    # Find image
    file_path = find_image(image_id)
    if not file_path:
        raise HTTPException(404, "Image not found")
    
    try:
        # Load image
        with Image.open(file_path) as img:
            # Resize if too large
            max_dim = settings.MAX_IMAGE_SIZE
            if max(img.size) > max_dim:
                ratio = max_dim / max(img.size)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                logger.info(f"Resized to {new_size}")
            
            img_array = np.array(img.convert("RGB"))
        
        logger.info(f"Analyzing {image_id}...")
        
        # 1. Depth estimation
        logger.info("1/3 Depth estimation...")
        depth_est = get_depth_estimator()
        depth_map = depth_est.estimate(img_array)
        logger.info(f"✓ Depth: {depth_map.shape}, range=[{depth_map.min():.3f}, {depth_map.max():.3f}]")
        
        # 2. Floor detection
        logger.info("2/3 Floor detection...")
        floor_det = get_floor_detector()
        floor_mask, floor_corners, floor_conf = floor_det.detect(img_array, depth_map)
        logger.info(f"✓ Floor: {np.sum(floor_mask)} pixels, conf={floor_conf:.3f}")
        
        # 3. Furniture segmentation
        logger.info("3/3 Furniture segmentation...")
        sam = get_sam_segmenter()
        furniture_masks = sam.segment_furniture(img_array)
        logger.info(f"✓ Furniture: {len(furniture_masks)} objects")
        
        # Cache results
        save_cache(image_id, {
            "floor_mask": floor_mask,
            "depth_map": depth_map,
            "furniture_masks": furniture_masks,
            "floor_confidence": floor_conf
        })
        
        return AnalysisResult(
            success=True,
            floor_mask_base64=array_to_base64(floor_mask),
            depth_map_base64=array_to_base64(depth_map),
            floor_confidence=floor_conf,
            furniture_count=len(furniture_masks),
            message="Analysis complete"
        )
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(500, f"Analysis failed: {str(e)}")

@app.post("/composite/realtime", response_model=CompositeResponse)
async def composite_realtime(request: CompositeRequest):
    """Real-time compositing"""
    
    try:
        # Load room
        file_path = find_image(request.room_image_id)
        if not file_path:
            raise HTTPException(404, "Room image not found")
        
        room_img = Image.open(file_path).convert("RGBA")
        
        # Resize if needed
        max_dim = settings.MAX_IMAGE_SIZE
        if max(room_img.size) > max_dim:
            ratio = max_dim / max(room_img.size)
            new_size = (int(room_img.width * ratio), int(room_img.height * ratio))
            room_img = room_img.resize(new_size, Image.Resampling.LANCZOS)
        
        # Load rug
        rug_img = base64_to_image(request.rug_data_url).convert("RGBA")
        
        # Get analysis
        analysis = load_cache(request.room_image_id)
        if not analysis:
            raise HTTPException(400, "Analysis not complete. Call /analyze/complete first")
        
        # Composite
        compositor = get_compositor()
        
        result = compositor.composite(
            room=room_img,
            rug=rug_img,
            depth_map=analysis["depth_map"] if request.use_depth else np.ones_like(analysis["depth_map"]) * 0.5,
            floor_mask=analysis["floor_mask"],
            position=(request.position_x, request.position_y),
            scale=request.base_scale,
            rotation=request.rotation,
            furniture_masks=analysis["furniture_masks"] if request.use_furniture_occlusion else None,
            use_depth_scaling=request.use_depth,
            use_occlusion=request.use_furniture_occlusion
        )
        
        return CompositeResponse(
            success=True,
            image_base64=image_to_base64(result),
            message="Composite successful"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Compositing failed: {e}", exc_info=True)
        raise HTTPException(500, f"Compositing failed: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Preload models on startup"""
    logger.info("=" * 60)
    logger.info("🏠 Rug Visualizer API v2")
    logger.info("=" * 60)
    logger.info(f"Device: {settings.DEVICE}")
    logger.info(f"Upload dir: {settings.UPLOAD_DIR}")
    logger.info(f"Cache dir: {settings.CACHE_DIR}")
    logger.info("=" * 60)
    
    # Preload models
    try:
        logger.info("Preloading models...")
        get_depth_estimator()
        get_sam_segmenter()
        get_compositor()
        logger.info("✓ All models loaded")
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        logger.error("Check that SAM checkpoint is downloaded!")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=False
    )