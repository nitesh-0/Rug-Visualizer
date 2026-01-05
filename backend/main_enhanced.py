"""
Enhanced Rug Visualizer Backend with Depth Estimation
======================================================
Production-ready API with RoomVo-quality features.
"""

import io
import os
import uuid
import base64
import logging
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Local modules
from segmentation import FloorSegmenter, FurnitureSegmenter
from depth_estimation import DepthEstimator, DepthAwareTransformer
from enhanced_compositing import DepthAwareRugCompositor

# ============================================
# CONFIGURATION
# ============================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
CACHE_DIR = Path("cache")

for directory in [UPLOAD_DIR, OUTPUT_DIR, CACHE_DIR]:
    directory.mkdir(exist_ok=True)

# ============================================
# FASTAPI APP
# ============================================

app = FastAPI(
    title="Enhanced Rug Visualizer API",
    description="Production-ready AI rug visualization with depth awareness",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# MODELS
# ============================================

class AnalysisResult(BaseModel):
    """Complete room analysis result."""
    success: bool
    floor_mask_base64: Optional[str] = None
    depth_map_base64: Optional[str] = None
    floor_confidence: float = 0.0
    furniture_count: int = 0
    furniture_masks: List[str] = []
    message: str = ""

class CompositeRequest(BaseModel):
    """Enhanced composite request with depth."""
    room_image_id: str
    rug_data_url: str  # Data URL of rug image
    position_x: float  # 0-1 normalized
    position_y: float  # 0-1 normalized
    base_scale: float = 1.0
    rotation: float = 0.0
    use_depth: bool = True
    use_furniture_occlusion: bool = True

class CompositeResponse(BaseModel):
    """Composite result."""
    success: bool
    image_base64: str
    debug_info: Dict = {}
    message: str = ""

# ============================================
# GLOBAL INSTANCES
# ============================================

_floor_segmenter: Optional[FloorSegmenter] = None
_furniture_segmenter: Optional[FurnitureSegmenter] = None
_depth_estimator: Optional[DepthEstimator] = None
_compositor: Optional[DepthAwareRugCompositor] = None

def get_floor_segmenter():
    global _floor_segmenter
    if _floor_segmenter is None:
        _floor_segmenter = FloorSegmenter()
    return _floor_segmenter

def get_furniture_segmenter():
    global _furniture_segmenter
    if _furniture_segmenter is None:
        _furniture_segmenter = FurnitureSegmenter()
    return _furniture_segmenter

def get_depth_estimator():
    global _depth_estimator
    if _depth_estimator is None:
        logger.info("Initializing depth estimator (this may take a moment)...")
        _depth_estimator = DepthEstimator(model_type="dpt_beit_large_512")
    return _depth_estimator

def get_compositor():
    global _compositor
    if _compositor is None:
        _compositor = DepthAwareRugCompositor()
    return _compositor

# ============================================
# UTILITY FUNCTIONS
# ============================================

def image_to_base64(img: Image.Image, format: str = "PNG") -> str:
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    img.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode()

def array_to_base64(arr: np.ndarray, normalize: bool = True) -> str:
    """Convert numpy array to base64 image."""
    if normalize:
        arr_min = arr.min()
        arr_max = arr.max()
        if arr_max > arr_min:
            arr = ((arr - arr_min) / (arr_max - arr_min) * 255).astype(np.uint8)
        else:
            arr = np.zeros_like(arr, dtype=np.uint8)
    else:
        arr = arr.astype(np.uint8)
    
    img = Image.fromarray(arr)
    return image_to_base64(img)

def base64_to_image(base64_str: str) -> Image.Image:
    """Convert base64 string to PIL Image."""
    # Handle data URLs
    if base64_str.startswith('data:'):
        base64_str = base64_str.split(',')[1]
    
    img_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(img_data))

def save_upload(file: UploadFile) -> Tuple[str, Path]:
    """Save uploaded file and return ID and path."""
    file_id = str(uuid.uuid4())
    ext = Path(file.filename).suffix or ".png"
    file_path = UPLOAD_DIR / f"{file_id}{ext}"
    
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    
    return file_id, file_path

def find_image_path(image_id: str) -> Optional[Path]:
    """Find image file by ID."""
    for ext in [".png", ".jpg", ".jpeg", ".webp"]:
        candidate = UPLOAD_DIR / f"{image_id}{ext}"
        if candidate.exists():
            return candidate
    return None

# ============================================
# CACHE MANAGEMENT
# ============================================

def get_cached_analysis(image_id: str) -> Optional[Dict]:
    """Get cached analysis results."""
    cache_file = CACHE_DIR / f"{image_id}_analysis.npz"
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

def cache_analysis(image_id: str, analysis: Dict):
    """Cache analysis results."""
    cache_file = CACHE_DIR / f"{image_id}_analysis.npz"
    np.savez_compressed(
        cache_file,
        floor_mask=analysis["floor_mask"],
        depth_map=analysis["depth_map"],
        furniture_masks=analysis["furniture_masks"],
        floor_confidence=analysis["floor_confidence"]
    )

# ============================================
# API ENDPOINTS
# ============================================

@app.get("/")
async def root():
    """Health check."""
    return {
        "status": "healthy",
        "service": "Enhanced Rug Visualizer API v2.0",
        "features": [
            "Depth-aware placement",
            "Furniture occlusion",
            "Perspective-correct warping",
            "Real-time manipulation"
        ]
    }

@app.post("/upload/room")
async def upload_room(file: UploadFile = File(...)):
    """
    Upload room image and trigger analysis.
    
    Returns image_id and dimensions.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    file_id, file_path = save_upload(file)
    
    with Image.open(file_path) as img:
        width, height = img.size
    
    logger.info(f"Room uploaded: {file_id} ({width}x{height})")
    
    return {
        "image_id": file_id,
        "width": width,
        "height": height,
        "message": "Upload successful. Call /analyze/complete for AI analysis."
    }

@app.post("/analyze/complete", response_model=AnalysisResult)
async def analyze_complete(image_id: str = Form(...)):
    """
    Complete AI analysis: floor segmentation, depth estimation, furniture detection.
    
    This is the main analysis endpoint that runs all AI models.
    """
    # Check cache first
    cached = get_cached_analysis(image_id)
    if cached:
        logger.info(f"Using cached analysis for {image_id}")
        
        return AnalysisResult(
            success=True,
            floor_mask_base64=array_to_base64(cached["floor_mask"] * 255, normalize=False),
            depth_map_base64=array_to_base64(cached["depth_map"]),
            floor_confidence=cached["floor_confidence"],
            furniture_count=len(cached["furniture_masks"]),
            furniture_masks=[array_to_base64(m * 255, normalize=False) for m in cached["furniture_masks"]],
            message="Analysis complete (cached)"
        )
    
    # Find image
    file_path = find_image_path(image_id)
    if not file_path:
        raise HTTPException(status_code=404, detail="Image not found")
    
    try:
        # Load image
        with Image.open(file_path) as img:
            img_array = np.array(img.convert("RGB"))
        
        logger.info(f"Starting complete analysis for {image_id}...")
        
        # 1. Floor Segmentation
        logger.info("1/3 Floor segmentation...")
        floor_seg = get_floor_segmenter()
        floor_mask, floor_corners, floor_conf = floor_seg.segment_floor(img_array)
        
        # 2. Depth Estimation
        logger.info("2/3 Depth estimation...")
        depth_est = get_depth_estimator()
        depth_map = depth_est.estimate(img_array, normalize=True)
        
        # 3. Furniture Detection
        logger.info("3/3 Furniture detection...")
        furn_seg = get_furniture_segmenter()
        furn_results = furn_seg.segment_furniture(img_array)
        furniture_masks = furn_results["masks"]
        
        logger.info(f"✓ Analysis complete: floor_conf={floor_conf:.2f}, "
                   f"furniture_count={len(furniture_masks)}")
        
        # Cache results
        cache_analysis(image_id, {
            "floor_mask": floor_mask,
            "depth_map": depth_map,
            "furniture_masks": furniture_masks,
            "floor_confidence": floor_conf
        })
        
        return AnalysisResult(
            success=True,
            floor_mask_base64=array_to_base64(floor_mask * 255, normalize=False),
            depth_map_base64=array_to_base64(depth_map),
            floor_confidence=floor_conf,
            furniture_count=len(furniture_masks),
            furniture_masks=[array_to_base64(m * 255, normalize=False) for m in furniture_masks],
            message="Analysis complete"
        )
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        return AnalysisResult(
            success=False,
            message=f"Analysis failed: {str(e)}"
        )

@app.post("/composite/realtime", response_model=CompositeResponse)
async def composite_realtime(request: CompositeRequest):
    """
    Real-time compositing with depth awareness.
    
    This is the main endpoint for live rug placement visualization.
    """
    try:
        # Load room image
        file_path = find_image_path(request.room_image_id)
        if not file_path:
            raise HTTPException(status_code=404, detail="Room image not found")
        
        room_img = Image.open(file_path).convert("RGBA")
        
        # Load rug from data URL
        rug_img = base64_to_image(request.rug_data_url).convert("RGBA")
        
        # Get analysis (should be cached)
        analysis = get_cached_analysis(request.room_image_id)
        if not analysis:
            # Run analysis if not cached
            img_array = np.array(room_img.convert("RGB"))
            
            floor_seg = get_floor_segmenter()
            floor_mask, _, floor_conf = floor_seg.segment_floor(img_array)
            
            depth_est = get_depth_estimator()
            depth_map = depth_est.estimate(img_array, normalize=True)
            
            furn_seg = get_furniture_segmenter()
            furn_results = furn_seg.segment_furniture(img_array)
            
            analysis = {
                "floor_mask": floor_mask,
                "depth_map": depth_map,
                "furniture_masks": furn_results["masks"],
                "floor_confidence": floor_conf
            }
            
            cache_analysis(request.room_image_id, analysis)
        
        # Composite with depth awareness
        compositor = get_compositor()
        
        furniture_masks = analysis["furniture_masks"] if request.use_furniture_occlusion else None
        
        result = compositor.composite_with_depth(
            room=room_img,
            rug=rug_img,
            depth_map=analysis["depth_map"] if request.use_depth else None,
            position=(request.position_x, request.position_y),
            floor_mask=analysis["floor_mask"],
            furniture_masks=furniture_masks,
            base_scale=request.base_scale,
            rotation=request.rotation
        )
        
        # Convert to base64
        result_base64 = image_to_base64(result, format="PNG")
        
        return CompositeResponse(
            success=True,
            image_base64=result_base64,
            debug_info={
                "depth_used": request.use_depth,
                "furniture_occlusion": request.use_furniture_occlusion,
                "furniture_count": len(furniture_masks) if furniture_masks else 0
            },
            message="Composite successful"
        )
        
    except Exception as e:
        logger.error(f"Compositing failed: {e}", exc_info=True)
        return CompositeResponse(
            success=False,
            image_base64="",
            message=f"Compositing failed: {str(e)}"
        )

@app.post("/depth/visualize")
async def visualize_depth(image_id: str = Form(...)):
    """
    Generate depth map visualization for debugging.
    
    Returns colored depth map image.
    """
    file_path = find_image_path(image_id)
    if not file_path:
        raise HTTPException(status_code=404, detail="Image not found")
    
    try:
        with Image.open(file_path) as img:
            img_array = np.array(img.convert("RGB"))
        
        depth_est = get_depth_estimator()
        depth_map = depth_est.estimate(img_array, normalize=True)
        
        # Create visualization
        depth_vis = depth_est.create_visualization(depth_map, colormap="magma")
        depth_img = Image.fromarray(depth_vis)
        
        return JSONResponse({
            "success": True,
            "depth_map_base64": image_to_base64(depth_img)
        })
        
    except Exception as e:
        logger.error(f"Depth visualization failed: {e}")
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("🏠 Enhanced Rug Visualizer Backend v2.0")
    print("=" * 60)
    print("Features:")
    print("  ✓ Depth-aware placement")
    print("  ✓ Furniture occlusion")
    print("  ✓ Perspective-correct warping")
    print("  ✓ Real-time manipulation")
    print("=" * 60)
    print(f"📁 Upload: {UPLOAD_DIR.absolute()}")
    print(f"📁 Cache: {CACHE_DIR.absolute()}")
    print(f"📁 Output: {OUTPUT_DIR.absolute()}")
    print("=" * 60)
    
    uvicorn.run(
        "main_enhanced:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )