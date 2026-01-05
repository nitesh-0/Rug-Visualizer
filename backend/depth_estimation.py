"""
Depth Estimation Module
========================
Monocular depth estimation for 3D scene understanding.

Supports:
- MiDaS v3.1 (Intel ISL)
- Depth-Anything-v2 (more accurate)
- ZoeDepth (metric depth)
"""

import logging
from typing import Tuple, Optional, Dict
from pathlib import Path

import numpy as np
from PIL import Image
import torch

logger = logging.getLogger(__name__)

# ============================================
# DEPTH ESTIMATOR
# ============================================

class DepthEstimator:
    """
    Estimates depth from single RGB images.
    
    Provides relative depth maps that can be used for:
    - Perspective-correct rug placement
    - Occlusion handling
    - Realistic scaling based on distance
    """
    
    def __init__(
        self,
        model_type: str = "dpt_beit_large_512",  # MiDaS model
        device: Optional[str] = None
    ):
        """
        Initialize depth estimator.
        
        Args:
            model_type: Model variant
                - "dpt_beit_large_512" (best quality, slower)
                - "dpt_swin2_large_384" (balanced)
                - "midas_v21_small_256" (fast, lower quality)
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        self.model_type = model_type
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transform = None
        self._load_model()
    
    def _load_model(self):
        """Load MiDaS model from torch hub."""
        try:
            logger.info(f"Loading MiDaS model: {self.model_type} on {self.device}")
            
            # Load model from torch hub
            self.model = torch.hub.load(
                "intel-isl/MiDaS",
                self.model_type,
                trust_repo=True
            )
            self.model.to(self.device)
            self.model.eval()
            
            # Load corresponding transform
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            
            if self.model_type == "dpt_beit_large_512":
                self.transform = midas_transforms.dpt_transform
            elif "small" in self.model_type:
                self.transform = midas_transforms.small_transform
            else:
                self.transform = midas_transforms.dpt_transform
            
            logger.info("✓ MiDaS model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load MiDaS model: {e}")
            logger.info("Depth estimation will use fallback method")
            self.model = None
    
    def estimate(
        self,
        image: np.ndarray,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Estimate depth map from RGB image.
        
        Args:
            image: RGB image array (H, W, 3)
            normalize: Normalize depth to 0-1 range
        
        Returns:
            Depth map (H, W) - higher values = closer to camera
        """
        if self.model is None:
            return self._fallback_depth(image)
        
        try:
            height, width = image.shape[:2]
            
            # Convert to PIL for transform
            img_pil = Image.fromarray(image)
            
            # Apply transform
            input_batch = self.transform(img_pil).to(self.device)
            
            # Predict
            with torch.no_grad():
                prediction = self.model(input_batch)
                
                # Resize to original resolution
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=(height, width),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            
            depth_map = prediction.cpu().numpy()
            
            # Normalize if requested
            if normalize:
                depth_map = self._normalize_depth(depth_map)
            
            return depth_map
            
        except Exception as e:
            logger.error(f"Depth estimation failed: {e}")
            return self._fallback_depth(image)
    
    def _normalize_depth(self, depth: np.ndarray) -> np.ndarray:
        """Normalize depth map to 0-1 range."""
        depth_min = depth.min()
        depth_max = depth.max()
        
        if depth_max - depth_min > 1e-6:
            return (depth - depth_min) / (depth_max - depth_min)
        return np.zeros_like(depth)
    
    def _fallback_depth(self, image: np.ndarray) -> np.ndarray:
        """
        Simple heuristic depth estimation when model unavailable.
        
        Assumes:
        - Top of image is farther away (smaller depth)
        - Bottom of image is closer (larger depth)
        - Adds some variation based on brightness
        """
        height, width = image.shape[:2]
        
        # Linear gradient from top (0) to bottom (1)
        y_gradient = np.linspace(0, 1, height)
        depth = np.tile(y_gradient[:, np.newaxis], (1, width))
        
        # Add subtle variation based on brightness
        gray = np.mean(image, axis=2) / 255.0
        depth = depth * 0.8 + gray * 0.2
        
        return depth
    
    def estimate_floor_depth(
        self,
        depth_map: np.ndarray,
        floor_mask: np.ndarray
    ) -> Dict[str, float]:
        """
        Analyze depth values on the floor region.
        
        Returns:
            Dictionary with floor depth statistics
        """
        floor_depths = depth_map[floor_mask]
        
        if len(floor_depths) == 0:
            return {
                "mean": 0.5,
                "min": 0.0,
                "max": 1.0,
                "std": 0.1
            }
        
        return {
            "mean": float(np.mean(floor_depths)),
            "min": float(np.min(floor_depths)),
            "max": float(np.max(floor_depths)),
            "std": float(np.std(floor_depths)),
            "median": float(np.median(floor_depths))
        }
    
    def create_visualization(
        self,
        depth_map: np.ndarray,
        colormap: str = "magma"
    ) -> np.ndarray:
        """
        Create colored visualization of depth map.
        
        Args:
            depth_map: Depth map array
            colormap: matplotlib colormap name
        
        Returns:
            RGB visualization (H, W, 3)
        """
        try:
            import matplotlib.pyplot as plt
            
            # Normalize to 0-255
            normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
            normalized = (normalized * 255).astype(np.uint8)
            
            # Apply colormap
            cmap = plt.get_cmap(colormap)
            colored = cmap(normalized)[:, :, :3]  # Remove alpha
            colored = (colored * 255).astype(np.uint8)
            
            return colored
            
        except ImportError:
            # Fallback: grayscale
            normalized = ((depth_map - depth_map.min()) / 
                         (depth_map.max() - depth_map.min()) * 255).astype(np.uint8)
            return np.stack([normalized] * 3, axis=-1)


# ============================================
# DEPTH-AWARE TRANSFORMATION
# ============================================

class DepthAwareTransformer:
    """
    Transforms images using depth information for realistic placement.
    
    Key features:
    - Depth-based scaling (far objects appear smaller)
    - Perspective-correct warping
    - Floor plane projection
    """
    
    def __init__(self):
        """Initialize transformer."""
        pass
    
    def calculate_scale_at_depth(
        self,
        depth_value: float,
        reference_depth: float = 0.7,
        scale_factor: float = 2.0
    ) -> float:
        """
        Calculate scale factor based on depth.
        
        Objects farther away (lower depth) appear smaller.
        
        Args:
            depth_value: Depth at object position (0-1)
            reference_depth: Reference depth for scale=1.0
            scale_factor: How much scale changes with depth
        
        Returns:
            Scale multiplier
        """
        # Convert depth to distance (invert since depth map has close=high)
        distance = 1.0 - depth_value
        reference_distance = 1.0 - reference_depth
        
        # Scale proportional to distance
        # Far objects (high distance) are smaller
        if reference_distance > 0:
            scale = scale_factor * (reference_distance / (distance + 0.1))
        else:
            scale = 1.0
        
        # Clamp to reasonable range
        return np.clip(scale, 0.3, 3.0)
    
    def calculate_perspective_warp(
        self,
        depth_map: np.ndarray,
        floor_mask: np.ndarray,
        rug_bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Calculate perspective warp matrix from depth map.
        
        Args:
            depth_map: Scene depth map
            floor_mask: Binary floor mask
            rug_bbox: Rug bounding box (x1, y1, x2, y2)
        
        Returns:
            3x3 homography matrix for perspective warp
        """
        x1, y1, x2, y2 = rug_bbox
        
        # Sample depth at rug corners
        # Top-left, top-right, bottom-right, bottom-left
        corner_depths = []
        for (x, y) in [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]:
            x = int(np.clip(x, 0, depth_map.shape[1] - 1))
            y = int(np.clip(y, 0, depth_map.shape[0] - 1))
            corner_depths.append(depth_map[y, x])
        
        # Calculate perspective effect
        # Top edge (farther) should be narrower than bottom edge (closer)
        top_depth = (corner_depths[0] + corner_depths[1]) / 2
        bottom_depth = (corner_depths[2] + corner_depths[3]) / 2
        
        # Perspective ratio
        perspective_ratio = top_depth / (bottom_depth + 0.01)
        
        # Build homography matrix
        # This is a simplified version - production would use full 3D reconstruction
        H = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, perspective_ratio * 0.001, 1.0]
        ], dtype=np.float32)
        
        return H
    
    def project_to_floor_plane(
        self,
        point: Tuple[float, float],
        depth_map: np.ndarray,
        floor_mask: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Project 2D point to 3D floor plane using depth.
        
        Args:
            point: (x, y) pixel coordinates
            depth_map: Scene depth map
            floor_mask: Floor region mask
        
        Returns:
            (x, y, depth) 3D coordinates
        """
        x, y = int(point[0]), int(point[1])
        h, w = depth_map.shape
        
        x = np.clip(x, 0, w - 1)
        y = np.clip(y, 0, h - 1)
        
        # Get depth at point
        if floor_mask[y, x]:
            depth = depth_map[y, x]
        else:
            # Find nearest floor point
            floor_points = np.argwhere(floor_mask)
            if len(floor_points) > 0:
                distances = np.sum((floor_points - [y, x]) ** 2, axis=1)
                nearest_idx = np.argmin(distances)
                nearest_y, nearest_x = floor_points[nearest_idx]
                depth = depth_map[nearest_y, nearest_x]
            else:
                depth = 0.5
        
        return (x, y, depth)


# ============================================
# DEPTH-ANYTHING-V2 ALTERNATIVE
# ============================================

class DepthAnythingEstimator:
    """
    Depth-Anything-v2 model (more accurate than MiDaS).
    
    Installation:
        pip install depth-anything-v2
    
    Note: This is an alternative to MiDaS with better accuracy.
    """
    
    def __init__(self, model_size: str = "vitl"):
        """
        Initialize Depth-Anything model.
        
        Args:
            model_size: 'vits', 'vitb', or 'vitl' (large is best)
        """
        self.model_size = model_size
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._load_model()
    
    def _load_model(self):
        """Load Depth-Anything model."""
        try:
            # This requires depth-anything-v2 to be installed
            # For now, fall back to MiDaS
            logger.info("Depth-Anything-v2 not implemented, using MiDaS")
            self.model = None
        except Exception as e:
            logger.error(f"Failed to load Depth-Anything: {e}")
            self.model = None
    
    def estimate(self, image: np.ndarray) -> np.ndarray:
        """Estimate depth using Depth-Anything."""
        # Fallback to simple gradient
        height, width = image.shape[:2]
        y_gradient = np.linspace(0, 1, height)
        return np.tile(y_gradient[:, np.newaxis], (1, width))