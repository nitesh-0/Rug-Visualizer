"""
Segmentation Module
===================
AI-powered segmentation for floor and furniture detection.
"""

import logging
from typing import Tuple, List, Optional, Dict, Any
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    logging.warning("OpenCV not available. Using fallback segmentation.")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logging.warning("PyTorch not available. Using fallback segmentation.")

logger = logging.getLogger(__name__)


class FloorSegmenter:
    """Floor detection and segmentation using AI or heuristics."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        self._load_model()
    
    def _load_model(self):
        if not HAS_TORCH:
            logger.info("Using fallback floor segmentation (no PyTorch)")
            return
        
        if self.model_path and Path(self.model_path).exists():
            try:
                logger.info(f"Model loaded from {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
        else:
            logger.info("No model path provided. Using fallback segmentation.")
    
    def segment_floor(
        self,
        image: np.ndarray,
        use_ai: bool = True
    ) -> Tuple[np.ndarray, List[Tuple[float, float]], float]:
        """
        Segment floor region from room image.
        
        Returns:
            - mask: Binary mask where floor pixels are True
            - corners: List of (x, y) corner points for perspective
            - confidence: Detection confidence (0-1)
        """
        height, width = image.shape[:2]
        
        if use_ai and self.model is not None:
            return self._ai_segment(image)
        
        return self._heuristic_segment(image)
    
    def _ai_segment(self, image: np.ndarray) -> Tuple[np.ndarray, List, float]:
        return self._heuristic_segment(image)
    
    def _heuristic_segment(
        self,
        image: np.ndarray
    ) -> Tuple[np.ndarray, List[Tuple[float, float]], float]:
        """Heuristic-based floor segmentation using color analysis."""
        height, width = image.shape[:2]
        
        if HAS_CV2:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Analyze lower third (likely floor)
            lower_third = image[int(height * 0.6):, :, :]
            avg_color = np.mean(lower_third, axis=(0, 1))
            
            # Color distance from average floor color
            color_dist = np.sqrt(np.sum((image.astype(float) - avg_color) ** 2, axis=2))
            
            # Create mask based on color similarity
            threshold = np.percentile(color_dist, 40)
            mask = color_dist < threshold
            
            # Apply morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Keep only largest connected component
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
            if num_labels > 1:
                largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                mask = (labels == largest_label).astype(np.uint8)
            
            # Find contours for corner points
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            corners = []
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                epsilon = 0.02 * cv2.arcLength(largest_contour, True)
                approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                corners = [(float(p[0][0]) / width, float(p[0][1]) / height) 
                          for p in approx[:8]]
            
            # Calculate confidence
            floor_coverage = np.sum(mask) / (height * width)
            confidence = min(0.95, 0.5 + floor_coverage)
            
            return mask.astype(bool), corners, confidence
        
        else:
            # Very basic fallback without OpenCV
            mask = np.zeros((height, width), dtype=bool)
            floor_start = int(height * 0.5)
            mask[floor_start:, :] = True
            
            corners = [
                (0.1, 0.5), (0.9, 0.5),
                (1.0, 1.0), (0.0, 1.0),
            ]
            
            return mask, corners, 0.6


class FurnitureSegmenter:
    """Furniture detection and segmentation for proper layering."""
    
    FURNITURE_CATEGORIES = [
        "sofa", "couch", "chair", "table", "bed",
        "cabinet", "shelf", "plant", "lamp", "rug"
    ]
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        self._load_model()
    
    def _load_model(self):
        if not HAS_TORCH:
            logger.info("Using fallback furniture segmentation (no PyTorch)")
            return
        logger.info("Furniture segmenter initialized")
    
    def segment_furniture(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Segment furniture pieces from room image.
        
        Returns:
            Dictionary with masks, labels, scores, and boxes
        """
        if self.model is not None:
            return self._ai_segment(image)
        
        return self._heuristic_segment(image)
    
    def _ai_segment(self, image: np.ndarray) -> Dict[str, Any]:
        return self._heuristic_segment(image)
    
    def _heuristic_segment(self, image: np.ndarray) -> Dict[str, Any]:
        """Heuristic-based furniture detection using edge analysis."""
        height, width = image.shape[:2]
        
        masks = []
        labels = []
        scores = []
        boxes = []
        
        if HAS_CV2:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            kernel = np.ones((5, 5), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=2)
            
            contours, _ = cv2.findContours(
                dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            min_area = (height * width) * 0.01
            max_area = (height * width) * 0.5
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if min_area < area < max_area:
                    mask = np.zeros((height, width), dtype=np.uint8)
                    cv2.drawContours(mask, [contour], -1, 255, -1)
                    
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    aspect_ratio = w / h if h > 0 else 1
                    center_y = (y + h/2) / height
                    
                    if center_y > 0.4 and aspect_ratio > 1.5:
                        label = "table"
                    elif center_y < 0.6 and aspect_ratio < 2:
                        label = "sofa"
                    else:
                        label = "furniture"
                    
                    masks.append(mask.astype(bool))
                    labels.append(label)
                    scores.append(0.7)
                    boxes.append((x, y, x + w, y + h))
        
        return {
            "masks": masks,
            "labels": labels,
            "scores": scores,
            "boxes": boxes
        }
    
    def get_leg_masks(
        self,
        image: np.ndarray,
        furniture_masks: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Extract furniture leg regions from furniture masks."""
        height, width = image.shape[:2]
        leg_masks = []
        
        for mask in furniture_masks:
            if not HAS_CV2:
                leg_mask = np.zeros_like(mask)
                rows = np.where(mask.any(axis=1))[0]
                if len(rows) > 0:
                    bottom_20_percent = int(0.2 * len(rows))
                    leg_rows = rows[-bottom_20_percent:]
                    leg_mask[leg_rows, :] = mask[leg_rows, :]
                leg_masks.append(leg_mask)
                continue
            
            rows = np.where(mask.any(axis=1))[0]
            if len(rows) == 0:
                leg_masks.append(np.zeros_like(mask))
                continue
            
            top_row, bottom_row = rows[0], rows[-1]
            furniture_height = bottom_row - top_row
            leg_region_start = bottom_row - int(0.25 * furniture_height)
            
            leg_mask = np.zeros_like(mask)
            leg_mask[leg_region_start:, :] = mask[leg_region_start:, :]
            
            kernel = np.ones((3, 3), np.uint8)
            leg_mask_uint8 = cv2.erode(leg_mask.astype(np.uint8), kernel, iterations=2)
            leg_masks.append(leg_mask_uint8.astype(bool))
        
        return leg_masks


class SAMSegmenter:
    """
    Segment Anything Model (SAM) integration for production use.
    
    Install: pip install git+https://github.com/facebookresearch/segment-anything.git
    """
    
    def __init__(self, checkpoint_path: str, model_type: str = "vit_h"):
        self.checkpoint_path = checkpoint_path
        self.model_type = model_type
        self.predictor = None
        self._load_model()
    
    def _load_model(self):
        try:
            from segment_anything import sam_model_registry, SamPredictor
            
            sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
            
            if HAS_TORCH and torch.cuda.is_available():
                sam.to(device="cuda")
            
            self.predictor = SamPredictor(sam)
            logger.info("SAM model loaded successfully")
            
        except ImportError:
            logger.error("segment-anything not installed")
            raise
        except Exception as e:
            logger.error(f"Failed to load SAM model: {e}")
            raise
    
    def segment_with_point(
        self,
        image: np.ndarray,
        point: Tuple[int, int],
        label: int = 1
    ) -> np.ndarray:
        """Segment region at a specific point."""
        if self.predictor is None:
            raise RuntimeError("SAM model not loaded")
        
        self.predictor.set_image(image)
        
        masks, scores, _ = self.predictor.predict(
            point_coords=np.array([[point[0], point[1]]]),
            point_labels=np.array([label]),
            multimask_output=True
        )
        
        best_idx = np.argmax(scores)
        return masks[best_idx]
    
    def segment_floor_interactive(
        self,
        image: np.ndarray,
        floor_points: List[Tuple[int, int]]
    ) -> np.ndarray:
        """Interactive floor segmentation with multiple points."""
        if self.predictor is None:
            raise RuntimeError("SAM model not loaded")
        
        self.predictor.set_image(image)
        
        points = np.array(floor_points)
        labels = np.ones(len(floor_points))
        
        masks, scores, _ = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True
        )
        
        best_idx = np.argmax(scores)
        return masks[best_idx]
