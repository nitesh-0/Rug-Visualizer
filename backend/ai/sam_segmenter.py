"""
SAM-based Segmentation
NO HEURISTICS - Pure AI
"""
import logging
import numpy as np
import torch
import cv2
from pathlib import Path
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

class SAMSegmenter:
    """Segment Anything Model segmenter"""
    
    def __init__(self, checkpoint_path: str, model_type: str = "vit_h", device: str = None):
        self.checkpoint_path = Path(checkpoint_path)
        self.model_type = model_type
        self.device = device or (
            "cuda"
            if torch.cuda.is_available() and torch.version.cuda is not None
            else "cpu"
        )

        self.predictor = None
        self._load_model()
    
    def _load_model(self):
        """Load SAM model - MUST succeed"""
        try:
            if not self.checkpoint_path.exists():
                raise FileNotFoundError(
                    f"SAM checkpoint not found: {self.checkpoint_path}\n"
                    f"Download from: https://github.com/facebookresearch/segment-anything#model-checkpoints"
                )
            
            logger.info(f"Loading SAM {self.model_type} on {self.device}...")
            
            from segment_anything import sam_model_registry, SamPredictor
            
            sam = sam_model_registry[self.model_type](checkpoint=str(self.checkpoint_path))
            try:
                sam.to(device=self.device)
            except AssertionError:
                logger.warning("CUDA requested but not supported by torch build, falling back to CPU")
                self.device = "cpu"
                sam.to(device="cpu")

            
            self.predictor = SamPredictor(sam)
            
            logger.info("✓ SAM model loaded successfully")
            
        except ImportError:
            raise RuntimeError(
                "segment-anything not installed. Run:\n"
                "pip install git+https://github.com/facebookresearch/segment-anything.git"
            )
        except Exception as e:
            logger.error(f"CRITICAL: Failed to load SAM: {e}")
            raise
    
    def segment_floor(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Segment floor using automatic mask generation
        
        Args:
            image: RGB image
            
        Returns:
            floor_mask: Binary mask
            confidence: Score
        """
        if self.predictor is None:
            raise RuntimeError("SAM not loaded")
        
        try:
            height, width = image.shape[:2]
            
            # Set image for SAM
            self.predictor.set_image(image)
            
            # Sample floor points (bottom 30% of image, grid pattern)
            floor_points = []
            y_start = int(height * 0.6)
            
            for y in range(y_start, height, height // 10):
                for x in range(width // 10, width, width // 10):
                    floor_points.append([x, y])
            
            if not floor_points:
                raise ValueError("No floor sample points generated")
            
            floor_points = np.array(floor_points)
            point_labels = np.ones(len(floor_points))
            
            # Predict with multiple points
            masks, scores, _ = self.predictor.predict(
                point_coords=floor_points,
                point_labels=point_labels,
                multimask_output=False
            )
            
            floor_mask = masks[0]
            confidence = float(scores[0])
            
            # Post-process: keep largest component
            floor_mask = self._largest_component(floor_mask)
            
            logger.info(f"Floor segmented: {np.sum(floor_mask)} pixels, conf={confidence:.3f}")
            
            return floor_mask, confidence
            
        except Exception as e:
            logger.error(f"Floor segmentation failed: {e}")
            raise
    
    def segment_furniture(self, image: np.ndarray, min_area: float = 0.01) -> List[np.ndarray]:
        """
        Segment furniture using automatic mask generation
        
        Args:
            image: RGB image
            min_area: Minimum area ratio (0-1)
            
        Returns:
            List of furniture masks
        """
        if self.predictor is None:
            raise RuntimeError("SAM not loaded")
        
        try:
            from segment_anything import SamAutomaticMaskGenerator
            
            height, width = image.shape[:2]
            min_area_pixels = int(height * width * min_area)
            
            # Create mask generator
            mask_generator = SamAutomaticMaskGenerator(
                model=self.predictor.model,
                points_per_side=32,
                pred_iou_thresh=0.86,
                stability_score_thresh=0.92,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=min_area_pixels,
            )
            
            # Generate masks
            masks = mask_generator.generate(image)
            
            # Filter furniture (exclude floor - bottom region)
            furniture_masks = []
            floor_threshold = int(height * 0.5)
            
            for mask_data in masks:
                mask = mask_data['segmentation']
                
                # Calculate center of mass
                y_coords, x_coords = np.where(mask)
                if len(y_coords) == 0:
                    continue
                
                center_y = np.mean(y_coords)
                area = mask_data['area']
                
                # Filter: not too low (likely floor), reasonable size
                if center_y < floor_threshold and area > min_area_pixels:
                    furniture_masks.append(mask)
            
            logger.info(f"Furniture detected: {len(furniture_masks)} objects")
            
            return furniture_masks
            
        except Exception as e:
            logger.error(f"Furniture segmentation failed: {e}")
            raise
    
    def _largest_component(self, mask: np.ndarray) -> np.ndarray:
        """Keep only largest connected component"""
        mask_uint8 = mask.astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8)
        
        if num_labels <= 1:
            return mask
        
        # Find largest (excluding background)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        return (labels == largest_label).astype(bool)


class FloorDetector:
    """Specialized floor detection"""
    
    def __init__(self, sam_segmenter: SAMSegmenter):
        self.sam = sam_segmenter
    
    def detect(self, image: np.ndarray, depth_map: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[Tuple[float, float]], float]:
        """
        Detect floor with corner extraction
        
        Returns:
            mask, corners (normalized), confidence
        """
        mask, confidence = self.sam.segment_floor(image)
        
        # Extract floor polygon corners
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        corners = []
        if contours:
            largest = max(contours, key=cv2.contourArea)
            epsilon = 0.02 * cv2.arcLength(largest, True)
            approx = cv2.approxPolyDP(largest, epsilon, True)
            
            height, width = image.shape[:2]
            corners = [
                (float(p[0][0]) / width, float(p[0][1]) / height)
                for p in approx[:8]
            ]
        
        return mask, corners, confidence