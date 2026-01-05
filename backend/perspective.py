"""
Perspective Transformation Module
=================================
Handles perspective warping of rug images to match room floor perspective.

Key concepts:
1. Homography - Maps points from one plane to another
2. Vanishing points - Where parallel lines converge in perspective
3. Affine transformations - Rotation, scaling, shearing
"""

import math
import logging
from typing import Tuple, List, Optional

import numpy as np
from PIL import Image

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

logger = logging.getLogger(__name__)

# ============================================
# PERSPECTIVE TRANSFORMER
# ============================================

class PerspectiveTransformer:
    """
    Transforms rug images to match room perspective.
    
    Provides multiple transformation methods:
    1. Simple affine (rotation, scale, shear)
    2. Full perspective (homography)
    3. Floor-guided perspective (uses detected floor corners)
    """
    
    def __init__(self):
        """Initialize transformer."""
        logger.info("Perspective transformer initialized")
    
    def transform(
        self,
        image: Image.Image,
        scale: float = 1.0,
        rotation: float = 0.0,
        skew_x: float = 0.0,
        skew_y: float = 0.0,
        perspective_strength: float = 0.0
    ) -> Image.Image:
        """
        Apply transformations to rug image.
        
        Args:
            image: Input PIL Image (RGBA)
            scale: Scale factor (1.0 = original size)
            rotation: Rotation angle in degrees
            skew_x: Horizontal skew (-1 to 1)
            skew_y: Vertical skew (-1 to 1)
            perspective_strength: Perspective effect strength (0 to 1)
        
        Returns:
            Transformed PIL Image
        """
        # Convert to numpy for processing
        img_array = np.array(image)
        height, width = img_array.shape[:2]
        
        # Calculate new dimensions
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        if HAS_CV2:
            # Use OpenCV for high-quality transforms
            
            # Center point for rotation
            center = (width // 2, height // 2)
            
            # Build transformation matrix
            # Start with rotation
            M = cv2.getRotationMatrix2D(center, rotation, scale)
            
            # Apply skew (shear)
            if skew_x != 0 or skew_y != 0:
                skew_matrix = np.array([
                    [1, skew_x, 0],
                    [skew_y, 1, 0]
                ], dtype=np.float32)
                M = np.vstack([M, [0, 0, 1]])
                skew_full = np.array([
                    [1, skew_x, 0],
                    [skew_y, 1, 0],
                    [0, 0, 1]
                ], dtype=np.float32)
                M = skew_full @ M
                M = M[:2, :]
            
            # Calculate output size to fit transformed image
            cos_angle = abs(math.cos(math.radians(rotation)))
            sin_angle = abs(math.sin(math.radians(rotation)))
            out_width = int(new_width * cos_angle + new_height * sin_angle)
            out_height = int(new_width * sin_angle + new_height * cos_angle)
            
            # Adjust translation in the matrix
            M[0, 2] += (out_width - width) / 2
            M[1, 2] += (out_height - height) / 2
            
            # Apply transformation
            transformed = cv2.warpAffine(
                img_array,
                M,
                (out_width, out_height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0)  # Transparent border
            )
            
            # Apply perspective if requested
            if perspective_strength > 0:
                transformed = self._apply_perspective(
                    transformed,
                    perspective_strength
                )
            
            return Image.fromarray(transformed)
        
        else:
            # PIL-only fallback
            result = image.copy()
            
            # Resize
            if scale != 1.0:
                result = result.resize(
                    (new_width, new_height),
                    Image.Resampling.LANCZOS
                )
            
            # Rotate
            if rotation != 0:
                result = result.rotate(
                    -rotation,  # PIL rotates counter-clockwise
                    expand=True,
                    resample=Image.Resampling.BICUBIC
                )
            
            # Skew using PIL's transform
            if skew_x != 0 or skew_y != 0:
                w, h = result.size
                # Affine transform coefficients
                coeffs = (
                    1, skew_x, -skew_x * h / 2,
                    skew_y, 1, -skew_y * w / 2
                )
                result = result.transform(
                    (int(w + abs(skew_x) * h), int(h + abs(skew_y) * w)),
                    Image.Transform.AFFINE,
                    coeffs,
                    resample=Image.Resampling.BICUBIC
                )
            
            return result
    
    def _apply_perspective(
        self,
        image: np.ndarray,
        strength: float
    ) -> np.ndarray:
        """
        Apply perspective transformation to simulate floor view.
        
        Makes the top of the image narrower (further away)
        and the bottom wider (closer).
        
        Args:
            image: Input image array (RGBA)
            strength: Effect strength (0 to 1)
        
        Returns:
            Transformed image array
        """
        height, width = image.shape[:2]
        
        # Define source points (rectangle)
        src_pts = np.float32([
            [0, 0],           # Top-left
            [width, 0],       # Top-right
            [width, height],  # Bottom-right
            [0, height]       # Bottom-left
        ])
        
        # Calculate perspective offset based on strength
        offset = int(width * strength * 0.15)
        
        # Define destination points (trapezoid)
        dst_pts = np.float32([
            [offset, 0],              # Top-left (moved in)
            [width - offset, 0],      # Top-right (moved in)
            [width, height],          # Bottom-right (unchanged)
            [0, height]               # Bottom-left (unchanged)
        ])
        
        # Calculate perspective transformation matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        # Apply transformation
        result = cv2.warpPerspective(
            image,
            M,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )
        
        return result
    
    def transform_to_floor(
        self,
        image: Image.Image,
        floor_corners: List[Tuple[float, float]],
        target_size: Tuple[int, int]
    ) -> Image.Image:
        """
        Transform rug to match detected floor perspective.
        
        Uses the floor corner points to calculate the exact
        perspective transformation needed.
        
        Args:
            image: Input rug image
            floor_corners: List of (x, y) normalized floor corners
            target_size: (width, height) of target room image
        
        Returns:
            Transformed rug image
        """
        if not HAS_CV2:
            logger.warning("OpenCV required for floor perspective matching")
            return image
        
        if len(floor_corners) < 4:
            logger.warning("Need at least 4 floor corners for perspective")
            return image
        
        img_array = np.array(image)
        rug_height, rug_width = img_array.shape[:2]
        target_width, target_height = target_size
        
        # Convert normalized floor corners to pixel coordinates
        floor_pts = np.float32([
            (c[0] * target_width, c[1] * target_height)
            for c in floor_corners[:4]
        ])
        
        # Source points (rug corners)
        src_pts = np.float32([
            [0, 0],
            [rug_width, 0],
            [rug_width, rug_height],
            [0, rug_height]
        ])
        
        # Calculate homography
        H, _ = cv2.findHomography(src_pts, floor_pts)
        
        # Apply transformation
        result = cv2.warpPerspective(
            img_array,
            H,
            (target_width, target_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )
        
        return Image.fromarray(result)
    
    def calculate_floor_homography(
        self,
        floor_corners: List[Tuple[float, float]],
        rug_size: Tuple[int, int],
        room_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Calculate homography matrix for floor perspective.
        
        Args:
            floor_corners: Normalized floor corner coordinates
            rug_size: (width, height) of rug image
            room_size: (width, height) of room image
        
        Returns:
            3x3 homography matrix
        """
        if not HAS_CV2:
            return np.eye(3)
        
        rug_width, rug_height = rug_size
        room_width, room_height = room_size
        
        # Rug corners (rectangle)
        src_pts = np.float32([
            [0, 0],
            [rug_width, 0],
            [rug_width, rug_height],
            [0, rug_height]
        ])
        
        # Floor corners (potentially irregular quadrilateral)
        dst_pts = np.float32([
            (c[0] * room_width, c[1] * room_height)
            for c in floor_corners[:4]
        ])
        
        H, _ = cv2.findHomography(src_pts, dst_pts)
        return H if H is not None else np.eye(3)


# ============================================
# ADVANCED PERSPECTIVE UTILITIES
# ============================================

class VanishingPointEstimator:
    """
    Estimates vanishing points from room images.
    
    Vanishing points help determine the 3D perspective of a scene,
    which is crucial for realistic rug placement.
    """
    
    def __init__(self):
        """Initialize estimator."""
        pass
    
    def estimate(
        self,
        image: np.ndarray
    ) -> List[Tuple[float, float]]:
        """
        Estimate vanishing points from image.
        
        Uses line detection and intersection analysis.
        
        Args:
            image: RGB image array
        
        Returns:
            List of (x, y) vanishing point coordinates
        """
        if not HAS_CV2:
            return []
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        height, width = gray.shape
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Line detection using Hough transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=100,
            minLineLength=50,
            maxLineGap=10
        )
        
        if lines is None:
            return []
        
        # Cluster lines by angle
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = math.atan2(y2 - y1, x2 - x1)
            angle_deg = math.degrees(angle)
            
            # Classify as roughly horizontal or vertical
            if abs(angle_deg) < 30 or abs(angle_deg) > 150:
                horizontal_lines.append((x1, y1, x2, y2))
            elif 60 < abs(angle_deg) < 120:
                vertical_lines.append((x1, y1, x2, y2))
        
        vanishing_points = []
        
        # Find horizontal vanishing point (intersection of horizontal lines)
        if len(horizontal_lines) >= 2:
            vp = self._find_vanishing_point(horizontal_lines)
            if vp:
                vanishing_points.append(vp)
        
        # Find vertical vanishing point
        if len(vertical_lines) >= 2:
            vp = self._find_vanishing_point(vertical_lines)
            if vp:
                vanishing_points.append(vp)
        
        return vanishing_points
    
    def _find_vanishing_point(
        self,
        lines: List[Tuple[int, int, int, int]]
    ) -> Optional[Tuple[float, float]]:
        """
        Find vanishing point from a set of lines.
        
        Uses least squares to find the point that minimizes
        distance to all lines.
        """
        if len(lines) < 2:
            return None
        
        # Convert lines to homogeneous representation
        homogeneous_lines = []
        for x1, y1, x2, y2 in lines:
            # Line through (x1, y1) and (x2, y2) in homogeneous coords
            a = y2 - y1
            b = x1 - x2
            c = x2 * y1 - x1 * y2
            homogeneous_lines.append((a, b, c))
        
        # Find intersection points of all line pairs
        intersections = []
        for i in range(len(homogeneous_lines)):
            for j in range(i + 1, len(homogeneous_lines)):
                a1, b1, c1 = homogeneous_lines[i]
                a2, b2, c2 = homogeneous_lines[j]
                
                det = a1 * b2 - a2 * b1
                if abs(det) > 1e-6:
                    x = (b1 * c2 - b2 * c1) / det
                    y = (a2 * c1 - a1 * c2) / det
                    intersections.append((x, y))
        
        if not intersections:
            return None
        
        # Return median intersection point (robust to outliers)
        xs = [p[0] for p in intersections]
        ys = [p[1] for p in intersections]
        return (np.median(xs), np.median(ys))


class FloorPlaneEstimator:
    """
    Estimates the 3D floor plane from a 2D room image.
    
    Used for accurate perspective matching when placing rugs.
    """
    
    def __init__(self):
        """Initialize estimator."""
        self.vanishing_estimator = VanishingPointEstimator()
    
    def estimate_plane(
        self,
        image: np.ndarray,
        floor_mask: np.ndarray
    ) -> dict:
        """
        Estimate floor plane parameters.
        
        Args:
            image: RGB image
            floor_mask: Binary floor mask
        
        Returns:
            Dictionary with:
            - normal: Estimated plane normal vector
            - vanishing_point: Main vanishing point
            - horizon_line: Y-coordinate of horizon
            - tilt: Estimated camera tilt
        """
        height, width = image.shape[:2]
        
        # Get vanishing points
        vps = self.vanishing_estimator.estimate(image)
        
        # Estimate horizon line from vanishing points
        horizon_y = height * 0.4  # Default
        if vps:
            horizon_y = np.mean([vp[1] for vp in vps])
        
        # Estimate camera tilt from floor mask shape
        floor_rows = np.where(floor_mask.any(axis=1))[0]
        if len(floor_rows) > 0:
            floor_top = floor_rows[0]
            # Tilt is related to where floor starts
            tilt = (floor_top - horizon_y) / height * 30  # degrees estimate
        else:
            tilt = 0
        
        return {
            "normal": (0, -math.sin(math.radians(tilt)), math.cos(math.radians(tilt))),
            "vanishing_point": vps[0] if vps else (width / 2, horizon_y),
            "horizon_line": horizon_y,
            "tilt": tilt
        }
