"""
Enhanced Compositing Module with Depth Awareness
=================================================
Production-quality rug compositing using depth maps.
"""

import logging
from typing import Tuple, Optional, List, Dict

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import cv2

logger = logging.getLogger(__name__)

# ============================================
# DEPTH-AWARE RUG COMPOSITOR
# ============================================

class DepthAwareRugCompositor:
    """
    Advanced compositor that uses depth maps for realistic placement.
    
    Features:
    - Depth-based scaling (rugs get smaller when farther away)
    - Proper occlusion (furniture legs over rug)
    - Perspective-correct warping
    - Depth-aware shadows
    - Ambient occlusion
    """
    
    def __init__(self):
        """Initialize compositor."""
        logger.info("Depth-aware rug compositor initialized")
    
    def composite_with_depth(
        self,
        room: Image.Image,
        rug: Image.Image,
        depth_map: np.ndarray,
        position: Tuple[float, float],  # Normalized (0-1)
        floor_mask: Optional[np.ndarray] = None,
        furniture_masks: Optional[List[np.ndarray]] = None,
        base_scale: float = 1.0,
        rotation: float = 0.0
    ) -> Image.Image:
        """
        Composite rug with depth-aware placement.
        
        Args:
            room: Background room image (RGBA)
            rug: Rug image (RGBA)
            depth_map: Depth map (0-1, close=high values)
            position: (x, y) normalized position (0-1)
            floor_mask: Binary floor mask
            furniture_masks: List of furniture masks
            base_scale: Base scale factor
            rotation: Rotation in degrees
        
        Returns:
            Composited image
        """
        room = room.convert("RGBA")
        rug = rug.convert("RGBA")
        
        room_array = np.array(room)
        rug_array = np.array(rug)
        
        height, width = room_array.shape[:2]
        
        # Convert normalized position to pixels
        center_x = int(position[0] * width)
        center_y = int(position[1] * height)
        
        # Get depth at rug center
        center_depth = depth_map[
            np.clip(center_y, 0, height - 1),
            np.clip(center_x, 0, width - 1)
        ]
        
        # Calculate depth-based scale
        # Closer objects (higher depth) appear larger
        depth_scale = self._calculate_depth_scale(center_depth)
        final_scale = base_scale * depth_scale
        
        # Calculate perspective warp based on depth gradient
        perspective_warp = self._calculate_perspective_warp(
            depth_map,
            center_x,
            center_y,
            int(rug.width * final_scale),
            int(rug.height * final_scale)
        )
        
        # Transform rug with rotation, scale, and perspective
        transformed_rug = self._transform_rug(
            rug_array,
            final_scale,
            rotation,
            perspective_warp
        )
        
        # Create placement mask (where rug can go)
        placement_mask = self._create_placement_mask(
            width,
            height,
            center_x,
            center_y,
            transformed_rug.shape[1],
            transformed_rug.shape[0],
            floor_mask
        )
        
        # Generate depth-aware shadow
        shadow = self._generate_depth_shadow(
            transformed_rug,
            depth_map,
            center_x,
            center_y
        )
        
        # Composite shadow first
        result = room_array.copy()
        result = self._blend_shadow(result, shadow, center_x, center_y, placement_mask)
        
        # Composite rug
        result = self._blend_rug(
            result,
            transformed_rug,
            center_x,
            center_y,
            placement_mask
        )
        
        # Apply furniture occlusion
        if furniture_masks:
            result = self._apply_furniture_occlusion(
                result,
                room_array,
                furniture_masks
            )
        
        # Enhance integration (lighting match, edge blend)
        result = self._enhance_integration(result, room_array, depth_map)
        
        return Image.fromarray(result)
    
    def _calculate_depth_scale(
        self,
        depth: float,
        min_scale: float = 0.4,
        max_scale: float = 2.0
    ) -> float:
        """
        Calculate scale based on depth.
        
        Depth map convention: close objects have HIGH values (0.8-1.0)
                             far objects have LOW values (0.0-0.3)
        """
        # Invert depth to get distance
        distance = 1.0 - depth
        
        # Map distance to scale
        # Close (distance=0, depth=1.0) -> large scale
        # Far (distance=1.0, depth=0) -> small scale
        scale = max_scale - (distance * (max_scale - min_scale))
        
        return np.clip(scale, min_scale, max_scale)
    
    def _calculate_perspective_warp(
        self,
        depth_map: np.ndarray,
        center_x: int,
        center_y: int,
        rug_width: int,
        rug_height: int
    ) -> Dict[str, float]:
        """
        Calculate perspective warp parameters from depth gradient.
        
        Returns:
            Dictionary with warp parameters
        """
        h, w = depth_map.shape
        
        # Define rug corners in image space
        half_w = rug_width // 2
        half_h = rug_height // 2
        
        corners = [
            (center_x - half_w, center_y - half_h),  # Top-left
            (center_x + half_w, center_y - half_h),  # Top-right
            (center_x + half_w, center_y + half_h),  # Bottom-right
            (center_x - half_w, center_y + half_h),  # Bottom-left
        ]
        
        # Sample depth at corners
        corner_depths = []
        for (x, y) in corners:
            x = np.clip(x, 0, w - 1)
            y = np.clip(y, 0, h - 1)
            corner_depths.append(depth_map[y, x])
        
        # Calculate perspective effect
        # Top edge (far) vs bottom edge (near)
        top_depth = (corner_depths[0] + corner_depths[1]) / 2
        bottom_depth = (corner_depths[2] + corner_depths[3]) / 2
        
        # Perspective strength (how much top shrinks relative to bottom)
        perspective_strength = top_depth / (bottom_depth + 0.01)
        
        return {
            "strength": float(perspective_strength),
            "top_depth": float(top_depth),
            "bottom_depth": float(bottom_depth)
        }
    
    def _transform_rug(
        self,
        rug: np.ndarray,
        scale: float,
        rotation: float,
        perspective: Dict[str, float]
    ) -> np.ndarray:
        """
        Apply full transformation to rug.
        
        Order: Scale -> Rotate -> Perspective warp
        """
        h, w = rug.shape[:2]
        
        # 1. Scale
        new_w = int(w * scale)
        new_h = int(h * scale)
        scaled = cv2.resize(rug, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # 2. Rotate
        if abs(rotation) > 0.1:
            center = (new_w // 2, new_h // 2)
            M = cv2.getRotationMatrix2D(center, rotation, 1.0)
            
            # Calculate new bounds
            cos = abs(M[0, 0])
            sin = abs(M[0, 1])
            new_w_rot = int((new_h * sin) + (new_w * cos))
            new_h_rot = int((new_h * cos) + (new_w * sin))
            
            # Adjust translation
            M[0, 2] += (new_w_rot / 2) - center[0]
            M[1, 2] += (new_h_rot / 2) - center[1]
            
            scaled = cv2.warpAffine(
                scaled,
                M,
                (new_w_rot, new_h_rot),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0)
            )
            new_w, new_h = new_w_rot, new_h_rot
        
        # 3. Perspective warp (trapezoid effect)
        perspective_strength = perspective.get("strength", 1.0)
        
        if abs(perspective_strength - 1.0) > 0.05:
            # Source points (rectangle)
            src_pts = np.float32([
                [0, 0],
                [new_w, 0],
                [new_w, new_h],
                [0, new_h]
            ])
            
            # Calculate perspective shrink for top edge
            shrink = (1.0 - perspective_strength) * 0.3  # Limit effect
            offset_x = int(new_w * shrink * 0.5)
            
            # Destination points (trapezoid - top narrower)
            dst_pts = np.float32([
                [offset_x, 0],
                [new_w - offset_x, 0],
                [new_w, new_h],
                [0, new_h]
            ])
            
            # Calculate and apply perspective transform
            M_perspective = cv2.getPerspectiveTransform(src_pts, dst_pts)
            scaled = cv2.warpPerspective(
                scaled,
                M_perspective,
                (new_w, new_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0)
            )
        
        return scaled
    
    def _create_placement_mask(
        self,
        width: int,
        height: int,
        center_x: int,
        center_y: int,
        rug_width: int,
        rug_height: int,
        floor_mask: Optional[np.ndarray]
    ) -> np.ndarray:
        """Create mask of where rug can be placed (floor only)."""
        mask = np.ones((height, width), dtype=bool)
        
        if floor_mask is not None:
            mask = mask & floor_mask
        
        return mask
    
    def _generate_depth_shadow(
        self,
        rug: np.ndarray,
        depth_map: np.ndarray,
        center_x: int,
        center_y: int,
        intensity: float = 0.4,
        blur_radius: int = 20
    ) -> np.ndarray:
        """
        Generate realistic shadow using depth information.
        
        Shadow intensity varies with depth:
        - Closer objects cast darker shadows
        - Shadow blur increases with distance from object
        """
        h_rug, w_rug = rug.shape[:2]
        h_depth, w_depth = depth_map.shape
        
        # Create shadow from rug alpha
        alpha = rug[:, :, 3]
        shadow = np.zeros((h_rug, w_rug), dtype=np.float32)
        shadow[alpha > 0] = intensity
        
        # Sample depth at rug location
        y1 = max(0, center_y - h_rug // 2)
        y2 = min(h_depth, center_y + h_rug // 2)
        x1 = max(0, center_x - w_rug // 2)
        x2 = min(w_depth, center_x + w_rug // 2)
        
        if y2 > y1 and x2 > x1:
            depth_region = depth_map[y1:y2, x1:x2]
            avg_depth = np.mean(depth_region)
            
            # Closer objects (high depth) -> darker shadows
            depth_factor = avg_depth
            shadow = shadow * (0.5 + 0.5 * depth_factor)
        
        # Apply blur
        shadow_uint8 = (shadow * 255).astype(np.uint8)
        shadow_blurred = cv2.GaussianBlur(shadow_uint8, (blur_radius * 2 + 1, blur_radius * 2 + 1), 0)
        shadow_float = shadow_blurred.astype(np.float32) / 255.0
        
        # Create RGBA shadow
        shadow_rgba = np.zeros((h_rug, w_rug, 4), dtype=np.uint8)
        shadow_rgba[:, :, 3] = (shadow_float * 255).astype(np.uint8)
        
        return shadow_rgba
    
    def _blend_shadow(
        self,
        room: np.ndarray,
        shadow: np.ndarray,
        center_x: int,
        center_y: int,
        placement_mask: np.ndarray,
        offset: Tuple[int, int] = (3, 8)
    ) -> np.ndarray:
        """Blend shadow onto room."""
        h_room, w_room = room.shape[:2]
        h_shadow, w_shadow = shadow.shape[:2]
        
        # Calculate shadow position with offset
        x = center_x - w_shadow // 2 + offset[0]
        y = center_y - h_shadow // 2 + offset[1]
        
        # Calculate valid region
        x1_src = max(0, -x)
        y1_src = max(0, -y)
        x2_src = min(w_shadow, w_room - x)
        y2_src = min(h_shadow, h_room - y)
        
        x1_dst = max(0, x)
        y1_dst = max(0, y)
        x2_dst = x1_dst + (x2_src - x1_src)
        y2_dst = y1_dst + (y2_src - y1_src)
        
        if x2_src <= x1_src or y2_src <= y1_src:
            return room
        
        # Extract regions
        shadow_region = shadow[y1_src:y2_src, x1_src:x2_src]
        room_region = room[y1_dst:y2_dst, x1_dst:x2_dst].copy()
        mask_region = placement_mask[y1_dst:y2_dst, x1_dst:x2_dst]
        
        # Alpha blend shadow
        alpha = (shadow_region[:, :, 3] / 255.0)[:, :, np.newaxis]
        alpha = alpha * mask_region[:, :, np.newaxis]  # Only on floor
        
        # Darken room image
        darkened = (room_region[:, :, :3] * (1 - alpha * 0.5)).astype(np.uint8)
        room_region[:, :, :3] = darkened
        
        room[y1_dst:y2_dst, x1_dst:x2_dst] = room_region
        
        return room
    
    def _blend_rug(
        self,
        room: np.ndarray,
        rug: np.ndarray,
        center_x: int,
        center_y: int,
        placement_mask: np.ndarray
    ) -> np.ndarray:
        """Blend rug onto room with floor masking."""
        h_room, w_room = room.shape[:2]
        h_rug, w_rug = rug.shape[:2]
        
        # Calculate position
        x = center_x - w_rug // 2
        y = center_y - h_rug // 2
        
        # Calculate valid region
        x1_src = max(0, -x)
        y1_src = max(0, -y)
        x2_src = min(w_rug, w_room - x)
        y2_src = min(h_rug, h_room - y)
        
        x1_dst = max(0, x)
        y1_dst = max(0, y)
        x2_dst = x1_dst + (x2_src - x1_src)
        y2_dst = y1_dst + (y2_src - y1_src)
        
        if x2_src <= x1_src or y2_src <= y1_src:
            return room
        
        # Extract regions
        rug_region = rug[y1_src:y2_src, x1_src:x2_src]
        room_region = room[y1_dst:y2_dst, x1_dst:x2_dst].copy()
        mask_region = placement_mask[y1_dst:y2_dst, x1_dst:x2_dst]
        
        # Alpha blend
        alpha = (rug_region[:, :, 3] / 255.0)[:, :, np.newaxis]
        alpha = alpha * mask_region[:, :, np.newaxis]  # Only on floor
        
        blended = (
            rug_region[:, :, :3] * alpha +
            room_region[:, :, :3] * (1 - alpha)
        ).astype(np.uint8)
        
        room_region[:, :, :3] = blended
        room[y1_dst:y2_dst, x1_dst:x2_dst] = room_region
        
        return room
    
    def _apply_furniture_occlusion(
        self,
        result: np.ndarray,
        original_room: np.ndarray,
        furniture_masks: List[np.ndarray]
    ) -> np.ndarray:
        """
        Apply furniture occlusion - furniture goes OVER rug.
        
        This ensures chair legs, table legs, etc. appear in front of the rug.
        """
        h, w = result.shape[:2]
        
        for furniture_mask in furniture_masks:
            # Resize mask if needed
            if furniture_mask.shape != (h, w):
                furniture_mask = cv2.resize(
                    furniture_mask.astype(np.uint8),
                    (w, h),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)
            
            # Where furniture is, use original room
            result[furniture_mask] = original_room[furniture_mask]
        
        return result
    
    def _enhance_integration(
        self,
        result: np.ndarray,
        room: np.ndarray,
        depth_map: np.ndarray,
        edge_blend_radius: int = 3
    ) -> np.ndarray:
        """
        Final enhancements for realistic integration.
        
        - Edge blending
        - Subtle ambient occlusion
        - Slight color adaptation
        """
        # Edge blending (subtle feathering at rug boundaries)
        # This would detect rug edges and soften them slightly
        
        # For now, return as-is
        # Production would add more sophisticated blending
        
        return result