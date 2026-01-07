"""
Production Rug Compositor with Depth Awareness
"""
import logging
import numpy as np
from PIL import Image
import cv2
from typing import Tuple, List, Optional

logger = logging.getLogger(__name__)

class RugCompositor:
    """Production-grade rug compositing"""
    
    def composite(
        self,
        room: Image.Image,
        rug: Image.Image,
        depth_map: np.ndarray,
        floor_mask: np.ndarray,
        position: Tuple[float, float],
        scale: float = 1.0,
        rotation: float = 0.0,
        furniture_masks: Optional[List[np.ndarray]] = None,
        use_depth_scaling: bool = True,
        use_occlusion: bool = True
    ) -> Image.Image:
        """
        Composite rug with depth-aware placement
        
        Args:
            room: Room image RGBA
            rug: Rug image RGBA
            depth_map: Normalized depth (0-1), close=high
            floor_mask: Binary floor mask
            position: (x, y) normalized 0-1
            scale: Base scale factor
            rotation: Degrees
            furniture_masks: Furniture occlusion masks
            use_depth_scaling: Scale by depth
            use_occlusion: Apply furniture occlusion
            
        Returns:
            Composited image
        """
        room_arr = np.array(room)
        rug_arr = np.array(rug)
        height, width = room_arr.shape[:2]
        
        # Convert position to pixels
        center_x = int(position[0] * width)
        center_y = int(position[1] * height)
        
        # Sample depth at center
        cx = np.clip(center_x, 0, width - 1)
        cy = np.clip(center_y, 0, height - 1)
        center_depth = depth_map[cy, cx]
        
        # Calculate depth-based scale
        if use_depth_scaling:
            depth_scale = self._depth_to_scale(center_depth)
            final_scale = scale * depth_scale
        else:
            final_scale = scale
        
        logger.info(f"Compositing: pos=({center_x},{center_y}), depth={center_depth:.3f}, scale={final_scale:.3f}")
        
        # Transform rug
        transformed_rug = self._transform_rug(
            rug_arr,
            scale=final_scale,
            rotation=rotation,
            depth_map=depth_map,
            center_x=center_x,
            center_y=center_y,
            room_size=(width, height)
        )
        
        # Generate shadow
        shadow = self._generate_shadow(transformed_rug, center_depth)
        
        # Start compositing
        result = room_arr.copy()
        
        # 1. Blend shadow
        result = self._blend_shadow(
            result, shadow, center_x, center_y, floor_mask
        )
        
        # 2. Blend rug
        result = self._blend_rug(
            result, transformed_rug, center_x, center_y, floor_mask
        )
        
        # 3. Apply furniture occlusion
        if use_occlusion and furniture_masks:
            result = self._apply_occlusion(
                result, room_arr, furniture_masks
            )
        
        return Image.fromarray(result)
    
    def _depth_to_scale(self, depth: float) -> float:
        """
        Convert depth to scale factor
        Close (depth=0.8-1.0) -> scale=1.2-1.5 (larger)
        Far (depth=0.2-0.4) -> scale=0.5-0.8 (smaller)
        """
        # Inverse relationship: close objects are large
        distance = 1.0 - depth
        
        # Map to scale range
        min_scale, max_scale = 0.5, 1.8
        scale = max_scale - (distance * (max_scale - min_scale))
        
        return np.clip(scale, min_scale, max_scale)
    
    def _transform_rug(
        self,
        rug: np.ndarray,
        scale: float,
        rotation: float,
        depth_map: np.ndarray,
        center_x: int,
        center_y: int,
        room_size: Tuple[int, int]
    ) -> np.ndarray:
        """Apply scale, rotation, and perspective warp"""
        h, w = rug.shape[:2]
        room_w, room_h = room_size
        
        # 1. Scale
        new_w = int(w * scale)
        new_h = int(h * scale)
        scaled = cv2.resize(rug, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # 2. Rotate
        if abs(rotation) > 0.5:
            center = (new_w // 2, new_h // 2)
            M = cv2.getRotationMatrix2D(center, rotation, 1.0)
            
            # Calculate bounds
            cos = abs(M[0, 0])
            sin = abs(M[0, 1])
            new_w_rot = int((new_h * sin) + (new_w * cos))
            new_h_rot = int((new_h * cos) + (new_w * sin))
            
            M[0, 2] += (new_w_rot / 2) - center[0]
            M[1, 2] += (new_h_rot / 2) - center[1]
            
            scaled = cv2.warpAffine(
                scaled, M, (new_w_rot, new_h_rot),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0)
            )
            new_w, new_h = new_w_rot, new_h_rot
        
        # 3. Perspective warp based on depth gradient
        perspective_strength = self._calculate_perspective_strength(
            depth_map, center_x, center_y, new_w, new_h, room_size
        )
        
        if perspective_strength > 0.05:
            scaled = self._apply_perspective(scaled, perspective_strength)
        
        return scaled
    
    def _calculate_perspective_strength(
        self,
        depth_map: np.ndarray,
        cx: int,
        cy: int,
        rug_w: int,
        rug_h: int,
        room_size: Tuple[int, int]
    ) -> float:
        """Calculate how much perspective to apply"""
        room_w, room_h = room_size
        
        # Sample depth at top and bottom of rug region
        y_top = max(0, cy - rug_h // 2)
        y_bottom = min(room_h - 1, cy + rug_h // 2)
        
        x_left = max(0, cx - rug_w // 4)
        x_right = min(room_w - 1, cx + rug_w // 4)
        
        # Average depth at top and bottom
        top_depth = np.mean(depth_map[y_top:y_top+5, x_left:x_right])
        bottom_depth = np.mean(depth_map[y_bottom-5:y_bottom, x_left:x_right])
        
        # Perspective strength from depth difference
        depth_diff = bottom_depth - top_depth
        
        # Normalize to 0-1 range (more diff = more perspective)
        strength = np.clip(depth_diff * 2.0, 0, 1.0)
        
        return strength
    
    def _apply_perspective(self, rug: np.ndarray, strength: float) -> np.ndarray:
        """Apply trapezoid perspective transform"""
        h, w = rug.shape[:2]
        
        # Source: rectangle
        src = np.float32([
            [0, 0], [w, 0], [w, h], [0, h]
        ])
        
        # Destination: trapezoid (top narrower)
        shrink = int(w * strength * 0.2)
        dst = np.float32([
            [shrink, 0], [w - shrink, 0], [w, h], [0, h]
        ])
        
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(
            rug, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )
        
        return warped
    
    def _generate_shadow(self, rug: np.ndarray, depth: float) -> np.ndarray:
        """Generate soft shadow from rug"""
        h, w = rug.shape[:2]
        
        # Extract alpha channel
        alpha = rug[:, :, 3]
        
        # Create shadow intensity
        intensity = 0.3 + (depth * 0.2)  # Closer = darker
        shadow = np.zeros((h, w), dtype=np.float32)
        shadow[alpha > 0] = intensity
        
        # Blur shadow
        blur_size = int(15 + depth * 10)
        if blur_size % 2 == 0:
            blur_size += 1
        
        shadow = cv2.GaussianBlur(shadow, (blur_size, blur_size), 0)
        
        # Convert to RGBA
        shadow_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        shadow_rgba[:, :, 3] = (shadow * 255).astype(np.uint8)
        
        return shadow_rgba
    
    def _blend_shadow(
        self,
        room: np.ndarray,
        shadow: np.ndarray,
        cx: int,
        cy: int,
        floor_mask: np.ndarray,
        offset: Tuple[int, int] = (3, 10)
    ) -> np.ndarray:
        """Blend shadow onto floor"""
        h_room, w_room = room.shape[:2]
        h_shadow, w_shadow = shadow.shape[:2]
        
        # Calculate position with offset
        x = cx - w_shadow // 2 + offset[0]
        y = cy - h_shadow // 2 + offset[1]
        
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
        
        # Blend
        shadow_region = shadow[y1_src:y2_src, x1_src:x2_src]
        room_region = room[y1_dst:y2_dst, x1_dst:x2_dst].copy()
        floor_region = floor_mask[y1_dst:y2_dst, x1_dst:x2_dst]
        
        alpha = (shadow_region[:, :, 3] / 255.0)[:, :, np.newaxis]
        alpha = alpha * floor_region[:, :, np.newaxis]
        
        # Darken
        darkened = (room_region[:, :, :3] * (1 - alpha * 0.6)).astype(np.uint8)
        room_region[:, :, :3] = darkened
        
        room[y1_dst:y2_dst, x1_dst:x2_dst] = room_region
        return room
    
    def _blend_rug(
        self,
        room: np.ndarray,
        rug: np.ndarray,
        cx: int,
        cy: int,
        floor_mask: np.ndarray
    ) -> np.ndarray:
        """Blend rug onto floor"""
        h_room, w_room = room.shape[:2]
        h_rug, w_rug = rug.shape[:2]
        
        x = cx - w_rug // 2
        y = cy - h_rug // 2
        
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
        
        rug_region = rug[y1_src:y2_src, x1_src:x2_src]
        room_region = room[y1_dst:y2_dst, x1_dst:x2_dst].copy()
        floor_region = floor_mask[y1_dst:y2_dst, x1_dst:x2_dst]
        
        alpha = (rug_region[:, :, 3] / 255.0)[:, :, np.newaxis]
        alpha = alpha * floor_region[:, :, np.newaxis]
        
        blended = (
            rug_region[:, :, :3] * alpha +
            room_region[:, :, :3] * (1 - alpha)
        ).astype(np.uint8)
        
        room_region[:, :, :3] = blended
        room[y1_dst:y2_dst, x1_dst:x2_dst] = room_region
        
        return room
    
    def _apply_occlusion(
        self,
        result: np.ndarray,
        original: np.ndarray,
        furniture_masks: List[np.ndarray]
    ) -> np.ndarray:
        """Apply furniture occlusion"""
        h, w = result.shape[:2]
        
        for mask in furniture_masks:
            if mask.shape != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            
            mask = (mask > 0)  # <-- FIXED: ensure boolean indexing
            
            result[mask] = original[mask]
        
        return result
