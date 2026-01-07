"""
Real Depth Estimation using MiDaS
NO FALLBACKS - Pure AI
"""
import logging
import numpy as np
import torch
import cv2

logger = logging.getLogger(__name__)

class DepthEstimator:
    """Production MiDaS depth estimator"""
    
    def __init__(self, model_type: str = "DPT_Large", device: str = None):
        self.model_type = model_type
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.transform = None
        self._load_model()
    
    def _load_model(self):
        """Load MiDaS model - MUST succeed"""
        try:
            logger.info(f"Loading MiDaS {self.model_type} on {self.device}...")
            
            self.model = torch.hub.load(
                "intel-isl/MiDaS",
                self.model_type,
                trust_repo=True,
                skip_validation=True
            )
            
            self.model.to(self.device)
            self.model.eval()
            
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            
            if self.model_type in ["DPT_Large", "DPT_Hybrid"]:
                self.transform = midas_transforms.dpt_transform
            else:
                self.transform = midas_transforms.small_transform
            
            logger.info("✓ MiDaS model loaded successfully")
            
        except Exception as e:
            logger.error(f"CRITICAL: Failed to load MiDaS model: {e}")
            raise RuntimeError(f"Cannot initialize depth estimator: {e}")
    
    def estimate(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate depth map
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        h, w = image.shape[:2]

        try:
            # MiDaS expects float32 numpy image
            img = image.astype(np.float32)

            transformed = self.transform(img)

            # 🔥 FINAL FIX: handle both MiDaS transform outputs
            if isinstance(transformed, dict):
                input_batch = transformed["image"]
            else:
                input_batch = transformed

            input_batch = input_batch.to(self.device)

            with torch.no_grad():
                prediction = self.model(input_batch)

                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=(h, w),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            depth_map = prediction.cpu().numpy()

            # Normalize
            d_min, d_max = depth_map.min(), depth_map.max()
            if d_max - d_min > 1e-6:
                depth_map = (depth_map - d_min) / (d_max - d_min)
            else:
                depth_map = np.full_like(depth_map, 0.5)

            return depth_map.astype(np.float32)

        except Exception as e:
            logger.error(f"Depth estimation failed: {e}")
            raise

    
    def visualize(self, depth_map: np.ndarray, colormap: int = cv2.COLORMAP_MAGMA) -> np.ndarray:
        """Create colored visualization"""
        depth_uint8 = (depth_map * 255).astype(np.uint8)
        colored = cv2.applyColorMap(depth_uint8, colormap)
        return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
