from typing import Optional
from pathlib import Path
import numpy as np
from PIL import Image
from ultralytics import YOLO
import torch
import logging

from app import config
from app.schemas.prediction import ClassificationResult

logger = logging.getLogger(__name__)


class CVService:
    """Computer Vision service for palm oil disease classification."""
    
    def __init__(self):
        """Initialize CV service with YOLO model."""
        self.model: Optional[YOLO] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"CV Service initialized. Device: {self.device}")
    
    def load_model(self) -> None:
        """
        Load YOLO model from disk.
        
        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model loading fails
        """
        try:
            if not config.MODEL_PATH.exists():
                raise FileNotFoundError(
                    f"Model file not found at: {config.MODEL_PATH}"
                )
            
            logger.info(f"Loading model from {config.MODEL_PATH}")
            self.model = YOLO(str(config.MODEL_PATH))
            self.model.to(self.device)
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None
    
    def predict(self, image: Image.Image) -> ClassificationResult:
        """
        Perform disease classification on image.
        
        Args:
            image: PIL Image object
            
        Returns:
            ClassificationResult with label and confidence
            
        Raises:
            RuntimeError: If model not loaded or prediction fails
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Convert image to RGB if needed
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # Run inference
            results = self.model.predict(
                source=image,
                conf=config.CONFIDENCE_THRESHOLD,
                iou=config.IOU_THRESHOLD,
                imgsz=config.IMAGE_SIZE,
                verbose=False
            )
            
            # Extract prediction
            if len(results) == 0 or len(results[0].boxes) == 0:
                # No detection - assume healthy or uncertain
                return ClassificationResult(
                    label="Healthy sample",
                    confidence=0.5
                )
            
            # Get highest confidence detection
            result = results[0]
            boxes = result.boxes
            
            # Find box with highest confidence
            confidences = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy()
            
            max_idx = np.argmax(confidences)
            max_conf = float(confidences[max_idx])
            class_id = int(class_ids[max_idx])
            
            # Get class name
            class_name = result.names[class_id]
            
            # Validate class name
            if class_name not in config.DISEASE_CLASSES:
                logger.warning(f"Unknown class detected: {class_name}")
                # Map to closest known class or healthy
                class_name = "Healthy sample"
                max_conf = 0.5
            
            logger.info(f"Prediction: {class_name} ({max_conf:.2f})")
            
            return ClassificationResult(
                label=class_name,
                confidence=max_conf
            )
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def validate_image(self, image: Image.Image) -> bool:
        """
        Validate image format and size.
        
        Args:
            image: PIL Image object
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check format
            if image.format not in ['JPEG', 'PNG', 'JPG']:
                return False
            
            # Check size (min 100x100, max 4096x4096)
            width, height = image.size
            if width < 100 or height < 100:
                return False
            if width > 4096 or height > 4096:
                return False
            
            return True
            
        except Exception:
            return False


# Global service instance
cv_service = CVService()