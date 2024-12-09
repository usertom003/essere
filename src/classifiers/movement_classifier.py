try:
    import tensorflow.keras as keras
    from tensorflow.keras import Model
    from tensorflow.keras.models import load_model
    import tensorflow as tf
except ImportError:
    raise ImportError("TensorFlow/Keras not installed. Run: pip install tensorflow>=2.8.0")

import numpy as np
from typing import Optional, List, Dict, Any
import logging
from pathlib import Path
from ..types import MovementPattern

logger = logging.getLogger(__name__)

class MovementClassifier:
    def __init__(self, model_path: Optional[str] = None):
        """Initialize movement classifier"""
        self.model: Optional[Model] = None
        self.class_names: List[str] = ['walking', 'running', 'sitting', 'standing', 'jumping']
        self._initialize_model(model_path)
        
    def _initialize_model(self, model_path: Optional[str]) -> None:
        try:
            if model_path is None:
                model_path = str(Path(__file__).parent / 'models' / 'movement_classifier.h5')
            self.model = load_model(model_path)
            if self.model is None:
                raise ValueError("Failed to load model")
        except Exception as e:
            logger.error(f"Model loading error: {e}")
            self.model = None

    def _extract_features(self, data: Dict[str, Any]) -> np.ndarray:
        """Extract features from input data"""
        if isinstance(data, dict) and 'landmarks' in data:
            return np.array(data['landmarks']).reshape(1, -1, 99)
        raise ValueError("Invalid input format")
        
    def _calculate_frequency(self, sequence: np.ndarray, class_idx: int) -> float:
        """Calculate movement frequency for given class"""
        try:
            # Calculate frequency from sequence data
            return float(np.mean(sequence[:, :, class_idx]))
        except Exception:
            return 0.0

    def _calculate_duration(self, sequence: np.ndarray, class_idx: int) -> float:
        """Calculate movement duration for given class"""
        try:
            # Calculate duration from sequence data
            return float(len(sequence) * 0.033)  # Assuming 30 FPS
        except Exception:
            return 0.0

    def _calculate_intensity(self, sequence: np.ndarray, class_idx: int) -> float:
        """Calculate movement intensity for given class"""
        try:
            # Calculate intensity from sequence data
            return float(np.max(sequence[:, :, class_idx]))
        except Exception:
            return 0.0

    def predict(self, sequence: np.ndarray) -> List[MovementPattern]:
        """Predict movement patterns"""
        if self.model is None:
            return []
            
        try:
            if len(sequence.shape) != 3:
                sequence = sequence.reshape(1, -1, 99)
                
            predictions = self.model.predict(sequence, verbose=0)
            return [
                MovementPattern(
                    name=name,
                    confidence=float(pred),
                    frequency=0.0,
                    duration=0.0,
                    intensity=0.0
                )
                for name, pred in zip(self.class_names, predictions[0])
            ]
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return []

    def get_movement_patterns(self) -> List[MovementPattern]:
        """Get current movement patterns"""
        try:
            dummy_input = np.zeros((1, 1, 99))  # Correct shape for model input
            predictions = self.model.predict(dummy_input, verbose=0)
            return [
                MovementPattern(
                    name=name,
                    confidence=float(pred),
                    frequency=0.0,
                    duration=0.0,
                    intensity=0.0
                )
                for name, pred in zip(self.class_names, predictions[0])
            ]
        except Exception as e:
            logger.error(f"Error getting patterns: {e}")
            return [] 