try:
    import tensorflow as tf
except ImportError:
    raise ImportError("TensorFlow not installed. Run: pip install tensorflow>=2.8.0")

import numpy as np
from typing import Optional, Dict, Any, List, Union
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class MovementClassifier:
    def __init__(self, model_path: Optional[str] = None):
        """Initialize movement classifier"""
        self.model: Optional[tf.keras.Model] = None
        self.class_names: List[str] = ['walking', 'running', 'sitting', 'standing', 'jumping']
        self._initialize_model(model_path)
        
    def _initialize_model(self, model_path: Optional[str]) -> None:
        """Initialize TensorFlow model"""
        try:
            if model_path is None:
                model_path = str(Path(__file__).parent / 'models' / 'movement_classifier.h5')
                
            self.model = tf.keras.models.load_model(model_path)
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Model loading error: {e}")
            self.model = None
            
    def predict(self, sequence: Union[np.ndarray, Dict[str, Any]]) -> Dict[str, Any]:
        """Predict movement patterns"""
        if self.model is None:
            raise RuntimeError("Model not initialized")
            
        try:
            # Convert input to numpy array
            if isinstance(sequence, dict):
                sequence = self._extract_features(sequence)
                
            sequence = np.asarray(sequence, dtype=np.float32)
            if len(sequence.shape) != 3:
                sequence = np.expand_dims(sequence, axis=0)
                
            # Make prediction
            predictions = self.model(sequence, training=False)
            if isinstance(predictions, tf.Tensor):
                predictions = predictions.numpy()
                
            return {
                'movement': self.class_names[np.argmax(predictions[0])],
                'confidence': float(predictions[0].max()),
                'predictions': {
                    name: float(pred) 
                    for name, pred in zip(self.class_names, predictions[0])
                }
            }
            
        except Exception as e:
            logger.error(f"Error predicting movement: {e}")
            return {
                'movement': 'unknown',
                'confidence': 0.0,
                'error': str(e)
            }
            
    def _extract_features(self, data: Dict[str, Any]) -> np.ndarray:
        """Extract features from landmarks data"""
        if 'pose_landmarks' not in data:
            raise ValueError("Missing pose_landmarks in input data")
            
        landmarks = np.array(data['pose_landmarks'])
        return landmarks.reshape(1, -1, 99)  # Reshape to expected format
        
    def classify_movement(self, landmarks: np.ndarray) -> Dict[str, Any]:
        """Classify movement from landmarks"""
        predictions = self.predict(landmarks)
        return {
            'type': predictions['movement'],
            'confidence': predictions['confidence']
        }
        
    def get_current_patterns(self) -> List[Dict[str, Any]]:
        """Get current movement patterns"""
        return [
            {
                'name': name,
                'confidence': self.model(np.zeros((1, 99)), training=False)[0][i]
            }
            for i, name in enumerate(self.class_names)
        ]