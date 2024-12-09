import numpy as np
from typing import Dict, Any
from datetime import datetime

class MockExpressionAnalyzer:
    def analyze_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        if frame is None or frame.size == 0:
            raise ValueError("Invalid frame")
            
        return {
            'eye_aspect_ratio': 0.5,
            'mouth_aspect_ratio': 0.6,
            'eyebrow_position': 0.7,
            'nose_wrinkle': 0.2,
            'confidence': 0.9,
            'timestamp': datetime.now().isoformat()
        }

class MockPupilAnalyzer:
    def analyze_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        if frame is None or frame.size == 0:
            raise ValueError("Invalid frame")
            
        return {
            'left_pupil_size': 0.3,
            'right_pupil_size': 0.3,
            'left_pupil_position': [0.4, 0.5],
            'right_pupil_position': [0.6, 0.5],
            'confidence': 0.85,
            'timestamp': datetime.now().isoformat()
        }

class MockPoseAnalyzer:
    def analyze_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        if frame is None or frame.size == 0:
            raise ValueError("Invalid frame")
            
        return {
            'balance_score': 0.8,
            'posture_quality': 0.7,
            'movement_intensity': 0.4,
            'joint_angles': {
                'neck': 15.0,
                'shoulders': [10.0, 12.0],
                'elbows': [85.0, 87.0],
                'spine': 5.0
            },
            'confidence': 0.9,
            'timestamp': datetime.now().isoformat()
        }

class MockEmotionClassifier:
    def __init__(self):
        self.emotion_labels = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'surprised']
        
    def predict(self, features: np.ndarray) -> str:
        if len(features.shape) != 2 or features.shape[1] != 7:
            raise ValueError("Invalid features shape")
            
        # Simula predizione
        return np.random.choice(self.emotion_labels)

class MockMovementClassifier:
    def __init__(self):
        self.movement_types = ['sitting', 'standing', 'walking', 'running']
        
    def predict(self, sequence: np.ndarray) -> str:
        if len(sequence.shape) != 3:
            raise ValueError("Invalid sequence shape")
            
        # Simula predizione
        return np.random.choice(self.movement_types)
