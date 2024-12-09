from typing import Dict, Any, Optional, List
import numpy as np
from dataclasses import dataclass

@dataclass
class MockResult:
    success: bool
    data: Dict[str, Any]
    error: Optional[str] = None

class MockExpressionAnalyzer:
    def analyze_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        if frame.size == 0 or frame.shape[0] < 10:
            raise ValueError("Invalid frame dimensions")
        return {
            'confidence': 0.95,
            'expressions': {
                'smile': 0.8,
                'blink': 0.2,
                'eyebrow_raise': 0.1
            }
        }

class MockPupilAnalyzer:
    def analyze_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        if frame.size == 0:
            raise ValueError("Empty frame")
        return {
            'left_pupil': {'x': 0.5, 'y': 0.5, 'size': 0.3},
            'right_pupil': {'x': 0.6, 'y': 0.5, 'size': 0.3},
            'confidence': 0.9
        }

class MockPoseAnalyzer:
    def analyze_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        if frame.size == 0:
            raise ValueError("Empty frame")
        return {
            'pose_landmarks': np.random.rand(33, 3),
            'confidence': 0.85
        } 