import numpy as np
from pathlib import Path
import cv2
from typing import Tuple, List, Dict, Any
import json
import logging

logger = logging.getLogger(__name__)

class TestHelper:
    @staticmethod
    def generate_test_frame(width: int = 640, height: int = 480) -> np.ndarray:
        """Genera un frame di test"""
        return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        
    @staticmethod
    def generate_test_sequence(length: int = 100, features: int = 99) -> np.ndarray:
        """Genera una sequenza di test"""
        return np.random.rand(1, length, features)
        
    @staticmethod
    def save_test_results(results: Dict[str, Any], test_name: str):
        """Salva i risultati del test"""
        output_dir = Path('tests/results')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"{test_name}_{results['timestamp']}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4) 

class MockAnalyzer:
    def __init__(self):
        self.face_mesh = None
        
    def close(self):
        pass
        
    def analyze_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        return {
            'confidence': 0.95,
            'metrics': {'value': 0.5}
        }

class MockExpressionAnalyzer(MockAnalyzer):
    pass

class MockPupilAnalyzer(MockAnalyzer):
    pass

class MockPoseAnalyzer(MockAnalyzer):
    def __init__(self):
        super().__init__()
        self.pose = None