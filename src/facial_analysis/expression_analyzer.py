from .base_analyzer import BaseAnalyzer
from typing import List, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ExpressionAnalyzer(BaseAnalyzer):
    def __init__(self):
        """Initialize MediaPipe Face Mesh"""
        self.face_mesh, self.mp_face_mesh, self.mp_drawing = self.init_mediapipe()

    def calculate_eye_aspect_ratio(self, eye_points: List[Tuple[float, float]]) -> float:
        """Calculate eye aspect ratio from points"""
        if len(eye_points) != 6:
            return 0.0
        
        try:
            # Convert points to numpy arrays
            points = np.array(eye_points)
            
            # Calculate vertical distances
            A = np.linalg.norm(points[1] - points[5])
            B = np.linalg.norm(points[2] - points[4])
            
            # Calculate horizontal distance
            C = np.linalg.norm(points[0] - points[3])
            
            if C == 0:
                return 0.0
            
            return float((A + B) / (2.0 * C))
            
        except Exception as e:
            logger.error(f"Error calculating EAR: {e}")
            return 0.0
