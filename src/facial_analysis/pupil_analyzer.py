from .base_analyzer import BaseAnalyzer
import cv2
import numpy as np
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class PupilAnalyzer(BaseAnalyzer):
    def __init__(self):
        self.face_mesh, self.mp_face_mesh, self.mp_drawing = self.init_mediapipe()
        self.baseline_data = {}
        
        # Punti degli occhi in MediaPipe Face Mesh
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]

    def _extract_eye_region(self, frame: np.ndarray, landmarks: Any, eye_points: list) -> Optional[np.ndarray]:
        """Extract eye region from frame using landmarks"""
        try:
            # Get eye region coordinates
            h, w = frame.shape[:2]
            x_coords = []
            y_coords = []
            
            for point in eye_points:
                landmark = landmarks.landmark[point]
                x_coords.append(int(landmark.x * w))
                y_coords.append(int(landmark.y * h))
            
            # Add padding
            padding = 10
            x = max(0, min(x_coords) - padding)
            y = max(0, min(y_coords) - padding)
            w = min(frame.shape[1] - x, max(x_coords) - min(x_coords) + 2*padding)
            h = min(frame.shape[0] - y, max(y_coords) - min(y_coords) + 2*padding)
            
            # Extract region
            eye_region = frame[y:y+h, x:x+w]
            return eye_region
            
        except Exception as e:
            logger.error(f"Error extracting eye region: {e}")
            return None

    def measure_pupil_size(self, eye_region: Optional[np.ndarray]) -> float:
        """Measure pupil size from eye region"""
        if eye_region is None or eye_region.size == 0:
            return 0.0
        
        try:
            gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=20,
                param1=50,
                param2=30,
                minRadius=5,
                maxRadius=int(min(eye_region.shape[:2]) / 2)
            )
            
            if circles is not None:
                # Convert to numpy array and handle types properly
                circles_array = np.asarray(circles[0], dtype=np.float32)
                if circles_array.size > 0:
                    # Get largest circle
                    largest_idx = np.argmax(circles_array[:, 2])
                    radius = circles_array[largest_idx, 2]
                    
                    # Calculate normalized area
                    eye_area = float(eye_region.shape[0] * eye_region.shape[1])
                    pupil_area = float(np.pi * (radius ** 2))
                    return pupil_area / eye_area
                    
            return 0.0
            
        except Exception as e:
            logger.error(f"Error measuring pupil size: {e}")
            return 0.0

    def analyze_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Analyze frame for pupil metrics"""
        results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if not results.multi_face_landmarks:
            return {'confidence': 0.0, 'error': 'No face detected'}
        
        landmarks = results.multi_face_landmarks[0]
        
        # Extract eye regions
        left_eye = self._extract_eye_region(frame, landmarks, self.LEFT_EYE)
        right_eye = self._extract_eye_region(frame, landmarks, self.RIGHT_EYE)
        
        return {
            'left_pupil': {'size': self.measure_pupil_size(left_eye)},
            'right_pupil': {'size': self.measure_pupil_size(right_eye)},
            'confidence': float(getattr(landmarks.landmark[0], 'visibility', 0.95))
        }
