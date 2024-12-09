"""Base class for facial analysis with MediaPipe initialization"""
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from typing import Tuple, Any, Dict, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)

class BaseAnalyzer:
    @staticmethod
    def init_mediapipe() -> Tuple[Any, Any, Any]:
        """Initialize MediaPipe components"""
        try:
            mp_face_mesh = mp.solutions.face_mesh
            face_mesh = mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            mp_drawing = mp.solutions.drawing_utils
            
            return face_mesh, mp_face_mesh, mp_drawing
            
        except Exception as e:
            logger.error(f"MediaPipe initialization error: {e}")
            raise ImportError(f"Failed to initialize MediaPipe: {e}")

    def analyze_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Base method for frame analysis"""
        raise NotImplementedError("Subclasses must implement analyze_frame")