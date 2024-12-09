from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np

@dataclass
class MovementPattern:
    name: str
    confidence: float
    frequency: float = 0.0
    duration: float = 0.0
    intensity: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'confidence': self.confidence,
            'frequency': self.frequency,
            'duration': self.duration,
            'intensity': self.intensity
        }

@dataclass
class BodyMetrics:
    pose_landmarks: Optional[np.ndarray]
    joint_angles: Dict[str, float]
    confidence: float
    landmarks: Dict[str, Any]

    def get_pose_landmarks(self) -> Optional[np.ndarray]:
        """Get pose landmarks safely"""
        if isinstance(self.pose_landmarks, np.ndarray):
            return self.pose_landmarks
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'pose_landmarks': self.pose_landmarks.tolist() if self.pose_landmarks is not None else None,
            'joint_angles': self.joint_angles,
            'confidence': self.confidence,
            'landmarks': self.landmarks
        }

@dataclass
class FacialMetrics:
    eye_aspect_ratio: float
    mouth_aspect_ratio: float
    eyebrow_position: float
    nose_wrinkle: float
    head_pose: Tuple[float, float, float] 

@dataclass
class ComponentHealth:
    name: str
    status: str
    error_count: int
    last_error: Optional[str] = None
    avg_response_time: float = 0.0
    memory_usage: float = 0.0

    def update_metrics(self, response_time: Union[float, np.floating], memory: Union[float, np.floating]) -> None:
        """Update metrics with explicit type conversion"""
        self.avg_response_time = float(response_time)
        self.memory_usage = float(memory)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'status': self.status,
            'error_count': int(self.error_count),
            'last_error': self.last_error,
            'avg_response_time': float(self.avg_response_time),
            'memory_usage': float(self.memory_usage)
        }

@dataclass
class JointAngles:
    left_elbow: float
    right_elbow: float
    left_knee: float
    right_knee: float
    left_hip: float
    right_hip: float
    left_shoulder: float
    right_shoulder: float

    def to_dict(self) -> Dict[str, float]:
        return {
            'left_elbow': self.left_elbow,
            'right_elbow': self.right_elbow,
            'left_knee': self.left_knee,
            'right_knee': self.right_knee,
            'left_hip': self.left_hip,
            'right_hip': self.right_hip,
            'left_shoulder': self.left_shoulder,
            'right_shoulder': self.right_shoulder
        }

@dataclass
class MovementMetrics:
    smoothness: float
    complexity: float
    periodicity: float
    symmetry: float
    coordination: float
    energy: float
    variability: float
    phase_coherence: float

    def to_dict(self) -> Dict[str, float]:
        return {
            'smoothness': self.smoothness,
            'complexity': self.complexity,
            'periodicity': self.periodicity,
            'symmetry': self.symmetry,
            'coordination': self.coordination,
            'energy': self.energy,
            'variability': self.variability,
            'phase_coherence': self.phase_coherence
        }