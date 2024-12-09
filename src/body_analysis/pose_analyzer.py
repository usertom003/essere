import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
import logging
from scipy.spatial import distance
from scipy.signal import find_peaks
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

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

@dataclass
class BodyMetrics:
    joint_angles: JointAngles
    spine_curvature: float
    balance_score: float
    movement_symmetry: float
    velocity: Dict[str, float]
    acceleration: Dict[str, float]

class PoseAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=2
        )
        
        # Deep learning model for movement analysis
        self.movement_model = self._create_movement_model()
        self.movement_history: List[np.ndarray] = []
        self.prev_landmarks = None
        self.frame_count = 0
        
    def _create_movement_model(self) -> Model:
        """Create deep learning model for movement pattern analysis"""
        input_shape = (100, 99)  # 33 landmarks x 3 coordinates
        
        inputs = Input(shape=input_shape)
        x = LSTM(128, return_sequences=True)(inputs)
        x = Dropout(0.3)(x)
        x = LSTM(64)(x)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(8, activation='softmax')(x)  # 8 movement patterns
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
        
    def calculate_joint_angles(self, landmarks) -> JointAngles:
        """Calculate angles between joints"""
        def calculate_angle(a, b, c) -> float:
            a = np.array([a.x, a.y, a.z])
            b = np.array([b.x, b.y, b.z])
            c = np.array([c.x, c.y, c.z])
            
            radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
                     np.arctan2(a[1]-b[1], a[0]-b[0])
            angle = np.abs(radians*180.0/np.pi)
            
            if angle > 180.0:
                angle = 360-angle
            return float(angle)
        
        # Calculate all joint angles
        left_elbow = calculate_angle(
            landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value],
            landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value],
            landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        )
        
        right_elbow = calculate_angle(
            landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
            landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value],
            landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        )
        
        left_knee = calculate_angle(
            landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value],
            landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value],
            landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        )
        
        right_knee = calculate_angle(
            landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value],
            landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value],
            landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        )
        
        left_hip = calculate_angle(
            landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value],
            landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value],
            landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        )
        
        right_hip = calculate_angle(
            landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
            landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value],
            landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        )
        
        left_shoulder = calculate_angle(
            landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value],
            landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value],
            landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        )
        
        right_shoulder = calculate_angle(
            landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value],
            landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
            landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        )
        
        return JointAngles(
            left_elbow=left_elbow,
            right_elbow=right_elbow,
            left_knee=left_knee,
            right_knee=right_knee,
            left_hip=left_hip,
            right_hip=right_hip,
            left_shoulder=left_shoulder,
            right_shoulder=right_shoulder
        )
        
    def calculate_spine_curvature(self, landmarks) -> float:
        """Calculate spine curvature using key points"""
        spine_points = [
            landmarks[self.mp_pose.PoseLandmark.NOSE.value],
            landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value],
            landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value],
            landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        ]
        
        points = np.array([[p.x, p.y, p.z] for p in spine_points])
        
        # Fit a polynomial to the spine points
        z = np.polyfit(points[:, 1], points[:, 0], 2)
        curvature = np.abs(z[0])  # Second derivative gives curvature
        
        return float(curvature)
        
    def calculate_balance_score(self, landmarks) -> float:
        """Calculate balance score based on center of mass and support base"""
        # Get feet and hip positions
        left_foot = np.array([
            landmarks[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,
            landmarks[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y
        ])
        right_foot = np.array([
            landmarks[self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,
            landmarks[self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y
        ])
        hips = np.array([
            landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
            landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y
        ])
        
        # Calculate center of mass (simplified)
        com = hips
        
        # Calculate base of support
        support_center = (left_foot + right_foot) / 2
        support_width = np.linalg.norm(right_foot - left_foot)
        
        # Calculate deviation from ideal balance
        com_deviation = np.linalg.norm(com - support_center)
        balance_score = 1.0 - min(com_deviation / support_width, 1.0)
        
        return float(balance_score)
        
    def calculate_movement_symmetry(self, landmarks) -> float:
        """Calculate movement symmetry between left and right body parts"""
        def get_limb_vector(landmark1, landmark2):
            return np.array([
                landmark2.x - landmark1.x,
                landmark2.y - landmark1.y,
                landmark2.z - landmark1.z
            ])
        
        # Compare arm movements
        left_arm = get_limb_vector(
            landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value],
            landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        )
        right_arm = get_limb_vector(
            landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
            landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        )
        
        # Compare leg movements
        left_leg = get_limb_vector(
            landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value],
            landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        )
        right_leg = get_limb_vector(
            landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value],
            landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        )
        
        # Calculate symmetry scores
        arm_symmetry = 1.0 - (np.linalg.norm(left_arm - right_arm) / 
                            (np.linalg.norm(left_arm) + np.linalg.norm(right_arm)))
        leg_symmetry = 1.0 - (np.linalg.norm(left_leg - right_leg) /
                            (np.linalg.norm(left_leg) + np.linalg.norm(right_leg)))
        
        return float((arm_symmetry + leg_symmetry) / 2.0)
        
    def calculate_velocity_acceleration(self, landmarks) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Calculate velocity and acceleration of key points"""
        if self.prev_landmarks is None:
            self.prev_landmarks = landmarks
            return {}, {}
            
        dt = 1/30  # Assuming 30 fps
        
        velocities = {}
        accelerations = {}
        
        for i, landmark in enumerate(landmarks):
            # Calculate velocity
            prev_pos = np.array([
                self.prev_landmarks[i].x,
                self.prev_landmarks[i].y,
                self.prev_landmarks[i].z
            ])
            curr_pos = np.array([landmark.x, landmark.y, landmark.z])
            velocity = (curr_pos - prev_pos) / dt
            
            # Store velocity magnitude
            velocities[f"point_{i}"] = float(np.linalg.norm(velocity))
            
            # Calculate acceleration if we have previous velocity
            if hasattr(self, 'prev_velocities'):
                prev_vel = self.prev_velocities[f"point_{i}"]
                acceleration = (velocities[f"point_{i}"] - prev_vel) / dt
                accelerations[f"point_{i}"] = float(acceleration)
            
        self.prev_landmarks = landmarks
        self.prev_velocities = velocities
        
        return velocities, accelerations
        
    def detect_repetitive_movements(self, landmarks) -> Dict[str, int]:
        """Detect repetitive movements using peak detection"""
        if len(self.movement_history) < 30:  # Need some history
            return {}
            
        movements = {}
        
        # Convert landmarks to flat array
        current_pose = np.array([[l.x, l.y, l.z] for l in landmarks]).flatten()
        self.movement_history.append(current_pose)
        
        if len(self.movement_history) > 300:  # Keep 10 seconds at 30fps
            self.movement_history.pop(0)
            
        # Analyze movement patterns
        movement_data = np.array(self.movement_history)
        
        # Find peaks in movement data
        for i in range(movement_data.shape[1]):
            peaks, _ = find_peaks(movement_data[:, i], distance=15)
            if len(peaks) > 1:
                frequency = len(peaks) / (len(movement_data) / 30)  # peaks per second
                if frequency > 0.5:  # More than 0.5 Hz
                    movements[f"point_{i}_frequency"] = float(frequency)
                    
        return movements
        
    def analyze_pose(self, frame) -> Optional[BodyMetrics]:
        """Main method to analyze pose in frame"""
        try:
            # Convert frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame
            results = self.pose.process(rgb_frame)
            
            if not results.pose_landmarks:
                return None
                
            landmarks = results.pose_landmarks.landmark
            
            # Calculate all metrics
            joint_angles = self.calculate_joint_angles(landmarks)
            spine_curvature = self.calculate_spine_curvature(landmarks)
            balance_score = self.calculate_balance_score(landmarks)
            movement_symmetry = self.calculate_movement_symmetry(landmarks)
            velocity, acceleration = self.calculate_velocity_acceleration(landmarks)
            
            # Detect repetitive movements
            repetitive_movements = self.detect_repetitive_movements(landmarks)
            
            # Update frame count
            self.frame_count += 1
            
            return BodyMetrics(
                joint_angles=joint_angles,
                spine_curvature=spine_curvature,
                balance_score=balance_score,
                movement_symmetry=movement_symmetry,
                velocity=velocity,
                acceleration=acceleration
            )
            
        except Exception as e:
            logger.error(f"Error analyzing pose: {str(e)}")
            return None
            
    def draw_pose_landmarks(self, frame, landmarks):
        """Draw pose landmarks and connections on frame"""
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        
        # Draw the pose landmarks
        mp_drawing.draw_landmarks(
            frame,
            landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )
        
        return frame

    def _calculate_balance_score(self, landmarks: np.ndarray) -> float:
        """Calcola un punteggio di equilibrio basato sulla posizione dei landmark"""
        try:
            # Estrai coordinate rilevanti
            hip_l = landmarks[23]  # Left hip
            hip_r = landmarks[24]  # Right hip
            shoulder_l = landmarks[11]  # Left shoulder
            shoulder_r = landmarks[12]  # Right shoulder
            ankle_l = landmarks[27]  # Left ankle
            ankle_r = landmarks[28]  # Right ankle
            
            # Calcola il centro di massa approssimativo
            com_x = np.mean([hip_l[0], hip_r[0], shoulder_l[0], shoulder_r[0]])
            com_y = np.mean([hip_l[1], hip_r[1], shoulder_l[1], shoulder_r[1]])
            
            # Calcola la base di supporto
            support_width = np.abs(ankle_r[0] - ankle_l[0])
            
            # Calcola la deviazione del COM dalla linea mediana
            midline = (ankle_l[0] + ankle_r[0]) / 2
            com_deviation = np.abs(com_x - midline)
            
            # Normalizza la deviazione rispetto alla base di supporto
            balance_score = 1.0 - (com_deviation / (support_width / 2))
            
            return float(np.clip(balance_score, 0, 1))
            
        except Exception as e:
            logger.error(f"Error calculating balance score: {str(e)}")
            return 0.0
            
    def _calculate_joint_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calcola l'angolo tra tre punti (landmarks)"""
        try:
            # Converti in vettori
            v1 = p1 - p2
            v2 = p3 - p2
            
            # Calcola l'angolo usando il prodotto scalare
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            
            return float(np.degrees(angle))
            
        except Exception as e:
            logger.error(f"Error calculating joint angle: {str(e)}")
            return 0.0
            
    def analyze_movement(self, landmarks: np.ndarray) -> Dict[str, Any]:
        """Analizza il movimento del corpo"""
        try:
            # Calcola angoli delle articolazioni principali
            angles = {
                'left_knee': self._calculate_joint_angle(
                    landmarks[23],  # Left hip
                    landmarks[25],  # Left knee
                    landmarks[27]   # Left ankle
                ),
                'right_knee': self._calculate_joint_angle(
                    landmarks[24],  # Right hip
                    landmarks[26],  # Right knee
                    landmarks[28]   # Right ankle
                ),
                'left_hip': self._calculate_joint_angle(
                    landmarks[11],  # Left shoulder
                    landmarks[23],  # Left hip
                    landmarks[25]   # Left knee
                ),
                'right_hip': self._calculate_joint_angle(
                    landmarks[12],  # Right shoulder
                    landmarks[24],  # Right hip
                    landmarks[26]   # Right knee
                ),
                'left_elbow': self._calculate_joint_angle(
                    landmarks[11],  # Left shoulder
                    landmarks[13],  # Left elbow
                    landmarks[15]   # Left wrist
                ),
                'right_elbow': self._calculate_joint_angle(
                    landmarks[12],  # Right shoulder
                    landmarks[14],  # Right elbow
                    landmarks[16]   # Right wrist
                )
            }
            
            # Calcola punteggio di equilibrio
            balance = self._calculate_balance_score(landmarks)
            
            # Calcola velocità del movimento (se disponibile storico)
            velocity = self._calculate_velocity(landmarks)
            
            return {
                'joint_angles': angles,
                'balance_score': balance,
                'velocity': velocity,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing movement: {str(e)}")
            return {}

    def analyze_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Analizza un frame per rilevare la postura"""
        try:
            if frame is None or frame.size == 0:
                raise ValueError("Invalid frame")
                
            # Converti in RGB se necessario
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
                
            # Rileva i landmark della postura
            results = self.pose.process(frame)
            
            if not results.pose_landmarks:
                return {
                    'balance_score': 0.0,
                    'joint_angles': {},
                    'movement_score': 0.0,
                    'confidence': 0.0,
                    'timestamp': datetime.now().isoformat()
                }
                
            # Estrai landmark
            landmarks = results.pose_landmarks.landmark
            
            # Calcola metriche
            balance_score = self._calculate_balance_score(landmarks)
            joint_angles = self._calculate_joint_angles(landmarks)
            movement_score = self.analyze_movement(landmarks)
            
            metrics = {
                'balance_score': float(balance_score),
                'joint_angles': joint_angles,
                'movement_score': float(movement_score),
                'confidence': 1.0 if landmarks else 0.0,
                'timestamp': datetime.now().isoformat()
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error analyzing frame: {str(e)}")
            raise
