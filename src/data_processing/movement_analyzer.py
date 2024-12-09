import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from scipy.signal import find_peaks, butter, filtfilt
from scipy.stats import entropy
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import json

logger = logging.getLogger(__name__)

@dataclass
class MovementMetrics:
    smoothness: float
    complexity: float
    periodicity: float
    symmetry: float
    coordination: float
    energy: float
    variability: Dict[str, float]
    phase_coherence: float

class MovementAnalyzer:
    def __init__(self):
        self.movement_history: List[np.ndarray] = []
        self.metrics_history: List[MovementMetrics] = []
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=3)
        
    def update_history(self, landmarks):
        """Update movement history with new landmarks"""
        features = np.array([[l.x, l.y, l.z] for l in landmarks]).flatten()
        self.movement_history.append(features)
        
        # Keep last 300 frames (10 seconds at 30fps)
        if len(self.movement_history) > 300:
            self.movement_history.pop(0)
            
    def calculate_smoothness(self, movement_data: np.ndarray) -> float:
        """Calculate movement smoothness using spectral arc length"""
        try:
            # Calculate velocity
            velocity = np.diff(movement_data, axis=0)
            
            # Normalize velocity
            velocity_norm = np.linalg.norm(velocity, axis=1)
            velocity_norm = velocity_norm / np.max(velocity_norm)
            
            # Calculate spectral arc length
            freq = np.fft.fftfreq(len(velocity_norm))
            spectrum = np.fft.fft(velocity_norm)
            
            # Calculate arc length in frequency domain
            sal = -np.sum(np.sqrt(1 + np.diff(np.abs(spectrum))**2))
            
            return float(sal)
            
        except Exception as e:
            logger.error(f"Error calculating smoothness: {str(e)}")
            return 0.0
            
    def calculate_complexity(self, movement_data: np.ndarray) -> float:
        """Calculate movement complexity using approximate entropy"""
        try:
            # Calculate velocity magnitude
            velocity = np.linalg.norm(np.diff(movement_data, axis=0), axis=1)
            
            # Calculate approximate entropy
            m = 2  # embedding dimension
            r = 0.2 * np.std(velocity)  # threshold
            
            def phi(m):
                patterns = np.array([velocity[i:i+m] for i in range(len(velocity)-m+1)])
                distances = np.abs(patterns[:, None] - patterns)
                return np.mean(np.log(np.mean(distances <= r, axis=1)))
                
            return float(abs(phi(m+1) - phi(m)))
            
        except Exception as e:
            logger.error(f"Error calculating complexity: {str(e)}")
            return 0.0
            
    def calculate_periodicity(self, movement_data: np.ndarray) -> float:
        """Calculate movement periodicity using autocorrelation"""
        try:
            # Calculate velocity
            velocity = np.diff(movement_data, axis=0)
            
            # Calculate autocorrelation
            n = len(velocity)
            velocity_norm = velocity - np.mean(velocity, axis=0)
            autocorr = np.correlate(velocity_norm.flatten(), velocity_norm.flatten(), mode='full')[n-1:]
            autocorr = autocorr / autocorr[0]
            
            # Find peaks in autocorrelation
            peaks, _ = find_peaks(autocorr, distance=15)
            
            if len(peaks) > 1:
                # Calculate mean peak height as periodicity measure
                periodicity = np.mean(autocorr[peaks])
            else:
                periodicity = 0.0
                
            return float(periodicity)
            
        except Exception as e:
            logger.error(f"Error calculating periodicity: {str(e)}")
            return 0.0
            
    def calculate_symmetry(self, landmarks) -> float:
        """Calculate movement symmetry between left and right body parts"""
        try:
            # Define corresponding left-right pairs
            pairs = [
                (11, 12),  # shoulders
                (13, 14),  # elbows
                (15, 16),  # wrists
                (23, 24),  # hips
                (25, 26),  # knees
                (27, 28)   # ankles
            ]
            
            symmetry_scores = []
            for left_idx, right_idx in pairs:
                left_point = np.array([
                    landmarks[left_idx].x,
                    landmarks[left_idx].y,
                    landmarks[left_idx].z
                ])
                right_point = np.array([
                    landmarks[right_idx].x,
                    landmarks[right_idx].y,
                    landmarks[right_idx].z
                ])
                
                # Calculate symmetry score for this pair
                distance = np.linalg.norm(left_point - right_point)
                max_dist = max(np.linalg.norm(left_point), np.linalg.norm(right_point))
                symmetry_scores.append(1.0 - (distance / (2 * max_dist)))
                
            return float(np.mean(symmetry_scores))
            
        except Exception as e:
            logger.error(f"Error calculating symmetry: {str(e)}")
            return 0.0
            
    def calculate_coordination(self, movement_data: np.ndarray) -> float:
        """Calculate movement coordination using phase coherence"""
        try:
            if len(movement_data) < 3:
                return 0.0
                
            # Calculate joint angles over time
            angles = []
            for i in range(0, len(movement_data[0]), 3):
                point1 = movement_data[:, i:i+3]
                point2 = movement_data[:, (i+3):(i+6)]
                if i+6 < len(movement_data[0]):
                    point3 = movement_data[:, (i+6):(i+9)]
                    # Calculate angle between three points
                    v1 = point1 - point2
                    v2 = point3 - point2
                    angle = np.arccos(np.sum(v1*v2, axis=1) / 
                                    (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)))
                    angles.append(angle)
                    
            if not angles:
                return 0.0
                
            # Calculate phase coherence between angle pairs
            coherence_scores = []
            angles = np.array(angles)
            for i in range(len(angles)):
                for j in range(i+1, len(angles)):
                    # Calculate phase difference
                    phase_diff = np.angle(np.exp(1j * (angles[i] - angles[j])))
                    # Calculate phase coherence
                    coherence = np.abs(np.mean(np.exp(1j * phase_diff)))
                    coherence_scores.append(coherence)
                    
            return float(np.mean(coherence_scores))
            
        except Exception as e:
            logger.error(f"Error calculating coordination: {str(e)}")
            return 0.0
            
    def calculate_energy(self, movement_data: np.ndarray) -> float:
        """Calculate movement energy using kinetic energy approximation"""
        try:
            # Calculate velocity
            velocity = np.diff(movement_data, axis=0)
            
            # Calculate kinetic energy (proportional to velocity squared)
            energy = np.mean(np.sum(velocity**2, axis=1))
            
            return float(energy)
            
        except Exception as e:
            logger.error(f"Error calculating energy: {str(e)}")
            return 0.0
            
    def calculate_variability(self, movement_data: np.ndarray) -> Dict[str, float]:
        """Calculate movement variability metrics"""
        try:
            variability = {}
            
            # Calculate velocity and acceleration
            velocity = np.diff(movement_data, axis=0)
            acceleration = np.diff(velocity, axis=0)
            
            # Calculate standard deviation of various metrics
            variability['position_std'] = float(np.std(movement_data))
            variability['velocity_std'] = float(np.std(velocity))
            variability['acceleration_std'] = float(np.std(acceleration))
            
            # Calculate entropy of movement distribution
            hist, _ = np.histogram(movement_data.flatten(), bins=20)
            variability['position_entropy'] = float(entropy(hist))
            
            return variability
            
        except Exception as e:
            logger.error(f"Error calculating variability: {str(e)}")
            return {}
            
    def analyze_movement(self, landmarks) -> Optional[MovementMetrics]:
        """Main method to analyze movement patterns"""
        try:
            # Update history
            self.update_history(landmarks)
            
            if len(self.movement_history) < 3:
                return None
                
            # Convert history to numpy array
            movement_data = np.array(self.movement_history)
            
            # Calculate all metrics
            smoothness = self.calculate_smoothness(movement_data)
            complexity = self.calculate_complexity(movement_data)
            periodicity = self.calculate_periodicity(movement_data)
            symmetry = self.calculate_symmetry(landmarks)
            coordination = self.calculate_coordination(movement_data)
            energy = self.calculate_energy(movement_data)
            variability = self.calculate_variability(movement_data)
            
            # Calculate phase coherence across all landmarks
            phase_coherence = self.calculate_coordination(movement_data)
            
            metrics = MovementMetrics(
                smoothness=smoothness,
                complexity=complexity,
                periodicity=periodicity,
                symmetry=symmetry,
                coordination=coordination,
                energy=energy,
                variability=variability,
                phase_coherence=phase_coherence
            )
            
            # Update metrics history
            self.metrics_history.append(metrics)
            if len(self.metrics_history) > 300:
                self.metrics_history.pop(0)
                
            return metrics
            
        except Exception as e:
            logger.error(f"Error analyzing movement: {str(e)}")
            return None
            
    def save_analysis(self, path: str):
        """Save analysis results to file"""
        try:
            analysis_data = {
                'metrics_history': [
                    {
                        'smoothness': m.smoothness,
                        'complexity': m.complexity,
                        'periodicity': m.periodicity,
                        'symmetry': m.symmetry,
                        'coordination': m.coordination,
                        'energy': m.energy,
                        'variability': m.variability,
                        'phase_coherence': m.phase_coherence
                    }
                    for m in self.metrics_history
                ]
            }
            
            with open(path, 'w') as f:
                json.dump(analysis_data, f, indent=2)
                
            logger.info(f"Analysis results saved to {path}")
            
        except Exception as e:
            logger.error(f"Error saving analysis results: {str(e)}")
            raise
            
    def load_analysis(self, path: str):
        """Load analysis results from file"""
        try:
            with open(path, 'r') as f:
                analysis_data = json.load(f)
                
            self.metrics_history = [
                MovementMetrics(**metrics)
                for metrics in analysis_data['metrics_history']
            ]
            
            logger.info(f"Analysis results loaded from {path}")
            
        except Exception as e:
            logger.error(f"Error loading analysis results: {str(e)}")
            raise
