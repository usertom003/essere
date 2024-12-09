import unittest
import numpy as np
import cv2
import sys
from pathlib import Path
import logging

# Aggiungi il percorso src al PYTHONPATH
src_path = Path(__file__).parent.parent / 'src'
sys.path.append(str(src_path))

from body_analysis.pose_analyzer import PoseAnalyzer, JointAngles, BodyMetrics

class TestPoseAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = PoseAnalyzer()
        self.test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
    def test_initialization(self):
        """Verifica corretta inizializzazione"""
        self.assertIsNotNone(self.analyzer.pose)
        self.assertIsNotNone(self.analyzer.movement_model)
        
    def test_joint_angles_calculation(self):
        """Test calcolo angoli articolazioni"""
        # Simula 3 punti per un angolo del gomito
        p1 = np.array([0, 0, 0])  # spalla
        p2 = np.array([1, 0, 0])  # gomito
        p3 = np.array([2, 1, 0])  # polso
        angle = self.analyzer._calculate_joint_angle(p1, p2, p3)
        self.assertGreater(angle, 0)
        self.assertLess(angle, 180)
        
    def test_movement_analysis(self):
        """Test analisi movimento"""
        # Genera sequenza di pose casuali
        sequence = np.random.rand(1, 100, 99)  # (batch, time_steps, features)
        movement = self.analyzer.analyze_movement(sequence)
        self.assertIsNotNone(movement)
        
    def test_balance_score(self):
        """Test calcolo punteggio equilibrio"""
        # Simula landmarks del corpo
        landmarks = np.random.rand(33, 3)  # 33 punti con coordinate x,y,z
        score = self.analyzer._calculate_balance_score(landmarks)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)
        
    def test_invalid_landmarks(self):
        """Test gestione landmarks non validi"""
        invalid_landmarks = np.array([])
        score = self.analyzer._calculate_balance_score(invalid_landmarks)
        self.assertEqual(score, 0.0)

class TestBodyMetrics(unittest.TestCase):
    def setUp(self):
        self.joint_angles = JointAngles(
            left_elbow=90.0,
            right_elbow=90.0,
            left_knee=180.0,
            right_knee=180.0,
            left_hip=90.0,
            right_hip=90.0,
            left_shoulder=0.0,
            right_shoulder=0.0
        )
        
    def test_joint_angles_validation(self):
        """Test validazione angoli articolazioni"""
        self.assertGreaterEqual(self.joint_angles.left_elbow, 0)
        self.assertLessEqual(self.joint_angles.left_elbow, 180)
        self.assertGreaterEqual(self.joint_angles.right_knee, 0)
        self.assertLessEqual(self.joint_angles.right_knee, 180)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    unittest.main()
