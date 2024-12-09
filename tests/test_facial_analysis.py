import unittest
import numpy as np
import cv2
import sys
from pathlib import Path
import logging

# Aggiungi il percorso src al PYTHONPATH
src_path = Path(__file__).parent.parent / 'src'
sys.path.append(str(src_path))

from facial_analysis.expression_analyzer import ExpressionAnalyzer
from facial_analysis.pupil_analyzer import PupilAnalyzer

class TestExpressionAnalyzer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Setup iniziale per tutti i test"""
        cls.test_frames = [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            for _ in range(5)
        ]
        
    def setUp(self):
        """Setup per ogni singolo test"""
        self.analyzer = ExpressionAnalyzer()
        self.test_frame = self.test_frames[0]
        
    def tearDown(self):
        """Cleanup dopo ogni test"""
        if hasattr(self, 'analyzer'):
            # Cleanup delle risorse MediaPipe
            self.analyzer.face_mesh.close()
        
    def test_initialization(self):
        """Verifica corretta inizializzazione"""
        self.assertIsNotNone(self.analyzer.face_mesh)
        
    def test_eye_aspect_ratio(self):
        """Test EAR calculation"""
        eye_points = [
            (0.0, 0.0), (1.0, 1.0), (2.0, 0.0),
            (4.0, 0.0), (3.0, 1.0), (2.0, 0.0)
        ]
        ear = self.analyzer.calculate_eye_aspect_ratio(eye_points)
        self.assertGreater(ear, 0)
        
    def test_invalid_eye_points(self):
        """Test gestione punti occhio non validi"""
        eye_points = [(0.0, 0.0), (1.0, 1.0)]  # Convert to float
        ear = self.analyzer.calculate_eye_aspect_ratio(eye_points)
        self.assertEqual(ear, 0.0)

class TestPupilAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = PupilAnalyzer()
        self.test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        # Disegna un cerchio nero su sfondo bianco per simulare una pupilla
        cv2.circle(self.test_image, (320, 240), 10, (255, 255, 255), -1)
        
    def test_initialization(self):
        """Verifica corretta inizializzazione"""
        self.assertIsNotNone(self.analyzer)
        
    def test_pupil_detection(self):
        """Test rilevamento pupilla"""
        left_eye = self.test_image[220:260, 300:340]
        size = self.analyzer.measure_pupil_size(left_eye)
        self.assertGreater(size, 0)
        
    def test_invalid_eye_region(self):
        """Test gestione regione occhio non valida"""
        invalid_eye = np.zeros((0, 0, 3), dtype=np.uint8)
        size = self.analyzer.measure_pupil_size(invalid_eye)
        self.assertEqual(size, 0.0)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    unittest.main()
