import unittest
import numpy as np
import cv2
import mediapipe as mp
from typing import Dict, List, Tuple
import sys
from pathlib import Path
import logging

# Aggiungi il percorso src al PYTHONPATH
src_path = Path(__file__).parent.parent / 'src'
sys.path.append(str(src_path))

from facial_analysis.pupil_analyzer import PupilAnalyzer

class TestPupilAnalyzer(unittest.TestCase):
    """Test suite per PupilAnalyzer"""
    
    @classmethod
    def setUpClass(cls):
        """Setup una volta per tutti i test"""
        logging.basicConfig(level=logging.ERROR)
    
    def setUp(self):
        """Setup per ogni test"""
        try:
            self.analyzer = PupilAnalyzer()
        except RuntimeError as e:
            self.skipTest(f"MediaPipe non inizializzato correttamente: {e}")
            
        # Crea frame di test realistici
        self.test_frames = []
        for _ in range(5):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # Disegna un cerchio per simulare un occhio
            cv2.circle(frame, (320, 240), 50, (255, 255, 255), -1)
            # Disegna un cerchio più piccolo per simulare l'iride
            cv2.circle(frame, (320, 240), 20, (128, 128, 128), -1)
            self.test_frames.append(frame)
        
    def test_initialization(self):
        """Test inizializzazione"""
        self.assertIsNotNone(self.analyzer.face_mesh)
        self.assertIsNotNone(getattr(self.analyzer, 'mp_face_mesh', None))
        self.assertIsNotNone(getattr(self.analyzer, 'mp_drawing', None))
        
    def test_frame_analysis(self):
        """Test analisi frame"""
        for frame in self.test_frames:
            results = self.analyzer.analyze_frame(frame)
            
            self.assertIsInstance(results, dict)
            self.assertIn("left_pupil", results)
            self.assertIn("right_pupil", results)
            self.assertIn("confidence", results)
            
            # Verifica che i valori siano nel range corretto
            self.assertGreaterEqual(results["confidence"], 0.0)
            self.assertLessEqual(results["confidence"], 1.0)
            
    def test_invalid_frame(self):
        """Test con frame invalido"""
        # Test con frame vuoto
        empty_frame = np.array([])
        results = self.analyzer.analyze_frame(empty_frame)
        self.assertIsInstance(results, dict)
        self.assertEqual(results["confidence"], 0.0)
        
        # Test con frame di dimensioni errate
        invalid_frame = np.zeros((10, 10), dtype=np.uint8)
        results = self.analyzer.analyze_frame(invalid_frame)
        self.assertIsInstance(results, dict)
        self.assertEqual(results["confidence"], 0.0)
        
    def test_pupil_metrics(self):
        """Test metriche pupille"""
        frame = self.test_frames[0]
        results = self.analyzer.analyze_frame(frame)
        
        for pupil in ["left_pupil", "right_pupil"]:
            metrics = results[pupil]
            self.assertIsInstance(metrics, dict)
            self.assertIn("size", metrics)
            self.assertIn("ratio", metrics)
            self.assertGreaterEqual(metrics["size"], 0.0)
            self.assertGreaterEqual(metrics["ratio"], 0.0)
            self.assertLessEqual(metrics["ratio"], 1.0)
            
    def test_pupil_detection(self):
        """Test rilevamento pupille"""
        frame = self.test_frames[0]
        results = self.analyzer.analyze_frame(frame)
        
        left_pupil = results["left_pupil"]
        right_pupil = results["right_pupil"]
        
        self.assertIsInstance(left_pupil, dict)
        self.assertIsInstance(right_pupil, dict)
        
        for pupil in [left_pupil, right_pupil]:
            self.assertIn("center", pupil)
            self.assertIn("size", pupil)
            self.assertIn("confidence", pupil)
            
    def test_pupil_tracking(self):
        """Test tracking pupille"""
        movements = []
        
        for frame in self.test_frames:
            results = self.analyzer.analyze_frame(frame)
            movement = self.analyzer.track_pupil_movement(results)
            movements.append(movement)
            
        self.assertEqual(len(movements), len(self.test_frames))
        
        for movement in movements:
            self.assertIsInstance(movement, dict)
            self.assertIn("left_velocity", movement)
            self.assertIn("right_velocity", movement)
            self.assertIn("gaze_direction", movement)
            
    def test_gaze_estimation(self):
        """Test stima direzione sguardo"""
        frame = self.test_frames[0]
        results = self.analyzer.analyze_frame(frame)
        
        gaze = self.analyzer.estimate_gaze(results)
        
        self.assertIsInstance(gaze, dict)
        self.assertIn("horizontal_angle", gaze)
        self.assertIn("vertical_angle", gaze)
        self.assertIn("confidence", gaze)
        
    def test_pupil_size(self):
        """Test dimensione pupille"""
        frame = self.test_frames[0]
        results = self.analyzer.analyze_frame(frame)
        
        sizes = self.analyzer.compute_pupil_size(results)
        
        self.assertIsInstance(sizes, dict)
        self.assertIn("left_size", sizes)
        self.assertIn("right_size", sizes)
        self.assertIn("average_size", sizes)
        
    def test_visualization(self):
        """Test visualizzazione"""
        frame = self.test_frames[0]
        results = self.analyzer.analyze_frame(frame)
        
        annotated = self.analyzer.draw_pupils(frame, results)
        
        self.assertIsInstance(annotated, np.ndarray)
        self.assertEqual(annotated.shape, frame.shape)
        
    def test_error_handling(self):
        """Test gestione errori"""
        # Frame vuoto
        with self.assertRaises(ValueError):
            self.analyzer.analyze_frame(np.array([]))
            
        # Frame dimensioni errate
        with self.assertRaises(ValueError):
            self.analyzer.analyze_frame(np.random.rand(100))
            
        # Frame tipo errato
        with self.assertRaises(ValueError):
            self.analyzer.analyze_frame([[1,2,3]])
            
    def test_confidence_scoring(self):
        """Test punteggi confidenza"""
        frame = self.test_frames[0]
        results = self.analyzer.analyze_frame(frame)
        
        confidence = self.analyzer.compute_confidence(results)
        
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0)
        self.assertLessEqual(confidence, 1)
        
    def test_pupil_normalization(self):
        """Test normalizzazione dimensioni pupille"""
        frame = self.test_frames[0]
        results = self.analyzer.analyze_frame(frame)
        
        normalized = self.analyzer.normalize_pupil_size(results)
        
        self.assertIsInstance(normalized, dict)
        self.assertIn("left_normalized", normalized)
        self.assertIn("right_normalized", normalized)
        
        for size in normalized.values():
            self.assertGreaterEqual(size, 0)
            self.assertLessEqual(size, 1)
            
    def test_temporal_filtering(self):
        """Test filtraggio temporale"""
        filtered_results = []
        
        for frame in self.test_frames:
            results = self.analyzer.analyze_frame(frame)
            filtered = self.analyzer.apply_temporal_filter(results)
            filtered_results.append(filtered)
            
        self.assertEqual(len(filtered_results), len(self.test_frames))
        
    def test_pupil_metrics(self):
        """Test metriche pupille"""
        frame = self.test_frames[0]
        results = self.analyzer.analyze_frame(frame)
        
        metrics = self.analyzer.compute_pupil_metrics(results)
        
        self.assertIsInstance(metrics, dict)
        self.assertIn("dilation", metrics)
        self.assertIn("symmetry", metrics)
        self.assertIn("stability", metrics)
        
if __name__ == '__main__':
    unittest.main()
