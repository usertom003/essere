import unittest
import numpy as np
import cv2
import mediapipe as mp
from typing import Dict, List, Tuple

from src.facial_analysis.expression_analyzer import ExpressionAnalyzer

class TestExpressionAnalyzer(unittest.TestCase):
    """Test suite per ExpressionAnalyzer"""
    
    def setUp(self):
        """Setup per ogni test"""
        self.analyzer = ExpressionAnalyzer()
        self.test_frames = [np.random.rand(480, 640, 3) for _ in range(10)]
        
    def test_initialization(self):
        """Test inizializzazione"""
        self.assertIsInstance(self.analyzer.face_mesh, mp.solutions.face_mesh.FaceMesh)
        self.assertIsNotNone(self.analyzer.mp_face_mesh)
        self.assertIsNotNone(self.analyzer.mp_drawing)
        
    def test_frame_analysis(self):
        """Test analisi frame"""
        for frame in self.test_frames:
            results = self.analyzer.analyze_frame(frame)
            
            self.assertIsInstance(results, dict)
            self.assertIn("landmarks", results)
            self.assertIn("expressions", results)
            self.assertIn("confidence", results)
            
    def test_landmark_extraction(self):
        """Test estrazione landmarks"""
        frame = self.test_frames[0]
        results = self.analyzer.analyze_frame(frame)
        
        landmarks = self.analyzer.extract_landmarks(results)
        
        self.assertIsInstance(landmarks, dict)
        self.assertGreater(len(landmarks), 0)
        
        for name, coords in landmarks.items():
            self.assertIsInstance(name, str)
            self.assertIsInstance(coords, np.ndarray)
            self.assertEqual(coords.shape, (3,))
            
    def test_expression_classification(self):
        """Test classificazione espressioni"""
        frame = self.test_frames[0]
        results = self.analyzer.analyze_frame(frame)
        
        expressions = self.analyzer.classify_expressions(results)
        
        self.assertIsInstance(expressions, dict)
        self.assertGreater(len(expressions), 0)
        
        for expr, score in expressions.items():
            self.assertIsInstance(expr, str)
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 1)
            
    def test_facial_features(self):
        """Test estrazione features facciali"""
        frame = self.test_frames[0]
        results = self.analyzer.analyze_frame(frame)
        
        features = self.analyzer.extract_facial_features(results)
        
        self.assertIsInstance(features, dict)
        self.assertIn("eyes", features)
        self.assertIn("mouth", features)
        self.assertIn("eyebrows", features)
        
    def test_expression_tracking(self):
        """Test tracking espressioni"""
        expressions = []
        
        for frame in self.test_frames:
            results = self.analyzer.analyze_frame(frame)
            expr = self.analyzer.track_expressions(results)
            expressions.append(expr)
            
        self.assertEqual(len(expressions), len(self.test_frames))
        
        for expr in expressions:
            self.assertIsInstance(expr, dict)
            self.assertIn("current", expr)
            self.assertIn("previous", expr)
            self.assertIn("change", expr)
            
    def test_visualization(self):
        """Test visualizzazione"""
        frame = self.test_frames[0]
        results = self.analyzer.analyze_frame(frame)
        
        annotated = self.analyzer.draw_expressions(frame, results)
        
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
        
    def test_expression_normalization(self):
        """Test normalizzazione espressioni"""
        frame = self.test_frames[0]
        results = self.analyzer.analyze_frame(frame)
        
        normalized = self.analyzer.normalize_expressions(results)
        
        self.assertIsInstance(normalized, dict)
        self.assertGreater(len(normalized), 0)
        
        total = sum(normalized.values())
        self.assertAlmostEqual(total, 1.0, places=5)
        
    def test_temporal_filtering(self):
        """Test filtraggio temporale"""
        filtered_results = []
        
        for frame in self.test_frames:
            results = self.analyzer.analyze_frame(frame)
            filtered = self.analyzer.apply_temporal_filter(results)
            filtered_results.append(filtered)
            
        self.assertEqual(len(filtered_results), len(self.test_frames))
        
    def test_expression_metrics(self):
        """Test metriche espressioni"""
        frame = self.test_frames[0]
        results = self.analyzer.analyze_frame(frame)
        
        metrics = self.analyzer.compute_expression_metrics(results)
        
        self.assertIsInstance(metrics, dict)
        self.assertIn("intensity", metrics)
        self.assertIn("duration", metrics)
        self.assertIn("variability", metrics)
        
if __name__ == '__main__':
    unittest.main()
