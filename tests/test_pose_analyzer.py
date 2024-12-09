import unittest
import numpy as np
import cv2
import mediapipe as mp
from typing import Dict, List, Tuple

from src.body_analysis.pose_analyzer import PoseAnalyzer

class TestPoseAnalyzer(unittest.TestCase):
    """Test suite per PoseAnalyzer"""
    
    def setUp(self):
        """Setup per ogni test"""
        self.analyzer = PoseAnalyzer()
        self.test_frames = [np.random.rand(480, 640, 3) for _ in range(10)]
        
    def test_initialization(self):
        """Test inizializzazione"""
        self.assertIsInstance(self.analyzer.pose, mp.solutions.pose.Pose)
        self.assertIsNotNone(self.analyzer.mp_pose)
        self.assertIsNotNone(self.analyzer.mp_drawing)
        
    def test_frame_analysis(self):
        """Test analisi frame"""
        for frame in self.test_frames:
            results = self.analyzer.analyze_frame(frame)
            
            self.assertIsInstance(results, dict)
            self.assertIn("landmarks", results)
            self.assertIn("pose_landmarks", results)
            self.assertIn("world_landmarks", results)
            
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
            
    def test_pose_classification(self):
        """Test classificazione postura"""
        frame = self.test_frames[0]
        results = self.analyzer.analyze_frame(frame)
        
        pose_class = self.analyzer.classify_pose(results)
        
        self.assertIsInstance(pose_class, str)
        self.assertIn(pose_class, self.analyzer.pose_classes)
        
    def test_joint_angles(self):
        """Test calcolo angoli"""
        frame = self.test_frames[0]
        results = self.analyzer.analyze_frame(frame)
        
        angles = self.analyzer.compute_joint_angles(results)
        
        self.assertIsInstance(angles, dict)
        self.assertGreater(len(angles), 0)
        
        for joint, angle in angles.items():
            self.assertIsInstance(joint, str)
            self.assertIsInstance(angle, float)
            self.assertGreaterEqual(angle, 0)
            self.assertLessEqual(angle, 360)
            
    def test_movement_tracking(self):
        """Test tracking movimento"""
        movements = []
        
        for frame in self.test_frames:
            results = self.analyzer.analyze_frame(frame)
            movement = self.analyzer.track_movement(results)
            movements.append(movement)
            
        self.assertEqual(len(movements), len(self.test_frames))
        
        for movement in movements:
            self.assertIsInstance(movement, dict)
            self.assertIn("velocity", movement)
            self.assertIn("acceleration", movement)
            
    def test_pose_visualization(self):
        """Test visualizzazione postura"""
        frame = self.test_frames[0]
        results = self.analyzer.analyze_frame(frame)
        
        annotated = self.analyzer.draw_pose(frame, results)
        
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
        
    def test_pose_normalization(self):
        """Test normalizzazione postura"""
        frame = self.test_frames[0]
        results = self.analyzer.analyze_frame(frame)
        
        normalized = self.analyzer.normalize_pose(results)
        
        self.assertIsInstance(normalized, dict)
        self.assertGreater(len(normalized), 0)
        
        for landmark in normalized.values():
            self.assertTrue(np.all(landmark >= -1))
            self.assertTrue(np.all(landmark <= 1))
            
    def test_temporal_filtering(self):
        """Test filtraggio temporale"""
        filtered_results = []
        
        for frame in self.test_frames:
            results = self.analyzer.analyze_frame(frame)
            filtered = self.analyzer.apply_temporal_filter(results)
            filtered_results.append(filtered)
            
        self.assertEqual(len(filtered_results), len(self.test_frames))
        
    def test_pose_metrics(self):
        """Test metriche postura"""
        frame = self.test_frames[0]
        results = self.analyzer.analyze_frame(frame)
        
        metrics = self.analyzer.compute_pose_metrics(results)
        
        self.assertIsInstance(metrics, dict)
        self.assertIn("symmetry", metrics)
        self.assertIn("stability", metrics)
        self.assertIn("smoothness", metrics)
        
if __name__ == '__main__':
    unittest.main()
