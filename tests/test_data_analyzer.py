import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.data_processing.data_analyzer import DataAnalyzer

class TestDataAnalyzer(unittest.TestCase):
    """Test suite per DataAnalyzer"""
    
    def setUp(self):
        """Setup per ogni test"""
        self.analyzer = DataAnalyzer()
        self.test_frames = [np.random.rand(480, 640, 3) for _ in range(10)]
        self.test_features = np.random.rand(100, 7)
        self.timestamps = [
            datetime.now() + timedelta(seconds=i)
            for i in range(100)
        ]
        
    def test_feature_extraction(self):
        """Test estrazione features"""
        features = self.analyzer.get_feature_matrix(self.test_frames)
        
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(features.shape[0], len(self.test_frames))
        self.assertEqual(features.shape[1], 7)
        self.assertTrue(np.all(features >= 0))
        self.assertTrue(np.all(features <= 1))
        
    def test_temporal_analysis(self):
        """Test analisi temporale"""
        patterns = self.analyzer.analyze_temporal_patterns(self.test_features)
        
        self.assertIsInstance(patterns, dict)
        self.assertIn("trends", patterns)
        self.assertIn("cycles", patterns)
        self.assertIn("anomalies", patterns)
        
        trends = patterns["trends"]
        self.assertEqual(len(trends), self.test_features.shape[1])
        
    def test_correlation_analysis(self):
        """Test analisi correlazioni"""
        correlations = self.analyzer.compute_correlations(self.test_features)
        
        self.assertIsInstance(correlations, np.ndarray)
        self.assertEqual(correlations.shape, (7,7))
        self.assertTrue(np.all(correlations >= -1))
        self.assertTrue(np.all(correlations <= 1))
        
    def test_feature_importance(self):
        """Test importanza features"""
        importance = self.analyzer.compute_feature_importance(
            self.test_features,
            np.random.randint(0, 2, 100)
        )
        
        self.assertIsInstance(importance, np.ndarray)
        self.assertEqual(len(importance), 7)
        self.assertTrue(np.all(importance >= 0))
        
    def test_anomaly_detection(self):
        """Test rilevamento anomalie"""
        anomalies = self.analyzer.detect_anomalies(self.test_features)
        
        self.assertIsInstance(anomalies, list)
        self.assertTrue(all(isinstance(a, dict) for a in anomalies))
        
        for anomaly in anomalies:
            self.assertIn("index", anomaly)
            self.assertIn("feature", anomaly)
            self.assertIn("score", anomaly)
            
    def test_pattern_mining(self):
        """Test mining pattern"""
        patterns = self.analyzer.mine_patterns(self.test_features)
        
        self.assertIsInstance(patterns, list)
        self.assertTrue(all(isinstance(p, dict) for p in patterns))
        
        for pattern in patterns:
            self.assertIn("features", pattern)
            self.assertIn("support", pattern)
            self.assertIn("confidence", pattern)
            
    def test_dimensionality_reduction(self):
        """Test riduzione dimensionalità"""
        reduced = self.analyzer.reduce_dimensions(self.test_features, n_components=3)
        
        self.assertIsInstance(reduced, np.ndarray)
        self.assertEqual(reduced.shape[0], self.test_features.shape[0])
        self.assertEqual(reduced.shape[1], 3)
        
    def test_feature_clustering(self):
        """Test clustering features"""
        clusters = self.analyzer.cluster_features(self.test_features, n_clusters=3)
        
        self.assertIsInstance(clusters, np.ndarray)
        self.assertEqual(len(clusters), self.test_features.shape[0])
        self.assertTrue(np.all(clusters >= 0))
        self.assertTrue(np.all(clusters < 3))
        
    def test_time_series_analysis(self):
        """Test analisi serie temporali"""
        forecast = self.analyzer.forecast_features(
            self.test_features,
            self.timestamps,
            horizon=10
        )
        
        self.assertIsInstance(forecast, np.ndarray)
        self.assertEqual(forecast.shape[1], self.test_features.shape[1])
        self.assertEqual(forecast.shape[0], 10)
        
    def test_feature_selection(self):
        """Test selezione features"""
        selected = self.analyzer.select_features(
            self.test_features,
            np.random.randint(0, 2, 100),
            k=5
        )
        
        self.assertIsInstance(selected, np.ndarray)
        self.assertEqual(selected.shape[0], self.test_features.shape[0])
        self.assertEqual(selected.shape[1], 5)
        
if __name__ == '__main__':
    unittest.main()
