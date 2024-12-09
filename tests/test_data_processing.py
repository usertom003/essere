import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path
import logging

# Aggiungi il percorso src al PYTHONPATH
src_path = Path(__file__).parent.parent / 'src'
sys.path.append(str(src_path))

from data_processing.data_analyzer import DataAnalyzer
from data_processing.metrics_manager import MetricsManager

class TestDataAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = DataAnalyzer()
        self.test_metrics = {
            'expression': {
                'eye_aspect_ratio': 0.3,
                'mouth_aspect_ratio': 0.4,
                'eyebrow_position': 0.1,
                'nose_wrinkle': 0.2
            },
            'pupil': {
                'left_pupil_size': 3.0,
                'right_pupil_size': 3.1,
                'pupil_ratio': 0.97
            }
        }
        
    def test_add_metrics(self):
        """Test aggiunta metriche"""
        self.analyzer.add_metrics(self.test_metrics)
        self.assertEqual(len(self.analyzer.metrics_history), 1)
        
    def test_feature_matrix(self):
        """Test creazione matrice features"""
        self.analyzer.add_metrics(self.test_metrics)
        features = self.analyzer.get_feature_matrix()
        self.assertEqual(features.shape[1], 7)  # 4 expression + 3 pupil features
        
    def test_trend_analysis(self):
        """Test analisi trend"""
        # Aggiungi multiple metriche
        for _ in range(10):
            metrics = self.test_metrics.copy()
            metrics['expression']['eye_aspect_ratio'] += np.random.normal(0, 0.1)
            self.analyzer.add_metrics(metrics)
            
        trends = self.analyzer.analyze_trends(pd.DataFrame(self.analyzer.metrics_history))
        self.assertIn('trends', trends)
        self.assertIn('patterns', trends)
        
    def test_empty_metrics(self):
        """Test gestione metriche vuote"""
        empty_metrics = {}
        self.analyzer.add_metrics(empty_metrics)
        features = self.analyzer.get_feature_matrix()
        self.assertEqual(features.shape[1], 7)  # dovrebbe comunque avere 7 features con valori 0

class TestMetricsManager(unittest.TestCase):
    def setUp(self):
        self.manager = MetricsManager()
        self.test_metrics = {
            'timestamp': datetime.now(),
            'emotion': 'happy',
            'movement': 'walking',
            'confidence': 0.95
        }
        
    def test_add_metrics(self):
        """Test aggiunta metriche"""
        self.manager.add_metrics(self.test_metrics)
        self.assertEqual(len(self.manager.metrics_history), 1)
        
    def test_get_recent_metrics(self):
        """Test recupero metriche recenti"""
        # Aggiungi metriche con timestamp diversi
        for i in range(5):
            metrics = self.test_metrics.copy()
            metrics['timestamp'] = datetime.now() - timedelta(minutes=i)
            self.manager.add_metrics(metrics)
            
        recent = self.manager.get_recent_metrics(minutes=3)
        self.assertLessEqual(len(recent), 3)
        
    def test_calculate_statistics(self):
        """Test calcolo statistiche"""
        # Aggiungi multiple metriche
        for _ in range(10):
            metrics = self.test_metrics.copy()
            metrics['confidence'] = np.random.uniform(0.8, 1.0)
            self.manager.add_metrics(metrics)
            
        stats = self.manager.calculate_statistics()
        self.assertIn('emotion_distribution', stats)
        self.assertIn('movement_distribution', stats)
        self.assertIn('avg_confidence', stats)
        
    def test_invalid_metrics(self):
        """Test gestione metriche non valide"""
        invalid_metrics = {'timestamp': 'invalid'}
        with self.assertRaises(ValueError):
            self.manager.add_metrics(invalid_metrics)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    unittest.main()
