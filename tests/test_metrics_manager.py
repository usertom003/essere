import unittest
import numpy as np
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List
import tempfile
from pathlib import Path

from src.data_processing.metrics_manager import MetricsManager
from src.facial_analysis.expression_analyzer import ExpressionAnalyzer
from src.facial_analysis.pupil_analyzer import PupilAnalyzer
from src.body_analysis.pose_analyzer import PoseAnalyzer
from test_base import BiometricTestBase

class TestMetricsManager(BiometricTestBase):
    """Test suite per MetricsManager"""
    
    def setUp(self):
        """Setup per ogni test"""
        super().setUp()
        self.manager = self.metrics_manager  # Use the one from BiometricTestBase
        self.test_metrics = {
            'timestamp': datetime.now().isoformat(),
            'confidence': 0.95,
            'expressions': {'smile': 0.8, 'blink': 0.2},
            'pupil_changes': {'left': 0.1, 'right': 0.1},
            'raw_metrics': {'ear': 0.3, 'mar': 0.5}
        }
        
        # Componenti di analisi
        self.expression_analyzer = ExpressionAnalyzer()
        self.pupil_analyzer = PupilAnalyzer()
        self.pose_analyzer = PoseAnalyzer()
        
    def test_metrics_collection(self):
        """Test raccolta metriche"""
        frame = self.test_frames[0]
        
        metrics = self.manager.collect_metrics(
            frame,
            self.expression_analyzer,
            self.pupil_analyzer,
            self.pose_analyzer
        )
        
        self.assertIsInstance(metrics, dict)
        self.assertIn("expressions", metrics)
        self.assertIn("pupils", metrics)
        self.assertIn("poses", metrics)
        self.assertIn("timestamp", metrics)
        
    def test_metrics_validation(self):
        """Test validazione metriche"""
        # Test metrica valida
        self.metrics_manager.add_metrics(self.test_metrics)
        self.assertEqual(len(self.metrics_manager.metrics_history), 1)
        
        # Test formato invalido
        with self.assertRaises(ValueError):
            invalid_metrics: Dict = []  # type: ignore
            self.metrics_manager.add_metrics(invalid_metrics)
            
        # Test timestamp invalido
        invalid_metrics = self.test_metrics.copy()
        invalid_metrics['timestamp'] = "invalid_date"
        with self.assertRaises(ValueError):
            self.metrics_manager.add_metrics(invalid_metrics)
            
    def test_metrics_aggregation(self):
        """Test aggregazione metriche"""
        metrics_list = []
        
        for frame in self.test_frames:
            metrics = self.manager.collect_metrics(
                frame,
                self.expression_analyzer,
                self.pupil_analyzer,
                self.pose_analyzer
            )
            metrics_list.append(metrics)
            
        aggregated = self.manager.aggregate_metrics(metrics_list)
        
        self.assertIsInstance(aggregated, dict)
        self.assertIn("expressions", aggregated)
        self.assertIn("pupils", aggregated)
        self.assertIn("poses", aggregated)
        
    def test_metrics_persistence(self):
        """Test persistenza metriche"""
        metrics = self.manager.collect_metrics(
            self.test_frames[0],
            self.expression_analyzer,
            self.pupil_analyzer,
            self.pose_analyzer
        )
        
        # Salva metriche
        filename = "test_metrics.json"
        self.manager.save_metrics(metrics, filename)
        
        # Verifica file
        self.assertTrue(os.path.exists(filename))
        
        # Carica metriche
        loaded = self.manager.load_metrics(filename)
        self.assertEqual(metrics, loaded)
        
        # Pulisci
        os.remove(filename)
        
    def test_metrics_filtering(self):
        """Test filtraggio metriche"""
        metrics_list = []
        
        for frame in self.test_frames:
            metrics = self.manager.collect_metrics(
                frame,
                self.expression_analyzer,
                self.pupil_analyzer,
                self.pose_analyzer
            )
            metrics_list.append(metrics)
            
        filtered = self.manager.filter_metrics(
            metrics_list,
            min_confidence=0.5
        )
        
        self.assertIsInstance(filtered, list)
        self.assertLessEqual(len(filtered), len(metrics_list))
        
    def test_temporal_analysis(self):
        """Test analisi temporale"""
        metrics_list = []
        
        for frame in self.test_frames:
            metrics = self.manager.collect_metrics(
                frame,
                self.expression_analyzer,
                self.pupil_analyzer,
                self.pose_analyzer
            )
            metrics_list.append(metrics)
            
        analysis = self.manager.analyze_temporal_patterns(metrics_list)
        
        self.assertIsInstance(analysis, dict)
        self.assertIn("trends", analysis)
        self.assertIn("changes", analysis)
        self.assertIn("frequencies", analysis)
        
    def test_error_handling(self):
        """Test gestione errori"""
        # Metriche invalide
        with self.assertRaises(ValueError):
            self.manager.validate_metrics(None)
            
        # File non esistente
        with self.assertRaises(FileNotFoundError):
            self.manager.load_metrics("nonexistent.json")
            
        # Formato file invalido
        with open("invalid.json", "w") as f:
            f.write("invalid json")
            
        with self.assertRaises(json.JSONDecodeError):
            self.manager.load_metrics("invalid.json")
            
        os.remove("invalid.json")
        
    def test_metrics_summary(self):
        """Test sommario metriche"""
        metrics_list = []
        
        for frame in self.test_frames:
            metrics = self.manager.collect_metrics(
                frame,
                self.expression_analyzer,
                self.pupil_analyzer,
                self.pose_analyzer
            )
            metrics_list.append(metrics)
            
        summary = self.manager.generate_summary(metrics_list)
        
        self.assertIsInstance(summary, dict)
        self.assertIn("start_time", summary)
        self.assertIn("end_time", summary)
        self.assertIn("total_frames", summary)
        self.assertIn("average_metrics", summary)
        
    def test_metrics_export(self):
        """Test esportazione metriche"""
        metrics_list = []
        
        for frame in self.test_frames:
            metrics = self.manager.collect_metrics(
                frame,
                self.expression_analyzer,
                self.pupil_analyzer,
                self.pose_analyzer
            )
            metrics_list.append(metrics)
            
        # CSV export
        csv_file = "test_metrics.csv"
        self.manager.export_to_csv(metrics_list, csv_file)
        self.assertTrue(os.path.exists(csv_file))
        os.remove(csv_file)
        
        # Excel export
        excel_file = "test_metrics.xlsx"
        self.manager.export_to_excel(metrics_list, excel_file)
        self.assertTrue(os.path.exists(excel_file))
        os.remove(excel_file)
        
    def test_realtime_stats(self):
        """Test statistiche real-time"""
        # Aggiungi multiple metriche
        for i in range(10):
            metrics = self.test_metrics.copy()
            metrics['expressions']['smile'] += np.random.normal(0, 0.1)
            metrics['pupil_changes']['left'] += np.random.normal(0, 0.05)
            self.metrics_manager.add_metrics(metrics)
            
        stats = self.metrics_manager.get_realtime_stats()
        
        self.assertIn('expression_intensity', stats)
        self.assertIn('pupil_activity', stats)
        self.assertIn('total_variation', stats)
        
    def test_recent_metrics(self):
        """Test recupero metriche recenti"""
        # Aggiungi metriche con timestamp diversi
        now = datetime.now()
        for minutes in range(10):
            metrics = self.test_metrics.copy()
            metrics['timestamp'] = (now - timedelta(minutes=minutes)).isoformat()
            self.metrics_manager.add_metrics(metrics)
            
        recent = self.metrics_manager.get_recent_metrics(minutes=5)
        self.assertEqual(len(recent), 6)  # 0-5 minuti
        
    def test_statistics_calculation(self):
        """Test calcolo statistiche"""
        # Aggiungi metriche con diverse emozioni e movimenti
        emotions = ['happy', 'neutral', 'sad']
        movements = ['still', 'walking', 'running']
        
        for emotion in emotions:
            for movement in movements:
                metrics = self.test_metrics.copy()
                metrics['emotion'] = emotion
                metrics['movement'] = movement
                self.metrics_manager.add_metrics(metrics)
                
        stats = self.metrics_manager.calculate_statistics()
        
        self.assertIn('emotion_distribution', stats)
        self.assertIn('movement_distribution', stats)
        self.assertEqual(sum(stats['emotion_distribution'].values()), 1.0)
        self.assertEqual(sum(stats['movement_distribution'].values()), 1.0)
        
    def test_session_persistence(self):
        """Test persistenza sessione"""
        # Aggiungi alcune metriche
        self.metrics_manager.add_metrics(self.test_metrics)
        
        # Test salvataggio
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_session.json"
            self.metrics_manager.save_session(str(filepath))
            
            # Verifica file salvato
            self.assertTrue(filepath.exists())
            
            # Crea nuovo manager e carica
            new_manager = type(self.metrics_manager)()
            success = new_manager.load_session(str(filepath))
            
            self.assertTrue(success)
            self.assertEqual(
                len(new_manager.metrics_history),
                len(self.metrics_manager.metrics_history)
            )
        
    def test_save_session(self):
        """Test salvataggio sessione"""
        filepath = "test_session.json"
        try:
            self.metrics_manager.save_session(filepath)
            self.assertTrue(Path(filepath).exists())
            
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            self.assertIn('metrics_history', data)
            self.assertIn('movement_patterns', data)
            self.assertIn('component_health', data)
            self.assertIn('timestamp', data)
            
        finally:
            if Path(filepath).exists():
                Path(filepath).unlink()
        
if __name__ == '__main__':
    unittest.main()
