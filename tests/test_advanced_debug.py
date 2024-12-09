import unittest
import numpy as np
import logging
import time
import psutil
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Tuple
from datetime import datetime
from pathlib import Path
import traceback
from unittest import TestCase

from src.data_processing.data_analyzer import DataAnalyzer
from src.data_processing.metrics_manager import MetricsManager
from tests.mocks import (
    MockExpressionAnalyzer, 
    MockPupilAnalyzer,
    MockPoseAnalyzer,
    MockResult
)
from src.ml_models.emotion_classifier import EmotionClassifier
from src.ml_models.movement_classifier import MovementClassifier

# Configurazione logging avanzato
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path('tests/logs/test_debug.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class SystemDebugger:
    """Classe per il debugging automatico del sistema"""
    
    def __init__(self):
        self.memory_snapshots = []
        self.performance_metrics = {}
        self.error_counts = {}
        self.component_status = {}
        
    def start_monitoring(self):
        """Avvia il monitoraggio delle risorse"""
        tracemalloc.start()
        self.start_time = time.time()
        self.initial_memory = psutil.Process().memory_info().rss
        
    def take_snapshot(self, label: str):
        """Cattura uno snapshot della memoria"""
        snapshot = tracemalloc.take_snapshot()
        self.memory_snapshots.append((label, snapshot))
        
    def analyze_memory_usage(self) -> List[str]:
        """Analizza l'utilizzo della memoria"""
        results = []
        for label, snapshot in self.memory_snapshots:
            top_stats = snapshot.statistics('lineno')
            results.append(f"\nMemory snapshot for {label}:")
            for stat in top_stats[:3]:
                results.append(f"{stat.count} blocks: {stat.size/1024:.1f} KiB")
        return results
        
    def log_error(self, component: str, error: Exception):
        """Registra un errore per componente"""
        if component not in self.error_counts:
            self.error_counts[component] = []
        self.error_counts[component].append({
            'error': str(error),
            'traceback': traceback.format_exc(),
            'timestamp': datetime.now().isoformat()
        })
        
    def get_system_health(self) -> Dict[str, Any]:
        """Restituisce lo stato di salute del sistema"""
        return {
            'memory_usage': psutil.Process().memory_info().rss / 1024 / 1024,
            'cpu_percent': psutil.Process().cpu_percent(),
            'error_counts': {k: len(v) for k, v in self.error_counts.items()},
            'uptime': time.time() - self.start_time
        }

class TestAdvancedDebug(TestCase):
    """Test suite avanzata con debugging automatico"""
    
    @classmethod
    def setUpClass(cls):
        """Setup for all tests"""
        cls.debugger = SystemDebugger()
        cls.debugger.start_monitoring()
        cls.metrics_manager = MetricsManager()
        
        # Initialize mock components
        cls.expression_analyzer = MockExpressionAnalyzer()
        cls.pupil_analyzer = MockPupilAnalyzer()
        cls.pose_analyzer = MockPoseAnalyzer()
        cls.emotion_classifier = EmotionClassifier()
        cls.movement_classifier = MovementClassifier()
        
        # Create test frame
        cls.test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
    def setUp(self):
        """Setup per ogni test"""
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss
        self.debugger.take_snapshot(f"Before {self._testMethodName}")
        
    def tearDown(self):
        """Cleanup dopo ogni test"""
        # Calcola metriche di performance
        execution_time = time.time() - self.start_time
        memory_used = (psutil.Process().memory_info().rss - self.start_memory) / 1024 / 1024
        
        logger.info(f"""
        Test: {self._testMethodName}
        Execution Time: {execution_time:.2f}s
        Memory Used: {memory_used:.2f}MB
        """)
        
        self.debugger.take_snapshot(f"After {self._testMethodName}")
        
    def test_concurrent_processing(self):
        """Test elaborazione concorrente"""
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for _ in range(10):
                futures.append(executor.submit(self.expression_analyzer.analyze_frame, self.test_frame))
                futures.append(executor.submit(self.pupil_analyzer.analyze_frame, self.test_frame))
                futures.append(executor.submit(self.pose_analyzer.analyze_frame, self.test_frame))
                
            results = [f.result() for f in futures]
            self.assertTrue(all(r is not None for r in results))
            
    def test_error_recovery(self):
        """Test recupero da errori"""
        # Simula un errore
        invalid_frame = np.zeros((1,1), dtype=np.uint8)
        
        try:
            self.expression_analyzer.analyze_frame(invalid_frame)
        except Exception as e:
            self.debugger.log_error('expression_analyzer', e)
            # Verifica che il sistema si riprenda
            valid_result = self.expression_analyzer.analyze_frame(self.test_frame)
            self.assertIsNotNone(valid_result)
            
    def test_memory_management(self):
        """Test gestione della memoria"""
        initial_memory = psutil.Process().memory_info().rss
        
        # Esegui operazioni intensive
        for _ in range(100):
            features = np.random.rand(1, 7)
            self.emotion_classifier.predict(features)
            
        final_memory = psutil.Process().memory_info().rss
        memory_diff = final_memory - initial_memory
        
        # Verifica che non ci siano memory leak significativi
        self.assertLess(memory_diff / initial_memory, 0.1)  # Max 10% incremento
        
    def test_system_stability(self):
        """Test stabilità del sistema"""
        start_health = self.debugger.get_system_health()
        
        # Esegui operazioni miste
        for _ in range(5):
            self.test_concurrent_processing()
            self.test_error_recovery()
            
        end_health = self.debugger.get_system_health()
        
        # Verifica stabilità
        self.assertLess(end_health['memory_usage'], start_health['memory_usage'] * 1.2)
        self.assertLess(len(self.debugger.error_counts), 3)  # Max 3 tipi di errori
        
    def test_edge_cases(self):
        """Test casi limite"""
        edge_cases = [
            np.zeros((480, 640, 3), dtype=np.uint8),  # Frame nero
            np.ones((480, 640, 3), dtype=np.uint8) * 255,  # Frame bianco
            np.random.rand(480, 640, 3) * 255  # Frame rumoroso
        ]
        
        for frame in edge_cases:
            try:
                facial_metrics = self.expression_analyzer.analyze_frame(frame)
                pupil_metrics = self.pupil_analyzer.analyze_frame(frame)
                pose_metrics = self.pose_analyzer.analyze_frame(frame)
                
                self.assertIsNotNone(facial_metrics)
                self.assertIsNotNone(pupil_metrics)
                self.assertIsNotNone(pose_metrics)
            except Exception as e:
                self.debugger.log_error('edge_case', e)
                self.fail(f"Edge case failed: {str(e)}")
                
    def test_data_consistency(self):
        """Test consistenza dei dati"""
        # Genera sequenza di metriche
        metrics_sequence = []
        for _ in range(10):
            metrics = {
                'facial_metrics': self.expression_analyzer.analyze_frame(self.test_frame),
                'pupil_metrics': self.pupil_analyzer.analyze_frame(self.test_frame),
                'pose_metrics': self.pose_analyzer.analyze_frame(self.test_frame),
                'timestamp': datetime.now().isoformat()
            }
            metrics_sequence.append(metrics)
            self.metrics_manager.add_metrics(metrics)
            
        # Verifica consistenza
        stats = self.metrics_manager.calculate_statistics()
        self.assertEqual(len(metrics_sequence), len(self.metrics_manager.metrics_history))
        self.assertIsNotNone(stats.get('avg_confidence'))
        
    @classmethod
    def tearDownClass(cls):
        """Cleanup finale"""
        memory_analysis = cls.debugger.analyze_memory_usage()
        logger.info("\n".join(memory_analysis))
        
        final_health = cls.debugger.get_system_health()
        logger.info(f"Final system health: {final_health}")
        
        cls.expression_analyzer.face_mesh.close()
        cls.pupil_analyzer.face_mesh.close()
        cls.pose_analyzer.pose.close()
        
if __name__ == '__main__':
    unittest.main()
