import unittest
import numpy as np
import time
import psutil
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
from datetime import datetime

from src.data_processing.data_analyzer import DataAnalyzer
from src.data_processing.metrics_manager import MetricsManager
from tests.mock_components import (
    MockExpressionAnalyzer, 
    MockPupilAnalyzer,
    MockPoseAnalyzer,
    MockEmotionClassifier,
    MockMovementClassifier
)
from src.utils.auto_debugger import AutoDebugger, ComponentHealth

class TestPerformance(unittest.TestCase):
    """Test suite per le prestazioni del sistema"""
    
    @classmethod
    def setUpClass(cls):
        """Setup iniziale"""
        cls.debugger = AutoDebugger()
        
        # Inizializza componenti
        cls.expression_analyzer = MockExpressionAnalyzer()
        cls.pupil_analyzer = MockPupilAnalyzer()
        cls.pose_analyzer = MockPoseAnalyzer()
        cls.emotion_classifier = MockEmotionClassifier()
        cls.movement_classifier = MockMovementClassifier()
        cls.metrics_manager = MetricsManager()
        cls.data_analyzer = DataAnalyzer()
        
        # Genera dati di test
        cls.test_frames = [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            for _ in range(100)
        ]
        cls.test_sequences = [
            np.random.rand(1, 100, 99)
            for _ in range(100)
        ]
        
    def test_throughput(self):
        """Test throughput del sistema"""
        start_time = time.time()
        processed_frames = 0
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            
            # Sottometti job per elaborazione frame
            for frame in self.test_frames[:20]:  # Test primi 20 frame
                futures.append(executor.submit(
                    self.debugger.monitor_component,
                    "expression_analyzer",
                    self.expression_analyzer.analyze_frame,
                    frame
                ))
                futures.append(executor.submit(
                    self.debugger.monitor_component,
                    "pupil_analyzer", 
                    self.pupil_analyzer.analyze_frame,
                    frame
                ))
                futures.append(executor.submit(
                    self.debugger.monitor_component,
                    "pose_analyzer",
                    self.pose_analyzer.analyze_frame,
                    frame
                ))
                processed_frames += 1
                
            # Attendi completamento
            results = [f.result() for f in futures]
            
        elapsed_time = time.time() - start_time
        fps = processed_frames / elapsed_time
        
        self.assertGreater(fps, 10)  # Minimo 10 FPS
        
    def test_latency(self):
        """Test latenza dei componenti"""
        latencies = {}
        
        # Test latenza expression analyzer
        start_time = time.time()
        self.debugger.monitor_component(
            "expression_analyzer",
            self.expression_analyzer.analyze_frame,
            self.test_frames[0]
        )
        latencies['expression_analyzer'] = time.time() - start_time
        
        # Test latenza pupil analyzer
        start_time = time.time()
        self.debugger.monitor_component(
            "pupil_analyzer",
            self.pupil_analyzer.analyze_frame,
            self.test_frames[0]
        )
        latencies['pupil_analyzer'] = time.time() - start_time
        
        # Test latenza pose analyzer
        start_time = time.time()
        self.debugger.monitor_component(
            "pose_analyzer",
            self.pose_analyzer.analyze_frame,
            self.test_frames[0]
        )
        latencies['pose_analyzer'] = time.time() - start_time
        
        # Verifica latenze
        for component, latency in latencies.items():
            self.assertLess(latency, 0.1)  # Max 100ms per componente
            
    def test_resource_usage(self):
        """Test utilizzo risorse"""
        initial_memory = psutil.Process().memory_info().rss
        initial_cpu = psutil.Process().cpu_percent()
        
        # Esegui operazioni intensive
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for frame in self.test_frames[:10]:
                futures.extend([
                    executor.submit(self.expression_analyzer.analyze_frame, frame),
                    executor.submit(self.pupil_analyzer.analyze_frame, frame),
                    executor.submit(self.pose_analyzer.analyze_frame, frame)
                ])
            [f.result() for f in futures]
            
        final_memory = psutil.Process().memory_info().rss
        final_cpu = psutil.Process().cpu_percent()
        
        # Verifica utilizzo risorse
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
        cpu_increase = final_cpu - initial_cpu
        
        self.assertLess(memory_increase, 100)  # Max 100MB incremento
        self.assertLess(cpu_increase, 50)  # Max 50% incremento CPU
        
    def test_stress(self):
        """Test stress del sistema"""
        error_count = 0
        total_operations = 1000
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            
            # Sottometti molte operazioni concorrenti
            for _ in range(total_operations):
                frame_idx = np.random.randint(0, len(self.test_frames))
                seq_idx = np.random.randint(0, len(self.test_sequences))
                
                # Randomizza operazione
                operation = np.random.choice([
                    lambda: self.expression_analyzer.analyze_frame(self.test_frames[frame_idx]),
                    lambda: self.pupil_analyzer.analyze_frame(self.test_frames[frame_idx]),
                    lambda: self.pose_analyzer.analyze_frame(self.test_frames[frame_idx]),
                    lambda: self.emotion_classifier.predict(np.random.rand(1, 7)),
                    lambda: self.movement_classifier.predict(self.test_sequences[seq_idx])
                ])
                
                futures.append(executor.submit(operation))
                
            # Raccogli risultati
            for future in futures:
                try:
                    future.result()
                except Exception:
                    error_count += 1
                    
        # Verifica affidabilità
        error_rate = error_count / total_operations
        self.assertLess(error_rate, 0.01)  # Max 1% errori
        
    def test_recovery(self):
        """Test recupero da errori"""
        # Simula errori in vari componenti
        components_to_test = [
            ('expression_analyzer', self.expression_analyzer),
            ('pupil_analyzer', self.pupil_analyzer),
            ('pose_analyzer', self.pose_analyzer),
            ('emotion_classifier', self.emotion_classifier),
            ('movement_classifier', self.movement_classifier)
        ]
        
        for component_name, component in components_to_test:
            print(f"\nTesting recovery for {component_name}")
            
            # Imposta lo stato del componente in errore
            self.debugger.component_health[component_name] = ComponentHealth(
                name=component_name,
                status="error",
                error_count=1,
                last_error="Test error",
                avg_response_time=0.0,
                memory_usage=0.0
            )
            
            # Verifica stato componente
            health = self.debugger.component_health.get(component_name)
            if health:
                print(f"Component status: {health.status}")
                print(f"Error count: {health.error_count}")
                print(f"Last error: {health.last_error}")
            else:
                print("No health record found")
                
            # Verifica recupero automatico
            recovered = self.debugger.auto_recover(component_name)
            print(f"Recovery result: {recovered}")
            
            if not recovered:
                self.fail(f"Failed to recover {component_name}")
                
            # Verifica funzionamento normale dopo recupero
            try:
                if hasattr(component, 'analyze_frame'):
                    result = self.debugger.monitor_component(
                        component_name,
                        component.analyze_frame,
                        self.test_frames[0]
                    )
                else:
                    result = self.debugger.monitor_component(
                        component_name,
                        component.predict,
                        np.random.rand(1, 7)
                    )
                self.assertIsNotNone(result)
                print("Component working normally after recovery")
            except Exception as e:
                self.fail(f"Component {component_name} failed after recovery: {str(e)}")
                
    def test_long_running(self):
        """Test esecuzione prolungata"""
        start_time = time.time()
        max_duration = 60  # 1 minuto
        operation_count = 0
        errors = []
        
        while time.time() - start_time < max_duration:
            try:
                # Esegui operazioni miste
                frame_idx = operation_count % len(self.test_frames)
                
                self.expression_analyzer.analyze_frame(self.test_frames[frame_idx])
                self.pupil_analyzer.analyze_frame(self.test_frames[frame_idx])
                self.pose_analyzer.analyze_frame(self.test_frames[frame_idx])
                
                if operation_count % 10 == 0:  # Ogni 10 frame
                    features = np.random.rand(1, 7)
                    self.emotion_classifier.predict(features)
                    
                if operation_count % 20 == 0:  # Ogni 20 frame
                    seq_idx = (operation_count // 20) % len(self.test_sequences)
                    self.movement_classifier.predict(self.test_sequences[seq_idx])
                    
                operation_count += 1
                
            except Exception as e:
                errors.append(str(e))
                
        # Verifica risultati
        error_rate = len(errors) / operation_count
        self.assertLess(error_rate, 0.01)  # Max 1% errori
        
        # Verifica stato sistema
        system_health = self.debugger.get_system_health()
        self.assertEqual(system_health['components']['expression_analyzer']['status'], 'healthy')
        self.assertEqual(system_health['components']['pupil_analyzer']['status'], 'healthy')
        self.assertEqual(system_health['components']['pose_analyzer']['status'], 'healthy')
        
    @classmethod
    def tearDownClass(cls):
        """Cleanup e report finale"""
        system_health = cls.debugger.get_system_health()
        print("\nSystem Health Report:")
        print(f"Uptime: {system_health['uptime']:.2f} seconds")
        print(f"Total Errors: {system_health['total_errors']}")
        print(f"Memory Usage: {system_health['memory_usage']:.2f} MB")
        print(f"CPU Usage: {system_health['cpu_percent']:.2f}%")
        
        print("\nComponent Status:")
        for name, stats in system_health['components'].items():
            print(f"\n{name}:")
            print(f"  Status: {stats['status']}")
            print(f"  Error Count: {stats['error_count']}")
            print(f"  Avg Response Time: {stats['avg_response_time']*1000:.2f}ms")
            print(f"  Memory Usage: {stats['memory_usage']/1024/1024:.2f}MB")
            
        suggestions = cls.debugger.suggest_fixes()
        if suggestions:
            print("\nSuggested Improvements:")
            for suggestion in suggestions:
                print(f"- {suggestion}")
                
if __name__ == '__main__':
    unittest.main()
