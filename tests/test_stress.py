from test_base import BiometricTestBase
from concurrent.futures import ThreadPoolExecutor
import time
import psutil
import numpy as np

class TestStress(BiometricTestBase):
    """Test di stress del sistema"""
    
    def setUp(self):
        """Setup per ogni test"""
        super().setUp()
        # Assicurati che i componenti siano inizializzati
        self.assertIsNotNone(self.expression_analyzer)
        self.assertIsNotNone(self.pupil_analyzer)
        self.assertIsNotNone(self.pose_analyzer)
        
    def test_sustained_load(self):
        """Test carico sostenuto"""
        duration = 60  # 1 minuto di test
        start_time = time.time()
        processed_frames = 0
        errors = []
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            while time.time() - start_time < duration:
                futures = []
                
                # Sottometti batch di frame
                for frame in self.test_frames:
                    future = executor.submit(
                        self.metrics_manager.process_frame,
                        frame,
                        self.expression_analyzer,
                        self.pupil_analyzer,
                        self.pose_analyzer
                    )
                    futures.append(future)
                    processed_frames += 1
                    
                # Raccogli risultati e errori
                for future in futures:
                    try:
                        metrics = future.result(timeout=10)
                        self.verify_metrics_format(metrics)
                    except Exception as e:
                        errors.append(str(e))
                        
        # Verifica risultati
        self.assertLess(len(errors) / processed_frames, 0.01)  # Max 1% errori
        
        # Verifica performance
        execution_time = time.time() - start_time
        fps = processed_frames / execution_time
        self.assertGreater(fps, 10)  # Minimo 10 FPS
        
        # Verifica utilizzo risorse
        process = psutil.Process()
        memory_percent = process.memory_percent()
        cpu_percent = process.cpu_percent()
        
        self.assertLess(memory_percent, 80)  # Max 80% memoria
        self.assertLess(cpu_percent, 80)  # Max 80% CPU 