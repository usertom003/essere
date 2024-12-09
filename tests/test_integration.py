from test_base import BiometricTestBase
from concurrent.futures import ThreadPoolExecutor
import time
import numpy as np

class TestIntegration(BiometricTestBase):
    """Test di integrazione del sistema"""
    
    def setUp(self):
        """Setup per ogni test"""
        self.start_time = time.time()
        self.debugger.take_snapshot(f"Before {self._testMethodName}")
        
    def tearDown(self):
        """Cleanup dopo ogni test"""
        self.debugger.take_snapshot(f"After {self._testMethodName}")
        execution_time = time.time() - self.start_time
        self.logger.info(f"Test {self._testMethodName} completed in {execution_time:.2f}s")
        
    def test_pipeline_integration(self):
        """Test integrazione completa della pipeline"""
        for frame in self.test_frames:
            with self.subTest(frame=frame):
                # Esegui pipeline completa
                metrics = self.metrics_manager.process_frame(
                    frame,
                    self.expression_analyzer,
                    self.pupil_analyzer,
                    self.pose_analyzer
                )
                
                # Verifica formato metriche
                self.verify_metrics_format(metrics)
                
                # Verifica salute componenti
                for component in ['expression', 'pupil', 'pose']:
                    self.verify_component_health(component)
