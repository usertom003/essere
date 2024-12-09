import unittest
import numpy as np
import logging
from pathlib import Path
import sys
from typing import Dict, List, Any, Optional
from dataclasses import asdict

from src.utils.auto_debugger import AutoDebugger, ComponentHealth
from src.data_processing.metrics_manager import MetricsManager
from src.facial_analysis.expression_analyzer import ExpressionAnalyzer
from src.facial_analysis.pupil_analyzer import PupilAnalyzer
from src.body_analysis.pose_analyzer import PoseAnalyzer

class BiometricTestBase(unittest.TestCase):
    """Classe base per i test del sistema biometrico"""
    
    @classmethod
    def setUpClass(cls):
        """Setup iniziale comune a tutti i test"""
        # Configura logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        cls.logger = logging.getLogger(cls.__name__)
        
        # Inizializza componenti
        cls.debugger = AutoDebugger()
        cls.metrics_manager = MetricsManager()
        cls.expression_analyzer = ExpressionAnalyzer()
        cls.pupil_analyzer = PupilAnalyzer()
        cls.pose_analyzer = PoseAnalyzer()
        
        # Genera dati di test
        cls.test_frames = cls.generate_test_frames()
        cls.test_features = cls.generate_test_features()
        
    @staticmethod
    def generate_test_frames(count: int = 10) -> list:
        """Genera frame di test standardizzati"""
        np.random.seed(42)  # Per riproducibilità
        return [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            for _ in range(count)
        ]
        
    @staticmethod
    def generate_test_features(count: int = 10) -> np.ndarray:
        """Genera feature di test standardizzate"""
        np.random.seed(42)
        return np.random.rand(count, 7)
        
    def verify_component_health(self, component_name: str) -> Dict[str, Any]:
        """Verifica lo stato di salute di un componente"""
        health = self.debugger.get_component_health(component_name)
        if health is None:
            raise ValueError(f"No health data for component {component_name}")
            
        health_dict = asdict(health)  # Converte ComponentHealth in dict
        self.assertEqual(health_dict['status'], 'healthy')
        return health_dict
        
    def verify_metrics_format(self, metrics: Dict[str, Any]):
        """Verifica il formato standard delle metriche"""
        required_fields = ['timestamp', 'confidence']
        for field in required_fields:
            self.assertIn(field, metrics)
            
    def verify_performance(self, start_time: float, end_time: float, 
                         operation_name: str = "Operation"):
        """Verifica le performance di un'operazione"""
        execution_time = end_time - start_time
        self.assertLess(
            execution_time, 
            30,  # Max 30 secondi
            f"{operation_name} took too long: {execution_time:.2f}s"
        ) 