import unittest
import numpy as np
import cv2
import sys
from pathlib import Path
import logging
import psutil
import time

# Aggiungi il percorso src al PYTHONPATH
src_path = Path(__file__).parent.parent / 'src'
sys.path.append(str(src_path))

from facial_analysis.expression_analyzer import ExpressionAnalyzer
from facial_analysis.pupil_analyzer import PupilAnalyzer
from body_analysis.pose_analyzer import PoseAnalyzer
from ml_models.emotion_classifier import EmotionClassifier
from ml_models.movement_classifier import MovementClassifier
from data_processing.data_analyzer import DataAnalyzer
from data_processing.metrics_manager import MetricsManager

class TestSystemIntegration(unittest.TestCase):
    def setUp(self):
        self.expression_analyzer = ExpressionAnalyzer()
        self.pupil_analyzer = PupilAnalyzer()
        self.pose_analyzer = PoseAnalyzer()
        self.emotion_classifier = EmotionClassifier()
        self.movement_classifier = MovementClassifier()
        self.data_analyzer = DataAnalyzer()
        self.metrics_manager = MetricsManager()
        
        # Crea un'immagine di test
        self.test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(self.test_frame, (320, 240), 100, (255, 255, 255), -1)
        
    def test_full_pipeline(self):
        """Test dell'intera pipeline di analisi"""
        try:
            # 1. Analisi espressione facciale
            facial_metrics = self.expression_analyzer.analyze_frame(self.test_frame)
            self.assertIsNotNone(facial_metrics)
            
            # 2. Analisi pupille
            pupil_metrics = self.pupil_analyzer.analyze_frame(self.test_frame)
            self.assertIsNotNone(pupil_metrics)
            
            # 3. Analisi postura
            pose_metrics = self.pose_analyzer.analyze_frame(self.test_frame)
            self.assertIsNotNone(pose_metrics)
            
            # 4. Classificazione emozioni
            features = np.random.rand(1, 7)  # features combinate
            emotion = self.emotion_classifier.predict(features)
            self.assertIsNotNone(emotion)
            
            # 5. Classificazione movimenti
            sequence = np.random.rand(1, 100, 99)
            movement = self.movement_classifier.predict(sequence)
            self.assertIsNotNone(movement)
            
            # 6. Aggregazione metriche
            combined_metrics = {
                'facial_metrics': facial_metrics,
                'pupil_metrics': pupil_metrics,
                'pose_metrics': pose_metrics,
                'emotion': emotion,
                'movement': movement
            }
            
            self.metrics_manager.add_metrics(combined_metrics)
            self.data_analyzer.add_metrics(combined_metrics)
            
        except Exception as e:
            self.fail(f"Pipeline failed: {str(e)}")
            
    def test_system_performance(self):
        """Test delle performance del sistema"""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        # Esegui pipeline multiple volte
        for _ in range(10):
            self.test_full_pipeline()
            
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss
        
        # Verifica performance
        execution_time = end_time - start_time
        memory_usage = (end_memory - start_memory) / 1024 / 1024  # MB
        
        print(f"\nPerformance Test Results:")
        print(f"Execution Time: {execution_time:.2f} seconds")
        print(f"Memory Usage: {memory_usage:.2f} MB")
        
        # Verifica limiti performance
        self.assertLess(execution_time, 30)  # non più di 30 secondi
        self.assertLess(memory_usage, 1000)  # non più di 1GB
        
    def test_error_handling(self):
        """Test della gestione errori"""
        # Test con frame invalido
        invalid_frame = np.array([])
        with self.assertRaises(Exception):
            self.expression_analyzer.analyze_frame(invalid_frame)
            
        # Test con features invalidi
        invalid_features = np.random.rand(1, 3)  # numero errato di features
        with self.assertRaises(ValueError):
            self.emotion_classifier.predict(invalid_features)
            
        # Test con sequenza invalida
        invalid_sequence = np.random.rand(1, 50, 99)  # lunghezza temporale errata
        with self.assertRaises(ValueError):
            self.movement_classifier.predict(invalid_sequence)

def run_diagnostics():
    """Esegue test diagnostici completi"""
    try:
        print("\n=== AVVIO TEST DIAGNOSTICI ===\n")
        
        # Test componenti
        components = {
            'Expression Analyzer': ExpressionAnalyzer,
            'Pupil Analyzer': PupilAnalyzer,
            'Pose Analyzer': PoseAnalyzer,
            'Emotion Classifier': EmotionClassifier,
            'Movement Classifier': MovementClassifier,
            'Data Analyzer': DataAnalyzer,
            'Metrics Manager': MetricsManager
        }
        
        for name, component in components.items():
            try:
                instance = component()
                print(f"✓ {name} inizializzato correttamente")
            except Exception as e:
                print(f"✗ Errore inizializzazione {name}: {str(e)}")
                
        # Test webcam
        print("\nTest webcam...")
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✓ Webcam funzionante ({frame.shape[1]}x{frame.shape[0]} px)")
            else:
                print("✗ Errore lettura frame")
            cap.release()
        else:
            print("✗ Webcam non accessibile")
            
        # Test performance
        print("\nTest performance...")
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        cpu_percent = psutil.cpu_percent()
        
        print(f"Memoria utilizzata: {memory_usage:.1f} MB")
        print(f"CPU utilizzata: {cpu_percent}%")
        
        print("\n=== TEST DIAGNOSTICI COMPLETATI ===")
        
    except Exception as e:
        print(f"\nERRORE CRITICO: {str(e)}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Esegui prima i test diagnostici
    run_diagnostics()
    
    # Poi esegui i test unitari
    unittest.main()
