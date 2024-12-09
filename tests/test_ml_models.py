import unittest
import numpy as np
import tensorflow as tf
import sys
from pathlib import Path
import logging

# Aggiungi il percorso src al PYTHONPATH
src_path = Path(__file__).parent.parent / 'src'
sys.path.append(str(src_path))

from ml_models.emotion_classifier import EmotionClassifier
from ml_models.movement_classifier import MovementClassifier, MovementPattern

class TestEmotionClassifier(unittest.TestCase):
    def setUp(self):
        self.classifier = EmotionClassifier()
        
    def test_initialization(self):
        """Verifica corretta inizializzazione"""
        self.assertIsNotNone(self.classifier)
        
    def test_prediction(self):
        """Test predizione emozioni"""
        features = np.random.rand(1, 7)  # 7 features
        emotion = self.classifier.predict(features)
        self.assertIn(emotion, ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgusted', 'surprised'])
        
    def test_invalid_features(self):
        """Test gestione features non validi"""
        invalid_features = np.random.rand(1, 5)  # numero errato di features
        with self.assertRaises(ValueError):
            self.classifier.predict(invalid_features)
            
    def test_batch_prediction(self):
        """Test predizione su batch"""
        batch_features = np.random.rand(10, 7)  # batch di 10 campioni
        emotions = [self.classifier.predict(features.reshape(1, -1)) for features in batch_features]
        self.assertEqual(len(emotions), 10)

class TestMovementClassifier(unittest.TestCase):
    def setUp(self):
        self.classifier = MovementClassifier()
        
    def test_initialization(self):
        """Verifica corretta inizializzazione"""
        self.assertIsNotNone(self.classifier.model)
        
    def test_prediction(self):
        """Test predizione movimenti"""
        sequence = np.random.rand(1, 100, 99)  # (batch, time_steps, features)
        movement = self.classifier.predict(sequence)
        self.assertIn(movement, self.classifier.movement_types)
        
    def test_invalid_sequence(self):
        """Test gestione sequenza non valida"""
        invalid_sequence = np.random.rand(1, 50, 99)  # lunghezza temporale errata
        with self.assertRaises(ValueError):
            self.classifier.predict(invalid_sequence)
            
    def test_movement_pattern(self):
        """Test pattern di movimento"""
        pattern = MovementPattern(
            name='walking',
            confidence=0.95,
            frequency=1.2,
            duration=5.0,
            intensity=0.7
        )
        self.assertEqual(pattern.name, 'walking')
        self.assertGreaterEqual(pattern.confidence, 0)
        self.assertLessEqual(pattern.confidence, 1)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    unittest.main()
