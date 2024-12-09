import unittest
import numpy as np
import tensorflow as tf
from typing import Dict, List

from src.ml_models.movement_classifier import MovementClassifier

class TestMovementClassifier(unittest.TestCase):
    """Test suite per MovementClassifier"""
    
    def setUp(self):
        """Setup per ogni test"""
        self.classifier = MovementClassifier()
        self.test_features = np.random.rand(100, 7)
        self.test_labels = np.random.randint(0, 5, 100)
        
    def test_model_initialization(self):
        """Test inizializzazione modello"""
        self.assertIsInstance(self.classifier.model, tf.keras.Model)
        
        # Verifica architettura
        layers = self.classifier.model.layers
        self.assertGreater(len(layers), 0)
        
        # Verifica input shape
        input_shape = layers[0].input_shape
        self.assertEqual(input_shape[1], 7)
        
        # Verifica output shape
        output_shape = layers[-1].output_shape
        self.assertEqual(output_shape[1], 5)  # 5 classi di movimento
        
    def test_prediction(self):
        """Test predizione"""
        predictions = self.classifier.predict(self.test_features)
        
        self.assertIsInstance(predictions, np.ndarray)
        self.assertEqual(predictions.shape[0], len(self.test_features))
        self.assertEqual(predictions.shape[1], 5)
        self.assertTrue(np.all(predictions >= 0))
        self.assertTrue(np.all(predictions <= 1))
        
    def test_training(self):
        """Test training"""
        history = self.classifier.train(
            self.test_features,
            self.test_labels,
            epochs=5,
            batch_size=32
        )
        
        self.assertIsInstance(history, dict)
        self.assertIn("loss", history)
        self.assertIn("accuracy", history)
        self.assertEqual(len(history["loss"]), 5)
        
    def test_evaluation(self):
        """Test valutazione"""
        metrics = self.classifier.evaluate(
            self.test_features,
            self.test_labels
        )
        
        self.assertIsInstance(metrics, dict)
        self.assertIn("loss", metrics)
        self.assertIn("accuracy", metrics)
        self.assertGreaterEqual(metrics["accuracy"], 0)
        self.assertLessEqual(metrics["accuracy"], 1)
        
    def test_feature_validation(self):
        """Test validazione features"""
        # Features valide
        valid = self.classifier.validate_features(self.test_features)
        self.assertTrue(valid)
        
        # Features invalide
        invalid_features = np.random.rand(100, 5)  # Dimensione errata
        valid = self.classifier.validate_features(invalid_features)
        self.assertFalse(valid)
        
        invalid_features = np.random.rand(100, 7) * 2 - 1  # Range errato
        valid = self.classifier.validate_features(invalid_features)
        self.assertFalse(valid)
        
    def test_prediction_confidence(self):
        """Test confidenza predizioni"""
        predictions, confidence = self.classifier.predict_with_confidence(
            self.test_features
        )
        
        self.assertIsInstance(predictions, np.ndarray)
        self.assertIsInstance(confidence, np.ndarray)
        self.assertEqual(len(confidence), len(predictions))
        self.assertTrue(np.all(confidence >= 0))
        self.assertTrue(np.all(confidence <= 1))
        
    def test_batch_prediction(self):
        """Test predizione batch"""
        batches = [
            self.test_features[i:i+32]
            for i in range(0, len(self.test_features), 32)
        ]
        
        for batch in batches:
            predictions = self.classifier.predict(batch)
            self.assertEqual(predictions.shape[0], len(batch))
            self.assertEqual(predictions.shape[1], 5)
            
    def test_model_saving(self):
        """Test salvataggio modello"""
        self.classifier.save_model("test_movement_model")
        
        # Carica modello
        loaded_classifier = MovementClassifier()
        loaded_classifier.load_model("test_movement_model")
        
        # Verifica predizioni
        orig_pred = self.classifier.predict(self.test_features)
        loaded_pred = loaded_classifier.predict(self.test_features)
        
        np.testing.assert_array_almost_equal(orig_pred, loaded_pred)
        
    def test_error_handling(self):
        """Test gestione errori"""
        # Feature matrix vuota
        with self.assertRaises(ValueError):
            self.classifier.predict(np.array([]))
            
        # Dimensioni errate
        with self.assertRaises(ValueError):
            self.classifier.predict(np.random.rand(100))
            
        # Tipo dati errato
        with self.assertRaises(ValueError):
            self.classifier.predict([[1,2,3,4,5,6,7]])
            
    def test_movement_mapping(self):
        """Test mapping movimenti"""
        movements = self.classifier.get_movement_labels()
        
        self.assertIsInstance(movements, list)
        self.assertEqual(len(movements), 5)
        self.assertTrue(all(isinstance(m, str) for m in movements))
        
    def test_sequence_prediction(self):
        """Test predizione sequenze"""
        sequence = np.random.rand(10, 7)  # 10 timesteps
        
        predictions = self.classifier.predict_sequence(sequence)
        
        self.assertIsInstance(predictions, np.ndarray)
        self.assertEqual(predictions.shape[0], len(sequence))
        self.assertEqual(predictions.shape[1], 5)
        
    def test_movement_analysis(self):
        """Test analisi movimento"""
        predictions = self.classifier.predict(self.test_features)
        
        analysis = self.classifier.analyze_movement_patterns(predictions)
        
        self.assertIsInstance(analysis, dict)
        self.assertIn("transitions", analysis)
        self.assertIn("durations", analysis)
        self.assertIn("frequencies", analysis)
        
if __name__ == '__main__':
    unittest.main()
