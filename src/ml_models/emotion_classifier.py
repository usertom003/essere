import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class EmotionClassifier:
    def __init__(self):
        """Inizializza il classificatore delle emozioni"""
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.emotion_labels = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgusted', 'surprised']
        self.n_features = 7
        
        # Addestra il modello con dati sintetici di base
        self._train_base_model()
        
    def _train_base_model(self):
        """Addestra il modello con dati sintetici di base"""
        try:
            # Genera dati sintetici per l'addestramento base
            n_samples = 1000
            X = np.random.rand(n_samples, self.n_features)
            y = np.random.choice(self.emotion_labels, size=n_samples)
            
            # Addestra il modello
            self.model.fit(X, y)
            logger.info("Base model trained successfully")
            
        except Exception as e:
            logger.error(f"Error training base model: {str(e)}")
            raise
            
    def preprocess_features(self, metrics: Dict[str, Any]) -> np.ndarray:
        """Preprocess raw metrics into feature vector"""
        features = []
        
        # Extract expression metrics
        if 'expression' in metrics:
            expr = metrics['expression']
            features.extend([
                expr.get('eye_aspect_ratio', 0.0),
                expr.get('mouth_aspect_ratio', 0.0),
                expr.get('eyebrow_position', 0.0),
                expr.get('nose_wrinkle', 0.0)
            ])
            
        # Extract pupil metrics
        if 'pupil' in metrics:
            pupil = metrics['pupil']
            features.extend([
                pupil.get('left_pupil_size', 0.0),
                pupil.get('right_pupil_size', 0.0),
                pupil.get('pupil_ratio', 0.0)
            ])
            
        return np.array(features).reshape(1, -1)
        
    def update_history(self, features: np.ndarray, emotion: str):
        """Update feature and emotion history"""
        self.feature_history.append(features.flatten().tolist())
        self.emotion_history.append(emotion)
        
        # Keep last 1000 samples
        if len(self.feature_history) > 1000:
            self.feature_history.pop(0)
            self.emotion_history.pop(0)
            
    def train(self, features: np.ndarray, emotions: List[str]):
        """Train the emotion classifier"""
        try:
            # Scale features
            scaled_features = StandardScaler().fit_transform(features)
            
            # Train classifier
            self.model.fit(scaled_features, emotions)
            logger.info("Emotion classifier trained successfully")
            
        except Exception as e:
            logger.error(f"Error training emotion classifier: {str(e)}")
            raise
            
    def predict(self, features: np.ndarray) -> str:
        """Predice l'emozione dai features"""
        try:
            if len(features.shape) != 2:
                raise ValueError("Features must be 2D array")
                
            if features.shape[1] != 7:  # Numero atteso di features
                raise ValueError(f"Expected 7 features, got {features.shape[1]}")
                
            # Normalizza features
            features = self._normalize_features(features)
            
            # Predici classe
            prediction = self.model.predict(features)
            
            # Gestisci sia predizioni numeriche che stringhe
            if isinstance(prediction[0], (int, np.integer)):
                prediction_idx = int(prediction[0])
            else:
                # Se è una stringa, cerca l'indice corrispondente
                try:
                    prediction_idx = self.emotion_labels.index(prediction[0])
                except ValueError:
                    logger.error(f"Invalid emotion label: {prediction[0]}")
                    raise ValueError(f"Invalid emotion label: {prediction[0]}")
            
            return self.emotion_labels[prediction_idx]
            
        except Exception as e:
            logger.error(f"Error predicting emotion: {str(e)}")
            raise
            
    def save_model(self, path: str):
        """Save model to disk"""
        try:
            model_data = {
                'model': self.model,
                'emotion_labels': self.emotion_labels,
                'n_features': self.n_features
            }
            joblib.dump(model_data, path)
            logger.info(f"Model saved to {path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
            
    def load_model(self, path: str):
        """Load model from disk"""
        try:
            model_data = joblib.load(path)
            self.model = model_data['model']
            self.emotion_labels = model_data['emotion_labels']
            self.n_features = model_data['n_features']
            logger.info(f"Model loaded from {path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        scaler = StandardScaler()
        return scaler.fit_transform(features)
