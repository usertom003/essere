import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import logging
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class DataAnalyzer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=3)
        self.metrics_history: List[Dict] = []
        self.analysis_results: Dict = {}
        
    def add_metrics(self, metrics: Dict):
        """Add new metrics to history"""
        timestamp = datetime.now().isoformat()
        metrics_with_time = {
            'timestamp': timestamp,
            **metrics
        }
        self.metrics_history.append(metrics_with_time)
        
    def get_feature_matrix(self) -> np.ndarray:
        """Converte le metriche in una matrice di features"""
        if not self.metrics_history:
            # Ritorna una matrice vuota con il numero corretto di colonne
            return np.zeros((0, 7))
            
        features = []
        for metrics in self.metrics_history:
            row = self._extract_features(metrics)
            features.append(row)
        return np.array(features)
        
    def _extract_features(self, metrics: Dict) -> List[float]:
        """Estrae features dalle metriche"""
        # Feature di default se le metriche sono vuote
        default_features = [0.0] * 7
        
        if not metrics:
            return default_features
            
        try:
            features = [
                metrics.get('eye_aspect_ratio', 0.0),
                metrics.get('mouth_aspect_ratio', 0.0),
                metrics.get('eyebrow_position', 0.0),
                metrics.get('nose_wrinkle', 0.0),
                metrics.get('pupil_size', 0.0),
                metrics.get('head_pose_x', 0.0),
                metrics.get('head_pose_y', 0.0)
            ]
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return default_features
            
    def analyze_trends(self, data: pd.DataFrame) -> Dict:
        """Analyze trends in metrics data"""
        trends = {}
        
        # Analizza trend delle pupille
        pupil_cols = ['left_pupil_size', 'right_pupil_size', 'pupil_ratio']
        for col in pupil_cols:
            if col in data.columns:
                trends[f'pupil_{col}_trend'] = self._calculate_trend(data[col])
                
        # Analizza trend delle espressioni
        expr_cols = ['eye_aspect_ratio', 'mouth_aspect_ratio', 'eyebrow_position', 'nose_wrinkle']
        for col in expr_cols:
            if col in data.columns:
                trends[f'expression_{col}_trend'] = self._calculate_trend(data[col])
                
        # Calcola pattern
        patterns = self._detect_patterns(data)
        
        return {
            'trends': trends,
            'patterns': patterns
        }
        
    def _calculate_trend(self, series: pd.Series) -> Dict:
        """Calculate trend statistics for a series"""
        return {
            'mean': float(series.mean()),
            'std': float(series.std()),
            'min': float(series.min()),
            'max': float(series.max()),
            'slope': float(np.polyfit(range(len(series)), series, 1)[0])
        }
        
    def _detect_patterns(self, data: pd.DataFrame) -> Dict:
        # Implementazione del metodo _detect_patterns
        # Questo metodo dovrebbe rilevare pattern nei dati
        # Per ora, restituisce un dizionario vuoto
        return {}
        
    def reduce_dimensions(self) -> Optional[np.ndarray]:
        """Perform PCA on feature matrix"""
        try:
            features = self.get_feature_matrix()
            if len(features) == 0:
                return None
                
            # Scale features
            scaled_features = self.scaler.fit_transform(features)
            
            # Apply PCA
            reduced_features = self.pca.fit_transform(scaled_features)
            
            self.analysis_results['pca'] = {
                'explained_variance_ratio': self.pca.explained_variance_ratio_.tolist(),
                'n_components': self.pca.n_components_
            }
            
            return reduced_features
            
        except Exception as e:
            logger.error(f"Error reducing dimensions: {str(e)}")
            return None
            
    def detect_anomalies(self, threshold: float = 2.0) -> List[Dict]:
        """Detect anomalous metrics using statistical methods"""
        try:
            features = self.get_feature_matrix()
            if len(features) == 0:
                return []
                
            # Scale features
            scaled_features = self.scaler.fit_transform(features)
            
            # Calculate Mahalanobis distance
            covariance = np.cov(scaled_features.T)
            inv_covariance = np.linalg.inv(covariance)
            mean = np.mean(scaled_features, axis=0)
            
            anomalies = []
            for i, sample in enumerate(scaled_features):
                diff = sample - mean
                dist = np.sqrt(diff.dot(inv_covariance).dot(diff))
                
                if dist > threshold:
                    anomalies.append({
                        'timestamp': self.metrics_history[i]['timestamp'],
                        'distance': float(dist),
                        'metrics': self.metrics_history[i]
                    })
                    
            self.analysis_results['anomalies'] = anomalies
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {str(e)}")
            return []
            
    def save_analysis(self, path: str):
        """Save analysis results to file"""
        try:
            with open(path, 'w') as f:
                json.dump(self.analysis_results, f, indent=2)
            logger.info(f"Analysis results saved to {path}")
            
        except Exception as e:
            logger.error(f"Error saving analysis results: {str(e)}")
            raise
            
    def load_analysis(self, path: str):
        """Load analysis results from file"""
        try:
            with open(path, 'r') as f:
                self.analysis_results = json.load(f)
            logger.info(f"Analysis results loaded from {path}")
            
        except Exception as e:
            logger.error(f"Error loading analysis results: {str(e)}")
            raise
