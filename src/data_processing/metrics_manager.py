import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import numpy as np
from uuid import uuid4
from datetime import timedelta
import pandas as pd

logger = logging.getLogger(__name__)

class MetricsManager:
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or str(uuid4())
        self.metrics_history: List[Dict[str, Any]] = []
        self.current_metrics: Dict[str, Any] = {}
        self.stats_window = 100  # Finestra per calcoli statistici
        
    def _calculate_expression_intensity(self) -> float:
        """Calcola l'intensità delle espressioni facciali"""
        if not self.metrics_history:
            return 0.0
            
        recent_metrics = self.get_recent_metrics(minutes=1)
        intensities = []
        
        for metric in recent_metrics:
            if 'facial_metrics' in metric:
                facial = metric['facial_metrics']
                # Calcola intensità media delle espressioni
                if 'expressions' in facial:
                    intensities.append(
                        np.mean(list(facial['expressions'].values()))
                    )
                    
        return float(np.mean(intensities)) if intensities else 0.0
        
    def _calculate_pupil_activity(self) -> float:
        """Calcola l'attività pupillare"""
        if not self.metrics_history:
            return 0.0
            
        recent_metrics = self.get_recent_metrics(minutes=1)
        pupil_sizes = []
        
        for metric in recent_metrics:
            if 'pupil_metrics' in metric:
                pupils = metric['pupil_metrics']
                if 'left_pupil' in pupils and 'right_pupil' in pupils:
                    left_size = pupils['left_pupil'].get('size', 0)
                    right_size = pupils['right_pupil'].get('size', 0)
                    pupil_sizes.append((left_size + right_size) / 2)
                    
        if not pupil_sizes:
            return 0.0
            
        # Calcola variabilità della dimensione pupillare
        return float(np.std(pupil_sizes))
        
    def _calculate_movement_intensity(self) -> float:
        """Calcola l'intensità del movimento"""
        if not self.metrics_history:
            return 0.0
            
        recent_metrics = self.get_recent_metrics(minutes=1)
        movements = []
        
        for metric in recent_metrics:
            if 'pose_metrics' in metric:
                pose = metric['pose_metrics']
                if 'pose_landmarks' in pose:
                    # Calcola velocità media dei landmark
                    landmarks = np.array(pose['pose_landmarks'])
                    if len(movements) > 0:
                        prev_landmarks = movements[-1]
                        movement = np.mean(np.abs(landmarks - prev_landmarks))
                        movements.append(movement)
                    movements.append(landmarks)
                    
        return float(np.mean(movements)) if len(movements) > 1 else 0.0
        
    def _calculate_emotion_distribution(self) -> Dict[str, float]:
        """Calcola la distribuzione delle emozioni"""
        if not self.metrics_history:
            return {}
            
        emotion_counts: Dict[str, int] = {}
        total = 0
        
        for metric in self.metrics_history:
            if 'facial_metrics' in metric and 'emotion' in metric['facial_metrics']:
                emotion = metric['facial_metrics']['emotion']
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                total += 1
                
        if total == 0:
            return {}
            
        return {
            emotion: count / total 
            for emotion, count in emotion_counts.items()
        }
        
    def _calculate_movement_distribution(self) -> Dict[str, float]:
        """Calcola la distribuzione dei movimenti"""
        if not self.metrics_history:
            return {}
            
        movement_counts: Dict[str, int] = {}
        total = 0
        
        for metric in self.metrics_history:
            if 'pose_metrics' in metric and 'movement' in metric['pose_metrics']:
                movement = metric['pose_metrics']['movement']
                movement_counts[movement] = movement_counts.get(movement, 0) + 1
                total += 1
                
        if total == 0:
            return {}
            
        return {
            movement: count / total 
            for movement, count in movement_counts.items()
        }
        
    def get_realtime_stats(self) -> Dict[str, float]:
        """Get realtime statistics"""
        if not self.metrics_history:
            return {}
        return {
            'expression_intensity': self._calculate_expression_intensity(),
            'pupil_activity': self._calculate_pupil_activity(),
            'movement_intensity': self._calculate_movement_intensity()
        }
        
    def get_recent_metrics(self, minutes: int = 5) -> List[Dict[str, Any]]:
        """Get metrics from last N minutes"""
        if not self.metrics_history:
            return []
        cutoff = datetime.now() - timedelta(minutes=minutes)
        return [
            m for m in self.metrics_history 
            if datetime.fromisoformat(m['timestamp']) > cutoff
        ]
        
    def calculate_statistics(self) -> Dict[str, Any]:
        """Calculate overall statistics"""
        if not self.metrics_history:
            return {}
        return {
            'emotion_distribution': self._calculate_emotion_distribution(),
            'movement_distribution': self._calculate_movement_distribution(),
            'avg_confidence': np.mean([m.get('confidence', 0) for m in self.metrics_history])
        }
        
    def store_session(self, filepath: str) -> None:
        """Store current session to file"""
        with open(filepath, 'w') as f:
            json.dump({
                'session_id': self.session_id,
                'metrics': self.metrics_history
            }, f, indent=2)
            
    def load_session(self, filepath: str) -> bool:
        """Load session from file"""
        try:
            with open(filepath) as f:
                data = json.load(f)
                self.session_id = data['session_id']
                self.metrics_history = data['metrics']
            return True
        except Exception as e:
            logger.error(f"Error loading session: {e}")
            return False
            
    def clear_old_metrics(self, max_age_days: int = 30) -> None:
        """Clear metrics older than max_age_days"""
        cutoff = datetime.now() - timedelta(days=max_age_days)
        self.metrics_history = [
            m for m in self.metrics_history
            if datetime.fromisoformat(m['timestamp']) > cutoff
        ]
        
    def update_metrics(self, metrics: Dict[str, Any]) -> None:
        """Update current metrics"""
        self.current_metrics = metrics
        self.metrics_history.append(metrics)
        
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        return self.current_metrics
        
    def process_frame(self, frame: np.ndarray,
                     expression_analyzer: Any,
                     pupil_analyzer: Any,
                     pose_analyzer: Any) -> Dict[str, Any]:
        """Processa un frame e raccoglie tutte le metriche"""
        try:
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'facial': expression_analyzer.analyze_frame(frame),
                'pupil': pupil_analyzer.analyze_frame(frame),
                'pose': pose_analyzer.analyze_frame(frame)
            }
            
            # Aggiorna le metriche correnti
            self.update_metrics(metrics)
            return metrics
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return {}
            
    def get_facial_metrics(self) -> Dict[str, float]:
        """Recupera le metriche facciali più recenti"""
        if not self.current_metrics:
            return {}
        return self.current_metrics.get('facial', {})
        
    def get_movement_metrics(self) -> Dict[str, float]:
        """Recupera le metriche di movimento più recenti"""
        if not self.current_metrics:
            return {}
        return self.current_metrics.get('movement', {})
        
    def collect_metrics(self, frame: np.ndarray,
                       expression_analyzer: Any,
                       pupil_analyzer: Any,
                       pose_analyzer: Any) -> Dict[str, Any]:
        """Collect metrics from all analyzers"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'expressions': expression_analyzer.analyze_frame(frame),
            'pupils': pupil_analyzer.analyze_frame(frame),
            'poses': pose_analyzer.analyze_frame(frame)
        }
        self.add_metrics(metrics)
        return metrics
        
    def aggregate_metrics(self, metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate multiple metrics"""
        if not metrics_list:
            return {}
            
        aggregated = {
            'expressions': {},
            'pupils': {},
            'poses': {}
        }
        
        for metrics in metrics_list:
            for category in ['expressions', 'pupils', 'poses']:
                if category in metrics:
                    for key, value in metrics[category].items():
                        if key not in aggregated[category]:
                            aggregated[category][key] = []
                        aggregated[category][key].append(value)
                        
        # Calculate averages
        for category in aggregated:
            for key in aggregated[category]:
                values = aggregated[category][key]
                if values:
                    aggregated[category][key] = float(np.mean(values))
                    
        return aggregated
        
    def filter_metrics(self, metrics_list: List[Dict[str, Any]], 
                      min_confidence: float = 0.0) -> List[Dict[str, Any]]:
        """Filter metrics based on confidence"""
        return [
            m for m in metrics_list 
            if m.get('confidence', 0.0) >= min_confidence
        ]
        
    def analyze_temporal_patterns(self, metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze temporal patterns in metrics"""
        if not metrics_list:
            return {}
            
        analysis = {
            'trends': {},
            'changes': {},
            'frequencies': {}
        }
        
        # Calculate trends
        for category in ['expressions', 'pupils', 'poses']:
            analysis['trends'][category] = self._calculate_trends(
                metrics_list, category
            )
            
        return analysis
        
    def validate_metrics(self, metrics: Optional[Dict[str, Any]]) -> bool:
        """Validate metrics format"""
        if metrics is None:
            return False
            
        required_fields = ['timestamp']
        return all(field in metrics for field in required_fields)
        
    def save_metrics(self, metrics: Dict[str, Any], filepath: str) -> None:
        """Save metrics to file"""
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
            
    def load_metrics(self, filepath: str) -> Dict[str, Any]:
        """Load metrics from file"""
        with open(filepath) as f:
            return json.load(f)
            
    def export_to_csv(self, metrics_list: List[Dict[str, Any]], filepath: str) -> None:
        """Export metrics to CSV"""
        df = pd.DataFrame(metrics_list)
        df.to_csv(filepath, index=False)
        
    def export_to_excel(self, metrics_list: List[Dict[str, Any]], filepath: str) -> None:
        """Export metrics to Excel"""
        df = pd.DataFrame(metrics_list)
        df.to_excel(filepath, index=False)
        
    def generate_summary(self, metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary of metrics"""
        if not metrics_list:
            return {}
            
        return {
            'start_time': metrics_list[0]['timestamp'],
            'end_time': metrics_list[-1]['timestamp'],
            'total_samples': len(metrics_list),
            'statistics': self.calculate_statistics()
        }
        
    def add_metrics(self, metrics: Dict[str, Any]) -> None:
        """Add new metrics to history"""
        if not isinstance(metrics, dict):
            raise ValueError("Metrics must be a dictionary")
        
        if 'timestamp' not in metrics:
            metrics['timestamp'] = datetime.now().isoformat()
        
        # Validate timestamp format
        try:
            datetime.fromisoformat(metrics['timestamp'])
        except ValueError:
            raise ValueError("Invalid timestamp format")
        
        self.metrics_history.append(metrics)
        self.current_metrics = metrics
        
    def _calculate_trends(self, metrics_list: List[Dict[str, Any]], 
                         category: str) -> Dict[str, float]:
        """Calculate trends for a specific category"""
        if not metrics_list:
            return {}
        
        values: Dict[str, List[float]] = {}
        
        for metrics in metrics_list:
            if category in metrics:
                for key, value in metrics[category].items():
                    if isinstance(value, (int, float)):
                        if key not in values:
                            values[key] = []
                        values[key].append(float(value))
                        
        trends = {}
        for key, series in values.items():
            if len(series) > 1:
                # Calculate linear regression slope
                x = np.arange(len(series))
                slope = np.polyfit(x, series, 1)[0]
                trends[key] = float(slope)
                
        return trends