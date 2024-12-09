import time
from typing import List, Dict, Any, Union, Sequence
import logging
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from ..types import MovementPattern, ComponentHealth
from ..database.unified_db import UnifiedDatabase
from dataclasses import asdict

logger = logging.getLogger(__name__)

class MetricsManager:
    def __init__(self, db_path: str = "data/metrics.db"):
        self.metrics_history: List[Dict[str, Any]] = []
        self.current_metrics: Dict[str, Dict[str, float]] = {
            'facial': {},
            'movement': {}
        }
        self.component_health: Dict[str, ComponentHealth] = {}
        self.movement_patterns: List[MovementPattern] = []
        self.db = UnifiedDatabase(db_path)

    def _prepare_data_for_storage(self, data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Prepare data for storage with support for both dict and list inputs"""
        def convert_value(v: Any) -> Any:
            if isinstance(v, np.ndarray):
                return v.tolist()
            if isinstance(v, (list, tuple)):
                return [convert_value(x) for x in v]
            if isinstance(v, dict):
                return {k: convert_value(val) for k, val in v.items()}
            if hasattr(v, 'to_dict'):
                return v.to_dict()
            if hasattr(v, 'dtype') and np.issubdtype(v.dtype, np.number):
                return float(v)
            return v
            
        if isinstance(data, list):
            return {'items': [convert_value(item) for item in data]}
        return {k: convert_value(v) for k, v in data.items()}

    def update_patterns(self, patterns: Sequence[MovementPattern]) -> None:
        """Update movement patterns"""
        try:
            self.movement_patterns = list(patterns)  # Create a new list from sequence
        except Exception as e:
            logger.error(f"Error updating patterns: {e}")
            self.movement_patterns = []

    def get_patterns(self) -> List[MovementPattern]:
        """Get current movement patterns"""
        return self.movement_patterns

    def store_session(self, filepath: str) -> None:
        """Store current session metrics to file"""
        try:
            data = {
                'metrics_history': [
                    {k: v.to_dict() if hasattr(v, 'to_dict') else v 
                     for k, v in metric.items()}
                    for metric in self.metrics_history
                ],
                'movement_patterns': [
                    pattern.__dict__ for pattern in self.movement_patterns
                ],
                'timestamp': datetime.now().isoformat()
            }
            
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error storing session: {e}")

    def get_facial_metrics(self) -> Dict[str, float]:
        """Get facial metrics from current metrics"""
        return self.current_metrics.get('facial', {})

    def get_movement_metrics(self) -> Dict[str, float]:
        """Get movement metrics from current metrics"""
        return self.current_metrics.get('movement', {})

    def clear_old_metrics(self) -> None:
        """Clear old metrics from history"""
        cutoff_time = time.time() - (60 * 60)  # 1 hour
        self.metrics_history = [
            m for m in self.metrics_history 
            if m.get('timestamp', 0) > cutoff_time
        ]

    def save_session(self, filepath: str) -> None:
        """Save current session metrics to file"""
        try:
            data = {
                'metrics_history': self._prepare_data_for_storage(self.metrics_history),
                'movement_patterns': [asdict(pattern) for pattern in self.movement_patterns],
                'component_health': {
                    name: asdict(health) 
                    for name, health in self.component_health.items()
                },
                'timestamp': datetime.now().isoformat()
            }
            
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Error saving session: {e}")

    def update_metrics(self, metrics: Dict[str, Any]) -> None:
        """Update current metrics"""
        try:
            self.current_metrics = {
                'facial': metrics.get('facial', {}),
                'movement': metrics.get('movement', {})
            }
            self.metrics_history.append({
                **metrics,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")

    def store_analysis_results(self, session_id: str, results: Dict[str, Any]) -> None:
        """Store analysis results"""
        try:
            processed_results = self._prepare_data_for_storage(results)
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO analysis_results 
                    (session_id, timestamp, data)
                    VALUES (?, ?, ?)
                """, (
                    session_id,
                    datetime.now().isoformat(),
                    json.dumps(processed_results)
                ))
        except Exception as e:
            logger.error(f"Error storing analysis results: {e}")