import json
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class ConfigManager:
    """Gestisce la configurazione globale del sistema"""
    
    DEFAULT_CONFIG = {
        'video': {
            'width': 640,
            'height': 480,
            'fps': 30
        },
        'analysis': {
            'min_detection_confidence': 0.5,
            'min_tracking_confidence': 0.5,
            'max_faces': 1
        },
        'performance': {
            'max_memory_percent': 80,
            'max_cpu_percent': 80,
            'metrics_window': 100,
            'cleanup_days': 30
        },
        'paths': {
            'database': 'data/biometric.db',
            'sessions': 'data/sessions',
            'logs': 'data/logs',
            'models': 'data/models'
        }
    }
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = Path(config_path)
        self.config = self.load_config()
        
    def load_config(self) -> Dict[str, Any]:
        """Carica la configurazione da file"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                return {**self.DEFAULT_CONFIG, **config}
            else:
                self.save_config(self.DEFAULT_CONFIG)
                return self.DEFAULT_CONFIG
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self.DEFAULT_CONFIG
            
    def save_config(self, config: Dict[str, Any]) -> bool:
        """Salva la configurazione su file"""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            return False
            
    def get(self, key: str, default: Any = None) -> Any:
        """Recupera un valore di configurazione"""
        try:
            keys = key.split('.')
            value = self.config
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
            
    def update(self, key: str, value: Any) -> bool:
        """Aggiorna un valore di configurazione"""
        try:
            keys = key.split('.')
            config = self.config
            for k in keys[:-1]:
                config = config[k]
            config[keys[-1]] = value
            return self.save_config(self.config)
        except Exception as e:
            logger.error(f"Error updating config: {e}")
            return False 