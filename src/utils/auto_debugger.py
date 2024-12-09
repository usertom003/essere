import logging
import traceback
import psutil
import time
import numpy as np
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ComponentHealth:
    """Stato di salute di un componente"""
    name: str
    status: str
    error_count: int
    last_error: Optional[str]
    avg_response_time: float
    memory_usage: float

class AutoDebugger:
    """Sistema di debug automatico"""
    
    def __init__(self):
        self.component_health: Dict[str, ComponentHealth] = {}
        self.error_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, List[float]] = {}
        self.start_time = time.time()
        self._cpu_history: List[float] = []
        self._last_memory_usage: Optional[int] = None
        
    def monitor_component(self, component_name: str, func, *args, **kwargs) -> Any:
        """Monitora l'esecuzione di una funzione di un componente"""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            memory_used = psutil.Process().memory_info().rss - start_memory
            
            self._update_component_health(
                component_name, 
                "healthy", 
                execution_time, 
                memory_used
            )
            
            return result
            
        except Exception as e:
            self._handle_error(component_name, e, traceback.format_exc())
            # Assicurati che il componente sia marcato come in errore
            if component_name not in self.component_health:
                self.component_health[component_name] = ComponentHealth(
                    name=component_name,
                    status="error",
                    error_count=1,
                    last_error=str(e),
                    avg_response_time=0.0,
                    memory_usage=0.0
                )
            else:
                self.component_health[component_name].status = "error"
                self.component_health[component_name].error_count += 1
                self.component_health[component_name].last_error = str(e)
            raise
            
    def _update_component_health(
        self, 
        component_name: str, 
        status: str,
        execution_time: float,
        memory_used: float
    ):
        """Aggiorna lo stato di salute di un componente"""
        if component_name not in self.performance_metrics:
            self.performance_metrics[component_name] = []
            
        self.performance_metrics[component_name].append(execution_time)
        
        # Calcola media mobile delle performance
        recent_metrics = self.performance_metrics[component_name][-100:]
        avg_time = np.mean(recent_metrics) if recent_metrics else 0
        
        health = self.component_health.get(component_name, ComponentHealth(
            name=component_name,
            status="unknown",
            error_count=0,
            last_error=None,
            avg_response_time=0,
            memory_usage=0
        ))
        
        health.status = status
        health.avg_response_time = avg_time
        health.memory_usage = memory_used
        
        self.component_health[component_name] = health
        
    def _handle_error(self, component_name: str, error: Exception, traceback_str: str):
        """Gestisce un errore di un componente"""
        error_info = {
            'component': component_name,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback_str,
            'timestamp': datetime.now().isoformat()
        }
        
        self.error_history.append(error_info)
        
        health = self.component_health.get(component_name)
        if health:
            health.error_count += 1
            health.last_error = str(error)
            health.status = "error"
            
        logger.error(f"Error in {component_name}: {str(error)}\n{traceback_str}")
        
    def get_system_health(self) -> Dict[str, Any]:
        """Restituisce lo stato di salute complessivo del sistema"""
        return {
            'uptime': time.time() - self.start_time,
            'total_errors': len(self.error_history),
            'memory_usage': psutil.Process().memory_info().rss / 1024 / 1024,
            'cpu_percent': psutil.Process().cpu_percent(),
            'components': {
                name: {
                    'status': health.status,
                    'error_count': health.error_count,
                    'avg_response_time': health.avg_response_time,
                    'memory_usage': health.memory_usage
                }
                for name, health in self.component_health.items()
            }
        }
        
    def analyze_errors(self) -> Dict[str, Any]:
        """Analizza gli errori per trovare pattern"""
        error_analysis = {
            'error_types': {},
            'components_affected': {},
            'error_frequency': {}
        }
        
        for error in self.error_history:
            # Analizza tipi di errore
            error_type = error['error_type']
            error_analysis['error_types'][error_type] = error_analysis['error_types'].get(error_type, 0) + 1
            
            # Analizza componenti affetti
            component = error['component']
            error_analysis['components_affected'][component] = error_analysis['components_affected'].get(component, 0) + 1
            
            # Analizza frequenza errori
            hour = error['timestamp'][:13]  # Raggruppa per ora
            error_analysis['error_frequency'][hour] = error_analysis['error_frequency'].get(hour, 0) + 1
            
        return error_analysis
        
    def suggest_fixes(self) -> List[str]:
        """Suggerisce possibili soluzioni basate sull'analisi degli errori"""
        suggestions = []
        error_analysis = self.analyze_errors()
        
        # Analizza componenti problematici
        for component, error_count in error_analysis['components_affected'].items():
            if error_count > 5:
                suggestions.append(f"High error rate in {component}. Consider checking error handling and input validation.")
                
        # Analizza tipi di errore comuni
        for error_type, count in error_analysis['error_types'].items():
            if count > 3:
                if "ValueError" in error_type:
                    suggestions.append(f"Multiple {error_type}s detected. Verify input data formats and validation.")
                elif "TimeoutError" in error_type:
                    suggestions.append(f"Multiple {error_type}s detected. Consider increasing timeouts or optimizing operations.")
                    
        # Analizza utilizzo memoria
        total_memory = sum(h.memory_usage for h in self.component_health.values())
        if total_memory > 1000 * 1024 * 1024:  # > 1GB
            suggestions.append("High memory usage detected. Consider implementing memory optimization techniques.")
            
        # Analizza performance
        slow_components = [
            name for name, health in self.component_health.items()
            if health.avg_response_time > 1.0  # > 1 secondo
        ]
        if slow_components:
            suggestions.append(f"Slow performance detected in components: {', '.join(slow_components)}")
            
        return suggestions
        
    def auto_recover(self, component_name: str) -> bool:
        """Tenta di recuperare automaticamente un componente problematico"""
        health = self.component_health.get(component_name)
        if not health or health.status != "error":
            return False
        
        try:
            recovery_strategies = {
                'analyzer': self._recover_analyzer,
                'classifier': self._recover_classifier,
                'manager': self._recover_manager
            }
            
            # Trova la strategia appropriata
            for key, strategy in recovery_strategies.items():
                if key in component_name.lower():
                    success = strategy(component_name)
                    if success:
                        logger.info(f"Successfully recovered {component_name}")
                        return True
                    
            logger.warning(f"No recovery strategy found for {component_name}")
            return False
            
        except Exception as e:
            logger.error(f"Error during auto recovery of {component_name}: {e}")
            return False
        
    def _recover_analyzer(self, component_name: str) -> bool:
        """Strategia di recovery per analizzatori"""
        try:
            health = self.component_health[component_name]
            
            # Reset contatori errori
            health.error_count = 0
            health.last_error = None
            
            # Reset metriche performance
            if component_name in self.performance_metrics:
                self.performance_metrics[component_name] = []
            
            health.status = "healthy"
            return True
            
        except Exception as e:
            logger.error(f"Error in analyzer recovery: {e}")
            return False
        
    def monitor_system_resources(self) -> Dict[str, float]:
        """Monitora le risorse di sistema"""
        try:
            process = psutil.Process()
            
            # CPU usage con media mobile
            cpu_percent = process.cpu_percent(interval=1.0)
            if not hasattr(self, '_cpu_history'):
                self._cpu_history = []
            self._cpu_history.append(cpu_percent)
            self._cpu_history = self._cpu_history[-10:]  # Keep last 10 samples
            
            # Memory usage con controllo leak
            memory_info = process.memory_info()
            if hasattr(self, '_last_memory_usage'):
                memory_growth = memory_info.rss - self._last_memory_usage
                if memory_growth > 100 * 1024 * 1024:  # 100MB growth
                    logger.warning("Possible memory leak detected")
            self._last_memory_usage = memory_info.rss
            
            return {
                'cpu_percent': np.mean(self._cpu_history),
                'memory_percent': process.memory_percent(),
                'memory_mb': memory_info.rss / 1024 / 1024,
                'num_threads': process.num_threads(),
                'io_counters': process.io_counters()._asdict()
            }
            
        except Exception as e:
            logger.error(f"Error monitoring system resources: {e}")
            return {}
        
    def get_component_health(self, component_name: str) -> Optional[ComponentHealth]:
        """Recupera lo stato di salute di un componente"""
        return self.component_health.get(component_name)
        
    def take_snapshot(self, label: str) -> Dict[str, Any]:
        """Cattura uno snapshot dello stato del sistema"""
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'label': label,
            'system_resources': self.monitor_system_resources(),
            'component_health': {
                name: vars(health)
                for name, health in self.component_health.items()
            }
        }
        return snapshot
        
    def _recover_classifier(self, component_name: str) -> bool:
        """Strategia di recovery per classificatori"""
        try:
            health = self.component_health[component_name]
            health.error_count = 0
            health.last_error = None
            health.status = "healthy"
            return True
        except Exception as e:
            logger.error(f"Error in classifier recovery: {e}")
            return False
            
    def _recover_manager(self, component_name: str) -> bool:
        """Strategia di recovery per manager"""
        try:
            health = self.component_health[component_name]
            health.error_count = 0
            health.last_error = None
            health.status = "healthy"
            if component_name in self.performance_metrics:
                self.performance_metrics[component_name] = []
            return True
        except Exception as e:
            logger.error(f"Error in manager recovery: {e}")
            return False

    def update_component_health(self, component_name: str, status: str, avg_time: float, memory_used: float) -> None:
        """Update component health metrics"""
        health = self.component_health.get(component_name)
        if not health:
            health = ComponentHealth(
                name=component_name,
                status='unknown',
                last_check=0.0,
                error_count=0,
                avg_response_time=0.0,
                memory_usage=0.0
            )
        
        health.status = status
        health.avg_response_time = float(avg_time)  # Explicit conversion
        health.memory_usage = float(memory_used)  # Explicit conversion
        
        self.component_health[component_name] = health

    def _create_health_record(self, component_name: str) -> ComponentHealth:
        """Create new health record"""
        return ComponentHealth(
            name=component_name,
            status='unknown',
            last_error=None,  # Initialize as None
            error_count=0,
            avg_response_time=0.0,
            memory_usage=0.0
        )
