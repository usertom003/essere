import sys
import traceback
import logging
from typing import Callable, Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path
from .auto_debugger import AutoDebugger

logger = logging.getLogger(__name__)

class GlobalErrorHandler:
    """Gestisce le eccezioni a livello globale"""
    
    def __init__(self, debugger: Optional[AutoDebugger] = None):
        self.debugger = debugger
        self.error_callbacks: List[Callable[[Exception, str], None]] = []
        self.error_history: List[Dict[str, Any]] = []
        
    def add_callback(self, callback: Callable[[Exception, str], None]) -> None:
        """Aggiunge un callback per la gestione degli errori"""
        if callback not in self.error_callbacks:
            self.error_callbacks.append(callback)
            
    def remove_callback(self, callback: Callable[[Exception, str], None]) -> None:
        """Rimuove un callback"""
        if callback in self.error_callbacks:
            self.error_callbacks.remove(callback)
            
    def install(self):
        """Installa il gestore delle eccezioni globale"""
        sys.excepthook = self.handle_exception
        
    def handle_exception(self, exc_type, exc_value, exc_traceback):
        """Gestisce un'eccezione non gestita"""
        try:
            # Log dell'errore
            error_msg = ''.join(traceback.format_exception(
                exc_type, exc_value, exc_traceback
            ))
            logger.error(f"Unhandled exception:\n{error_msg}")
            
            # Notifica il debugger
            if self.debugger:
                self.debugger._handle_error(
                    "system",
                    exc_value,
                    error_msg
                )
                
            # Esegui callback personalizzati
            for callback in self.error_callbacks:
                try:
                    callback(exc_value, error_msg)
                except Exception as e:
                    logger.error(f"Error in error callback: {e}")
                    
            # Salva crash report
            self.save_crash_report(error_msg)
            
        except Exception as e:
            # Fallback per errori nel gestore errori
            print(f"Error in error handler: {e}", file=sys.stderr)
            
    def save_crash_report(self, error_msg: str):
        """Salva un report del crash"""
        try:
            crash_dir = Path("data/crashes")
            crash_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            crash_path = crash_dir / f"crash_{timestamp}.txt"
            
            with open(crash_path, 'w') as f:
                f.write(f"Crash Report - {datetime.now().isoformat()}\n")
                f.write("-" * 80 + "\n")
                f.write(error_msg)
                
        except Exception as e:
            logger.error(f"Error saving crash report: {e}") 