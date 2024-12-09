import sys
from pathlib import Path

# Aggiungi la directory root al PYTHONPATH
root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
from PyQt5.QtCore import Qt
import cv2
import numpy as np
import logging
from datetime import datetime

# Importazioni corrette usando il percorso completo
from src.classifiers.emotion_classifier import EmotionClassifier
from src.classifiers.movement_classifier import MovementClassifier
from src.facial_analysis.expression_analyzer import ExpressionAnalyzer
from src.facial_analysis.pupil_analyzer import PupilAnalyzer
from src.video.video_thread import VideoThread
from src.metrics.metrics_manager import MetricsManager

logger = logging.getLogger(__name__)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Biometric Analysis")
        self.setup_ui()
        self.setup_video()
        self.setup_analyzers()
        self.setup_metrics()
        
    def setup_ui(self):
        """Setup dell'interfaccia utente"""
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Usa QVBoxLayout invece di QBoxLayout
        layout = QVBoxLayout()
        self.central_widget.setLayout(layout)
        
    def setup_video(self):
        """Setup del thread video"""
        self.video_thread = VideoThread()
        self.video_thread.change_pixmap_signal.connect(self.update_video)
        self.video_thread.error_signal.connect(self.handle_video_error)
        self.video_thread.start()
        
    def setup_analyzers(self):
        """Setup degli analizzatori"""
        try:
            self.expression_analyzer = ExpressionAnalyzer()
            self.pupil_analyzer = PupilAnalyzer()
            self.emotion_classifier = EmotionClassifier()
            self.movement_classifier = MovementClassifier()
        except Exception as e:
            logger.error(f"Error setting up analyzers: {e}")
            
    def setup_metrics(self):
        """Setup del gestore metriche"""
        self.metrics_manager = MetricsManager()
        
    def update_video(self, frame: np.ndarray):
        """Aggiorna il display video con il frame analizzato"""
        try:
            # Analizza il frame
            facial_metrics = self.expression_analyzer.analyze_frame(frame)
            pupil_metrics = self.pupil_analyzer.analyze_frame(frame)
            
            # Aggiorna le metriche
            self.metrics_manager.update_metrics({
                'facial': facial_metrics,
                'pupil': pupil_metrics,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            
    def handle_video_error(self, error_msg: str):
        """Gestisce gli errori video"""
        logger.error(f"Video error: {error_msg}")
        
    def closeEvent(self, event):
        """Gestisce la chiusura dell'applicazione"""
        self.video_thread.stop()
        event.accept()

def main():
    """Funzione principale"""
    logging.basicConfig(level=logging.INFO)
    
    app = QApplication([])
    window = MainWindow()
    window.show()
    return app.exec_()

if __name__ == "__main__":
    sys.exit(main())
