from typing import Optional, Union
import numpy as np
import cv2
import logging

try:
    from PyQt5.QtCore import QThread, pyqtSignal
except ImportError:
    try:
        from PyQt6.QtCore import QThread, pyqtSignal
    except ImportError:
        raise ImportError("Neither PyQt5 nor PyQt6 found")

logger = logging.getLogger(__name__)

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    error_signal = pyqtSignal(str)
    
    def __init__(self, source: int = 0):
        super().__init__()
        self.source = source
        self._capture: Optional[cv2.VideoCapture] = None
        self.running = True
        
    def run(self) -> None:
        try:
            self._capture = cv2.VideoCapture(self.source)
            if not self._capture.isOpened():
                raise RuntimeError("Failed to open video capture")
                
            while self.running:
                ret, frame = self._capture.read()
                if ret and isinstance(frame, np.ndarray):
                    self.change_pixmap_signal.emit(frame)
                elif not ret:
                    logger.warning("Failed to read frame")
                    
        except Exception as e:
            logger.error(f"Video thread error: {e}")
            self.error_signal.emit(str(e))
        finally:
            self.release_capture()
            
    def stop(self) -> None:
        self.running = False
        self.release_capture()
        
    def release_capture(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None