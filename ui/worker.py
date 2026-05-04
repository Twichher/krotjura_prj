"""Qt Worker для запуска VideoProcessor в фоновом потоке.

Сигналы:
    progress(int) — процент обработки (0–100)
    finished(Session) — результат готов
    error(str) — ошибка во время обработки
"""

from PyQt6.QtCore import QThread, pyqtSignal
from core.processor import VideoProcessor
from core.session import Session


class ProcessingWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(object)  # Session
    error = pyqtSignal(str)

    def __init__(self, video_path: str, road_polygon: list, max_duration_sec=None, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.road_polygon = road_polygon
        self.max_duration_sec = max_duration_sec

    def run(self):
        try:
            processor = VideoProcessor(
                self.video_path,
                self.road_polygon,
                progress_callback=self._on_progress,
                max_duration_sec=self.max_duration_sec,
                should_stop=lambda: self.isInterruptionRequested(),
            )
            session = processor.process()
            if not self.isInterruptionRequested():
                self.finished.emit(session)
        except Exception as e:
            if not self.isInterruptionRequested():
                self.error.emit(str(e))

    def _on_progress(self, pct: int):
        self.progress.emit(pct)
