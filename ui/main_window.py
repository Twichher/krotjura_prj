from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QSplitter,
)
from PyQt6.QtCore import Qt

from ui.left_panel import LeftPanel
from ui.center_panel import CenterPanel
from ui.right_panel import RightPanel


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Krotjura — Анализ дорожного движения")
        self.setMinimumSize(1200, 700)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        self.left_panel = LeftPanel()
        splitter.addWidget(self.left_panel)

        self.center_panel = CenterPanel()
        splitter.addWidget(self.center_panel)

        self.right_panel = RightPanel()
        splitter.addWidget(self.right_panel)

        splitter.setSizes([250, 700, 250])
        layout.addWidget(splitter)

        # Подключаем сигналы
        self.center_panel.video_loaded.connect(self.on_video_loaded)
        self.center_panel.road_confirmed.connect(self.on_road_confirmed)
        self.center_panel.processing_finished.connect(self.on_processing_finished)
        self.center_panel.frame_stats_changed.connect(self.left_panel.update_stats)

    def on_video_loaded(self, path: str):
        """Вызывается после выбора файла и отображения 1-го кадра."""
        self.left_panel.reset()
        self.right_panel.reset()

    def on_road_confirmed(self, polygon: list):
        """Вызывается после подтверждения полигона дороги пользователем."""
        pass

    def on_processing_finished(self, session):
        """Вызывается после окончания pre-processing."""
        self.left_panel.set_enabled(True)
        self.right_panel.set_session(session)
