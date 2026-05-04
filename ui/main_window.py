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

        # Центральный виджет с горизонтальным layout
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Сплиттер для ресайза панелей
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Левая панель — текущая статистика
        self.left_panel = LeftPanel()
        splitter.addWidget(self.left_panel)

        # Центральная панель — видео + контролы
        self.center_panel = CenterPanel()
        splitter.addWidget(self.center_panel)

        # Правая панель — история
        self.right_panel = RightPanel()
        splitter.addWidget(self.right_panel)

        # Пропорции: левая 1, центр 3, правая 1
        splitter.setSizes([250, 700, 250])
        layout.addWidget(splitter)

        # Подключаем сигналы
        self.center_panel.video_loaded.connect(self.on_video_loaded)
        self.center_panel.processing_finished.connect(self.on_processing_finished)

    def on_video_loaded(self, path: str):
        """Вызывается после выбора файла и отображения 1-го кадра."""
        self.left_panel.reset()
        self.right_panel.reset()

    def on_processing_finished(self, results: dict):
        """Вызывается после окончания pre-processing."""
        self.left_panel.set_enabled(True)
        self.right_panel.set_results(results)
