from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QStackedWidget, QProgressBar, QSlider, QFileDialog,
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QPixmap, QImage
import cv2
import numpy as np
import os

from core.video_loader import VideoLoader
from core.road_detector import RoadDetector
from ui.polygon_editor import PolygonEditor


def cv2_to_pixmap(frame: np.ndarray) -> QPixmap:
    """BGR (OpenCV) → RGB QPixmap."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


class CenterPanel(QWidget):
    video_loaded = pyqtSignal(str)
    road_confirmed = pyqtSignal(list)   # List[Tuple[int, int]]
    processing_finished = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.video_path = None
        self.loader = None
        self.polygon_editor = None
        self.frame_results = None

        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self.stack = QStackedWidget()

        # 0: EMPTY
        self.empty_widget = self._build_empty_state()
        self.stack.addWidget(self.empty_widget)

        # 1: POLYGON — контейнер, куда динамически вставляется редактор
        self.polygon_container = QWidget()
        self.polygon_layout = QVBoxLayout(self.polygon_container)
        self.polygon_layout.setContentsMargins(0, 0, 0, 0)
        self.stack.addWidget(self.polygon_container)

        # 2: PROCESSING
        self.processing_widget = self._build_processing_state()
        self.stack.addWidget(self.processing_widget)

        # 3: PLAYER
        self.player_widget = self._build_player_state()
        self.stack.addWidget(self.player_widget)

        layout.addWidget(self.stack, stretch=1)

        # Кнопка удалить
        bottom = QHBoxLayout()
        bottom.addStretch()
        self.delete_btn = QPushButton("🗑 Удалить видео")
        self.delete_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc2626; color: white;
                border-radius: 6px; padding: 8px 16px; font-weight: bold;
            }
            QPushButton:hover { background-color: #b91c1c; }
        """)
        self.delete_btn.setVisible(False)
        self.delete_btn.clicked.connect(self._on_delete)
        bottom.addWidget(self.delete_btn)
        layout.addLayout(bottom)

        self.setStyleSheet("background-color: #121212;")

    # ------------------------------------------------------------------
    # STATE 0: EMPTY
    # ------------------------------------------------------------------
    def _build_empty_state(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.upload_btn = QPushButton("📁 Загрузить видео")
        self.upload_btn.setFont(QFont("", 16))
        self.upload_btn.setMinimumSize(300, 80)
        self.upload_btn.setStyleSheet("""
            QPushButton {
                background-color: #2563eb; color: white;
                border-radius: 12px; font-weight: bold;
            }
            QPushButton:hover { background-color: #1d4ed8; }
        """)
        self.upload_btn.clicked.connect(self._on_upload)
        layout.addWidget(self.upload_btn)
        return w

    def _on_upload(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите видео", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )
        if path:
            self.load_video(path)

    # ------------------------------------------------------------------
    # STATE 1: POLYGON EDITOR
    # ------------------------------------------------------------------
    def _show_polygon_editor(self, frame: np.ndarray, polygon: list):
        """Создать/пересоздать редактор полигона поверх 1-го кадра."""
        # Очистить контейнер
        while self.polygon_layout.count():
            item = self.polygon_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        pixmap = cv2_to_pixmap(frame)
        self.polygon_editor = PolygonEditor(pixmap, polygon)
        self.polygon_editor.polygon_changed.connect(self._on_polygon_changed)
        self.polygon_layout.addWidget(self.polygon_editor)

        # Кнопки внизу: Подтвердить (слева) + Удалить (справа)
        btn_row = QHBoxLayout()

        confirm = QPushButton("✅ Подтвердить дорогу и начать анализ")
        confirm.setFont(QFont("", 12))
        confirm.setStyleSheet("""
            QPushButton {
                background-color: #16a34a; color: white;
                border-radius: 8px; padding: 10px 20px; font-weight: bold;
            }
            QPushButton:hover { background-color: #15803d; }
        """)
        confirm.clicked.connect(self._on_confirm_road)
        btn_row.addWidget(confirm)

        btn_row.addStretch()

        delete = QPushButton("🗑 Удалить видео")
        delete.setFont(QFont("", 12))
        delete.setStyleSheet("""
            QPushButton {
                background-color: #dc2626; color: white;
                border-radius: 8px; padding: 10px 20px; font-weight: bold;
            }
            QPushButton:hover { background-color: #b91c1c; }
        """)
        delete.clicked.connect(self._on_delete)
        btn_row.addWidget(delete)

        self.polygon_layout.addLayout(btn_row)

        self.stack.setCurrentIndex(1)

    def _on_polygon_changed(self, polygon: list):
        """Вызывается при drag-вершин или добавлении точки."""
        pass  # Можно логировать или показывать preview

    def _on_confirm_road(self):
        if self.polygon_editor is None:
            return
        confirmed_polygon = self.polygon_editor.get_polygon()
        self.road_confirmed.emit(confirmed_polygon)
        self.stack.setCurrentIndex(2)  # PROCESSING
        self._start_processing(confirmed_polygon)

    # ------------------------------------------------------------------
    # STATE 2: PROCESSING
    # ------------------------------------------------------------------
    def _build_processing_state(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.progress_label = QLabel("⏳ Анализ видео...")
        self.progress_label.setFont(QFont("", 16))
        self.progress_label.setStyleSheet("color: #ffffff;")
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.progress_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setMinimumWidth(400)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #374151; border-radius: 8px;
                text-align: center; color: white; height: 24px;
            }
            QProgressBar::chunk { background-color: #2563eb; border-radius: 6px; }
        """)
        layout.addWidget(self.progress_bar)
        return w

    def _start_processing(self, polygon: list):
        # TODO: заменить на реальную обработку в QThread
        self.progress_bar.setValue(0)
        self._simulate_processing()

    def _simulate_processing(self):
        import random
        val = self.progress_bar.value()
        val = min(val + random.randint(5, 15), 100)
        self.progress_bar.setValue(val)
        if val < 100:
            QTimer.singleShot(200, self._simulate_processing)
        else:
            self.stack.setCurrentIndex(3)  # PLAYER
            self.delete_btn.setVisible(True)
            self.processing_finished.emit({})

    # ------------------------------------------------------------------
    # STATE 3: PLAYER
    # ------------------------------------------------------------------
    def _build_player_state(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: #000000; border-radius: 8px;")
        self.video_label.setMinimumSize(640, 360)
        layout.addWidget(self.video_label, stretch=1)

        controls = QHBoxLayout()
        controls.setSpacing(12)

        self.play_btn = QPushButton("▶")
        self.play_btn.setFixedSize(40, 40)
        self.play_btn.setStyleSheet("font-size: 16px;")

        self.timeline = QSlider(Qt.Orientation.Horizontal)
        self.timeline.setRange(0, 100)

        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setStyleSheet("color: #aaaaaa;")
        self.time_label.setMinimumWidth(100)

        controls.addWidget(self.play_btn)
        controls.addWidget(self.timeline, stretch=1)
        controls.addWidget(self.time_label)
        layout.addLayout(controls)
        return w

    # ------------------------------------------------------------------
    # Публичные методы
    # ------------------------------------------------------------------
    def load_video(self, path: str):
        self.video_path = path
        self.loader = VideoLoader(path)

        # Читаем 1-й кадр
        frame = self.loader.read_first_frame()

        # Авто-полигон дороги
        detector = RoadDetector()
        polygon = detector.detect(frame)

        # Показываем редактор
        self._show_polygon_editor(frame, polygon)
        self.video_loaded.emit(path)

    def _on_delete(self):
        self.video_path = None
        if self.loader:
            self.loader.release()
            self.loader = None
        self.polygon_editor = None
        self.progress_bar.setValue(0)

        # Очистить контейнер редактора полигона
        while self.polygon_layout.count():
            item = self.polygon_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                self._clear_layout(item.layout())

        self.stack.setCurrentIndex(0)
        self.delete_btn.setVisible(False)

    def _clear_layout(self, layout):
        """Рекурсивная очистка layout."""
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                self._clear_layout(item.layout())
