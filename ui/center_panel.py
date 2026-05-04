from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QStackedWidget, QProgressBar, QSlider, QFileDialog,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QPixmap, QIcon
import cv2
import os


class CenterPanel(QWidget):
    video_loaded = pyqtSignal(str)
    processing_finished = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.video_path = None
        self.cap = None
        self.total_frames = 0
        self.fps = 30.0

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # === Stacked widget с 4 состояниями ===
        self.stack = QStackedWidget()

        # 0: EMPTY — кнопка загрузки
        self.empty_widget = self._build_empty_state()
        self.stack.addWidget(self.empty_widget)

        # 1: POLYGON — редактор дороги
        self.polygon_widget = self._build_polygon_state()
        self.stack.addWidget(self.polygon_widget)

        # 2: PROCESSING — прогресс
        self.processing_widget = self._build_processing_state()
        self.stack.addWidget(self.processing_widget)

        # 3: PLAYER — плеер
        self.player_widget = self._build_player_state()
        self.stack.addWidget(self.player_widget)

        layout.addWidget(self.stack, stretch=1)

        # === Нижняя панель (кнопка удалить) ===
        bottom = QHBoxLayout()
        bottom.addStretch()
        self.delete_btn = QPushButton("🗑 Удалить видео")
        self.delete_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc2626;
                color: white;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
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
        self.upload_btn.setFont(QFont(16))
        self.upload_btn.setMinimumSize(300, 80)
        self.upload_btn.setStyleSheet("""
            QPushButton {
                background-color: #2563eb;
                color: white;
                border-radius: 12px;
                font-weight: bold;
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
    # STATE 1: POLYGON
    # ------------------------------------------------------------------
    def _build_polygon_state(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.polygon_label = QLabel("🛣 Здесь будет редактор полигона дороги")
        self.polygon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.polygon_label.setFont(QFont(14))
        self.polygon_label.setStyleSheet("color: #aaaaaa;")
        layout.addWidget(self.polygon_label)

        self.confirm_road_btn = QPushButton("✅ Подтвердить дорогу и начать анализ")
        self.confirm_road_btn.setStyleSheet("""
            QPushButton {
                background-color: #16a34a;
                color: white;
                border-radius: 8px;
                padding: 10px 20px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #15803d; }
        """)
        self.confirm_road_btn.clicked.connect(self._on_confirm_road)
        layout.addWidget(self.confirm_road_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        return w

    def _on_confirm_road(self):
        # TODO: запустить pre-processing в отдельном потоке
        self.stack.setCurrentIndex(2)  # PROCESSING
        self._simulate_processing()

    # ------------------------------------------------------------------
    # STATE 2: PROCESSING
    # ------------------------------------------------------------------
    def _build_processing_state(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.progress_label = QLabel("⏳ Анализ видео...")
        self.progress_label.setFont(QFont(16))
        self.progress_label.setStyleSheet("color: #ffffff;")
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.progress_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setMinimumWidth(400)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #374151;
                border-radius: 8px;
                text-align: center;
                color: white;
                height: 24px;
            }
            QProgressBar::chunk {
                background-color: #2563eb;
                border-radius: 6px;
            }
        """)
        layout.addWidget(self.progress_bar)
        return w

    def _simulate_processing(self):
        # TODO: заменить на реальную обработку в QThread
        import random
        val = self.progress_bar.value()
        val = min(val + random.randint(5, 15), 100)
        self.progress_bar.setValue(val)
        if val < 100:
            from PyQt6.QtCore import QTimer
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

        # Видео-область
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: #000000; border-radius: 8px;")
        self.video_label.setMinimumSize(640, 360)
        layout.addWidget(self.video_label, stretch=1)

        # Контролы
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
    # Общие методы
    # ------------------------------------------------------------------
    def load_video(self, path: str):
        self.video_path = path
        self.cap = cv2.VideoCapture(path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        # Показать 1-й кадр в POLYGON-режиме
        ret, frame = self.cap.read()
        if ret:
            # TODO: отрисовать кадр в polygon_label с возможностью редактирования полигона
            self.polygon_label.setText(
                f"Загружено: {os.path.basename(path)}\n"
                f"Кадров: {self.total_frames}, FPS: {self.fps:.1f}\n"
                f"(Здесь будет редактор полигона поверх кадра)"
            )

        self.stack.setCurrentIndex(1)  # POLYGON
        self.video_loaded.emit(path)

    def _on_delete(self):
        self.video_path = None
        if self.cap:
            self.cap.release()
            self.cap = None
        self.total_frames = 0
        self.progress_bar.setValue(0)
        self.stack.setCurrentIndex(0)  # EMPTY
        self.delete_btn.setVisible(False)
