from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QStackedWidget, QProgressBar, QSlider, QFileDialog, QMessageBox,
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QPixmap, QImage
import cv2
import numpy as np
import os

from core.video_loader import VideoLoader
from core.road_detector import RoadDetector
from core.session import Session
from ui.polygon_editor import PolygonEditor
from ui.worker import ProcessingWorker
from core.visualizer import draw_overlay


def cv2_to_pixmap(frame: np.ndarray) -> QPixmap:
    """BGR (OpenCV) → RGB QPixmap."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


class CenterPanel(QWidget):
    video_loaded = pyqtSignal(str)
    video_deleted = pyqtSignal()
    road_confirmed = pyqtSignal(list)   # List[Tuple[int, int]]
    processing_finished = pyqtSignal(object)  # Session
    frame_stats_changed = pyqtSignal(int, int, int)  # moving, stopped, parked

    def __init__(self):
        super().__init__()
        self.video_path = None
        self.loader = None
        self.polygon_editor = None
        self.session = None
        self.worker = None
        self._timer = None
        self._current_frame = 0
        self._last_read_frame = -1  # для оптимизации последовательного чтения

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

        # Кнопка удалить (под плеером)
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
        while self.polygon_layout.count():
            item = self.polygon_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                self._clear_layout(item.layout())

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
        pass

    def _on_confirm_road(self):
        if self.polygon_editor is None:
            return
        confirmed_polygon = self.polygon_editor.get_polygon()
        self.road_confirmed.emit(confirmed_polygon)

        # Проверка длительности
        duration = self.loader.duration_sec if self.loader else 0.0
        if duration > 60.0:
            msg = QMessageBox(self)
            msg.setWindowTitle("Длительное видео")
            msg.setText(f"Видео длится {duration:.0f} секунд.\nОбрезать до 1 минуты для ускорения?")
            msg.setStandardButtons(
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            msg.button(QMessageBox.StandardButton.Yes).setText("✂️ Обрезать до 1 минуты")
            msg.button(QMessageBox.StandardButton.No).setText("📼 Обрабатывать полностью")
            msg.setDefaultButton(QMessageBox.StandardButton.Yes)
            reply = msg.exec()
            if reply == QMessageBox.StandardButton.Yes:
                self.stack.setCurrentIndex(2)
                self._start_processing(confirmed_polygon, max_duration_sec=60.0)
            else:
                self.stack.setCurrentIndex(2)
                self._start_processing(confirmed_polygon)
        else:
            self.stack.setCurrentIndex(2)
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

        # Кнопка отмены обработки
        cancel_btn = QPushButton("🛑 Отменить обработку")
        cancel_btn.setFont(QFont("", 12))
        cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc2626; color: white;
                border-radius: 8px; padding: 10px 20px; font-weight: bold;
            }
            QPushButton:hover { background-color: #b91c1c; }
        """)
        cancel_btn.clicked.connect(self._on_cancel_processing)
        layout.addWidget(cancel_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        return w

    def _start_processing(self, polygon: list, max_duration_sec=None):
        self.progress_bar.setValue(0)
        self.progress_label.setText("⏳ Анализ видео...")
        self.progress_label.setStyleSheet("color: #ffffff;")
        self.worker = ProcessingWorker(self.video_path, polygon, max_duration_sec)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_processing_done)
        self.worker.error.connect(self._on_processing_error)
        self.worker.start()

    def _on_progress(self, pct: int):
        self.progress_bar.setValue(pct)

    def _on_processing_done(self, session: Session):
        self.session = session
        self.stack.setCurrentIndex(3)  # PLAYER
        self.delete_btn.setVisible(True)
        self._init_player()
        self.processing_finished.emit(session)

    def _on_processing_error(self, msg: str):
        self.progress_label.setText(f"❌ Ошибка: {msg}")
        self.progress_label.setStyleSheet("color: #f87171;")

    def _on_cancel_processing(self):
        """Прервать обработку и сбросить всё."""
        if self.worker and self.worker.isRunning():
            self.worker.requestInterruption()
            self.worker.wait(3000)
            if self.worker.isRunning():
                self.worker.terminate()
            self.worker = None
        self._on_delete()

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
        self.play_btn.clicked.connect(self._toggle_play)

        self.timeline = QSlider(Qt.Orientation.Horizontal)
        self.timeline.setRange(0, 100)
        self.timeline.sliderReleased.connect(self._on_seek)

        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setStyleSheet("color: #aaaaaa;")
        self.time_label.setMinimumWidth(100)

        controls.addWidget(self.play_btn)
        controls.addWidget(self.timeline, stretch=1)
        controls.addWidget(self.time_label)
        layout.addLayout(controls)
        return w

    def _init_player(self):
        """Подготовить плеер после окончания обработки."""
        if self.session is None:
            return
        self._current_frame = 0
        self._last_read_frame = -1
        self.loader.seek(0)
        self._last_read_frame = 0
        self.timeline.setRange(0, self.session.total_frames - 1)
        self._update_time_label()
        self._show_frame(0)

    def _toggle_play(self):
        if self._timer is None:
            self._timer = QTimer(self)
            self._timer.timeout.connect(self._next_frame)
        if self._timer.isActive():
            self._timer.stop()
            self.play_btn.setText("▶")
        else:
            interval = int(1000 / self.session.fps) if self.session.fps > 0 else 33
            self._timer.start(interval)
            self.play_btn.setText("⏸")

    def _next_frame(self):
        if self.session is None:
            return
        if self._current_frame >= self.session.total_frames - 1:
            self._timer.stop()
            self.play_btn.setText("▶")
            return
        self._current_frame += 1
        self.timeline.setValue(self._current_frame)
        # Последовательное чтение (быстро) vs seek (медленно)
        if self._current_frame == self._last_read_frame + 1:
            frame = self.loader.read_frame()
            if frame is not None:
                self._last_read_frame = self._current_frame
                self._display_frame(frame)
        else:
            self._show_frame(self._current_frame)

    def _on_seek(self):
        self._current_frame = self.timeline.value()
        if self._timer and self._timer.isActive():
            self._timer.stop()
            self.play_btn.setText("▶")
        self._last_read_frame = self._current_frame
        self.loader.seek(self._current_frame)
        self._show_frame(self._current_frame)

    def _show_frame(self, frame_id: int):
        """Показать кадр с overlay через seek (для перемотки)."""
        if self.loader is None:
            return
        frame = self.loader.seek(frame_id)
        self._last_read_frame = frame_id
        if frame is None:
            return
        self._display_frame(frame)

    def _display_frame(self, frame: np.ndarray):
        """Отрисовать overlay и показать кадр (frame уже прочитан)."""
        frame_id = self._current_frame
        # Overlay
        if self.session and frame_id < len(self.session.frame_results):
            dets = self.session.frame_results[frame_id].detections
            frame = draw_overlay(
                frame, dets, self.session.road_polygon,
                draw_road=True, draw_boxes=True, draw_labels=True,
            )

        pixmap = cv2_to_pixmap(frame)
        self.video_label.setPixmap(pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))
        self._update_time_label()

        # Обновить текущую статистику
        if self.session:
            m, s, p = self.session.get_stats_at_frame(frame_id)
            self.frame_stats_changed.emit(m, s, p)

    def _update_time_label(self):
        if self.session is None:
            return
        cur = self._current_frame / self.session.fps if self.session.fps > 0 else 0
        tot = self.session.total_frames / self.session.fps if self.session.fps > 0 else 0
        self.time_label.setText(f"{cur:.1f}s / {tot:.1f}s")

    # ------------------------------------------------------------------
    # Публичные методы
    # ------------------------------------------------------------------
    def load_video(self, path: str):
        self.video_path = path
        self.loader = VideoLoader(path)
        frame = self.loader.read_first_frame()
        detector = RoadDetector()
        polygon = detector.detect(frame)
        self._show_polygon_editor(frame, polygon)
        self.video_loaded.emit(path)

    def _on_delete(self):
        self.video_path = None
        if self.loader:
            self.loader.release()
            self.loader = None
        self.polygon_editor = None
        self.session = None
        self.progress_bar.setValue(0)
        self.progress_label.setText("⏳ Анализ видео...")
        self.progress_label.setStyleSheet("color: #ffffff;")
        self._current_frame = 0
        self._last_read_frame = -1

        if self._timer and self._timer.isActive():
            self._timer.stop()
            self._timer = None

        while self.polygon_layout.count():
            item = self.polygon_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                self._clear_layout(item.layout())

        self.stack.setCurrentIndex(0)
        self.delete_btn.setVisible(False)
        self.video_deleted.emit()

    def _clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                self._clear_layout(item.layout())
