from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont


class HistoryRow(QFrame):
    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)

        self.title = QLabel(label)
        self.title.setFont(QFont("", 11))
        self.title.setStyleSheet("color: #cccccc;")

        self.value = QLabel("0")
        self.value.setFont(QFont("", 14, QFont.Weight.Bold))
        self.value.setStyleSheet("color: #ffffff;")
        self.value.setAlignment(Qt.AlignmentFlag.AlignRight)

        layout.addWidget(self.title)
        layout.addWidget(self.value)

    def set_value(self, val: int):
        self.value.setText(str(val))


class RightPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumWidth(220)
        self.setMaximumWidth(320)
        self.setStyleSheet("background-color: #1e1e1e;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        header = QLabel("📊 История")
        header.setFont(QFont("", 14, QFont.Weight.Bold))
        header.setStyleSheet("color: #ffffff; margin-bottom: 8px;")
        layout.addWidget(header)

        self.total_moving = HistoryRow("Всего проехало")
        self.total_stopped = HistoryRow("Всего остановилось")
        self.total_parked = HistoryRow("Пик припаркованных")
        self.total_tracks = HistoryRow("Всего машин (треков)")

        layout.addWidget(self.total_moving)
        layout.addWidget(self.total_stopped)
        layout.addWidget(self.total_parked)
        layout.addWidget(self.total_tracks)

        layout.addStretch()
        self.setEnabled(False)

    def reset(self):
        self.total_moving.set_value(0)
        self.total_stopped.set_value(0)
        self.total_parked.set_value(0)
        self.total_tracks.set_value(0)
        self.setEnabled(False)

    def set_session(self, session):
        """Заполнить статистику из Session."""
        self.setEnabled(True)
        self.total_moving.set_value(session.total_moving_tracks)
        self.total_stopped.set_value(session.total_stopped_tracks)
        self.total_parked.set_value(session.total_parked_tracks)
        self.total_tracks.set_value(session.total_unique_tracks)
