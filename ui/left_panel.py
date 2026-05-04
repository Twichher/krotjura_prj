from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QFrame,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont


class StatRow(QFrame):
    """Одна строка статистики: цветной индикатор + название + значение."""

    def __init__(self, label: str, color: str, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)

        self.title = QLabel(label)
        self.title.setStyleSheet(f"color: {color}; font-weight: bold;")
        self.title.setFont(QFont(11))

        self.value = QLabel("0")
        self.value.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.value.setFont(QFont(24, QFont.Weight.Bold))
        self.value.setStyleSheet("color: #ffffff;")

        layout.addWidget(self.title)
        layout.addWidget(self.value)

    def set_value(self, val: int):
        self.value.setText(str(val))


class LeftPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumWidth(220)
        self.setMaximumWidth(320)
        self.setStyleSheet("background-color: #1e1e1e;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        header = QLabel("🎯 Текущий кадр")
        header.setFont(QFont(14, QFont.Weight.Bold))
        header.setStyleSheet("color: #ffffff; margin-bottom: 8px;")
        layout.addWidget(header)

        self.moving = StatRow("Едет", "#4ade80")
        self.stopped = StatRow("Стоит", "#facc15")
        self.parked = StatRow("Припарковано", "#f87171")

        layout.addWidget(self.moving)
        layout.addWidget(self.stopped)
        layout.addWidget(self.parked)

        layout.addStretch()
        self.setEnabled(False)

    def reset(self):
        self.moving.set_value(0)
        self.stopped.set_value(0)
        self.parked.set_value(0)
        self.setEnabled(False)

    def set_enabled(self, enabled: bool):
        self.setEnabled(enabled)

    def update_stats(self, moving: int, stopped: int, parked: int):
        self.moving.set_value(moving)
        self.stopped.set_value(stopped)
        self.parked.set_value(parked)
