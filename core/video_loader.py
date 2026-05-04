"""Чистая обертка над OpenCV VideoCapture.

Не зависит от UI. Принимает путь — возвращает кадры как numpy arrays.
"""

from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np


class VideoLoader:
    def __init__(self, path: str):
        self.path = Path(path)
        self._cap: Optional[cv2.VideoCapture] = None
        self._fps: float = 0.0
        self._frame_count: int = 0
        self._width: int = 0
        self._height: int = 0
        self._is_open: bool = False

        self._open()

    def _open(self) -> None:
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Не удалось открыть видео: {self.path}")

        self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._is_open = True

    # ------------------------------------------------------------------
    # Свойства
    # ------------------------------------------------------------------
    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def duration_sec(self) -> float:
        return self._frame_count / self._fps if self._fps > 0 else 0.0

    @property
    def size(self) -> Tuple[int, int]:
        """(width, height)"""
        return self._width, self._height

    # ------------------------------------------------------------------
    # Чтение кадров
    # ------------------------------------------------------------------
    def read_frame(self) -> Optional[np.ndarray]:
        """Следующий кадр (BGR). None если конец."""
        if not self._is_open or self._cap is None:
            return None
        ret, frame = self._cap.read()
        return frame if ret else None

    def seek(self, frame_id: int) -> Optional[np.ndarray]:
        """Перейти к кадру по номеру и прочитать его."""
        if not self._is_open or self._cap is None:
            return None
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(frame_id, self._frame_count - 1)))
        return self.read_frame()

    def read_first_frame(self) -> np.ndarray:
        """Прочитать 1-й кадр. Выбросит ошибку, если не удалось."""
        frame = self.seek(0)
        if frame is None:
            raise RuntimeError("Не удалось прочитать 1-й кадр")
        return frame

    # ------------------------------------------------------------------
    # Утилиты
    # ------------------------------------------------------------------
    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._is_open = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()
        return False
