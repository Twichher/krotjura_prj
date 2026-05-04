"""Классификация состояния трека: moving / stopped / parked.

Логика основана на скорости, положении относительно полигона дороги
и длительности остановки.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import cv2
import numpy as np


@dataclass
class TrackState:
    track_id: int
    positions: List[Tuple[int, int]] = field(default_factory=list)  # центры bbox
    timestamps: List[float] = field(default_factory=list)  # секунды от начала видео
    states: List[str] = field(default_factory=list)  # история состояний
    last_state: str = "moving"
    stop_start_time: Optional[float] = None


class StateClassifier:
    def __init__(
        self,
        road_polygon: List[Tuple[int, int]],
        move_threshold_px: float = 1.5,      # px/кадр (30 FPS, камера высоко)
        stop_timeout_sec: float = 2.0,       # секунд до 'stopped'
        park_timeout_sec: float = 4.0,       # секунд до 'parked' на дороге
        speed_window: int = 3,               # усреднение по N кадрам
    ):
        self.road_polygon = np.array(road_polygon, dtype=np.int32)
        self.move_threshold = move_threshold_px
        self.stop_timeout = stop_timeout_sec
        self.park_timeout = park_timeout_sec
        self.speed_window = speed_window

        self.tracks: Dict[int, TrackState] = {}

    def update(self, frame_id: int, timestamp: float, detections: List) -> List[dict]:
        """Обновить состояния для текущего кадра.

        Args:
            detections: список объектов с .track_id и .bbox

        Returns:
            Список словарей: [{track_id, bbox, state, speed}, ...]
        """
        results = []
        for det in detections:
            tid = det.track_id
            x1, y1, x2, y2 = det.bbox
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            if tid not in self.tracks:
                self.tracks[tid] = TrackState(track_id=tid)

            track = self.tracks[tid]
            track.positions.append((cx, cy))
            track.timestamps.append(timestamp)

            # Скорость (px/кадр, усредненная)
            speed = self._calc_speed(track)

            # Внутри ли дороги?
            on_road = cv2.pointPolygonTest(self.road_polygon, (cx, cy), False) >= 0

            # Классификация
            state = self._classify(track, speed, on_road, timestamp)
            track.last_state = state
            track.states.append(state)

            results.append({
                "track_id": tid,
                "bbox": det.bbox,
                "state": state,
                "speed": speed,
                "cls_name": det.cls_name,
            })

        return results

    def _calc_speed(self, track: TrackState) -> float:
        """Средняя скорость за последние speed_window позиций (px/кадр)."""
        if len(track.positions) < 2:
            return 999.0  # считаем движущимся по умолчанию

        window = min(self.speed_window, len(track.positions))
        total_dist = 0.0
        for i in range(1, window):
            x1, y1 = track.positions[-i - 1]
            x2, y2 = track.positions[-i]
            total_dist += ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        return total_dist / (window - 1)

    def _classify(
        self, track: TrackState, speed: float, on_road: bool, now: float
    ) -> str:
        """Вернуть одно из: moving, stopped, parked."""
        if speed > self.move_threshold:
            track.stop_start_time = None
            return "moving"

        # Стоит — фиксируем время начала остановки
        if track.stop_start_time is None:
            track.stop_start_time = now

        stopped_for = now - track.stop_start_time

        # Если вне дороги — сразу parked (скорее всего парковка/обочина)
        if not on_road:
            return "parked"

        # На дороге: короткая остановка = stopped, долгая = parked
        if stopped_for < self.stop_timeout:
            return "stopped"
        elif stopped_for < self.park_timeout:
            return "stopped"
        else:
            return "parked"

    def get_track_history(self, track_id: int) -> TrackState:
        return self.tracks[track_id]
