"""Классификация состояния трека: moving / stopped / parked.

Логика основана на скорости (с гистерезисом), положении относительно
полигона дороги и длительности остановки.
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
        # Гистерезис скорости
        move_threshold_low: float = 1.0,     # px/кадр: moving → stopped
        move_threshold_high: float = 3.0,    # px/кадр: stopped → moving
        # Таймауты остановки
        stop_timeout_sec: float = 2.0,       # секунд до 'stopped' (светофор)
        park_timeout_on_road: float = 20.0,  # секунд до 'parked' на дороге
        park_timeout_off_road: float = 3.0,  # секунд до 'parked' вне дороги
        speed_window: int = 3,               # усреднение по N кадрам
    ):
        self.road_polygon = np.array(road_polygon, dtype=np.int32)
        self.move_threshold_low = move_threshold_low
        self.move_threshold_high = move_threshold_high
        self.stop_timeout = stop_timeout_sec
        self.park_timeout_on_road = park_timeout_on_road
        self.park_timeout_off_road = park_timeout_off_road
        self.speed_window = speed_window

        self.tracks: Dict[int, TrackState] = {}

    def update(self, frame_id: int, timestamp: float, detections: List) -> List[dict]:
        """Обновить состояния для текущего кадра."""
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

            speed = self._calc_speed(track)
            on_road = self._is_on_road(x1, y1, x2, y2)
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

    # ------------------------------------------------------------------
    # Геометрия
    # ------------------------------------------------------------------
    def _is_on_road(self, x1: int, y1: int, x2: int, y2: int) -> bool:
        """>= 3 из 5 точек (центр + углы) внутри полигона → 'на дороге'."""
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        points = [(cx, cy), (x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        inside = sum(
            1 for p in points if cv2.pointPolygonTest(self.road_polygon, p, False) >= 0
        )
        return inside >= 3

    # ------------------------------------------------------------------
    # Скорость
    # ------------------------------------------------------------------
    def _calc_speed(self, track: TrackState) -> float:
        if len(track.positions) < 2:
            return 999.0
        window = min(self.speed_window, len(track.positions))
        total_dist = 0.0
        for i in range(1, window):
            x1, y1 = track.positions[-i - 1]
            x2, y2 = track.positions[-i]
            total_dist += ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        return total_dist / (window - 1)

    # ------------------------------------------------------------------
    # Классификация с гистерезисом
    # ------------------------------------------------------------------
    def _classify(
        self, track: TrackState, speed: float, on_road: bool, now: float
    ) -> str:
        was_moving = track.last_state == "moving"

        if was_moving:
            # Двигалась — проверяем, остановилась ли (низкий порог)
            if speed >= self.move_threshold_low:
                return "moving"
            # Остановилась — фиксируем время
            if track.stop_start_time is None:
                track.stop_start_time = now
        else:
            # Стояла — проверяем, поехала ли (высокий порог)
            if speed > self.move_threshold_high:
                track.stop_start_time = None
                return "moving"
            # Продолжает стоять
            if track.stop_start_time is None:
                track.stop_start_time = now

        stopped_for = now - track.stop_start_time if track.stop_start_time else 0.0

        if on_road:
            if stopped_for < self.stop_timeout:
                return "stopped"
            elif stopped_for < self.park_timeout_on_road:
                return "stopped"
            else:
                return "parked"
        else:
            if stopped_for < self.park_timeout_off_road:
                return "stopped"
            else:
                return "parked"

    def get_track_history(self, track_id: int) -> TrackState:
        return self.tracks[track_id]
