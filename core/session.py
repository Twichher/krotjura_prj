"""Хранение результатов анализа одного видео.

Session живёт от загрузки видео до удаления. Содержит полигон дороги,
результаты по каждому кадру и агрегированную статистику.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class FrameResult:
    frame_id: int
    timestamp: float
    detections: List[dict] = field(default_factory=list)


@dataclass
class Session:
    video_path: str
    fps: float
    total_frames: int
    width: int
    height: int
    road_polygon: List[Tuple[int, int]] = field(default_factory=list)
    frame_results: List[FrameResult] = field(default_factory=list)
    offset_sec: float = 0.0  # смещение чанка от начала видео

    # Агрегированная статистика (заполняется после обработки)
    total_moving_tracks: int = 0
    total_stopped_tracks: int = 0
    total_parked_tracks: int = 0
    max_parked_at_once: int = 0
    total_unique_tracks: int = 0

    def get_stats_at_frame(self, frame_id: int) -> Tuple[int, int, int]:
        """Вернуть (moving, stopped, parked) на конкретном кадре."""
        if frame_id < 0 or frame_id >= len(self.frame_results):
            return 0, 0, 0
        dets = self.frame_results[frame_id].detections
        moving = sum(1 for d in dets if d["state"] == "moving")
        stopped = sum(1 for d in dets if d["state"] == "stopped")
        parked = sum(1 for d in dets if d["state"] == "parked")
        return moving, stopped, parked
