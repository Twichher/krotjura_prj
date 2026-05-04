"""Движок pre-processing: VideoLoader → Tracker → StateClassifier → Session.

Чистый Python, не зависит от Qt.
"""

from typing import List, Tuple, Callable, Optional
import numpy as np

from core.video_loader import VideoLoader
from core.tracker import VehicleTracker
from core.state_classifier import StateClassifier
from core.session import Session, FrameResult


class VideoProcessor:
    def __init__(
        self,
        video_path: str,
        road_polygon: List[Tuple[int, int]],
        progress_callback: Callable[[int], None] = None,
        max_duration_sec: Optional[float] = None,
        should_stop: Optional[Callable[[], bool]] = None,
        start_offset_sec: float = 0.0,
    ):
        self.video_path = video_path
        self.road_polygon = road_polygon
        self.progress_callback = progress_callback
        self.max_duration_sec = max_duration_sec
        self.should_stop = should_stop
        self.start_offset_sec = start_offset_sec

        self.loader = VideoLoader(video_path)
        self.tracker = VehicleTracker()
        self.classifier = StateClassifier(road_polygon)

        # Прыгаем на стартовую позицию
        if start_offset_sec > 0 and self.loader.fps > 0:
            start_frame = int(start_offset_sec * self.loader.fps)
            self.loader.seek(start_frame)

        total_frames = self.loader.frame_count
        remaining = total_frames - int(start_offset_sec * self.loader.fps) if self.loader.fps > 0 else total_frames
        total_frames = remaining

        if max_duration_sec is not None and self.loader.fps > 0:
            total_frames = min(total_frames, int(max_duration_sec * self.loader.fps))

        self.session = Session(
            video_path=video_path,
            fps=self.loader.fps,
            total_frames=total_frames,
            width=self.loader.size[0],
            height=self.loader.size[1],
            road_polygon=road_polygon,
            offset_sec=start_offset_sec,
        )

    def process(self) -> Session:
        """Обработать видео кадр за кадром (с ограничением по длительности)."""
        total = self.session.total_frames
        frame_id = 0

        while True:
            if self.should_stop and self.should_stop():
                break

            frame = self.loader.read_frame()
            if frame is None:
                break

            # Ограничение по длительности
            if frame_id >= total:
                break

            timestamp = self.start_offset_sec + (frame_id / self.loader.fps if self.loader.fps > 0 else 0.0)

            # Детекция + трекинг
            detections = self.tracker.track(frame)

            # Классификация состояния
            classified = self.classifier.update(frame_id, timestamp, detections)

            # Сохранить результат кадра
            self.session.frame_results.append(
                FrameResult(frame_id=frame_id, timestamp=timestamp, detections=classified)
            )

            # Прогресс
            if self.progress_callback and total > 0:
                pct = int((frame_id + 1) / total * 100)
                self.progress_callback(pct)

            frame_id += 1

        self.loader.release()
        self._compute_aggregate_stats()
        return self.session

    def _compute_aggregate_stats(self):
        """Посчитать накопительную статистику по всем кадрам."""
        unique_moving = set()
        unique_stopped = set()
        unique_parked = set()
        max_parked = 0

        for fr in self.session.frame_results:
            for det in fr.detections:
                tid = det["track_id"]
                state = det["state"]
                if state == "moving":
                    unique_moving.add(tid)
                elif state == "stopped":
                    unique_stopped.add(tid)
                elif state == "parked":
                    unique_parked.add(tid)

            parked_now = sum(1 for d in fr.detections if d["state"] == "parked")
            max_parked = max(max_parked, parked_now)

        self.session.total_moving_tracks = len(unique_moving)
        self.session.total_stopped_tracks = len(unique_stopped)
        self.session.total_parked_tracks = len(unique_parked)
        self.session.max_parked_at_once = max_parked
        self.session.total_unique_tracks = len(
            unique_moving | unique_stopped | unique_parked
        )
