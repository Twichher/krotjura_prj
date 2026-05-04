"""Движок pre-processing: VideoLoader → Tracker → StateClassifier → Session.

Чистый Python, не зависит от Qt.
"""

from typing import List, Tuple, Callable
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
    ):
        self.video_path = video_path
        self.road_polygon = road_polygon
        self.progress_callback = progress_callback

        self.loader = VideoLoader(video_path)
        self.tracker = VehicleTracker()
        self.classifier = StateClassifier(road_polygon)

        self.session = Session(
            video_path=video_path,
            fps=self.loader.fps,
            total_frames=self.loader.frame_count,
            width=self.loader.size[0],
            height=self.loader.size[1],
            road_polygon=road_polygon,
        )

    def process(self) -> Session:
        """Обработать всё видео кадр за кадром."""
        total = self.loader.frame_count
        frame_id = 0

        while True:
            frame = self.loader.read_frame()
            if frame is None:
                break

            timestamp = frame_id / self.loader.fps if self.loader.fps > 0 else 0.0

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
