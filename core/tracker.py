"""Обёртка над YOLOv8 + ByteTrack для покадровой обработки.

Не зависит от UI. Принимает numpy-кадр, возвращает список Detection.
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    track_id: int
    bbox: tuple  # (x1, y1, x2, y2)
    conf: float
    cls: int     # COCO class id
    cls_name: str


class VehicleTracker:
    """Покадровый трекер ТС.

    Использует YOLOv8n (nano) + ByteTrack.
    Отслеживает только транспорт: car, motorcycle, bus, truck.
    """

    # COCO IDs: 2=car, 3=motorcycle, 5=bus, 7=truck
    VEHICLE_CLASSES = {2, 3, 5, 7}
    CLASS_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

    def __init__(self, model_path: str = "yolov8s.pt"):
        self.model = YOLO(model_path)
        self.model.fuse()  # ускорение инференса

    def track(self, frame: np.ndarray) -> List[Detection]:
        """Обработать один кадр (BGR).

        Returns:
            Список Detection с уникальными track_id.
        """
        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
            classes=list(self.VEHICLE_CLASSES),
        )

        if results[0].boxes.id is None:
            return []

        detections = []
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        clss = results[0].boxes.cls.cpu().numpy().astype(int)
        ids = results[0].boxes.id.cpu().numpy().astype(int)

        for box, conf, cls, tid in zip(boxes, confs, clss, ids):
            x1, y1, x2, y2 = map(int, box)
            detections.append(Detection(
                track_id=int(tid),
                bbox=(x1, y1, x2, y2),
                conf=float(conf),
                cls=int(cls),
                cls_name=self.CLASS_NAMES.get(int(cls), "vehicle"),
            ))

        return detections
