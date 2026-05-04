"""Рисование overlay на кадр: рамки, маска дороги, подписи.

Чистая функция: принимает BGR-кадр + детекции + полигон,
возвращает BGR-кадр с нарисованным overlay.
"""

from typing import List, Tuple
import cv2
import numpy as np


COLORS = {
    "moving": (0, 255, 0),    # 🟢 зелёный
    "stopped": (0, 255, 255), # 🟡 жёлтый
    "parked": (0, 0, 255),    # 🔴 красный
}

ROAD_COLOR = (255, 200, 0)   # голубой полупрозрачный


def draw_overlay(
    frame: np.ndarray,
    detections: List[dict],
    road_polygon: List[Tuple[int, int]],
    draw_road: bool = True,
    draw_boxes: bool = True,
    draw_labels: bool = True,
) -> np.ndarray:
    """Нарисовать overlay на кадр.

    Args:
        frame: BGR кадр (numpy array)
        detections: список словарей с ключами:
            track_id, bbox (x1,y1,x2,y2), state, cls_name
        road_polygon: список (x, y) точек полигона дороги
        draw_road: рисовать ли полигон дороги
        draw_boxes: рисовать ли рамки
        draw_labels: рисовать ли подписи

    Returns:
        Кадр с overlay (BGR)
    """
    out = frame.copy()
    h, w = out.shape[:2]

    # --- Полигон дороги ---
    if draw_road and road_polygon:
        pts = np.array(road_polygon, dtype=np.int32).reshape((-1, 1, 2))
        overlay = out.copy()
        cv2.fillPoly(overlay, [pts], color=ROAD_COLOR)
        cv2.polylines(overlay, [pts], True, ROAD_COLOR, 2)
        out = cv2.addWeighted(out, 1.0, overlay, 0.25, 0)

    # --- Рамки и подписи ---
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        state = det.get("state", "moving")
        color = COLORS.get(state, (255, 255, 255))
        tid = det.get("track_id", 0)
        cls_name = det.get("cls_name", "vehicle")

        # Рамка
        if draw_boxes:
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # Подпись
        if draw_labels:
            label = f"#{tid} {cls_name} [{state}]"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            # Фон под текст
            tx1, ty1 = x1, y1 - th - 6
            if ty1 < 0:
                ty1 = y1 + th + 6
                tx1 = x1
            cv2.rectangle(
                out, (tx1, ty1), (tx1 + tw + 6, ty1 + th + 6), color, -1
            )
            cv2.putText(
                out, label, (tx1 + 3, ty1 + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2
            )

    return out
