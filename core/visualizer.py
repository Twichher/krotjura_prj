"""Рисование overlay на кадр: рамки, маска дороги, подписи.

Адаптивный под разрешение видео.
"""

from typing import List, Tuple
import cv2
import numpy as np


COLORS = {
    "moving": (0, 200, 0),     # 🟢 приглушённый зелёный
    "stopped": (0, 200, 200),  # 🟡 приглушённый жёлтый (BGR)
    "parked": (200, 0, 0),     # 🔴 приглушённый красный
}

ROAD_COLOR = (255, 200, 0)    # голубой


def _adaptive_scale(frame: np.ndarray) -> Tuple[float, int, int]:
    """Вернуть (font_scale, line_thickness, text_thickness) относительно размера кадра."""
    h, w = frame.shape[:2]
    short_side = min(w, h)
    # Базовый масштаб: 640px = scale=1.0
    scale = short_side / 640.0
    font_scale = max(0.35, 0.5 * scale)
    line_thickness = max(1, int(1.5 * scale))
    text_thickness = max(1, int(1.2 * scale))
    return font_scale, line_thickness, text_thickness


def draw_overlay(
    frame: np.ndarray,
    detections: List[dict],
    road_polygon: List[Tuple[int, int]],
    draw_road: bool = True,
    draw_boxes: bool = True,
    draw_labels: bool = True,
) -> np.ndarray:
    """Нарисовать overlay на кадр."""
    out = frame.copy()
    h, w = out.shape[:2]
    font_scale, line_thickness, text_thickness = _adaptive_scale(out)

    # --- Полигон дороги ---
    if draw_road and road_polygon:
        pts = np.array(road_polygon, dtype=np.int32).reshape((-1, 1, 2))
        overlay = out.copy()
        cv2.fillPoly(overlay, [pts], color=ROAD_COLOR)
        cv2.polylines(overlay, [pts], True, ROAD_COLOR, line_thickness)
        out = cv2.addWeighted(out, 1.0, overlay, 0.2, 0)

    # --- Рамки и подписи ---
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        state = det.get("state", "moving")
        color = COLORS.get(state, (180, 180, 180))
        tid = det.get("track_id", 0)

        # Рамка (тонкая)
        if draw_boxes:
            cv2.rectangle(out, (x1, y1), (x2, y2), color, line_thickness)

        # Подпись — только ID (коротко)
        if draw_labels:
            label = f"#{tid}"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness
            )

            # Фон под текст — полупрозрачный чёрный
            pad = max(2, int(3 * min(w, h) / 640))
            tx1, ty1 = x1, y1 - th - pad * 2
            if ty1 < 0:
                ty1 = y1 + th + pad * 2

            overlay_text = out.copy()
            cv2.rectangle(
                overlay_text,
                (tx1, ty1),
                (tx1 + tw + pad * 2, ty1 + th + pad * 2),
                (0, 0, 0),
                -1,
            )
            out = cv2.addWeighted(out, 0.7, overlay_text, 0.3, 0)

            # Сам текст
            cv2.putText(
                out, label, (tx1 + pad, ty1 + th + pad - 1),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, text_thickness,
            )

    return out
