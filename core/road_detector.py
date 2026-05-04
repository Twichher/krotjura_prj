"""Автоматическое определение полигона проезжей части на 1-м кадре.

Использует классический CV (цвет асфальта + контуры).
Возвращает список точек [(x, y), ...] — полигон дороги.
"""

from typing import List, Tuple
import cv2
import numpy as np


class RoadDetector:
    def __init__(self,
                 blur_kernel: Tuple[int, int] = (7, 7),
                 morph_kernel: Tuple[int, int] = (21, 21),
                 min_area_ratio: float = 0.05,
                 max_points: int = 8):
        """
        Args:
            blur_kernel: размер Gaussian blur
            morph_kernel: размер морфологического закрытия
            min_area_ratio: минимальная площадь контура относительно площади кадра
            max_points: максимальное число точек в полигоне
        """
        self.blur_kernel = blur_kernel
        self.morph_kernel = morph_kernel
        self.min_area_ratio = min_area_ratio
        self.max_points = max_points

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """Принимает BGR-кадр, возвращает полигон дороги."""
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, self.blur_kernel, 0)

        # Адаптивная бинаризация: дорога обычно темнее окружения (тротуары, трава)
        # Инвертируем, чтобы дорога стала белой
        thresh = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=31,
            C=10
        )

        # Морфологическое закрытие — убираем мелкие дырки
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.morph_kernel)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Находим контуры
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return self._fallback_polygon(w, h)

        # Фильтруем по площади и положению (дорога обычно в центре-низу кадра)
        min_area = w * h * self.min_area_ratio
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            # Центроид должен быть в нижней половине кадра (для камеры сверху)
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cy = int(M["m01"] / M["m00"])
            if cy > h * 0.2:  # не слишком высоко
                candidates.append((area, cnt))

        if not candidates:
            return self._fallback_polygon(w, h)

        # Берем самый большой кандидат
        candidates.sort(key=lambda x: x[0], reverse=True)
        best = candidates[0][1]

        # Упрощаем до полигона
        epsilon = 0.01 * cv2.arcLength(best, True)
        approx = cv2.approxPolyDP(best, epsilon, True)
        pts = [(int(p[0][0]), int(p[0][1])) for p in approx]

        # Если точек слишком много — еще упрощаем
        if len(pts) > self.max_points:
            epsilon = 0.02 * cv2.arcLength(best, True)
            approx = cv2.approxPolyDP(best, epsilon, True)
            pts = [(int(p[0][0]), int(p[0][1])) for p in approx]

        # Если всё еще много — берем bounding box как fallback
        if len(pts) > self.max_points:
            return self._fallback_polygon(w, h)

        return pts

    @staticmethod
    def _fallback_polygon(w: int, h: int) -> List[Tuple[int, int]]:
        """Прямоугольник по центру кадра — если автоматика не сработала."""
        margin_x = int(w * 0.15)
        margin_y = int(h * 0.15)
        return [
            (margin_x, margin_y),
            (w - margin_x, margin_y),
            (w - margin_x, h - margin_y),
            (margin_x, h - margin_y),
        ]
