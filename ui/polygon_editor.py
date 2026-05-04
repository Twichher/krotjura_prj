"""Интерактивный редактор полигона дороги.

QGraphicsView + draggable вершины. Принимает QPixmap (кадр) и начальный полигон.
Возвращает отредактированный полигон по сигналу.
"""

from typing import List, Tuple
from PyQt6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QGraphicsPolygonItem, QGraphicsEllipseItem,
)
from PyQt6.QtCore import Qt, pyqtSignal, QPointF
from PyQt6.QtGui import QPixmap, QPen, QBrush, QColor, QPolygonF, QPainter


class DraggableVertex(QGraphicsEllipseItem):
    """Точка-вершина полигона, которую можно тягать мышью."""

    RADIUS = 6

    def __init__(self, x: float, y: float, index: int, parent=None):
        r = self.RADIUS
        super().__init__(-r, -r, r * 2, r * 2, parent)
        self.index = index
        self.setPos(x, y)
        self.setBrush(QBrush(QColor("#38bdf8")))
        self.setPen(QPen(QColor("#0ea5e9"), 2))
        self.setFlags(
            QGraphicsEllipseItem.GraphicsItemFlag.ItemIsMovable |
            QGraphicsEllipseItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def itemChange(self, change, value):
        if change == QGraphicsEllipseItem.GraphicsItemChange.ItemPositionChange:
            # Сообщить родительскому полигону, что вершина сдвинулась
            scene = self.scene()
            if isinstance(scene, PolygonScene):
                scene.update_polygon_vertex(self.index, value)
        return super().itemChange(change, value)


class PolygonScene(QGraphicsScene):
    """Сцена с изображением и редактируемым полигоном."""

    polygon_changed = pyqtSignal(list)  # List[Tuple[int, int]]

    def __init__(self, pixmap: QPixmap, polygon: List[Tuple[int, int]], parent=None):
        super().__init__(parent)
        self._polygon_points: List[QPointF] = [QPointF(x, y) for x, y in polygon]

        # Фон — кадр видео
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.addItem(self.pixmap_item)

        # Полигон дороги
        self.polygon_item = QGraphicsPolygonItem(QPolygonF(self._polygon_points))
        self.polygon_item.setPen(QPen(QColor("#38bdf8"), 3, Qt.PenStyle.DashLine))
        self.polygon_item.setBrush(QBrush(QColor(56, 189, 248, 60)))  # полупрозрачный
        self.addItem(self.polygon_item)

        # Вершины
        self.vertices: List[DraggableVertex] = []
        for i, pt in enumerate(self._polygon_points):
            v = DraggableVertex(pt.x(), pt.y(), i)
            self.addItem(v)
            self.vertices.append(v)

        # Размер сцены = размер изображения
        self.setSceneRect(self.pixmap_item.boundingRect())

    def update_polygon_vertex(self, index: int, new_pos: QPointF):
        """Вызывается при перетаскивании вершины."""
        if 0 <= index < len(self._polygon_points):
            self._polygon_points[index] = new_pos
            self.polygon_item.setPolygon(QPolygonF(self._polygon_points))
            self.polygon_changed.emit(self.get_polygon_int())

    def get_polygon_int(self) -> List[Tuple[int, int]]:
        return [(int(p.x()), int(p.y())) for p in self._polygon_points]

    def mousePressEvent(self, event):
        # Клик по пустому месту — добавить новую вершину
        if self.itemAt(event.scenePos(), self.views()[0].transform()) is None:
            pos = event.scenePos()
            self._add_vertex(pos)
        super().mousePressEvent(event)

    def _add_vertex(self, pos: QPointF):
        """Добавить вершину в полигон."""
        # Вставляем в ближайшее ребро
        idx = self._find_insert_index(pos)
        self._polygon_points.insert(idx, pos)

        # Пересоздаем полигон
        self.polygon_item.setPolygon(QPolygonF(self._polygon_points))

        # Пересоздаем вершины (перенумеровываем)
        for v in self.vertices:
            self.removeItem(v)
        self.vertices.clear()
        for i, pt in enumerate(self._polygon_points):
            v = DraggableVertex(pt.x(), pt.y(), i)
            self.addItem(v)
            self.vertices.append(v)

        self.polygon_changed.emit(self.get_polygon_int())

    def _find_insert_index(self, pos: QPointF) -> int:
        """Найти, между какими вершинами вставить новую точку."""
        pts = self._polygon_points
        n = len(pts)
        if n < 2:
            return n

        best_idx = n
        best_dist = float("inf")
        for i in range(n):
            a = pts[i]
            b = pts[(i + 1) % n]
            # Расстояние от точки до отрезка ab
            dist = self._point_to_segment_dist(pos, a, b)
            if dist < best_dist:
                best_dist = dist
                best_idx = i + 1
        return best_idx

    @staticmethod
    def _point_to_segment_dist(p: QPointF, a: QPointF, b: QPointF) -> float:
        ab = QPointF(b.x() - a.x(), b.y() - a.y())
        ap = QPointF(p.x() - a.x(), p.y() - a.y())
        ab_len2 = ab.x() * ab.x() + ab.y() * ab.y()
        if ab_len2 == 0:
            dx = p.x() - a.x()
            dy = p.y() - a.y()
            return (dx * dx + dy * dy) ** 0.5
        t = max(0.0, min(1.0, (ap.x() * ab.x() + ap.y() * ab.y()) / ab_len2))
        closest = QPointF(a.x() + t * ab.x(), a.y() + t * ab.y())
        dx = p.x() - closest.x()
        dy = p.y() - closest.y()
        return (dx * dx + dy * dy) ** 0.5


class PolygonEditor(QGraphicsView):
    """Виджет-редактор. Оборачивает PolygonScene."""

    polygon_changed = pyqtSignal(list)

    def __init__(self, pixmap: QPixmap, polygon: List[Tuple[int, int]], parent=None):
        super().__init__(parent)
        self.scene = PolygonScene(pixmap, polygon)
        self.scene.polygon_changed.connect(self.polygon_changed.emit)
        self.setScene(self.scene)

        # Настройки отображения
        self.setRenderHints(QPainter.RenderHint.Antialiasing)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setStyleSheet("border: none; background-color: #000000;")

    def get_polygon(self) -> List[Tuple[int, int]]:
        return self.scene.get_polygon_int()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
