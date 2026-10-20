"""Растеризация страницы, поворот на 90°, выравнивание перекоса, обратное преобразование bbox."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import pymupdf

from pdfdoc.models import BBox


def render_gray(page: pymupdf.Page, dpi: int) -> np.ndarray:
    """Страница (как отображается, с учётом /Rotate) → оттенки серого uint8 (H, W)."""
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY, alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.stride)
    return arr[:, : pix.width].copy()


class PixelTransform:
    """Отображение пикселей обработанного изображения → пункты страницы.

    Хранит аффинную матрицу 3x3 «обработанное → исходный рендер»; каждое
    преобразование изображения (поворот на 90°, выравнивание) дописывает к ней
    обратное.
    """

    def __init__(self, dpi: int, page_w: float, page_h: float):
        self.scale = 72.0 / dpi
        self.page_w = page_w
        self.page_h = page_h
        self.m = np.eye(3)

    def push(self, forward_2x3: np.ndarray) -> None:
        """Изображение преобразовано матрицей forward (исходное → новое)."""
        f = np.vstack([forward_2x3, [0, 0, 1]])
        self.m = self.m @ np.linalg.inv(f)

    def to_points(self, x0: float, y0: float, x1: float, y1: float) -> BBox:
        pts = np.array([[x0, y0, 1], [x1, y0, 1], [x0, y1, 1], [x1, y1, 1]], dtype=float).T
        src = self.m @ pts
        xs = src[0] * self.scale
        ys = src[1] * self.scale
        bx0 = float(np.clip(xs.min(), 0, self.page_w))
        by0 = float(np.clip(ys.min(), 0, self.page_h))
        bx1 = float(np.clip(xs.max(), 0, self.page_w))
        by1 = float(np.clip(ys.max(), 0, self.page_h))
        return (round(bx0, 2), round(by0, 2), round(max(bx0, bx1), 2), round(max(by0, by1), 2))


def rotate90_image(img: np.ndarray, clockwise_deg: int) -> np.ndarray:
    """Повернуть на 90/180/270 по часовой без учёта в PixelTransform (для вырезок)."""
    k = (clockwise_deg // 90) % 4
    codes = {1: cv2.ROTATE_90_CLOCKWISE, 2: cv2.ROTATE_180, 3: cv2.ROTATE_90_COUNTERCLOCKWISE}
    return cv2.rotate(img, codes[k]) if k else img


def rotate90(img: np.ndarray, clockwise_deg: int, tr: PixelTransform) -> np.ndarray:
    """Повернуть на 90/180/270 по часовой стрелке и учесть это в tr."""
    k = (clockwise_deg // 90) % 4
    if k == 0:
        return img
    h, w = img.shape[:2]
    if k == 1:  # по часовой: (x, y) → (h-1-y, x)
        f = np.array([[0, -1, h - 1], [1, 0, 0]], dtype=float)
    elif k == 2:
        f = np.array([[-1, 0, w - 1], [0, -1, h - 1]], dtype=float)
    else:  # против часовой: (x, y) → (y, w-1-x)
        f = np.array([[0, 1, 0], [-1, 0, w - 1]], dtype=float)
    tr.push(f)
    return rotate90_image(img, clockwise_deg)


def _ink(gray: np.ndarray, dpi: int) -> np.ndarray:
    block = max(15, int(dpi / 10) | 1)
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block, 10
    )


def estimate_skew(gray: np.ndarray, dpi: int, max_deg: float = 5.0) -> float:
    """Угол перекоса (град., против часовой > 0), при котором строки текста и
    линии таблиц становятся строго горизонтальными.

    Метод: максимум дисперсии горизонтальной проекции «чернил» по углам —
    сначала грубо (шаг 0,2°), затем точно (0,02°). Работает и по тексту,
    и по линиям сетки.
    """
    ink = _ink(gray, dpi)
    scale = 0.25 if dpi >= 250 else 0.4
    small = cv2.resize(ink, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if small.mean() < 0.5:  # пустая страница
        return 0.0
    h, w = small.shape
    center = (w / 2, h / 2)

    def score(angle: float) -> float:
        m = cv2.getRotationMatrix2D(center, angle, 1.0)
        rot = cv2.warpAffine(small, m, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        return float(np.var(rot.sum(axis=1, dtype=np.float64)))

    base = score(0.0)
    best, best_s = 0.0, base
    for a in np.arange(-max_deg, max_deg + 1e-9, 0.2):
        s = score(float(a))
        if s > best_s:
            best, best_s = float(a), s
    lo, hi = best - 0.2, best + 0.2
    for a in np.arange(lo, hi + 1e-9, 0.02):
        s = score(float(a))
        if s > best_s:
            best, best_s = float(a), s
    # слабый выигрыш — шум, не поворачиваем
    if base > 0 and best_s / base < 1.02:
        return 0.0
    return round(best, 2)


def rotate_small(gray: np.ndarray, angle: float, tr: PixelTransform) -> np.ndarray:
    """Повернуть на малый угол (против часовой > 0) с белым фоном, размер сохраняется."""
    if abs(angle) < 0.05:
        return gray
    h, w = gray.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    out = cv2.warpAffine(
        gray, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255
    )
    tr.push(m)
    return out


def _reading_score(img: np.ndarray, engine) -> float:
    """Сколько уверенно прочитанных слов (вес — уверенность × длина) на изображении.

    psm 6 (без разбора раскладки): в psm 3 Tesseract сам находит вертикальные
    блоки и читает повёрнутый на 90° текст почти так же, как прямой.
    """
    try:
        words = engine.words(img, psm=6)
    except RuntimeError:
        return 0.0
    return sum(w.conf / 100 * len(w.text) for w in words if sum(c.isalnum() for c in w.text) >= 3 and w.conf >= 50)


def fix_orientation(
    gray: np.ndarray, engine, tr: PixelTransform, min_conf: float = 2.0
) -> tuple[np.ndarray, Optional[int], bool]:
    """Определить поворот на 90/180/270 через Tesseract OSD и исправить.

    Возвращает (изображение, поворот | None, поворот под сомнением). На таблицах
    с редким текстом уверенность OSD бывает ниже порога (у перевёрнутых на 180°
    листов «Приложения» — 2,5…2,7 при пороге 2): тогда предложенный поворот
    сверяется с исходным по пробному распознаванию.
    """
    small = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    res = engine.orientation(small)
    if not res:
        return gray, None, False
    rotate, conf = res
    if rotate == 0:
        return gray, 0, False
    if conf >= min_conf:
        return rotate90(gray, rotate, tr), rotate, False
    as_is = _reading_score(small, engine)
    turned = _reading_score(rotate90_image(small, rotate), engine)
    if turned > 1.5 * as_is and turned > 0:
        return rotate90(gray, rotate, tr), rotate, False
    if as_is > 1.5 * turned and as_is > 0:
        return gray, 0, False
    return gray, None, True
