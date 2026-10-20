"""Поиск таблиц на скане по линиям сетки и восстановление объединённых ячеек.

Отличия от исходного pdf_table_ocr (сверка 1.2):
* таблиц на странице может быть несколько — каждая ищется как связная
  компонента маски линий, а не одна сетка на всю страницу;
* объединения ищутся в обе стороны (rowspan и colspan) через union-find
  по отсутствующим границам, а не только по вертикали;
* рамка страницы (например, ГОСТ-рамка со штампом) не превращает весь лист в
  одну таблицу: такая область разбирается заново без внешней рамки;
* линии сетки не обязаны быть строго вертикальными/горизонтальными: у каждой
  линии свой наклон. Скан листа, протянутого со сдвигом, даёт вертикали
  с наклоном ~0,5° при ровных горизонталях (поворотом страницы это не
  исправить), а на высоте таблицы это ~20 px — больше допуска проверки границ.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class GridCell:
    r0: int
    c0: int
    r1: int  # не включительно
    c1: int

    @property
    def rowspan(self) -> int:
        return self.r1 - self.r0

    @property
    def colspan(self) -> int:
        return self.c1 - self.c0


@dataclass
class TableGrid:
    """Сетка таблицы.

    xs/ys — положения линий в центре таблицы; вертикаль i проходит по
    x = xs[i] + vslopes[i]·(y − yc), горизонталь j — по y = ys[j] + hslopes[j]·(x − xc).
    """

    x0: int
    y0: int
    x1: int
    y1: int
    xs: list[int]
    ys: list[int]
    cells: list[GridCell] = field(default_factory=list)
    penalty: float = 1.0
    vslopes: list[float] = field(default_factory=list)
    hslopes: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.vslopes) != len(self.xs):
            self.vslopes = [0.0] * len(self.xs)
        if len(self.hslopes) != len(self.ys):
            self.hslopes = [0.0] * len(self.ys)

    @property
    def n_rows(self) -> int:
        return len(self.ys) - 1

    @property
    def n_cols(self) -> int:
        return len(self.xs) - 1

    @property
    def xc(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def yc(self) -> float:
        return (self.y0 + self.y1) / 2

    def x_at(self, i: int, y):
        return self.xs[i] + self.vslopes[i] * (y - self.yc)

    def y_at(self, j: int, x):
        return self.ys[j] + self.hslopes[j] * (x - self.xc)

    def cell_box(self, c: GridCell) -> tuple[int, int, int, int]:
        """Прямоугольник ячейки по линиям, взятым в её центре."""
        xm = (self.xs[c.c0] + self.xs[c.c1]) / 2
        ym = (self.ys[c.r0] + self.ys[c.r1]) / 2
        y0, y1 = self.y_at(c.r0, xm), self.y_at(c.r1, xm)
        ym = (y0 + y1) / 2
        x0, x1 = self.x_at(c.c0, ym), self.x_at(c.c1, ym)
        return int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))


@dataclass
class LineMasks:
    ink: np.ndarray
    h: np.ndarray
    v: np.ndarray
    kernel: int


def line_masks(gray: np.ndarray, dpi: int) -> LineMasks:
    block = max(15, int(dpi / 10) | 1)
    ink = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block, 10
    )
    L = max(25, int(dpi * 0.2))  # 0,2 дюйма ≈ 5 мм: длиннее любой буквы до ~14 пт
    h = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1)))
    v = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, L)))
    # склеить разрывы бледных линий скана
    gap = max(3, int(dpi * 0.03))
    h = cv2.morphologyEx(h, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (gap, 1)))
    v = cv2.morphologyEx(v, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, gap)))
    return LineMasks(ink=ink, h=h, v=v, kernel=L)


def _peaks(proj: np.ndarray, thr: float, min_gap: int) -> list[int]:
    coords: list[int] = []
    i, n = 0, len(proj)
    while i < n:
        if proj[i] >= thr:
            j = i
            while j < n and proj[j] >= thr:
                j += 1
            coords.append((i + j - 1) // 2)
            i = j
        else:
            i += 1
    merged: list[int] = []
    for c in coords:
        if merged and c - merged[-1] < min_gap:
            merged[-1] = (merged[-1] + c) // 2
        else:
            merged.append(c)
    return merged


MAX_SLOPE = 0.0524  # tan 3°: больше — уже не перекос скана, а не та линия


def _border_coverage(mask: np.ndarray, along: np.ndarray, across: np.ndarray, tol: int, vertical: bool) -> float:
    """Доля точек линии (along[i], across[i]), рядом с которыми (±tol поперёк) есть маска."""
    offs = np.arange(-tol, tol + 1)
    lim = mask.shape[1] if vertical else mask.shape[0]
    cross = np.clip(np.rint(across).astype(int)[:, None] + offs[None, :], 0, lim - 1)
    a = along[:, None]
    hit = mask[a, cross] if vertical else mask[cross, a]
    return float((hit > 0).any(axis=1).mean())


def _border_h(mask: np.ndarray, g: TableGrid, j: int, xa: int, xb: int, tol: int) -> bool:
    xa, xb = max(0, xa), min(mask.shape[1], xb)
    if xb - xa < 3:
        return True
    xs = np.arange(xa, xb)
    return _border_coverage(mask, xs, g.y_at(j, xs), tol, vertical=False) >= 0.5


def _border_v(mask: np.ndarray, g: TableGrid, i: int, ya: int, yb: int, tol: int) -> bool:
    ya, yb = max(0, ya), min(mask.shape[0], yb)
    if yb - ya < 3:
        return True
    ys = np.arange(ya, yb)
    return _border_coverage(mask, ys, g.x_at(i, ys), tol, vertical=True) >= 0.5


def _dominant_slope(sub: np.ndarray, vertical: bool, min_len: int) -> float:
    """Общий наклон длинных линий маски: dx/dy для вертикалей, dy/dx для горизонталей."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(sub, connectivity=8)
    slopes, weights = [], []
    for k in range(1, n):
        x, y, w, h, _ = stats[k]
        length, width = (h, w) if vertical else (w, h)
        if length < min_len or width > length * 0.2:
            continue
        py, px = np.nonzero(labels[y : y + h, x : x + w] == k)
        step = max(1, py.size // 4000)
        t, p = (py, px) if vertical else (px, py)
        s = float(np.polyfit(t[::step].astype(float), p[::step].astype(float), 1)[0])
        if abs(s) <= MAX_SLOPE:
            slopes.append(s)
            weights.append(float(length))
    if not slopes:
        return 0.0
    order = np.argsort(slopes)
    cum = np.cumsum(np.asarray(weights)[order])
    return float(np.asarray(slopes)[order][np.searchsorted(cum, cum[-1] / 2)])


def _sheared_projection(sub: np.ndarray, slope: float, vertical: bool) -> np.ndarray:
    """Проекция маски вдоль наклонных линий (позиции — в центре области)."""
    rh, rw = sub.shape
    if abs(slope) < 1e-4:
        return (sub > 0).sum(axis=0 if vertical else 1)
    if vertical:  # x' = x − s·(y − yc)
        mat = np.float32([[1, -slope, slope * rh / 2], [0, 1, 0]])
    else:  # y' = y − s·(x − xc)
        mat = np.float32([[1, 0, 0], [-slope, 1, slope * rw / 2]])
    warped = cv2.warpAffine(sub, mat, (rw, rh), flags=cv2.INTER_NEAREST, borderValue=0)
    return (warped > 0).sum(axis=0 if vertical else 1)


def _fit_line(t: np.ndarray, p: np.ndarray, span: int, pos: float, slope: float, half: int,
              tol: int) -> tuple[float, float]:
    """Уточнить (положение в центре, наклон) одной линии по пикселям маски рядом с ней.

    t — координата вдоль линий (от центра области), p — поперёк.
    """
    for width in (half, tol):
        near = np.abs(p - (pos + slope * t)) <= width
        # мало пикселей (линия короткая или её нет) — оставляем как есть
        if near.sum() < 0.2 * span or np.ptp(t[near]) < 0.2 * span:
            break
        s, c = np.polyfit(t[near], p[near], 1)
        if abs(s) > MAX_SLOPE:
            break
        slope, pos = float(s), float(c)
    return pos, slope


def _lines(sub: np.ndarray, vertical: bool, m: LineMasks, dpi: int, min_gap: int) -> tuple[list[float], list[float]]:
    rh, rw = sub.shape
    span = rh if vertical else rw
    slope = _dominant_slope(sub, vertical, max(3 * m.kernel, int(0.2 * span)))
    proj = _sheared_projection(sub, slope, vertical)
    # Короткие линии тоже берём: лишние разрезы потом склеит union-find
    peaks = _peaks(proj, max(m.kernel * 0.8, 0.08 * span), min_gap)
    tol = max(3, int(dpi * 0.015))
    py, px = np.nonzero(sub)
    t, q = (py, px) if vertical else (px, py)
    t = t.astype(float) - span / 2
    q = q.astype(float)
    out_pos, out_slope = [], []
    for k, p in enumerate(peaks):
        gaps = [p - peaks[k - 1]] if k else []
        gaps += [peaks[k + 1] - p] if k + 1 < len(peaks) else []
        half = int(min([max(2 * tol, dpi * 0.04)] + [g / 2 - 1 for g in gaps]))
        pos, s = _fit_line(t, q, span, float(p), slope, max(tol, half), tol)
        out_pos.append(pos)
        out_slope.append(s)
    return out_pos, out_slope


def _build_grid(m: LineMasks, x0: int, y0: int, x1: int, y1: int, dpi: int) -> TableGrid | None:
    hsub = m.h[y0:y1, x0:x1]
    vsub = m.v[y0:y1, x0:x1]
    min_gap = max(6, int(dpi * 0.06))
    vx, vs = _lines(vsub, True, m, dpi, min_gap)
    hy, hs = _lines(hsub, False, m, dpi, min_gap)
    xs = [x0 + int(round(p)) for p in vx]
    ys = [y0 + int(round(p)) for p in hy]
    # крайняя линия наклонена — до рамки области сравниваем её крайнюю точку
    rh, rw = y1 - y0, x1 - x0
    if not xs or xs[0] - abs(vs[0]) * rh / 2 - x0 > min_gap:
        xs.insert(0, x0)
        vs.insert(0, 0.0)
    if x1 - (xs[-1] + abs(vs[-1]) * rh / 2) > min_gap:
        xs.append(x1)
        vs.append(0.0)
    if not ys or ys[0] - abs(hs[0]) * rw / 2 - y0 > min_gap:
        ys.insert(0, y0)
        hs.insert(0, 0.0)
    if y1 - (ys[-1] + abs(hs[-1]) * rw / 2) > min_gap:
        ys.append(y1)
        hs.append(0.0)
    if len(xs) < 2 or len(ys) < 2:
        return None
    return TableGrid(x0=x0, y0=y0, x1=x1, y1=y1, xs=xs, ys=ys, vslopes=vs, hslopes=hs)


def _merge_cells(g: TableGrid, m: LineMasks, dpi: int) -> None:
    """Union-find по отсутствующим границам → прямоугольные объединённые ячейки."""
    R, C = g.n_rows, g.n_cols
    tol = max(3, int(dpi * 0.015))
    pad = tol + 2
    parent = list(range(R * C))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for r in range(R):
        for c in range(C):
            if r + 1 < R:
                y = g.ys[r + 1]
                xa = int(round(g.x_at(c, y))) + pad
                xb = int(round(g.x_at(c + 1, y))) - pad
                if not _border_h(m.h, g, r + 1, xa, xb, tol):
                    union(r * C + c, (r + 1) * C + c)
            if c + 1 < C:
                x = g.xs[c + 1]
                ya = int(round(g.y_at(r, x))) + pad
                yb = int(round(g.y_at(r + 1, x))) - pad
                if not _border_v(m.v, g, c + 1, ya, yb, tol):
                    union(r * C + c, r * C + c + 1)

    groups: dict[int, list[tuple[int, int]]] = {}
    for r in range(R):
        for c in range(C):
            groups.setdefault(find(r * C + c), []).append((r, c))

    rects = []
    for cells in groups.values():
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        rects.append([min(rs), min(cs), max(rs) + 1, max(cs) + 1, len(cells)])

    # непрямоугольные группы (L-образные) → объединяем всё, что пересекает их рамку
    changed = True
    while changed:
        changed = False
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                a, b = rects[i], rects[j]
                if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
                    rects[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]), a[4] + b[4]]
                    rects.pop(j)
                    g.penalty = min(g.penalty, 0.9)
                    changed = True
                    break
            if changed:
                break
    for r0, c0, r1, c1, n in rects:
        if (r1 - r0) * (c1 - c0) != n:
            g.penalty = min(g.penalty, 0.9)
    g.cells = sorted((GridCell(r0, c0, r1, c1) for r0, c0, r1, c1, _ in rects), key=lambda c: (c.r0, c.c0))


def _components(mask: np.ndarray, dpi: int) -> list[tuple[int, int, int, int]]:
    k = max(3, int(dpi * 0.01))
    grid = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    n, _, stats, _ = cv2.connectedComponentsWithStats(grid, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        if w >= dpi * 0.8 and h >= dpi * 0.2:
            out.append((int(x), int(y), int(x + w), int(y + h)))
    return out


def find_tables(gray: np.ndarray, dpi: int, min_cells: int = 2, _masks: LineMasks | None = None,
                _region: tuple[int, int, int, int] | None = None, _depth: int = 0) -> list[TableGrid]:
    """Все таблицы страницы (координаты в пикселях изображения gray)."""
    m = _masks or line_masks(gray, dpi)
    H, W = gray.shape[:2]
    combined = cv2.bitwise_or(m.h, m.v)
    if _region is not None:
        rx0, ry0, rx1, ry1 = _region
        sub = np.zeros_like(combined)
        sub[ry0:ry1, rx0:rx1] = combined[ry0:ry1, rx0:rx1]
        combined = sub

    tables: list[TableGrid] = []
    for x0, y0, x1, y1 in _components(combined, dpi):
        hs = (m.h[y0:y1, x0:x1] > 0).any(axis=1).sum()
        vs = (m.v[y0:y1, x0:x1] > 0).any(axis=0).sum()
        if hs == 0 or vs == 0:
            continue  # одиночная черта (подпись) — не таблица
        g = _build_grid(m, x0, y0, x1, y1, dpi)
        if g is None:
            continue
        _merge_cells(g, m, dpi)

        # рамка листа: область почти во всю страницу, где одна ячейка занимает
        # больше половины площади → разбираем внутренность без рамки
        area = (x1 - x0) * (y1 - y0)
        biggest = max(((g.xs[c.c1] - g.xs[c.c0]) * (g.ys[c.r1] - g.ys[c.r0]) for c in g.cells), default=0)
        if area > 0.6 * W * H and biggest > 0.4 * area:
            if _depth == 0:
                band = max(m.kernel // 2, int(dpi * 0.05))
                inner = (x0 + band, y0 + band, x1 - band, y1 - band)
                tables.extend(find_tables(gray, dpi, min_cells, m, inner, _depth + 1))
            continue  # остаток рамки (двойная рамка, края скана) — тоже не таблица

        if len(g.cells) < min_cells:
            continue  # рамка вокруг текста (1x1) — не таблица
        tables.append(g)
    if _depth == 0:
        # таблица внутри рамки находится и при разборе рамки, и как отдельный компонент
        unique: list[TableGrid] = []
        for t in tables:
            if not any(_same_box(t, u) for u in unique):
                unique.append(t)
        tables = unique
    tables.sort(key=lambda t: (t.y0, t.x0))
    return tables


def _same_box(a: TableGrid, b: TableGrid) -> bool:
    ix = max(0, min(a.x1, b.x1) - max(a.x0, b.x0))
    iy = max(0, min(a.y1, b.y1) - max(a.y0, b.y0))
    inter = ix * iy
    union = (a.x1 - a.x0) * (a.y1 - a.y0) + (b.x1 - b.x0) * (b.y1 - b.y0) - inter
    return union > 0 and inter / union > 0.9
