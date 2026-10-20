"""OCR одной страницы-скана: поворот, выравнивание, таблицы, текст вне таблиц."""

from __future__ import annotations

import logging
import statistics
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import pymupdf

from pdfdoc._raw import RawBlock
from pdfdoc.config import PipelineConfig
from pdfdoc.models import BlockSource, BlockType
from pdfdoc.ocr.cells import fix_bar_ones, latin_cells_to_reread, ocr_cell, regroup_thousands, reread_russian
from pdfdoc.ocr.engine import Tesseract, Word, words_confidence
from pdfdoc.ocr.grid import TableGrid, find_tables, line_masks
from pdfdoc.ocr.image import (
    PixelTransform,
    estimate_skew,
    fix_orientation,
    render_gray,
    rotate90_image,
    rotate_small,
)
from pdfdoc.tables import CellData, TableData
from pdfdoc.text_utils import (
    alnum_count,
    clean_line,
    fix_homoglyphs,
    fix_ocr_confusions,
    join_cell_lines,
    starts_list_item,
)

log = logging.getLogger("pdfdoc")

RUS_REREAD_MIN_CONF = 0.5


@dataclass
class OcrPageResult:
    blocks: list[RawBlock] = field(default_factory=list)
    body_size: Optional[float] = None
    skew_angle: Optional[float] = None
    orientation: Optional[int] = None
    layout_width: Optional[float] = None
    layout_height: Optional[float] = None
    warnings: list[str] = field(default_factory=list)


def _lbox(x0: float, y0: float, x1: float, y1: float, scale: float):
    return (round(x0 * scale, 2), round(y0 * scale, 2), round(x1 * scale, 2), round(y1 * scale, 2))


def _ocr_table(
    g: TableGrid,
    gray: np.ndarray,
    engine: Tesseract,
    cfg: PipelineConfig,
    pool: ThreadPoolExecutor,
) -> tuple[TableData, int]:
    pad = max(3, int(cfg.dpi * 0.012))
    crops = []
    for c in g.cells:
        x0, y0, x1, y1 = g.cell_box(c)
        crops.append(gray[y0 + pad : max(y0 + pad + 1, y1 - pad), x0 + pad : max(x0 + pad + 1, x1 - pad)])

    def run(crop: np.ndarray) -> tuple[str, Optional[float], bool]:
        try:
            t, conf = ocr_cell(
                crop,
                engine,
                cfg.dpi,
                numeric_recheck=cfg.numeric_recheck,
                homoglyphs=cfg.fix_homoglyphs,
            )
            return t, conf, False
        except RuntimeError as exc:  # таймаут pytesseract
            log.warning("cell OCR failed: %s", exc)
            return "", 0.0, True

    results = list(pool.map(run, crops))
    table = TableData(n_rows=g.n_rows, n_cols=g.n_cols, structure_penalty=g.penalty)
    failed = 0
    for c, (text, conf, err) in zip(g.cells, results):
        failed += err
        table.cells.append(
            CellData(row=c.r0, col=c.c0, rowspan=c.rowspan, colspan=c.colspan, text=text, confidence=conf)
        )
    if "rus" in engine.lang.split("+"):
        idx = latin_cells_to_reread(table.cells)

        def reread(i: int) -> tuple[str, Optional[float]]:
            try:
                return reread_russian(crops[i], engine, cfg.dpi)
            except RuntimeError:
                return "", None

        for i, (text, conf) in zip(idx, pool.map(reread, idx)):
            cell = table.cells[i]
            log.debug("latin cell %r (%s) -> %r (%s)", cell.text, cell.confidence, text, conf)
            if text and conf is not None and conf >= RUS_REREAD_MIN_CONF:
                cell.text, cell.confidence = text, conf
    fix_bar_ones(table.cells)
    regroup_thousands(table.cells)
    return table, failed


def _erase_grid(gray: np.ndarray, masks, dpi: int) -> np.ndarray:
    """Сетку стираем целиком по маске линий страницы (длинные, склеенные линии —
    устойчиво к шуму), а не пытаемся вычищать обрывки в каждой ячейке."""
    k = max(3, int(dpi * 0.012))
    grid_lines = cv2.dilate(cv2.bitwise_or(masks.h, masks.v), np.ones((k, k), np.uint8))
    out = gray.copy()
    out[grid_lines > 0] = 255
    return out


def _table_turn(g: TableGrid, cells_img: np.ndarray, engine: Tesseract, dpi: int) -> int:
    """Поворот таблицы относительно страницы (альбомная таблица на книжном листе).

    OSD по вырезке без линий; предложенный поворот всегда сверяется пробным
    распознаванием — на небольшой таблице OSD ошибается и с высокой уверенностью.
    """
    crop = cells_img[g.y0 : g.y1 + 1, g.x0 : g.x1 + 1]
    h, w = crop.shape[:2]
    _, turn, _ = fix_orientation(crop, engine, PixelTransform(dpi, w, h), min_conf=float("inf"))
    return turn or 0


def _ocr_turned_table(
    g: TableGrid,
    gray: np.ndarray,
    turn: int,
    engine: Tesseract,
    cfg: PipelineConfig,
    pool: ThreadPoolExecutor,
) -> Optional[tuple[TableData, int]]:
    """Повернуть вырезку таблицы на turn° и заново найти в ней сетку."""
    m = int(cfg.dpi * 0.05)
    crop = gray[max(0, g.y0 - m) : g.y1 + m, max(0, g.x0 - m) : g.x1 + m]
    crop = rotate90_image(crop, turn)
    masks = line_masks(crop, cfg.dpi)
    grids = find_tables(crop, cfg.dpi, cfg.min_table_cells, masks)
    if not grids:
        return None
    sub = max(grids, key=lambda t: (t.x1 - t.x0) * (t.y1 - t.y0))
    return _ocr_table(sub, _erase_grid(crop, masks, cfg.dpi), engine, cfg, pool)


def _hline_segments(h_mask: np.ndarray, dpi: int, width: int) -> list[tuple[int, int, int, int]]:
    n, _, stats, _ = cv2.connectedComponentsWithStats(h_mask, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        if dpi * 0.4 <= w <= width * 0.7 and h <= dpi * 0.03:
            out.append((int(x), int(y), int(x + w), int(y + h)))
    return out


@dataclass
class _Line:
    words: list[Word]

    @property
    def text(self) -> str:
        return " ".join(w.text for w in sorted(self.words, key=lambda w: w.left))

    @property
    def x0(self) -> int:
        return min(w.left for w in self.words)

    @property
    def y0(self) -> int:
        return min(w.top for w in self.words)

    @property
    def x1(self) -> int:
        return max(w.right for w in self.words)

    @property
    def y1(self) -> int:
        return max(w.bottom for w in self.words)

    @property
    def height(self) -> float:
        # высота всей строки (заглавные + выносные), а не медиана слов: у слов
        # «по», «на» высота меньше, чем у «Подразделение», и медиана скачет
        return float(self.y1 - self.y0)


def _group_paragraphs(words: list[Word]) -> list[list[_Line]]:
    paras: dict[tuple[int, int], dict[int, list[Word]]] = {}
    order: list[tuple[int, int]] = []
    for w in words:
        key = (w.block, w.par)
        if key not in paras:
            paras[key] = {}
            order.append(key)
        paras[key].setdefault(w.line, []).append(w)
    out = []
    for key in order:
        lines = [_Line(ws) for _, ws in sorted(paras[key].items())]
        lines.sort(key=lambda ln: ln.y0)
        out.append(lines)
    return out


def _split_lines(lines: list[_Line]) -> list[list[_Line]]:
    """Разбить абзац Tesseract при скачке высоты строки или большом зазоре."""
    groups: list[list[_Line]] = []
    for ln in lines:
        if not groups:
            groups.append([ln])
            continue
        prev = groups[-1][-1]
        h, ph = ln.height, prev.height
        gap = ln.y0 - prev.y1
        if abs(h - ph) / max(h, ph, 1) > 0.3 or gap > 1.5 * max(h, ph) or starts_list_item(ln.text):
            groups.append([ln])
        else:
            groups[-1].append(ln)
    return groups


def ocr_page(
    page: pymupdf.Page,
    cfg: PipelineConfig,
    engine: Tesseract,
    pool: ThreadPoolExecutor,
) -> OcrPageResult:
    engine.ensure()
    res = OcrPageResult()
    pno = page.number + 1
    dpi = cfg.dpi
    gray = render_gray(page, dpi)
    tr = PixelTransform(dpi, page.rect.width, page.rect.height)

    if cfg.detect_orientation:
        gray, res.orientation, doubtful = fix_orientation(gray, engine, tr)
        if res.orientation:
            res.warnings.append(f"page {pno}: scan rotated by {res.orientation}° (orientation fixed)")
        elif doubtful:
            res.warnings.append(f"page {pno}: scan orientation is uncertain, page processed as is")
    if cfg.deskew:
        angle = estimate_skew(gray, dpi, cfg.max_skew_deg)
        gray = rotate_small(gray, angle, tr)
        res.skew_angle = angle

    scale = 72.0 / dpi
    res.layout_height = round(gray.shape[0] * scale, 2)
    res.layout_width = round(gray.shape[1] * scale, 2)
    masks = line_masks(gray, dpi)
    grids = find_tables(gray, dpi, cfg.min_table_cells, masks) if cfg.detect_tables else []

    # ---------------------------------------------------------------- таблицы
    if grids:
        cells_img = _erase_grid(gray, masks, dpi)
    for g in grids:
        turn = 0
        # таблицу почти во всю страницу уже развернул поворот страницы
        if cfg.detect_orientation and (g.x1 - g.x0) * (g.y1 - g.y0) < 0.5 * gray.shape[0] * gray.shape[1]:
            turn = _table_turn(g, cells_img, engine, dpi)
        turned = _ocr_turned_table(g, gray, turn, engine, cfg, pool) if turn else None
        if turned:
            table, failed = turned
            res.warnings.append(f"page {pno}: table rotated by {turn}° on the page (orientation fixed)")
        else:
            table, failed = _ocr_table(g, cells_img, engine, cfg, pool)
        if not any(c.text for c in table.cells):
            continue
        if failed:
            res.warnings.append(f"page {pno}: OCR timeout in {failed} table cells")
        res.blocks.append(
            RawBlock(
                bbox=tr.to_points(g.x0, g.y0, g.x1, g.y1),
                layout_bbox=_lbox(g.x0, g.y0, g.x1, g.y1, scale),
                source=BlockSource.TABLE_OCR,
                confidence=round(table.confidence() or 0.0, 3),
                type=BlockType.TABLE,
                table=table,
            )
        )

    # ---------------------------------------------------------------- текст вне таблиц
    masked = gray.copy()
    h_free = masks.h.copy()
    ink_free = masks.ink.copy()
    margin = max(3, int(dpi * 0.01))
    for g in grids:
        sl = np.s_[max(0, g.y0 - margin) : g.y1 + margin, max(0, g.x0 - margin) : g.x1 + margin]
        masked[sl] = 255
        h_free[sl] = 0
        ink_free[sl] = 0
    if float(ink_free.mean()) / 255.0 < 0.0005:
        return res

    try:
        words = engine.words(masked, psm=3)
    except RuntimeError as exc:
        res.warnings.append(f"page {pno}: text OCR failed: {exc}")
        return res

    hsegs = _hline_segments(h_free, dpi, gray.shape[1])
    line_heights: list[float] = []
    for para in _group_paragraphs(words):
        for group in _split_lines(para):
            text = join_cell_lines("\n".join(clean_line(ln.text) for ln in group))
            if cfg.fix_homoglyphs:
                text = fix_ocr_confusions(fix_homoglyphs(text))
            all_words = [w for ln in group for w in ln.words]
            conf = words_confidence(all_words) or 0.0
            if alnum_count(text) == 0 and "___" not in text:
                continue  # «|», «~», «—» от пятен и линий
            if conf < 0.3 and alnum_count(text) < 4:
                continue
            x0 = min(ln.x0 for ln in group)
            y0 = min(ln.y0 for ln in group)
            x1 = max(ln.x1 for ln in group)
            y1 = max(ln.y1 for ln in group)
            lh = statistics.median(ln.height for ln in group)
            line_heights.extend(ln.height for ln in group)
            near = any(
                sy1 >= y0 - lh and sy0 <= y1 + lh * 0.8 and sx1 >= x0 - 3 * lh and sx0 <= x1 + 3 * lh
                for sx0, sy0, sx1, sy1 in hsegs
            )
            res.blocks.append(
                RawBlock(
                    bbox=tr.to_points(x0, y0, x1, y1),
                    layout_bbox=_lbox(x0, y0, x1, y1, scale),
                    source=BlockSource.OCR,
                    confidence=round(conf, 3),
                    text=text,
                    font_size=round(lh * scale, 2),
                    n_lines=len(group),
                    near_hline=near,
                )
            )
    if line_heights:
        res.body_size = round(statistics.median(line_heights) * scale, 2)
    return res
