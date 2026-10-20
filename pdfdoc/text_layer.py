"""Извлечение блоков и таблиц из текстового слоя PDF (PyMuPDF)."""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass
from typing import Optional

import pymupdf

from pdfdoc._raw import RawBlock
from pdfdoc.config import PipelineConfig
from pdfdoc.models import BBox, BlockSource, BlockType
from pdfdoc.tables import table_from_cell_bboxes
from pdfdoc.text_utils import (
    clean_line,
    garbage_ratio,
    join_cell_lines,
    misread_cyrillic_ratio,
    starts_list_item,
)

log = logging.getLogger("pdfdoc")

_BOLD_FLAG = 16


@dataclass
class PageAssessment:
    mode: str  # "text_layer" | "ocr" | "empty"
    n_chars: int
    garbage: float
    hidden_ratio: float
    image_coverage: float
    reason: str = ""


def _to_display(page: pymupdf.Page, bbox) -> BBox:
    """Координаты PyMuPDF (неповёрнутая страница) → как страница отображается."""
    r = pymupdf.Rect(bbox)
    if page.rotation:
        r = r * page.rotation_matrix
    r.normalize()
    pr = page.rect
    x0, y0 = max(pr.x0, r.x0), max(pr.y0, r.y0)
    x1, y1 = min(pr.x1, r.x1), min(pr.y1, r.y1)
    if x1 < x0:
        x1 = x0
    if y1 < y0:
        y1 = y0
    return (round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2))


def _unrotated_rect(page: pymupdf.Page) -> pymupdf.Rect:
    r = page.rect * page.derotation_matrix if page.rotation else pymupdf.Rect(page.rect)
    r.normalize()
    return r


def _image_coverage(page: pymupdf.Page) -> tuple[float, list[pymupdf.Rect]]:
    """Доля площади под картинками и их прямоугольники (неповёрнутые координаты)."""
    prect = _unrotated_rect(page)
    area = abs(prect)
    rects: list[pymupdf.Rect] = []
    covered = 0.0
    try:
        infos = page.get_image_info()
    except Exception:  # noqa: BLE001
        infos = []
    for info in infos:
        r = pymupdf.Rect(info["bbox"]) & prect
        if r.is_empty:
            continue
        rects.append(r)
        covered += abs(r)
    return (min(1.0, covered / area) if area else 0.0), rects


def _hidden_ratio(page: pymupdf.Page) -> float:
    """Доля невидимого текста (render mode 3 / нулевая прозрачность) — слой чужого OCR."""
    total = hidden = 0
    try:
        for span in page.get_texttrace():
            n = len(span.get("chars", ()))
            total += n
            if span.get("type") == 3 or span.get("opacity", 1) == 0:
                hidden += n
    except Exception:  # noqa: BLE001
        return 0.0
    return hidden / total if total else 0.0


def assess_page(page: pymupdf.Page, cfg: PipelineConfig) -> PageAssessment:
    """Решить, читать страницу из текстового слоя или распознавать."""
    text = page.get_text("text")
    n_chars = sum(1 for c in text if not c.isspace())
    garbage = garbage_ratio(text)
    coverage, _ = _image_coverage(page)
    hidden = _hidden_ratio(page) if n_chars else 0.0

    if cfg.force_ocr:
        return PageAssessment("ocr", n_chars, garbage, hidden, coverage, "force_ocr")
    if n_chars < cfg.min_text_chars:
        has_graphics = coverage > 0.02
        if not has_graphics:
            try:
                has_graphics = len(page.get_drawings()) > 20  # текст «в кривых»
            except Exception:  # noqa: BLE001
                has_graphics = False
        if has_graphics:
            return PageAssessment("ocr", n_chars, garbage, hidden, coverage, "no_text_layer")
        if n_chars == 0:
            return PageAssessment("empty", n_chars, garbage, hidden, coverage, "empty")
        return PageAssessment("text_layer", n_chars, garbage, hidden, coverage, "short_text")
    if garbage > cfg.max_garbage_ratio:
        return PageAssessment("ocr", n_chars, garbage, hidden, coverage, "damaged_text_layer")
    if cfg.check_hidden_text_layer and hidden > 0.5 and misread_cyrillic_ratio(text):
        return PageAssessment("ocr", n_chars, garbage, hidden, coverage, "misread_hidden_layer")
    return PageAssessment("text_layer", n_chars, garbage, hidden, coverage, "text_layer")


# ---------------------------------------------------------------- таблицы


def _line_turn(line: dict) -> Optional[int]:
    """На сколько градусов по часовой повернуть страницу, чтобы строка шла слева направо."""
    dx, dy = line.get("dir", (1, 0))
    if dx >= 0.9:
        return 0
    if dx <= -0.9:
        return 180
    if dy <= -0.9:  # строка идёт снизу вверх
        return 90
    if dy >= 0.9:  # сверху вниз
        return 270
    return None


def _turn_weights(data: dict, keep=None) -> dict[int, int]:
    """Число символов в строках каждого направления; keep(bbox) — отбор строк."""
    weight = {0: 0, 90: 0, 180: 0, 270: 0}
    for blk in data.get("blocks", []):
        for line in blk.get("lines", []):
            if keep is not None and not keep(line["bbox"]):
                continue
            turn = _line_turn(line)
            if turn is not None:
                weight[turn] += sum(len(s.get("text", "").strip()) for s in line.get("spans", []))
    return weight


def _dominant(weight: dict[int, int]) -> int:
    best = max(weight, key=lambda t: weight[t])
    return best if weight[best] > weight[0] else 0


def _page_turn(data: dict, table_rects: list[pymupdf.Rect]) -> int:
    """Направление страницы — по тексту вне таблиц (у таблиц своё); если его нет — по всему."""
    outside = _turn_weights(data, lambda bb: not _inside_any(bb, table_rects))
    return _dominant(outside if any(outside.values()) else _turn_weights(data))


class _Frame:
    """Системы координат страницы.

    * unrot — координаты текстового слоя PyMuPDF (страница без /Rotate);
    * display — страница как отображается (этот bbox попадает в JSON);
    * layout — страница, повёрнутая так, чтобы преобладающие строки текста шли
      слева направо: в ней строится сетка таблиц и считается порядок чтения.
      Определяется по направлению строк, а не по /Rotate: альбомную таблицу,
      вставленную в книжный лист поворотом содержимого, флаг не выдаёт.

    find_tables, в отличие от get_text, возвращает координаты display.
    Направление layout задаётся через set_turn после поиска таблиц.
    """

    def __init__(self, page: pymupdf.Page, data: dict):
        self.page = page
        self.rot = page.rotation
        self.data = data
        self.unrot_rect = _unrotated_rect(page)
        self.set_turn(0)

    def set_turn(self, turn: int) -> None:
        self.turn = turn
        ur = self.unrot_rect
        if turn in (90, 270):
            self.layout_w, self.layout_h = ur.height, ur.width
        else:
            self.layout_w, self.layout_h = ur.width, ur.height

    def display(self, unrot) -> BBox:
        return _to_display(self.page, unrot)

    def turned(self, unrot, turn: int) -> BBox:
        """Прямоугольник unrot на странице, повёрнутой на turn° по часовой."""
        r = pymupdf.Rect(unrot)
        ur = self.unrot_rect
        x0, y0, x1, y1 = r.x0 - ur.x0, r.y0 - ur.y0, r.x1 - ur.x0, r.y1 - ur.y0
        w, h = ur.width, ur.height
        if turn == 90:
            x0, y0, x1, y1 = h - y1, x0, h - y0, x1
        elif turn == 180:
            x0, y0, x1, y1 = w - x1, h - y1, w - x0, h - y0
        elif turn == 270:
            x0, y0, x1, y1 = y0, w - x1, y1, w - x0
        return (round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2))

    def layout(self, unrot) -> BBox:
        return self.turned(unrot, self.turn)

    def disp_to_unrot(self, disp) -> pymupdf.Rect:
        r = pymupdf.Rect(disp)
        if self.rot:
            r = r * self.page.derotation_matrix
            r.normalize()
        return r


def _extract_tables(
    page: pymupdf.Page, cfg: PipelineConfig, base_conf: float, frame: _Frame
) -> tuple[list[RawBlock], list[pymupdf.Rect], list[str]]:
    blocks: list[RawBlock] = []
    rects: list[pymupdf.Rect] = []  # в координатах unrot — для исключения текста таблиц
    try:
        found = page.find_tables(strategy=cfg.text_table_strategy)
    except Exception as exc:  # noqa: BLE001
        log.warning("find_tables failed on page %s: %s", page.number + 1, exc)
        return blocks, rects, [f"page {page.number + 1}: table detection failed: {exc}"]

    for tab in found.tables:
        cells_unrot = [frame.disp_to_unrot(c) for c in (tab.cells or []) if c is not None]
        if len(cells_unrot) < cfg.min_table_cells:
            continue
        texts = []
        for r in cells_unrot:
            inner = pymupdf.Rect(r.x0 + 0.5, r.y0 + 0.5, r.x1 - 0.5, r.y1 - 0.5)
            texts.append(join_cell_lines(page.get_textbox(inner) or ""))
        tab_unrot = frame.disp_to_unrot(tab.bbox)
        # у таблицы своё направление: повёрнутая таблица на обычной странице
        turn = _dominant(_turn_weights(frame.data, lambda bb: _inside_any(bb, [tab_unrot])))
        table = table_from_cell_bboxes([frame.turned(r, turn) for r in cells_unrot], texts)
        if table is None or not any(c.text for c in table.cells):
            continue
        if table.n_rows * table.n_cols < cfg.min_table_cells:
            continue
        g = garbage_ratio(table.all_text())
        conf = max(0.0, 0.95 * base_conf * (1 - min(0.5, 2 * g)) * table.structure_penalty)
        rects.append(tab_unrot)
        blocks.append(
            RawBlock(
                bbox=frame.display(tab_unrot),
                source=BlockSource.TABLE_EXTRACTOR,
                confidence=round(conf, 3),
                type=BlockType.TABLE,
                table=table,
            )
        )
    return blocks, rects, []


# ---------------------------------------------------------------- текст


def _inside_any(bbox, rects: list[pymupdf.Rect]) -> bool:
    x = (bbox[0] + bbox[2]) / 2
    y = (bbox[1] + bbox[3]) / 2
    return any(r.x0 - 1 <= x <= r.x1 + 1 and r.y0 - 1 <= y <= r.y1 + 1 for r in rects)


def _line_info(line: dict) -> tuple[str, float, float]:
    """(текст, кегль, доля жирного) строки, кегль и доля — по числу символов."""
    text = ""
    size_w = 0.0
    n = 0
    bold = 0
    for span in line.get("spans", []):
        t = span.get("text", "")
        k = len(t.strip())
        text += t
        size_w += span.get("size", 0) * k
        n += k
        font = span.get("font", "") or ""
        if span.get("flags", 0) & _BOLD_FLAG or "bold" in font.lower():
            bold += k
    size = size_w / n if n else 0.0
    return clean_line(text), size, (bold / n if n else 0.0)


def _split_groups(lines: list[tuple]) -> list[list[tuple]]:
    """Разбить блок PyMuPDF при смене кегля или большом вертикальном зазоре.

    Элемент: (текст, кегль, доля жирного, bbox_unrot, bbox_layout); зазоры — по layout.
    """
    groups: list[list[tuple]] = []
    for ln in lines:
        if not groups:
            groups.append([ln])
            continue
        prev = groups[-1][-1]
        size, bb = ln[1], ln[4]
        psize, pbb = prev[1], prev[4]
        line_h = max(1.0, pbb[3] - pbb[1])
        gap = bb[1] - pbb[3]
        size_jump = psize and size and abs(size - psize) / max(size, psize) > 0.15
        if size_jump or gap > 1.2 * line_h or starts_list_item(ln[0]):
            groups.append([ln])
        else:
            groups[-1].append(ln)
    return groups


def _union(bbs) -> tuple[float, float, float, float]:
    bbs = list(bbs)
    return (min(b[0] for b in bbs), min(b[1] for b in bbs), max(b[2] for b in bbs), max(b[3] for b in bbs))


@dataclass
class TextLayerResult:
    blocks: list[RawBlock]
    body_size: Optional[float]
    warnings: list[str]
    layout_width: float
    layout_height: float


def extract_text_layer_page(
    page: pymupdf.Page, cfg: PipelineConfig, assessment: PageAssessment
) -> TextLayerResult:
    """Блоки страницы из текстового слоя."""
    warnings: list[str] = []
    base_conf = cfg.hidden_text_layer_confidence if assessment.hidden_ratio > 0.5 else 0.99
    blocks: list[RawBlock] = []
    table_rects: list[pymupdf.Rect] = []

    data = page.get_text("dict", flags=pymupdf.TEXT_PRESERVE_WHITESPACE | pymupdf.TEXT_MEDIABOX_CLIP)
    frame = _Frame(page, data)

    if cfg.detect_tables:
        tblocks, table_rects, tw = _extract_tables(page, cfg, base_conf, frame)
        blocks.extend(tblocks)
        warnings.extend(tw)
    frame.set_turn(_page_turn(data, table_rects))
    for b, r in zip(blocks, table_rects):
        b.layout_bbox = frame.layout(r)

    sizes: list[float] = []
    for blk in data.get("blocks", []):
        if blk.get("type") != 0:
            continue
        lines = []
        for line in blk.get("lines", []):
            if _inside_any(line["bbox"], table_rects):
                continue
            text, size, bold = _line_info(line)
            if not text:
                continue
            ub = tuple(line["bbox"])
            lines.append((text, size, bold, ub, frame.layout(ub)))
            sizes.extend([size] * max(1, len(text) // 10))
        for group in _split_groups(lines):
            texts = [g[0] for g in group]
            text = join_cell_lines("\n".join(texts))
            if not text:
                continue
            nchar = sum(len(t) for t in texts) or 1
            size = sum(g[1] * len(g[0]) for g in group) / nchar
            bold = sum(g[2] * len(g[0]) for g in group) / nchar
            conf = base_conf * (1 - min(0.5, 2 * garbage_ratio(text)))
            ub = _union(g[3] for g in group)
            blocks.append(
                RawBlock(
                    bbox=frame.display(ub),
                    layout_bbox=frame.layout(ub),
                    source=BlockSource.TEXT_LAYER,
                    confidence=round(conf, 3),
                    text=text,
                    font_size=round(size, 2),
                    bold_ratio=round(bold, 2),
                    n_lines=len(group),
                )
            )

    # картинки (логотипы, печати); фон скана под невидимым слоем пропускаем
    _, img_rects = _image_coverage(page)
    page_area = abs(page.rect) or 1.0
    for r in img_rects:
        share = abs(r) / page_area
        if share < 0.02 or (share > 0.8 and assessment.hidden_ratio > 0.5):
            continue
        blocks.append(
            RawBlock(
                bbox=frame.display(r),
                layout_bbox=frame.layout(r),
                source=BlockSource.TEXT_LAYER,
                confidence=1.0,
                type=BlockType.IMAGE,
            )
        )

    body = statistics.median(sizes) if sizes else None
    return TextLayerResult(blocks, body, warnings, round(frame.layout_w, 2), round(frame.layout_h, 2))
