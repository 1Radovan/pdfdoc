"""Определение типа блока и порядок чтения (эвристики, без ML)."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Optional

from pdfdoc._raw import RawBlock, RawPage
from pdfdoc.models import BlockSource, BlockType
from pdfdoc.text_utils import alnum_count, upper_ratio

_PAGE_NUM_RE = re.compile(
    r"^(?:[-–—]\s*)?\d{1,4}(?:\s*[-–—])?$"
    r"|^(?:стр\.?|страница|лист|page|p\.)\s*\d{1,4}(?:\s*(?:из|of|/)\s*\d{1,4})?$",
    re.IGNORECASE,
)
_SIGN_ROLE_RE = re.compile(
    r"(подпись|м\.\s?п\.|руководител|директор|бухгалтер|исполнител|ответственн|"
    r"начальник|председател|член комиссии|утверждаю|согласовано|кассир|"
    r"сдал|принял|получил|уполномоченн|представител|заведующ)",
    re.IGNORECASE,
)
_INITIALS_RE = re.compile(
    r"[А-ЯЁ][а-яё]+\s+[А-ЯЁ]\.\s?[А-ЯЁ]\.|[А-ЯЁ]\.\s?[А-ЯЁ]\.\s?[А-ЯЁ][а-яё]+"
)
_UNDERLINE_RE = re.compile(r"_{3,}|\.{8,}|[—–-]{4,}")
_SLASH_NAME_RE = re.compile(r"/\s*[А-ЯЁA-Z][^/\n]{2,40}/")
_LIST_RE = re.compile(
    r"^(?:[•●▪◦‣■□\-–—*]\s+\S|\d{1,2}[.)]\s+\S|\d{1,2}\.\d{1,2}\.?\s+\S|[а-яa-z]\)\s+\S)"
)


def _is_signature(b: RawBlock, text: str) -> bool:
    if len(text) > 300:
        return False
    low = text.casefold()
    role = bool(_SIGN_ROLE_RE.search(low))
    initials = bool(_INITIALS_RE.search(text))
    slash = bool(_SLASH_NAME_RE.search(text))
    underline = bool(_UNDERLINE_RE.search(text))
    if "подпись" in low or re.search(r"\bм\.\s?п\.?", low):
        return True
    if underline and (role or initials or slash):
        return True
    if slash and (role or initials):
        return True
    if b.near_hline and (role or initials) and len(text) <= 150:
        return True
    return False


def _is_header(b: RawBlock, text: str, body_size: Optional[float]) -> bool:
    n = len(text)
    if n > 250 or b.n_lines > 4 or alnum_count(text) < 2:
        return False
    if text.rstrip().endswith((",", ";")):
        return False
    is_ocr = b.source == BlockSource.OCR
    if body_size and b.font_size:
        # у OCR размер — высота строки; у заглавных букв она ниже, отсюда другой порог
        ratio = b.font_size / body_size
        if ratio >= (1.2 if is_ocr else 1.18):
            return True
    letters = sum(c.isalpha() for c in text)
    if letters >= 4 and upper_ratio(text) >= 0.8 and n <= 160 and b.n_lines <= 3:
        return True
    if not is_ocr and b.bold_ratio >= 0.6 and n <= 160 and b.n_lines <= 3:
        return True
    return False


def classify_block(b: RawBlock, page: RawPage) -> BlockType:
    if b.type is not None:
        return b.type
    text = (b.text or "").strip()
    if not text:
        return BlockType.TEXT
    ph = page.layout_height or page.height
    y_center = (b.lb[1] + b.lb[3]) / 2
    in_margin = y_center < ph * 0.08 or y_center > ph * 0.92
    if in_margin and _PAGE_NUM_RE.match(text):
        return BlockType.PAGE_NUMBER
    if _is_signature(b, text):
        return BlockType.SIGNATURE
    if _is_header(b, text, page.body_size):
        return BlockType.HEADER
    if _LIST_RE.match(text):
        return BlockType.LIST_ITEM
    return BlockType.TEXT


def classify_page(page: RawPage) -> None:
    for b in page.blocks:
        b.type = classify_block(b, page)


def mark_running_headers(pages: list[RawPage]) -> None:
    """Колонтитулы: текст у верхнего/нижнего края, повторяющийся на страницах.

    Цифры при сравнении заменяются на «#», поэтому «Стр. 1» и «Стр. 2» совпадут.
    """
    usable = [p for p in pages if p.blocks]
    if len(usable) < 2:
        return
    seen: dict[tuple[str, str], set[int]] = defaultdict(set)
    cand: list[tuple[RawPage, RawBlock, tuple[str, str]]] = []
    for p in usable:
        for b in p.blocks:
            if b.table is not None or not b.text:
                continue
            ph = p.layout_height or p.height
            if b.lb[3] <= ph * 0.1:
                zone = "top"
            elif b.lb[1] >= ph * 0.9:
                zone = "bottom"
            else:
                continue
            key = (zone, re.sub(r"\d+", "#", re.sub(r"\s+", " ", b.text.casefold())).strip())
            seen[key].add(p.page)
            cand.append((p, b, key))
    need = max(2, (len(usable) + 1) // 2)
    for p, b, key in cand:
        if len(seen[key]) >= need and b.type != BlockType.PAGE_NUMBER:
            b.type = BlockType.PAGE_HEADER if key[0] == "top" else BlockType.PAGE_FOOTER


def reading_order(blocks: list[RawBlock], line_tol: float = 4.0) -> list[RawBlock]:
    """Сверху вниз; блоки на одной высоте (±line_tol пт) — слева направо."""
    ordered = sorted(blocks, key=lambda b: (b.lb[1], b.lb[0]))
    out: list[RawBlock] = []
    band: list[RawBlock] = []
    band_y: Optional[float] = None
    for b in ordered:
        if band_y is not None and abs(b.lb[1] - band_y) <= line_tol:
            band.append(b)
            continue
        out.extend(sorted(band, key=lambda x: x.lb[0]))
        band, band_y = [b], b.lb[1]
    out.extend(sorted(band, key=lambda x: x.lb[0]))
    return out
