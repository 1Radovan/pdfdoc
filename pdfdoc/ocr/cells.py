"""Распознавание одной ячейки таблицы.

Исправления по сравнению с исходным pdf_table_ocr (сверка 1.2), проверенные
на синтетических сканах:
* нет принудительного увеличения ×2: при 300 dpi цифры и так ~28 px, после
  увеличения Tesseract путает 5→9, 4→2; увеличиваем только реально мелкий
  текст (< 20 px);
* вокруг текста добавляется белое поле — без него Tesseract хуже читает
  одиночные цифры;
* обрывки линий сетки удаляются только у краёв ячейки — штрихи цифр в
  тесных строках не стираются;
* одиночный прочерк распознаётся геометрически («-»), а не Tesseract'ом
  (тот читал его как «7»);
* уверенность берётся из image_to_data, а не теряется;
* числа (и непрочитанные ячейки) перечитываются с более светлым порогом
  бинаризации — на размытом скане тёмный порог стирает засечки и тонкие
  части цифр, а слова лучше читаются с прежним;
* «1» без засечек (бледный скан, шрифт Times) Tesseract читает как «|», «I»,
  «!», «]», «J» или не читает вовсе; в числовой колонке такие «палочки»
  заменяются на «1» (fix_bar_ones).
"""

from __future__ import annotations

import math
import re
from typing import Optional

import cv2
import numpy as np

from pdfdoc.ocr.engine import Tesseract, words_confidence, words_to_text
from pdfdoc.tables import CellData
from pdfdoc.text_utils import (
    clean_line,
    fix_homoglyphs,
    fix_number_separators,
    fix_ocr_confusions,
    is_numeric_like,
    join_cell_lines,
)

NUMERIC_WHITELIST = "0123456789.,-–—()%/:"
BAR_TEXT_RE = re.compile(r"[|Il!\[\]J]{1,3}")
BAR_ONE_CONFIDENCE = 0.6  # «1», восстановленная по контексту колонки, а не прочитанная


def _remove_edge_lines(ink: np.ndarray, dpi: int) -> np.ndarray:
    """Стереть остатки линий сетки, прилегающие к краям ячейки.

    Два прохода: (1) длинные горизонтальные штрихи у верхнего/нижнего края и
    вертикальные у левого/правого (морфология) — так убираются и «уголки»
    (верх + левый край, склеенные в одну компоненту); (2) тонкие компоненты,
    касающиеся края.

    Вертикальные штрихи цифр в верхней/нижней полосе низкой ячейки — не линии.
    Обрывок вертикали тянется почти во всю высоту ячейки, а штрих буквы у
    левого края («к» в «кг», «1») короче — его не трогаем.
    """
    h, w = ink.shape
    band = max(4, int(dpi * 0.03))
    hl = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(8, int(w * 0.3)), 1)))
    vl = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(8, int(h * 0.6)))))
    tb = np.zeros_like(ink)
    tb[:band, :] = 255
    tb[h - band :, :] = 255
    lr = np.zeros_like(ink)
    lr[:, :band] = 255
    lr[:, w - band :] = 255
    lines = cv2.bitwise_or(cv2.bitwise_and(hl, tb), cv2.bitwise_and(vl, lr))
    out = ink.copy()
    if lines.any():
        lines = cv2.dilate(lines, np.ones((3, 3), np.uint8))
        out[lines > 0] = 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        # обрывок вертикали прилегает к левому/правому краю, горизонтали — к
        # верхнему/нижнему: «1» в низкой ячейке касается низа, но это не линия
        touch_lr = x <= band or x + cw >= w - band
        touch_tb = y <= band or y + ch >= h - band
        thin_v = touch_lr and cw <= band and ch >= 0.6 * h
        thin_h = touch_tb and ch <= band and cw >= 0.3 * w
        if thin_v or thin_h:
            out[labels == i] = 0
    return out


def _remove_specks(ink: np.ndarray, dpi: int) -> np.ndarray:
    min_area = max(4, int((dpi / 300) ** 2 * 12))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    out = ink.copy()
    for i in range(1, n):
        if stats[i][4] < min_area:
            out[labels == i] = 0
    return out


def _text_lines(ink: np.ndarray) -> list[tuple[int, int]]:
    rows = np.where(ink.sum(axis=1) > 0)[0]
    if rows.size == 0:
        return []
    runs = []
    start = prev = rows[0]
    for r in rows[1:]:
        if r - prev > 2:
            runs.append((start, prev + 1))
            start = r
        prev = r
    runs.append((start, prev + 1))
    return runs


def prepare_cell(gray: np.ndarray, dpi: int, thin_strokes: bool = False) -> tuple[Optional[np.ndarray], str, int]:
    """Подготовить ячейку к OCR.

    thin_strokes — более светлый порог для цифр (см. ниже).
    Возвращает (изображение | None, особый случай: '' | 'empty' | 'dash', число строк).
    """
    if gray.shape[0] < 6 or gray.shape[1] < 6:
        return None, "empty", 0
    # Порог Otsu, но не светлее «фон − 40»: на пустой шумной ячейке Otsu
    # делит сам шум пополам и «находит» текст.
    # thin_strokes: посередине между ними. Otsu на размытом скане оставляет
    # только сердцевину штрихов: у «1» пропадают засечки (Tesseract читает «|»,
    # «J» или ничего), у «0» рвётся контур («0,00» → «9.00»). Для слов светлый
    # порог хуже: штрихи толстеют, «р» слипается в «з»/«т».
    # Медиана 3x3 гасит «соль-перец» шум, но съедает тонкий прочерк — поэтому
    # только если фон заметно шумит (СКО светлых пикселей > 6).
    bg0 = float(np.median(gray))
    light = gray[gray > bg0 - 30]
    noisy = light.size > 0 and float(light.std()) > 6.0
    smooth = cv2.medianBlur(gray, 3) if noisy else gray
    bg = float(np.median(smooth))
    otsu, _ = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = min(otsu, bg - 40)
    if thin_strokes:
        thr = (thr + bg - 40) / 2
    if thr <= 0:
        return None, "empty", 0
    ink = np.where(smooth <= thr, 255, 0).astype(np.uint8)  # как THRESH_BINARY_INV
    ink = _remove_edge_lines(ink, dpi)
    ink = _remove_specks(ink, dpi)
    ys, xs = np.nonzero(ink)
    # порог абсолютный (в пикселях при данном dpi), а не доля площади: прочерк
    # «-» при 300 dpi — всего ~30 px, в широкой ячейке это < 0,1% площади
    if ys.size < max(12, int((dpi / 300) ** 2 * 20)):
        return None, "empty", 0

    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    ih, iw = y1 - y0, x1 - x0
    n_comp = cv2.connectedComponents(ink[y0:y1, x0:x1], connectivity=8)[0] - 1
    if n_comp == 1 and ih <= max(3, dpi * 0.025) and iw >= 2.5 * ih:
        return None, "dash", 1

    lines = _text_lines(ink[y0:y1, x0:x1])
    m = max(6, int(dpi * 0.04))
    crop = 255 - ink[max(0, y0 - m) : y1 + m, max(0, x0 - m) : x1 + m]
    crop = cv2.copyMakeBorder(crop, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255)
    line_h = float(np.median([b - a for a, b in lines])) if lines else ih
    if 0 < line_h < 20:
        k = min(4, math.ceil(32 / line_h))
        crop = cv2.resize(crop, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
    return crop, "", len(lines)


def _read(img: np.ndarray, n_lines: int, engine: Tesseract, numeric_recheck: bool) -> tuple[str, Optional[float]]:
    psm = 7 if n_lines <= 1 else 6
    words = engine.words(img, psm=psm)
    text = clean_line(join_cell_lines(words_to_text(words)))
    conf = words_confidence(words)

    if not text and n_lines <= 1:
        # одиночный символ: psm 10 («один знак»)
        words = engine.words(img, psm=10)
        text = clean_line(words_to_text(words, " "))
        conf = words_confidence(words)

    if numeric_recheck and text and is_numeric_like(text):
        words2 = engine.words(img, psm=psm, whitelist=NUMERIC_WHITELIST, lang="eng")
        text2 = clean_line(join_cell_lines(words_to_text(words2)))
        conf2 = words_confidence(words2)
        if text2 and conf2 is not None and (conf is None or conf2 >= conf - 0.02):
            text, conf = text2, conf2
    return text, conf


def _digits_like(text: str) -> bool:
    return is_numeric_like(text) or bool(BAR_TEXT_RE.fullmatch(text))


def ocr_cell(gray: np.ndarray, engine: Tesseract, dpi: int, *, numeric_recheck: bool = True,
             homoglyphs: bool = True) -> tuple[str, Optional[float]]:
    """(текст, уверенность 0..1 | None для пустой ячейки)."""
    img, special, n_lines = prepare_cell(gray, dpi)
    if special == "empty":
        return "", None
    if special == "dash":
        return "-", 0.9
    assert img is not None

    text, conf = _read(img, n_lines, engine, numeric_recheck)
    if not text or _digits_like(text):
        # число (или не прочитано): перечитать со светлым порогом — так
        # сохраняются засечки «1» и тонкие части «0»
        img2, special2, n2 = prepare_cell(gray, dpi, thin_strokes=True)
        if special2 == "" and img2 is not None:
            text2, conf2 = _read(img2, n2, engine, numeric_recheck)
            if text2 and _digits_like(text2):
                img, text, conf = img2, text2, conf2

    if text and is_numeric_like(text):
        text = fix_number_separators(text)
    elif homoglyphs and text:
        text = fix_ocr_confusions(fix_homoglyphs(text))
    if not text:
        n = bar_glyphs(img)
        if n:
            return "|" * n, 0.3
        # чернила есть, а текст не прочитан: явно низкая уверенность
        return "", 0.0
    return text, conf


def bar_glyphs(img: np.ndarray) -> int:
    """Сколько в подготовленной ячейке вертикальных «палочек» одной высоты (0 — не только они)."""
    ink = (img < 128).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    comps = [stats[i] for i in range(1, n)]
    if not comps:
        return 0
    hmax = max(s[3] for s in comps)
    main = [s for s in comps if s[3] >= 0.6 * hmax]
    rest = sum(int(s[4]) for s in comps if s[3] < 0.6 * hmax)
    if len(main) > 3 or rest > 0.3 * sum(int(s[4]) for s in main):
        return 0
    tops = [int(s[1]) for s in main]
    if max(tops) - min(tops) > 0.3 * hmax:
        return 0
    if any(s[2] > 0.5 * s[3] for s in main):
        return 0
    return len(main)


LATIN_SHORT_RE = re.compile(r"[A-Za-z]{1,4}\.?")
CYR_SHORT_RE = re.compile(r"[А-Яа-яЁё]{1,5}\.?")
_CYR_RE = re.compile(r"[А-Яа-яЁё]")
_LAT_RE = re.compile(r"[A-Za-z]")


def latin_cells_to_reread(cells: list[CellData]) -> list[int]:
    """Индексы коротких латинских ячеек («Kr», «UIT.») в русской таблице.

    С rus+eng Tesseract читает одиночные короткие слова латиницей («кг» → «Kr»,
    «шт.» → «UIT.»): контекста нет, и английская модель выигрывает. В таблице,
    где кириллицы большинство, такие ячейки стоит перечитать только по-русски.
    """
    text = " ".join(c.text for c in cells if c.text)
    cyr, lat = len(_CYR_RE.findall(text)), len(_LAT_RE.findall(text))
    if cyr < 20 or cyr < 2 * lat:
        return []
    return [i for i, c in enumerate(cells) if c.text and LATIN_SHORT_RE.fullmatch(c.text)]


def reread_russian(gray: np.ndarray, engine: Tesseract, dpi: int) -> tuple[str, Optional[float]]:
    """Перечитать ячейку только русской моделью; ('', None), если не вышло слово кириллицей."""
    img, special, n_lines = prepare_cell(gray, dpi)
    if special or img is None or n_lines > 1:
        return "", None
    words = engine.words(img, psm=7, lang="rus")
    text = clean_line(words_to_text(words, " "))
    if not CYR_SHORT_RE.fullmatch(text):
        return "", None
    return text, words_confidence(words)


GROUPED_RE = re.compile(r"-?\d{1,3}(?: \d{3})+[.,]\d{1,2}")
UNGROUPED_RE = re.compile(r"(-?)(\d{1,3}(?: ?\d{3})*\d{0,3})([.,]\d{1,2})")


def _group_thousands(m: re.Match) -> str:
    digits = m.group(2).replace(" ", "")
    head = len(digits) % 3 or 3
    parts = [digits[:head]] + [digits[i : i + 3] for i in range(head, len(digits), 3)]
    return m.group(1) + " ".join(parts) + m.group(3)


def regroup_thousands(cells: list[CellData]) -> int:
    """Пробелы между тысячами — как у большинства сумм в колонке.

    Узкий пробел после первой цифры OCR часто теряет: в колонке «1 490,00»,
    «27 525,00» встречается «1490,00» или «1317 500,00». Если в колонке не
    меньше половины сумм записано с пробелами, остальные приводятся к тому же
    виду; значение числа не меняется. Возвращает число исправленных ячеек.
    """
    cols: dict[int, list[CellData]] = {}
    for c in cells:
        m = UNGROUPED_RE.fullmatch(c.text) if c.colspan == 1 and c.text else None
        if m and len(m.group(2).replace(" ", "")) >= 4:  # суммы от 1000
            cols.setdefault(c.col, []).append(c)
    fixed = 0
    for col_cells in cols.values():
        grouped = sum(1 for c in col_cells if GROUPED_RE.fullmatch(c.text))
        if grouped < 2 or grouped < 0.5 * len(col_cells):
            continue
        for c in col_cells:
            new = UNGROUPED_RE.sub(_group_thousands, c.text)
            if new != c.text:
                c.text = new
                fixed += 1
    return fixed


def fix_bar_ones(cells: list[CellData]) -> int:
    """В колонках, где почти всё — числа, «палочки» («|», «I», «]», …) → «1».

    Возвращает число исправленных ячеек. Уверенность таких ячеек снижается до
    BAR_ONE_CONFIDENCE: цифра восстановлена по контексту, а не прочитана.
    """
    cols: dict[int, list[CellData]] = {}
    for c in cells:
        if c.colspan == 1 and c.text:
            cols.setdefault(c.col, []).append(c)
    fixed = 0
    for col_cells in cols.values():
        bars = [c for c in col_cells if BAR_TEXT_RE.fullmatch(c.text)]
        if not bars:
            continue
        others = [c for c in col_cells if not BAR_TEXT_RE.fullmatch(c.text)]
        numeric = sum(1 for c in others if is_numeric_like(c.text))
        if numeric < 3 or numeric < 0.6 * len(others):
            continue
        for c in bars:
            c.text = "1" * len(c.text)
            c.confidence = BAR_ONE_CONFIDENCE
            fixed += 1
    return fixed
