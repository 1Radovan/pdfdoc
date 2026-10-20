from __future__ import annotations

import numpy as np
import pytest

from pdfdoc._raw import RawBlock, RawPage
from pdfdoc.classify import classify_block
from pdfdoc.models import BlockSource, BlockType, PageSource
from pdfdoc.ocr.cells import (
    BAR_ONE_CONFIDENCE,
    bar_glyphs,
    fix_bar_ones,
    latin_cells_to_reread,
    prepare_cell,
    regroup_thousands,
)
from pdfdoc.ocr.grid import find_tables
from pdfdoc.tables import CellData, table_from_cell_bboxes
from pdfdoc.text_utils import (
    detect_language,
    fix_homoglyphs,
    fix_number_separators,
    garbage_ratio,
    join_cell_lines,
    misread_cyrillic_ratio,
    starts_list_item,
)


def test_homoglyphs():
    assert fix_homoglyphs("Cтанoк тoкарный") == "Станок токарный"
    assert fix_homoglyphs("Главный бухгалтер / Смирнова A.B. /") == "Главный бухгалтер / Смирнова А.В. /"
    assert fix_homoglyphs("ООО Ромашка LLC") == "ООО Ромашка LLC"
    assert fix_homoglyphs("Отдел IT и PDF") == "Отдел IT и PDF"


def test_ocr_confusions():
    from pdfdoc.text_utils import fix_ocr_confusions as f

    assert f("№ пп") == f("№ n/n") == "№ п/п"
    assert f("№ 5") == "№ 5"
    assert f("установка (монтаж}") == "установка (монтаж)"
    assert f("Пресс гидравлический [16330") == "Пресс гидравлический П6330"
    assert f("см. [12] и [123]") == "см. [12] и [123]"


def test_language():
    assert detect_language("Итого к выплате") == "ru"
    assert detect_language("Total amount") == "en"
    assert detect_language("12 345,00") is None


def test_garbage_ratio_detects_mojibake():
    assert garbage_ratio("Ïðèâåò ìèð") > 0.5
    assert garbage_ratio("Привет, мир! «Цитата» 100 %") == 0


def test_join_cell_lines():
    assert join_cell_lines("Вертикально-\nфрезерный станок") == "Вертикально-фрезерный станок"
    assert join_cell_lines("Иванов\nИ.И.") == "Иванов И.И."
    assert join_cell_lines("Болт ГОСТ 1000-\n99") == "Болт ГОСТ 1000-99"
    assert join_cell_lines("Санкт-\nПетербург") == "Санкт-Петербург"
    assert join_cell_lines("Цена -\n100") == "Цена - 100"  # тире, а не перенос
    # число, перенесённое в узкой ячейке
    assert join_cell_lines("4 321.5\n6") == "4 321.56"
    assert join_cell_lines("0.02\n5") == "0.025"
    assert join_cell_lines("98 765.\n43") == "98 765.43"
    assert join_cell_lines("2.5\n10") == "2.5 10"  # два значения, а не перенос
    assert join_cell_lines("Итого\n3") == "Итого 3"


def test_starts_list_item():
    for s in ("2. Срок поставки", "3.1. Оплата", "4) Гарантия", "• Доставка", "– Монтаж"):
        assert starts_list_item(s), s
    for s in ("5.1 Договора изменить", "— это лицо", "2024 года", "12.03.2024 г."):
        assert not starts_list_item(s), s


def test_fix_number_separators():
    assert fix_number_separators("2. 340,00") == "2 340,00"
    assert fix_number_separators("2, 340,00") == "2 340,00"
    assert fix_number_separators("2,340,00") == "2 340,00"
    assert fix_number_separators("4.765 120,50") == "4 765 120,50"
    for ok in ("2.340,00", "2,340.00", "2 340,00", "12.03.2024", "1,5"):
        assert fix_number_separators(ok) == ok


def test_misread_cyrillic_layer():
    # «Акт сдачи работ по договору от …» без русской модели
    assert misread_cyrillic_ratio("Akr cgaun pa6or no goroBopy or 01.02.2024 r. Ha cymmy " * 4) > 0.9
    english = ("The seller shall deliver the goods to the buyer within ten days from the date "
               "of payment and the buyer is obliged to accept them in accordance with this contract.")
    assert misread_cyrillic_ratio(english) == 0
    assert misread_cyrillic_ratio("Акт сдачи работ по договору от 01.02.2024 на сумму " * 4) == 0
    assert misread_cyrillic_ratio("Akr cgaun") == 0  # мало слов — не судим
    # страница с одними подписями
    assert misread_cyrillic_ratio("Pykosoqutenb Merposa A. B. 6yxranrep Merposa A. B.") > 0.9
    assert misread_cyrillic_ratio("Signed by the director of the company") == 0


def test_regroup_thousands_follows_column():
    texts = ["Сумма", "2 340,00", "12 600,50", "4810,00", "1234 567,00", "75,00", "Итого"]
    cells = [CellData(row=i, col=0, text=t) for i, t in enumerate(texts)]
    other = [CellData(row=i, col=1, text=t) for i, t in enumerate(["Цена", "2340,00", "4810,00", "1 200,00"])]
    assert regroup_thousands(cells + other) == 2
    assert [c.text for c in cells] == ["Сумма", "2 340,00", "12 600,50", "4 810,00", "1 234 567,00", "75,00", "Итого"]
    assert [c.text for c in other][1:3] == ["2340,00", "4810,00"]  # в колонке большинство без пробелов


def test_latin_cells_to_reread_only_in_russian_table():
    ru = ["Наименование товара", "Ед. изм.", "Количество", "Болт стальной", "Kr", "UIT.", "M12", "ISO"]
    cells = [CellData(row=i, col=0, text=t) for i, t in enumerate(ru)]
    assert [cells[i].text for i in latin_cells_to_reread(cells)] == ["Kr", "UIT.", "ISO"]
    en = ["Description", "Unit", "Quantity", "Steel bolt", "kg", "pcs"]
    assert latin_cells_to_reread([CellData(row=i, col=0, text=t) for i, t in enumerate(en)]) == []


def test_table_spans_md_html():
    # 2x3: верхняя строка — объединение колонок 1..2, левая колонка — объединение строк
    boxes = [(0, 0, 10, 20), (10, 0, 30, 10), (10, 10, 20, 20), (20, 10, 30, 20)]
    t = table_from_cell_bboxes(boxes, ["A", "B", "c", "d"])
    assert (t.n_rows, t.n_cols) == (2, 3)
    assert t.matrix() == [["A", "B", ""], ["", "c", "d"]]
    assert t.matrix("repeat") == [["A", "B", "B"], ["A", "c", "d"]]
    spans = {(s.row, s.col, s.rowspan, s.colspan) for s in t.spans()}
    assert spans == {(0, 0, 2, 1), (0, 1, 1, 2)}
    assert 'rowspan="2"' in t.to_html() and 'colspan="2"' in t.to_html()
    assert t.to_markdown().splitlines()[1] == "|---|---|---|"
    assert t.empty_cells_by_column() == {}  # пусты только перекрытые ячейки


def _page():
    return RawPage(page=1, width=595, height=842, rotation=0, source=PageSource.TEXT_LAYER, body_size=11)


def _rb(text, y=300, size=11, bold=0.0, source=BlockSource.TEXT_LAYER):
    return RawBlock(bbox=(72, y, 400, y + 12), source=source, confidence=0.99, text=text,
                    font_size=size, bold_ratio=bold)


def test_classifier():
    p = _page()
    assert classify_block(_rb("ВЕДОМОСТЬ начисления", size=14), p) == BlockType.HEADER
    assert classify_block(_rb("Главный бухгалтер ______ / Смирнова А.В. /"), p) == BlockType.SIGNATURE
    assert classify_block(_rb("Стр. 2 из 5", y=810), p) == BlockType.PAGE_NUMBER
    assert classify_block(_rb("Стр. 2 из 5", y=400), p) == BlockType.TEXT
    assert classify_block(_rb("1. Начисления произведены согласно расписанию."), p) == BlockType.LIST_ITEM
    assert classify_block(_rb("Обычный абзац текста, который ничем не выделен."), p) == BlockType.TEXT


def test_prepare_cell_empty_noise_and_dash():
    rng = np.random.default_rng(0)
    noise = np.clip(rng.normal(245, 8, (80, 200)), 0, 255).astype(np.uint8)
    assert prepare_cell(noise, 300)[1] == "empty"
    dash = noise.copy()
    dash[38:42, 90:115] = 20
    assert prepare_cell(dash, 300)[1] == "dash"


def test_prepare_cell_keeps_one_touching_bottom():
    # «1» без засечек в низкой ячейке (строка номеров граф) касается нижнего
    # края, но это не обрывок вертикали сетки — те прилегают к левому/правому краю
    cell = np.full((40, 99), 255, np.uint8)
    cell[6:31, 46:53] = 30
    for thin in (False, True):
        img, special, _ = prepare_cell(cell, 300, thin_strokes=thin)
        assert special == "" and img is not None
        assert bar_glyphs(img) == 1


def test_prepare_cell_keeps_letter_stem_at_edge():
    # вертикальный штрих буквы («к», «1») у левого края ячейки: короче и толще
    # обрывка линии сетки — не стирается; обрывок во всю высоту — стирается
    cell = np.full((60, 120), 255, np.uint8)
    cell[15:45, 2:8] = 30
    img, special, _ = prepare_cell(cell, 300)
    assert special == "" and img is not None
    line = np.full((60, 120), 255, np.uint8)
    line[:, 0:2] = 30
    line[20:40, 50:56] = 30
    img, special, _ = prepare_cell(line, 300)
    assert special == "" and img is not None
    assert bar_glyphs(img) == 1  # остался только штрих в середине


def _sheared_grid(xs, ys, tilt_deg, thickness=4):
    """Сетка, у которой вертикали наклонены (скан листа со сдвигом), а горизонтали ровные."""
    h, w = ys[-1] + 100, xs[-1] + 100
    img = np.full((h, w), 255, np.uint8)
    k = np.tan(np.radians(tilt_deg))
    yc = (ys[0] + ys[-1]) / 2
    for y in ys:
        xa = int(round(xs[0] + k * (y - yc)))
        xb = int(round(xs[-1] + k * (y - yc))) + thickness
        img[y : y + thickness, xa:xb] = 0
    rows = np.arange(ys[0], ys[-1] + thickness)
    for x in xs:
        cx = np.rint(x + k * (rows - yc)).astype(int)
        for d in range(thickness):
            img[rows, cx + d] = 0
    return img


@pytest.mark.parametrize("tilt", [0.45, -0.45, 1.5])
def test_grid_on_sheared_scan(tilt):
    # как в «приложении 1»: 7 колонок, 16 строк на ~2950 px, вертикали под 0,45°
    # (смещение ±12 px от середины к краям таблицы — больше допуска проверки границ)
    xs = [200, 305, 822, 1179, 1536, 1775, 2015, 2375]
    ys = [270 + 185 * i for i in range(17)]
    img = _sheared_grid(xs, ys, tilt)
    tables = find_tables(img, 300)
    assert len(tables) == 1
    g = tables[0]
    assert (g.n_rows, g.n_cols) == (16, 7)
    assert all(c.rowspan == 1 and c.colspan == 1 for c in g.cells)
    assert len(g.cells) == 16 * 7
    assert all(abs(np.degrees(np.arctan(s)) - tilt) < 0.05 for s in g.vslopes)
    # ячейка в углу таблицы вырезается по реальным линиям, а не по средним
    corner = next(c for c in g.cells if c.r0 == 0 and c.c0 == 0)
    x0, _, x1, _ = g.cell_box(corner)
    expect = xs[0] + np.tan(np.radians(tilt)) * ((ys[0] + ys[1]) / 2 - (ys[0] + ys[-1]) / 2)
    assert abs(x0 - expect) <= 4


def test_table_inside_page_frame_found_once():
    # края скана дают рамку во весь лист; таблица внутри — отдельный компонент:
    # её не должно быть дважды, а сама рамка — не таблица
    img = np.full((2480, 3508), 255, np.uint8)
    img[60:64, 30:3480] = 0
    img[2360:2364, 30:3480] = 0
    img[60:2364, 30:34] = 0
    img[60:2364, 3476:3480] = 0
    for y in (800, 950, 1100):
        img[y : y + 4, 400:3200] = 0
    for x in (400, 1200, 2400, 3196):
        img[800:1104, x : x + 4] = 0
    tables = find_tables(img, 300)
    assert len(tables) == 1
    assert (tables[0].n_rows, tables[0].n_cols) == (2, 3)


def test_fix_bar_ones_only_in_numeric_columns():
    nums = [CellData(row=r, col=0, text=t, confidence=0.95) for r, t in enumerate(["Кол-во", "2", "4", "10"])]
    bars = [CellData(row=4, col=0, text="|", confidence=0.9), CellData(row=5, col=0, text="]", confidence=0.3),
            CellData(row=6, col=0, text="||", confidence=0.3)]
    words = [CellData(row=r, col=1, text=t, confidence=0.95) for r, t in enumerate(["Этап", "подготовка", "монтаж"])]
    roman = CellData(row=3, col=1, text="I", confidence=0.9)
    assert fix_bar_ones(nums + bars + words + [roman]) == 3
    assert [c.text for c in bars] == ["1", "1", "11"]
    assert all(c.confidence == BAR_ONE_CONFIDENCE for c in bars)
    assert roman.text == "I"
