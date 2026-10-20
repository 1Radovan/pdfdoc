from __future__ import annotations

import json

import pytest
from conftest import needs_tesseract

from pdfdoc import (
    BlockSource,
    BlockType,
    Document,
    PageSource,
    PdfOpenError,
    PipelineConfig,
    json_path_for,
    process_folder,
    process_pdf,
)

PEOPLE_ROWS = [
    ["№", "ФИО", "ИНН", "Должность", "Сумма"],
    ["1", "Иванов И.И.", "770123456701", "Менеджер", "1000.00"],
    ["2", "Петров П.П.", "770123456702", "Бухгалтер", "2000.00"],
    ["3", "Сидоров С.С.", "770123456703", "Инженер", "1500.00"],
    ["4", "Козлов К.К.", "770123456704", "", "0.00"],
]
DEDUCTIONS = [
    ["Вид удержания", "Период", "", "Сумма"],
    ["НДФЛ", "январь", "2025", "585.00"],
    ["", "февраль", "2025", "0.00"],
    ["Алименты", "январь", "2025", "250.00"],
]


def _check_common(doc: Document):
    assert doc.total_pages == 2
    tables = doc.tables()
    assert len(tables) == 2
    assert tables[0].table_cells == PEOPLE_ROWS
    assert tables[1].table_cells == DEDUCTIONS
    spans = {(s.row, s.col, s.rowspan, s.colspan) for s in tables[1].table_spans}
    assert spans == {(0, 1, 1, 2), (1, 0, 2, 1)}
    assert any("column 4 has 1 empty cell" in w for w in doc.warnings)
    sig = doc.blocks_of_type("signature")
    assert len(sig) == 2 and "Смирнова А.В." in sig[0].text
    assert doc.blocks_of_type("page_number")
    # навигация
    assert [p.page for p in doc.pages] == [1, 2]
    for p in doc.pages:
        assert p.block_ids == [b.block_id for b in doc.page_blocks(p.page)]


def test_text_layer_pdf(fresh_copy):
    pdf = fresh_copy("text")
    doc = process_pdf(pdf)
    _check_common(doc)
    assert {p.source for p in doc.pages} == {PageSource.TEXT_LAYER}
    assert doc.blocks[0].type == BlockType.HEADER
    assert doc.blocks[0].text.startswith("ВЕДОМОСТЬ")
    assert doc.tables()[0].source == BlockSource.TABLE_EXTRACTOR
    # JSON сохранён рядом и читается обратно
    out = json_path_for(pdf)
    assert out == pdf.with_suffix(".json") and out.exists()
    loaded = Document.model_validate_json(out.read_text(encoding="utf-8"))
    assert loaded == doc
    assert json.loads(out.read_text(encoding="utf-8"))["blocks"][0]["source"] == "text_layer"


def test_text_layer_page_with_rotate(sample_pdfs):
    """Страница с /Rotate 90: find_tables и get_text дают разные системы координат."""
    import pymupdf

    from pdfdoc import process_bytes

    src = pymupdf.open(sample_pdfs["text"])
    src[0].set_rotation(90)
    doc = process_bytes(src.tobytes(), "rot.pdf")
    p1 = doc.page_blocks(1)
    assert p1[0].type == BlockType.HEADER and p1[0].text.startswith("ВЕДОМОСТЬ")
    tables = doc.tables(1)
    assert tables[0].table_cells == PEOPLE_ROWS and tables[1].table_cells == DEDUCTIONS
    w, h = doc.pages[0].width, doc.pages[0].height
    assert (w, h) == (842.0, 595.0)
    assert all(b.bbox[2] <= w and b.bbox[3] <= h for b in p1)


def test_skip_existing_and_force(fresh_copy):
    pdf = fresh_copy("text")
    d1 = process_pdf(pdf)
    d2 = process_pdf(pdf)  # из готового JSON
    assert d2.processing_time_ms == d1.processing_time_ms
    d3 = process_pdf(pdf, PipelineConfig(skip_existing=False))
    assert d3.file_hash == d1.file_hash


def test_output_dir_and_no_save(fresh_copy, tmp_path):
    pdf = fresh_copy("text")
    out_dir = tmp_path / "json_out"
    process_pdf(pdf, PipelineConfig(output_dir=str(out_dir)))
    assert (out_dir / "vedomost_text.json").exists()
    pdf2 = fresh_copy("text")
    pdf2.with_suffix(".json").unlink(missing_ok=True)
    process_pdf(pdf2, save=False)
    assert not pdf2.with_suffix(".json").exists()


def test_broken_pdf(tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a pdf")
    with pytest.raises(PdfOpenError):
        process_pdf(bad)
    rep = process_folder(tmp_path, workers=1)
    assert rep.failed == 1 and rep.files[0].status == "failed"


@needs_tesseract
def test_scan_pdf(fresh_copy):
    doc = process_pdf(fresh_copy("scan"))
    _check_common(doc)
    assert {p.source for p in doc.pages} == {PageSource.OCR}
    assert all(abs(p.skew_angle + 0.8) < 0.15 for p in doc.pages)
    assert doc.tables()[0].source == BlockSource.TABLE_OCR
    assert doc.tables()[0].confidence > 0.85
    assert doc.tables()[0].table_cell_confidence is not None
    assert doc.blocks[0].type == BlockType.HEADER


@needs_tesseract
def test_sheared_scan(fresh_copy):
    """Лист протянут со сдвигом: вертикали сетки наклонены, текст и горизонтали ровные."""
    doc = process_pdf(fresh_copy("sheared"))
    _check_common(doc)
    assert all(abs(p.skew_angle or 0.0) < 0.15 for p in doc.pages)


@pytest.mark.parametrize("key", ["text_rot90", "text_rot270"])
def test_text_layer_content_rotated(fresh_copy, key):
    """Содержимое повёрнуто внутри альбомного листа без /Rotate: таблицы читаются по строкам текста."""
    doc = process_pdf(fresh_copy(key))
    _check_common(doc)
    assert doc.blocks[0].type == BlockType.HEADER
    assert doc.blocks[0].text.startswith("ВЕДОМОСТЬ")


@pytest.mark.parametrize(
    "key",
    [
        "rotated_table",
        "rotated_table_270",
        pytest.param("rotated_table_scan", marks=needs_tesseract),
        pytest.param("rotated_table_270_scan", marks=needs_tesseract),
    ],
)
def test_rotated_table_on_upright_page(fresh_copy, key):
    """Альбомная таблица на книжном листе: текст вокруг таблицы не повёрнут."""
    doc = process_pdf(fresh_copy(key))
    assert [b.text[:8] for b in doc.blocks[:2]] == ["СВЕДЕНИЯ", "Таблица "]
    tables = doc.tables()
    assert len(tables) == 1 and tables[0].table_cells == PEOPLE_ROWS
    if key.endswith("_scan"):
        assert doc.pages[0].orientation == 0
        assert any("table rotated by" in w for w in doc.warnings)


@needs_tesseract
@pytest.mark.parametrize("key,turn", [("rot90", 90), ("rot180", 180), ("rot270", 270)])
def test_rotated_scan(fresh_copy, key, turn):
    doc = process_pdf(fresh_copy(key))
    assert doc.pages[0].orientation == turn
    assert doc.blocks[0].text.startswith("ВЕДОМОСТЬ")
    assert doc.tables()[0].table_cells == PEOPLE_ROWS


@needs_tesseract
@pytest.mark.parametrize("turned", [False, True])
@pytest.mark.parametrize("osd_turn", [90, 180, 270])
def test_low_confidence_osd_is_verified(sample_pdfs, osd_turn, turned):
    """Неуверенный ответ OSD принимается, только если после поворота текст читается лучше."""
    import cv2
    import numpy as np

    from pdfdoc.ocr.engine import Tesseract
    from pdfdoc.ocr.image import PixelTransform, fix_orientation
    from samples import page_images

    gray = np.asarray(page_images(sample_pdfs["text"])[0])
    upright = gray.shape[:2]
    if turned:  # лист повёрнут так, что исправляет его именно osd_turn
        back = {90: cv2.ROTATE_90_COUNTERCLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_CLOCKWISE}
        gray = cv2.rotate(gray, back[osd_turn])
    engine = Tesseract(PipelineConfig())
    engine.orientation = lambda img: (osd_turn, 1.0)
    tr = PixelTransform(300, 595, 842)
    out, rotation, doubtful = fix_orientation(gray, engine, tr)
    assert rotation == (osd_turn if turned else 0) and not doubtful
    assert out.shape[:2] == upright


@needs_tesseract
def test_mixed_pdf(fresh_copy):
    doc = process_pdf(fresh_copy("mixed"))
    assert [p.source for p in doc.pages] == [PageSource.TEXT_LAYER, PageSource.OCR]
    _check_common(doc)
