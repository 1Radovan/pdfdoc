from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from pdfdoc import Block, BlockType, Document

SPEC = Path(__file__).resolve().parents[1] / "examples" / "example_from_spec.json"


def test_spec_example_validates():
    doc = Document.model_validate_json(SPEC.read_text(encoding="utf-8"))
    assert doc.file_name == "vedomost_2025_01.pdf"
    assert doc.total_pages == 2
    assert [b.type for b in doc.blocks] == [
        BlockType.HEADER, BlockType.TEXT, BlockType.TABLE, BlockType.TEXT, BlockType.SIGNATURE
    ]
    t = doc.tables()[0]
    assert (t.n_rows, t.n_cols) == (4, 5)
    assert doc.pages == []  # необязательное расширение


def test_roundtrip_and_cyrillic(tmp_path):
    doc = Document.model_validate_json(SPEC.read_text(encoding="utf-8"))
    s = doc.to_json()
    assert "ВЕДОМОСТЬ" in s  # кириллица без \u-экранирования
    assert Document.model_validate_json(s) == doc
    p = doc.save(tmp_path / "x.json")
    assert Document.load(p) == doc
    import os
    import stat

    if os.name == "posix":
        assert stat.S_IMODE(p.stat().st_mode) == 0o644  # не 0600 от mkstemp


def test_navigation_helpers():
    doc = Document.model_validate_json(SPEC.read_text(encoding="utf-8"))
    assert [b.block_id for b in doc.page_blocks(2)] == ["p2_b01", "p2_b02"]
    assert doc.get_block("p1_b03").is_table
    assert [b.block_id for b in doc.low_confidence(0.7)] == ["p2_b02"]
    assert dict((p, len(bs)) for p, bs in doc.iter_pages()) == {1: 3, 2: 2}
    assert "| № | ФИО |" in doc.full_text(page=1)


def test_unknown_fields_ignored():
    doc = Document.model_validate_json(SPEC.read_text(encoding="utf-8"))
    raw = doc.model_dump(mode="json")
    raw["future_field"] = 1
    raw["blocks"][0]["future_block_field"] = "x"
    assert Document.model_validate(raw).blocks[0].text == doc.blocks[0].text


def _block(**kw):
    base = dict(page=1, block_id="p1_b01", type="text", bbox=[0, 0, 10, 10], confidence=0.9,
                source="ocr", text="a")
    base.update(kw)
    return base


def test_rejects_bad_bbox():
    with pytest.raises(ValidationError):
        Block.model_validate(_block(bbox=[10, 0, 0, 10]))


def test_rejects_bad_confidence_and_type():
    with pytest.raises(ValidationError):
        Block.model_validate(_block(confidence=1.5))
    with pytest.raises(ValidationError):
        Block.model_validate(_block(type="paragraph"))


def test_rejects_ragged_table():
    with pytest.raises(ValidationError):
        Block.model_validate(_block(type="table", text=None, table_cells=[["a", "b"], ["c"]]))


def test_rejects_duplicate_ids_and_bad_page():
    b = _block()
    with pytest.raises(ValidationError):
        Document.model_validate(dict(file_name="f", file_hash="h", total_pages=1, pipeline_version="1",
                                     processing_time_ms=0, blocks=[b, b]))
    with pytest.raises(ValidationError):
        Document.model_validate(dict(file_name="f", file_hash="h", total_pages=1, pipeline_version="1",
                                     processing_time_ms=0, blocks=[_block(page=2)]))
