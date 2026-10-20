"""Настройки пайплайна."""

from __future__ import annotations

import os
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


def _default_threads() -> int:
    return max(1, min(4, os.cpu_count() or 1))


class PipelineConfig(BaseModel):
    """Все параметры распознавания. Значения по умолчанию подходят для
    русскоязычных деловых документов (сканы 200–400 dpi и PDF с текстовым слоем).

    Любой параметр можно переопределить: ``PipelineConfig(dpi=400, force_ocr=True)``.
    Часть параметров читается из переменных окружения (см. ``from_env``).
    """

    model_config = ConfigDict(extra="forbid")

    # --- сохранение результата ---
    save_json: bool = Field(True, description="Сохранять JSON после обработки")
    output_dir: Optional[str] = Field(
        None, description="Папка для JSON; None — рядом с исходным PDF"
    )
    json_suffix: str = Field(".json", description="Суффикс имени JSON: report.pdf → report.json")
    json_indent: Optional[int] = Field(2, description="Отступ JSON; None — в одну строку")
    skip_existing: bool = Field(
        True,
        description="Не обрабатывать заново, если рядом уже есть JSON с тем же file_hash "
        "и pipeline_version",
    )

    max_pages: Optional[int] = Field(
        None, ge=1, description="Обрабатывать не больше N первых страниц (None — все)"
    )

    # --- выбор способа обработки страницы ---
    force_ocr: bool = Field(False, description="Игнорировать текстовый слой, всегда OCR")
    min_text_chars: int = Field(
        25, description="Меньше символов в текстовом слое страницы → страница считается сканом"
    )
    max_garbage_ratio: float = Field(
        0.2, description="Доля «битых» символов, при которой текстовый слой отбрасывается"
    )
    hidden_text_layer_confidence: float = Field(
        0.9,
        description="Уверенность для невидимого текстового слоя (его сделал чужой OCR, "
        "качество неизвестно)",
    )
    check_hidden_text_layer: bool = Field(
        True,
        description="Распознавать заново страницы, у которых невидимый слой чужого OCR похож "
        "на кириллицу, прочитанную латиницей («Cqer Ha onnary»), а не на русский/английский",
    )

    # --- таблицы ---
    detect_tables: bool = True
    text_table_strategy: Literal["lines", "lines_strict", "text"] = Field(
        "lines",
        description="Стратегия PyMuPDF find_tables для PDF с текстовым слоем: 'lines' — по "
        "линиям сетки; 'text' — для таблиц без линий (больше ложных срабатываний)",
    )
    merged_cell_fill: Literal["top_left", "repeat"] = Field(
        "top_left",
        description="Как заполнять объединённые ячейки в table_cells/table_md: текст только "
        "в левой верхней ('top_left') или повтор во всех перекрытых ('repeat')",
    )
    table_html: bool = Field(True, description="Заполнять html таблиц (с rowspan/colspan)")
    min_table_cells: int = Field(2, description="Таблица из меньшего числа ячеек — не таблица")
    warn_empty_cells: bool = Field(True, description="Предупреждать о пустых ячейках в колонках")

    # --- OCR ---
    ocr_lang: str = Field("rus+eng", description="Языки Tesseract")
    dpi: int = Field(300, ge=150, le=600, description="Разрешение растеризации для OCR")
    tesseract_cmd: Optional[str] = Field(None, description="Путь к tesseract; None — из PATH")
    tessdata_dir: Optional[str] = Field(None, description="Папка tessdata; None — по умолчанию")
    tesseract_timeout_s: float = Field(
        60.0, description="Таймаут одного вызова Tesseract, секунд (0 — без таймаута)"
    )
    ocr_threads: int = Field(
        default_factory=_default_threads,
        description="Параллельных вызовов Tesseract внутри одного документа",
    )
    deskew: bool = Field(True, description="Исправлять перекос скана")
    max_skew_deg: float = Field(5.0, description="Максимальный искомый перекос, градусы")
    detect_orientation: bool = Field(
        True, description="Определять поворот скана на 90/180/270 (нужен osd.traineddata)"
    )
    fix_homoglyphs: bool = Field(
        True,
        description="Заменять латинские двойники (A, B, C, E, H, K, M, O, P, T, X, a, c, e, o, p, x, y) "
        "внутри кириллических слов в результатах OCR",
    )
    numeric_recheck: bool = Field(
        True,
        description="Повторно распознавать числовые ячейки с белым списком цифр и брать "
        "более уверенный вариант",
    )

    # --- качество ---
    low_confidence_threshold: float = Field(
        0.7, description="Блоки ниже этой уверенности попадают в warnings"
    )

    @classmethod
    def from_env(cls, **overrides) -> "PipelineConfig":
        """Конфиг с учётом переменных окружения PDFDOC_* (явные overrides важнее).

        PDFDOC_DPI, PDFDOC_OCR_LANG, PDFDOC_OCR_THREADS, PDFDOC_FORCE_OCR,
        PDFDOC_OUTPUT_DIR, TESSERACT_CMD, TESSDATA_PREFIX.
        """
        env: dict = {}
        if v := os.environ.get("PDFDOC_DPI"):
            env["dpi"] = int(v)
        if v := os.environ.get("PDFDOC_OCR_LANG"):
            env["ocr_lang"] = v
        if v := os.environ.get("PDFDOC_OCR_THREADS"):
            env["ocr_threads"] = int(v)
        if v := os.environ.get("PDFDOC_FORCE_OCR"):
            env["force_ocr"] = v.strip().lower() in {"1", "true", "yes"}
        if v := os.environ.get("PDFDOC_OUTPUT_DIR"):
            env["output_dir"] = v
        if v := os.environ.get("TESSERACT_CMD"):
            env["tesseract_cmd"] = v
        if v := os.environ.get("TESSDATA_PREFIX"):
            env["tessdata_dir"] = v
        env.update(overrides)
        return cls(**env)
