"""Модели результата распознавания PDF (pydantic v2).

Основной сценарий использования в другом приложении::

    from pdfdoc import Document

    doc = Document.model_validate_json(json_str)     # или Document.load(path)
    for block in doc.page_blocks(1):
        ...

Порядок и имена полей ``Document`` и ``Block`` совпадают с согласованным
форматом JSON. Дополнительные поля (``pages``, ``table_spans``,
``table_cell_confidence``, ``schema_version``) необязательны: JSON без них
тоже проходит валидацию.
"""

from __future__ import annotations

import os
import tempfile
from enum import Enum
from pathlib import Path
from typing import Iterable, Iterator, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "1.0"

BBox = tuple[float, float, float, float]


class BlockType(str, Enum):
    """Тип блока."""

    HEADER = "header"  # заголовок документа / раздела
    TEXT = "text"  # обычный абзац
    LIST_ITEM = "list_item"  # пункт списка
    TABLE = "table"  # таблица (см. table_cells / table_md / html)
    SIGNATURE = "signature"  # блок подписи («Главный бухгалтер ____ / Смирнова А.В. /»)
    PAGE_HEADER = "page_header"  # верхний колонтитул (повторяется на страницах)
    PAGE_FOOTER = "page_footer"  # нижний колонтитул (повторяется на страницах)
    PAGE_NUMBER = "page_number"  # номер страницы
    IMAGE = "image"  # растровое изображение в PDF с текстовым слоем (логотип, печать)


class BlockSource(str, Enum):
    """Откуда получено содержимое блока."""

    TEXT_LAYER = "text_layer"  # текстовый слой PDF
    OCR = "ocr"  # распознавание изображения страницы (Tesseract)
    TABLE_EXTRACTOR = "table_extractor"  # таблица из текстового слоя + векторных линий
    TABLE_OCR = "table_ocr"  # таблица со скана: сетка по линиям + OCR ячеек


class PageSource(str, Enum):
    """Как обработана страница целиком."""

    TEXT_LAYER = "text_layer"
    OCR = "ocr"
    EMPTY = "empty"  # нет ни текста, ни изображений
    FAILED = "failed"  # ошибка обработки (причина — в warnings документа)


class _Model(BaseModel):
    # extra="ignore": JSON от более новой версии пайплайна с новыми полями
    # читается старой версией без ошибок.
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class CellSpan(_Model):
    """Объединённая ячейка таблицы: левый верхний угол и размеры (с нуля)."""

    row: int = Field(ge=0)
    col: int = Field(ge=0)
    rowspan: int = Field(default=1, ge=1)
    colspan: int = Field(default=1, ge=1)


class Block(_Model):
    """Логический блок страницы: абзац, заголовок, таблица, подпись и т.п."""

    page: int = Field(ge=1, description="Номер страницы, с 1")
    block_id: str = Field(description="Идентификатор вида p{page}_b{NN}")
    type: BlockType
    bbox: BBox = Field(
        description="[x0, y0, x1, y1] в пунктах PDF (1/72 дюйма), начало координат — "
        "левый верхний угол страницы в том виде, как она отображается (с учётом /Rotate)"
    )
    text: Optional[str] = None
    table_md: Optional[str] = None
    table_cells: Optional[list[list[str]]] = None
    html: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    source: BlockSource
    language: Optional[str] = Field(
        default=None, description="'ru', 'en', 'mixed' или null (нет букв)"
    )
    # --- расширения формата (необязательные) ---
    table_spans: Optional[list[CellSpan]] = Field(
        default=None, description="Только объединённые ячейки (rowspan/colspan > 1)"
    )
    table_cell_confidence: Optional[list[list[Optional[float]]]] = Field(
        default=None,
        description="Уверенность OCR по каждой ячейке (та же форма, что table_cells); "
        "null для пустых и перекрытых объединением ячеек",
    )

    @field_validator("bbox")
    @classmethod
    def _check_bbox(cls, v: BBox) -> BBox:
        x0, y0, x1, y1 = v
        if x1 < x0 or y1 < y0:
            raise ValueError(f"bbox должен быть [x0, y0, x1, y1] с x0<=x1, y0<=y1: {v}")
        return v

    @model_validator(mode="after")
    def _check_table(self) -> "Block":
        if self.table_cells is not None:
            widths = {len(r) for r in self.table_cells}
            if len(widths) > 1:
                raise ValueError(
                    f"{self.block_id}: строки table_cells разной длины {sorted(widths)}"
                )
        return self

    # --- удобства ---
    @property
    def is_table(self) -> bool:
        return self.type == BlockType.TABLE

    @property
    def n_rows(self) -> int:
        return len(self.table_cells or [])

    @property
    def n_cols(self) -> int:
        return len(self.table_cells[0]) if self.table_cells else 0

    def as_text(self) -> str:
        """Текст блока; для таблиц — markdown-представление."""
        if self.text is not None:
            return self.text
        if self.table_md is not None:
            return self.table_md
        return ""


class PageInfo(_Model):
    """Навигация по странице: размеры, способ обработки и блоки страницы."""

    page: int = Field(ge=1)
    width: float = Field(description="Ширина страницы в пунктах (как отображается)")
    height: float = Field(description="Высота страницы в пунктах (как отображается)")
    rotation: int = Field(default=0, description="/Rotate страницы в PDF")
    source: PageSource
    confidence: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Средняя уверенность по блокам страницы"
    )
    skew_angle: Optional[float] = Field(
        default=None, description="Исправленный перекос скана, градусы (только для OCR)"
    )
    orientation: Optional[int] = Field(
        default=None, description="Поворот скана на 90/180/270, найденный OSD (только для OCR)"
    )
    block_ids: list[str] = Field(default_factory=list)
    table_ids: list[str] = Field(default_factory=list)


class Document(_Model):
    """Распознанный PDF-документ."""

    file_name: str
    file_hash: str = Field(description="SHA-256 содержимого PDF, hex")
    total_pages: int = Field(ge=0)
    pipeline_version: str
    processing_time_ms: int = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)
    blocks: list[Block] = Field(default_factory=list)
    # --- расширения формата (необязательные) ---
    pages: list[PageInfo] = Field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    @model_validator(mode="after")
    def _check_consistency(self) -> "Document":
        seen: set[str] = set()
        for b in self.blocks:
            if b.block_id in seen:
                raise ValueError(f"повторяющийся block_id: {b.block_id}")
            seen.add(b.block_id)
            if self.total_pages and b.page > self.total_pages:
                raise ValueError(
                    f"{b.block_id}: page={b.page} больше total_pages={self.total_pages}"
                )
        return self

    # ------------------------------------------------------------ навигация
    def page_blocks(self, page: int) -> list[Block]:
        """Блоки страницы в порядке чтения."""
        return [b for b in self.blocks if b.page == page]

    def iter_pages(self) -> Iterator[tuple[int, list[Block]]]:
        """(номер страницы, блоки) для всех страниц, включая пустые."""
        by_page: dict[int, list[Block]] = {p: [] for p in range(1, self.total_pages + 1)}
        for b in self.blocks:
            by_page.setdefault(b.page, []).append(b)
        yield from sorted(by_page.items())

    def page_info(self, page: int) -> Optional[PageInfo]:
        return next((p for p in self.pages if p.page == page), None)

    def get_block(self, block_id: str) -> Optional[Block]:
        return next((b for b in self.blocks if b.block_id == block_id), None)

    def blocks_of_type(self, *types: Union[BlockType, str]) -> list[Block]:
        wanted = {BlockType(t) for t in types}
        return [b for b in self.blocks if b.type in wanted]

    def tables(self, page: Optional[int] = None) -> list[Block]:
        return [
            b
            for b in self.blocks
            if b.type == BlockType.TABLE and (page is None or b.page == page)
        ]

    def low_confidence(self, threshold: float = 0.7) -> list[Block]:
        return [b for b in self.blocks if b.confidence < threshold]

    def full_text(self, page: Optional[int] = None, include_tables: bool = True) -> str:
        """Текст документа (или страницы) в порядке чтения; таблицы — в markdown."""
        parts: list[str] = []
        for b in self.blocks:
            if page is not None and b.page != page:
                continue
            if b.type == BlockType.TABLE and not include_tables:
                continue
            t = b.as_text()
            if t:
                parts.append(t)
        return "\n\n".join(parts)

    # ------------------------------------------------------------ JSON
    def to_json(self, indent: Optional[int] = 2) -> str:
        return self.model_dump_json(indent=indent)

    def save(self, path: Union[str, os.PathLike], indent: Optional[int] = 2) -> Path:
        """Атомарно записать JSON (через временный файл в той же папке)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_json(indent=indent)
        fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(data)
                fh.write("\n")
            # mkstemp создаёт файл с правами 0600 — без этого JSON прочитает только владелец
            try:
                mode = path.stat().st_mode & 0o777 if path.exists() else 0o644
                os.chmod(tmp, mode)
            except OSError:
                pass
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return path

    @classmethod
    def load(cls, path: Union[str, os.PathLike]) -> "Document":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


def iter_blocks(docs: Iterable[Document], *types: Union[BlockType, str]) -> Iterator[tuple[Document, Block]]:
    """Пройти по блокам нескольких документов (например, всех таблиц пачки)."""
    wanted = {BlockType(t) for t in types} if types else None
    for d in docs:
        for b in d.blocks:
            if wanted is None or b.type in wanted:
                yield d, b
