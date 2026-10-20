"""Промежуточные структуры между извлечением страницы и итоговыми моделями."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from pdfdoc.models import BBox, BlockSource, BlockType, PageSource
from pdfdoc.tables import TableData


@dataclass
class RawBlock:
    bbox: BBox  # пункты, координаты отображаемой страницы
    source: BlockSource
    confidence: float
    type: Optional[BlockType] = None  # None — определится классификатором
    text: Optional[str] = None
    table: Optional[TableData] = None
    font_size: Optional[float] = None  # пт: кегль (текстовый слой) / высота строки (OCR)
    bold_ratio: float = 0.0
    n_lines: int = 1
    near_hline: bool = False  # OCR: рядом горизонтальная черта (место подписи)
    # bbox в «выпрямленных» координатах (после поворота скана на 90° и выравнивания);
    # по нему считаются порядок чтения и поля страницы. None — совпадает с bbox.
    layout_bbox: Optional[BBox] = None

    @property
    def lb(self) -> BBox:
        return self.layout_bbox or self.bbox


@dataclass
class RawPage:
    page: int
    width: float
    height: float
    rotation: int
    source: PageSource
    blocks: list[RawBlock] = field(default_factory=list)
    body_size: Optional[float] = None
    layout_width: Optional[float] = None  # размеры страницы в выпрямленных координатах
    layout_height: Optional[float] = None
    skew_angle: Optional[float] = None
    orientation: Optional[int] = None
    warnings: list[str] = field(default_factory=list)
