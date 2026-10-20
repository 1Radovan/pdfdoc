"""pdfdoc — распознавание PDF (текстовый слой + OCR сканов) в структурированный JSON.

Быстрый старт::

    from pdfdoc import process_pdf, Document

    doc = process_pdf("vedomost_2025_01.pdf")      # JSON сохранится рядом: vedomost_2025_01.json
    json_str = doc.to_json()
    same = Document.model_validate_json(json_str)  # загрузка в другом месте приложения
"""

__version__ = "0.1.3"

from pdfdoc.config import PipelineConfig  # noqa: E402
from pdfdoc.errors import PdfDocError, PdfOpenError, TesseractNotAvailableError  # noqa: E402
from pdfdoc.models import (  # noqa: E402
    SCHEMA_VERSION,
    Block,
    BlockSource,
    BlockType,
    CellSpan,
    Document,
    PageInfo,
    PageSource,
    iter_blocks,
)
from pdfdoc.pipeline import (  # noqa: E402
    BatchReport,
    FileResult,
    find_pdfs,
    json_path_for,
    process_bytes,
    process_folder,
    process_pdf,
)

__all__ = [
    "__version__",
    "SCHEMA_VERSION",
    "BatchReport",
    "Block",
    "BlockSource",
    "BlockType",
    "CellSpan",
    "Document",
    "FileResult",
    "PageInfo",
    "PageSource",
    "PdfDocError",
    "PdfOpenError",
    "PipelineConfig",
    "TesseractNotAvailableError",
    "find_pdfs",
    "iter_blocks",
    "json_path_for",
    "process_bytes",
    "process_folder",
    "process_pdf",
]
