"""Точка входа: PDF → Document (+ JSON рядом с документом)."""

from __future__ import annotations

import hashlib
import logging
import os
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Optional, Union

import pymupdf
from pydantic import BaseModel, Field

from pdfdoc import __version__
from pdfdoc._raw import RawBlock, RawPage
from pdfdoc.classify import classify_page, mark_running_headers, reading_order
from pdfdoc.config import PipelineConfig
from pdfdoc.errors import PdfDocError, PdfOpenError, TesseractNotAvailableError
from pdfdoc.models import Block, BlockType, Document, PageInfo, PageSource
from pdfdoc.text_layer import assess_page, extract_text_layer_page
from pdfdoc.text_utils import detect_language

log = logging.getLogger("pdfdoc")

if hasattr(pymupdf, "no_recommend_layout"):
    pymupdf.no_recommend_layout()  # не печатать рекламу pymupdf_layout в stdout приложения

PathLike = Union[str, os.PathLike]


# ---------------------------------------------------------------- утилиты


def file_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_path_for(pdf_path: PathLike, config: Optional[PipelineConfig] = None) -> Path:
    """Куда сохраняется JSON для данного PDF: рядом с ним или в config.output_dir."""
    cfg = config or PipelineConfig()
    p = Path(pdf_path)
    folder = Path(cfg.output_dir) if cfg.output_dir else p.parent
    return folder / f"{p.stem}{cfg.json_suffix}"


def _load_cached(json_path: Path, file_hash: str) -> Optional[Document]:
    if not json_path.exists():
        return None
    try:
        doc = Document.load(json_path)
    except Exception:  # noqa: BLE001 — битый/чужой JSON: просто перерабатываем
        return None
    if doc.file_hash == file_hash and doc.pipeline_version == __version__:
        return doc
    return None


# ---------------------------------------------------------------- сборка документа


def _to_block(rb: RawBlock, page: int, block_id: str, cfg: PipelineConfig) -> Block:
    if rb.table is not None:
        t = rb.table
        lang_src = t.all_text()
        has_conf = any(c.confidence is not None for c in t.cells)
        return Block(
            page=page,
            block_id=block_id,
            type=BlockType.TABLE,
            bbox=rb.bbox,
            text=None,
            table_md=t.to_markdown(cfg.merged_cell_fill),
            table_cells=t.matrix(cfg.merged_cell_fill),
            html=t.to_html() if cfg.table_html else None,
            confidence=round(min(1.0, max(0.0, rb.confidence)), 3),
            source=rb.source,
            language=detect_language(lang_src),
            table_spans=t.spans() or None,
            table_cell_confidence=t.confidence_matrix() if has_conf else None,
        )
    return Block(
        page=page,
        block_id=block_id,
        type=rb.type or BlockType.TEXT,
        bbox=rb.bbox,
        text=rb.text,
        confidence=round(min(1.0, max(0.0, rb.confidence)), 3),
        source=rb.source,
        language=detect_language(rb.text),
    )


def _block_warnings(b: Block, rb: RawBlock, cfg: PipelineConfig) -> list[str]:
    out: list[str] = []
    if b.confidence < cfg.low_confidence_threshold and b.type != BlockType.IMAGE:
        kind = "OCR" if b.source.value in ("ocr", "table_ocr") else "extraction"
        out.append(
            f"page {b.page}: {kind} confidence below {cfg.low_confidence_threshold:g} "
            f"on {b.type.value} block {b.block_id} ({b.confidence:.2f})"
        )
    if rb.table is not None:
        if cfg.warn_empty_cells:
            for col, n in sorted(rb.table.empty_cells_by_column().items()):
                out.append(
                    f"page {b.page}: table {b.block_id} column {col + 1} has {n} empty "
                    f"cell{'s' if n != 1 else ''}"
                )
        unread = rb.table.unreadable_cells()
        if unread:
            out.append(f"page {b.page}: table {b.block_id} has {unread} unreadable cells")
    return out


def _process_page(
    page: pymupdf.Page,
    cfg: PipelineConfig,
    engine_box: dict,
    pool: ThreadPoolExecutor,
) -> RawPage:
    pno = page.number + 1
    rp = RawPage(
        page=pno,
        width=round(page.rect.width, 2),
        height=round(page.rect.height, 2),
        rotation=page.rotation,
        source=PageSource.EMPTY,
    )
    a = assess_page(page, cfg)
    if a.mode == "empty":
        return rp
    if a.mode == "text_layer":
        rp.source = PageSource.TEXT_LAYER
        tl = extract_text_layer_page(page, cfg, a)
        rp.blocks, rp.body_size = tl.blocks, tl.body_size
        rp.layout_width, rp.layout_height = tl.layout_width, tl.layout_height
        rp.warnings.extend(tl.warnings)
        if a.hidden_ratio > 0.5:
            rp.warnings.append(
                f"page {pno}: invisible text layer (made by another OCR), used as is"
            )
        return rp

    # OCR
    if a.reason == "damaged_text_layer":
        rp.warnings.append(
            f"page {pno}: text layer looks damaged ({a.garbage:.0%} bad characters), OCR used"
        )
    elif a.reason == "misread_hidden_layer":
        rp.warnings.append(
            f"page {pno}: invisible text layer looks like Cyrillic misread as Latin, OCR used"
        )
    if "engine" not in engine_box:
        from pdfdoc.ocr.engine import Tesseract

        engine_box["engine"] = Tesseract(cfg)
    from pdfdoc.ocr.page import ocr_page

    res = ocr_page(page, cfg, engine_box["engine"], pool)
    rp.source = PageSource.OCR
    rp.blocks = res.blocks
    rp.body_size = res.body_size
    rp.skew_angle = res.skew_angle
    rp.orientation = res.orientation
    rp.layout_width = res.layout_width
    rp.layout_height = res.layout_height
    rp.warnings.extend(res.warnings)
    return rp


def _build_document(
    raw_pages: list[RawPage],
    file_name: str,
    file_hash: str,
    total_pages: int,
    t0: float,
    cfg: PipelineConfig,
    extra_warnings: list[str],
) -> Document:
    for rp in raw_pages:
        classify_page(rp)
    mark_running_headers(raw_pages)

    blocks: list[Block] = []
    warnings: list[str] = list(extra_warnings)
    pages: list[PageInfo] = []
    for rp in raw_pages:
        warnings.extend(rp.warnings)
        ordered = reading_order(rp.blocks)
        ids: list[str] = []
        tids: list[str] = []
        confs: list[float] = []
        for i, rb in enumerate(ordered, start=1):
            bid = f"p{rp.page}_b{i:02d}"
            b = _to_block(rb, rp.page, bid, cfg)
            blocks.append(b)
            ids.append(bid)
            if b.type == BlockType.TABLE:
                tids.append(bid)
            if b.type != BlockType.IMAGE:
                confs.append(b.confidence)
            warnings.extend(_block_warnings(b, rb, cfg))
        if rp.source in (PageSource.OCR, PageSource.TEXT_LAYER) and not ordered:
            warnings.append(f"page {rp.page}: no content recognized")
        pages.append(
            PageInfo(
                page=rp.page,
                width=rp.width,
                height=rp.height,
                rotation=rp.rotation,
                source=rp.source,
                confidence=round(statistics.fmean(confs), 3) if confs else None,
                skew_angle=rp.skew_angle,
                orientation=rp.orientation,
                block_ids=ids,
                table_ids=tids,
            )
        )
    return Document(
        file_name=file_name,
        file_hash=file_hash,
        total_pages=total_pages,
        pipeline_version=__version__,
        processing_time_ms=int((time.perf_counter() - t0) * 1000),
        warnings=warnings,
        blocks=blocks,
        pages=pages,
    )


# ---------------------------------------------------------------- публичный API


def process_bytes(
    data: bytes,
    file_name: str,
    config: Optional[PipelineConfig] = None,
    *,
    pages: Optional[Iterable[int]] = None,
) -> Document:
    """Распознать PDF из памяти. JSON не сохраняет (см. process_pdf).

    pages — номера страниц с 1 (по умолчанию все).
    """
    cfg = config or PipelineConfig()
    t0 = time.perf_counter()
    file_hash = file_sha256(data)
    try:
        pdf = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:  # noqa: BLE001
        raise PdfOpenError(f"{file_name}: не удалось открыть PDF: {exc}") from exc
    try:
        if pdf.needs_pass and not pdf.authenticate(""):
            raise PdfOpenError(f"{file_name}: PDF защищён паролем")
        total = pdf.page_count
        wanted = sorted({p for p in pages if 1 <= p <= total}) if pages else list(range(1, total + 1))
        raw_pages: list[RawPage] = []
        extra: list[str] = []
        if cfg.max_pages and len(wanted) > cfg.max_pages:
            extra.append(
                f"document: {len(wanted)} pages, only first {cfg.max_pages} processed (max_pages)"
            )
            wanted = wanted[: cfg.max_pages]
        engine_box: dict = {}
        with ThreadPoolExecutor(max_workers=max(1, cfg.ocr_threads)) as pool:
            for pno in wanted:
                page = pdf[pno - 1]
                try:
                    raw_pages.append(_process_page(page, cfg, engine_box, pool))
                except TesseractNotAvailableError:
                    raise
                except Exception as exc:  # noqa: BLE001 — страница не ломает документ
                    log.exception("%s: page %s failed", file_name, pno)
                    extra.append(f"page {pno}: processing failed: {type(exc).__name__}: {exc}")
                    raw_pages.append(
                        RawPage(
                            page=pno,
                            width=round(page.rect.width, 2),
                            height=round(page.rect.height, 2),
                            rotation=page.rotation,
                            source=PageSource.FAILED,
                        )
                    )
        return _build_document(raw_pages, file_name, file_hash, total, t0, cfg, extra)
    finally:
        pdf.close()


def process_pdf(
    path: PathLike,
    config: Optional[PipelineConfig] = None,
    *,
    save: Optional[bool] = None,
) -> Document:
    """Распознать PDF-файл и (по умолчанию) сохранить JSON рядом с ним.

    Если JSON уже есть и в нём тот же file_hash и pipeline_version, файл не
    обрабатывается повторно (config.skip_existing).
    """
    return _process_pdf(path, config or PipelineConfig(), save)[0]


def _process_pdf(path: PathLike, cfg: PipelineConfig, save: Optional[bool]) -> tuple[Document, bool]:
    """(документ, взят_из_готового_JSON)."""
    path = Path(path)
    data = path.read_bytes()
    do_save = cfg.save_json if save is None else save
    out = json_path_for(path, cfg)
    if do_save and cfg.skip_existing:
        cached = _load_cached(out, file_sha256(data))
        if cached is not None:
            log.info("%s: already processed, %s", path.name, out)
            return cached, True
    doc = process_bytes(data, path.name, cfg)
    if do_save:
        doc.save(out, indent=cfg.json_indent)
        log.info("%s: saved %s (%d blocks, %d ms)", path.name, out, len(doc.blocks), doc.processing_time_ms)
    return doc, False


# ---------------------------------------------------------------- пакетная обработка


class FileResult(BaseModel):
    file: str
    status: str  # "processed" | "skipped" | "failed"
    json_path: Optional[str] = None
    error: Optional[str] = None
    pages: int = 0
    blocks: int = 0
    warnings: int = 0
    processing_time_ms: int = 0


class BatchReport(BaseModel):
    total: int = 0
    processed: int = 0
    skipped: int = 0
    failed: int = 0
    elapsed_ms: int = 0
    files: list[FileResult] = Field(default_factory=list)


def _worker(path_str: str, cfg_dict: dict) -> FileResult:
    cfg = PipelineConfig(**cfg_dict)
    path = Path(path_str)
    try:
        out = json_path_for(path, cfg)
        doc, skipped = _process_pdf(path, cfg, save=True)
        return FileResult(
            file=path_str,
            status="skipped" if skipped else "processed",
            json_path=str(out),
            pages=doc.total_pages,
            blocks=len(doc.blocks),
            warnings=len(doc.warnings),
            processing_time_ms=doc.processing_time_ms,
        )
    except PdfDocError as exc:  # ожидаемая ошибка (битый/зашифрованный PDF, нет Tesseract)
        log.error("%s: %s", path_str, exc)
        return FileResult(file=path_str, status="failed", error=f"{type(exc).__name__}: {exc}")
    except Exception as exc:  # noqa: BLE001
        log.exception("%s failed", path_str)
        return FileResult(file=path_str, status="failed", error=f"{type(exc).__name__}: {exc}")


def find_pdfs(folder: PathLike, recursive: bool = True) -> list[Path]:
    root = Path(folder)
    it = root.rglob("*") if recursive else root.glob("*")
    return sorted(p for p in it if p.is_file() and p.suffix.lower() == ".pdf")


def process_folder(
    folder: Union[PathLike, Iterable[PathLike]],
    config: Optional[PipelineConfig] = None,
    *,
    recursive: bool = True,
    workers: Optional[int] = None,
) -> BatchReport:
    """Обработать все PDF в папке (или список файлов), JSON — рядом с каждым PDF.

    workers — число процессов (по умолчанию: число ядер, но не больше файлов).
    Внутри процесса Tesseract вызывается в 1 поток, чтобы не было переподписки ядер.
    Ошибка в одном файле не останавливает пачку: она попадёт в отчёт.

    При workers > 1 используется multiprocessing: код, вызывающий функцию,
    должен быть защищён ``if __name__ == "__main__":``.
    """
    cfg = config or PipelineConfig()
    if isinstance(folder, (str, os.PathLike)):
        files = find_pdfs(folder, recursive) if Path(folder).is_dir() else [Path(folder)]
    else:
        files = [Path(p) for p in folder]
    t0 = time.perf_counter()
    report = BatchReport(total=len(files))
    if not files:
        return report
    n = workers or os.cpu_count() or 1
    n = max(1, min(n, len(files)))
    cfg_dict = cfg.model_dump()
    cfg_dict["save_json"] = True
    if n > 1:
        cfg_dict["ocr_threads"] = max(1, (os.cpu_count() or 1) // n)

    results: list[FileResult] = []
    if n == 1:
        for f in files:
            results.append(_worker(str(f), cfg_dict))
    else:
        with ProcessPoolExecutor(max_workers=n) as ex:
            futs = {ex.submit(_worker, str(f), cfg_dict): f for f in files}
            for fut in as_completed(futs):
                results.append(fut.result())
    results.sort(key=lambda r: r.file)
    report.files = results
    report.processed = sum(r.status == "processed" for r in results)
    report.skipped = sum(r.status == "skipped" for r in results)
    report.failed = sum(r.status == "failed" for r in results)
    report.elapsed_ms = int((time.perf_counter() - t0) * 1000)
    return report
