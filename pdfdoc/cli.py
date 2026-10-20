"""Командная строка: python -m pdfdoc <файлы или папки> [опции]."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from pdfdoc import __version__
from pdfdoc.config import PipelineConfig
from pdfdoc.models import Document
from pdfdoc.pipeline import find_pdfs, process_folder


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="pdfdoc",
        description="Распознать PDF в JSON (рядом с каждым PDF или в --out).",
    )
    ap.add_argument("paths", nargs="*", help="PDF-файлы и/или папки с PDF")
    ap.add_argument("--out", help="Папка для JSON (по умолчанию — рядом с PDF)")
    ap.add_argument("--workers", type=int, default=None, help="Параллельных процессов")
    ap.add_argument("--no-recursive", action="store_true", help="Не заходить во вложенные папки")
    ap.add_argument("--force", action="store_true", help="Переобработать, даже если JSON уже есть")
    ap.add_argument("--force-ocr", action="store_true", help="Игнорировать текстовый слой")
    ap.add_argument("--dpi", type=int, default=None, help="DPI растеризации для OCR (300)")
    ap.add_argument("--lang", default=None, help="Языки Tesseract (rus+eng)")
    ap.add_argument("--no-tables", action="store_true", help="Не искать таблицы")
    ap.add_argument("--report", help="Сохранить отчёт пачки в JSON")
    ap.add_argument("--schema", action="store_true", help="Вывести JSON Schema документа и выйти")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--version", action="version", version=f"pdfdoc {__version__}")
    args = ap.parse_args(argv)

    if args.schema:
        print(json.dumps(Document.model_json_schema(), ensure_ascii=False, indent=2))
        return 0
    if not args.paths:
        ap.error("укажите PDF-файлы или папки")

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    overrides: dict = {"skip_existing": not args.force}
    if args.out:
        overrides["output_dir"] = args.out
    if args.force_ocr:
        overrides["force_ocr"] = True
    if args.dpi:
        overrides["dpi"] = args.dpi
    if args.lang:
        overrides["ocr_lang"] = args.lang
    if args.no_tables:
        overrides["detect_tables"] = False
    cfg = PipelineConfig.from_env(**overrides)

    files: list[Path] = []
    for p in map(Path, args.paths):
        if p.is_dir():
            files.extend(find_pdfs(p, recursive=not args.no_recursive))
        elif p.is_file():
            files.append(p)
        else:
            print(f"нет такого файла или папки: {p}", file=sys.stderr)
    if not files:
        print("PDF не найдены", file=sys.stderr)
        return 2

    report = process_folder(files, cfg, workers=args.workers)
    for r in report.files:
        if r.status == "failed":
            print(f"FAIL  {r.file}: {r.error}")
        else:
            print(
                f"{r.status.upper():9} {r.file} → {r.json_path} "
                f"({r.pages} стр., {r.blocks} блоков, {r.warnings} предупр., {r.processing_time_ms} мс)"
            )
    print(
        f"Итого: {report.total}, обработано {report.processed}, пропущено {report.skipped}, "
        f"ошибок {report.failed}, {report.elapsed_ms / 1000:.1f} с"
    )
    if args.report:
        Path(args.report).write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
