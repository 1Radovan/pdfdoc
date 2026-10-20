"""Исключения пакета."""


class PdfDocError(Exception):
    """Базовое исключение pdfdoc."""


class PdfOpenError(PdfDocError):
    """PDF не открывается: повреждён, зашифрован или это не PDF."""


class TesseractNotAvailableError(PdfDocError):
    """Нужен OCR, но Tesseract или языковые данные не установлены.

    Не перехватывается внутри пайплайна: без OCR результат по сканам был бы
    пустым, а причина — скрытой.
    """
