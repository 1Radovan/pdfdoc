# pdfdoc

Библиотека: PDF (текстовый слой или скан) → объект `Document` с блоками, таблицами и уверенностью.
Используется как импорт функций в своём коде.

## Подключение

```bash
sudo apt-get install -y tesseract-ocr tesseract-ocr-rus tesseract-ocr-eng tesseract-ocr-osd
pip install -r requirements.txt
pip install -e /путь/к/pdfdoc
```

```python
from pdfdoc import process_pdf, process_bytes, process_folder, PipelineConfig, Document, PdfDocError
```

## Обработка файла → `doc`

```python
from pdfdoc import process_pdf, PipelineConfig

doc = process_pdf("docs/file.pdf")                                   # JSON сохранится рядом с PDF
doc = process_pdf("docs/file.pdf", save=False)                       # без записи JSON
doc = process_pdf("docs/file.pdf", PipelineConfig(output_dir="out")) # JSON в папку out/
```

Повторный вызов на том же файле берёт готовый JSON (если совпадают хеш файла и версия пайплайна).
Пересчитать заново: `PipelineConfig(skip_existing=False)`.


```python
for t in doc.tables():                    # все таблицы (или doc.tables(page=1))
    print(t.page, t.table_cells)          # матрица ячеек: list[list[str]]
    print(t.table_md, t.html)             # то же в markdown / HTML с объединениями
    print(t.table_cell_confidence)        # уверенность по ячейкам (для сканов)

text = doc.full_text()                    # весь текст документа
blocks = doc.page_blocks(1)               # блоки страницы 1
heads = doc.blocks_of_type("header")      # блоки нужного типа
for b in doc.low_confidence(0.7):         # что стоит проверить вручную
    print(b.block_id, b.as_text())
print(doc.warnings)                       # предупреждения пайплайна
```

## Готовый результат без повторной обработки

```python
from pdfdoc import Document

doc = Document.load("docs/file.json")            # из файла
json_str = doc.to_json()                         # в строку (БД, API)
doc = Document.model_validate_json(json_str)     # из строки
```

## Из памяти (загрузка по HTTP и т. п.)

```python
from pdfdoc import process_bytes

doc = process_bytes(uploaded_bytes, "upload.pdf")   # JSON не сохраняется
```

## Папка целиком

`process_folder` запускает несколько процессов, поэтому нужен `if __name__ == "__main__":`.

```python
from pdfdoc import process_folder

if __name__ == "__main__":
    report = process_folder("/data/docs", workers=8)   # JSON рядом с каждым PDF
    print(report.processed, report.skipped, report.failed)
```

## Ошибки

```python
from pdfdoc import process_pdf, PdfOpenError, TesseractNotAvailableError

try:
    doc = process_pdf(path)
except PdfOpenError:                 # битый / зашифрованный / не PDF
    ...
except TesseractNotAvailableError:   # для скана не найден Tesseract или язык rus
    ...
```
