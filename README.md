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

doc = process_pdf("docs/file.pdf")                                   
doc = process_pdf("docs/file.pdf", save=False)                       
doc = process_pdf("docs/file.pdf", PipelineConfig(output_dir="out")) 
```

Повторный вызов на том же файле берёт готовый JSON (если совпадают хеш файла и версия пайплайна).
Пересчитать заново: `PipelineConfig(skip_existing=False)`.


```python
for t in doc.tables():                    
    print(t.page, t.table_cells)          
    print(t.table_md, t.html)             
    print(t.table_cell_confidence)        

text = doc.full_text()                   
blocks = doc.page_blocks(1)              
heads = doc.blocks_of_type("header")      
for b in doc.low_confidence(0.7):         
    print(b.block_id, b.as_text())
print(doc.warnings)                     
```

## Готовый результат без повторной обработки

```python
from pdfdoc import Document

doc = Document.load("docs/file.json")           
json_str = doc.to_json()                         
doc = Document.model_validate_json(json_str)     
```

## Из памяти (загрузка по HTTP и т. п.)

```python
from pdfdoc import process_bytes

doc = process_bytes(uploaded_bytes, "upload.pdf")   # JSON не сохраняется
```

## Папка целиком

```python
from pdfdoc import process_folder

if __name__ == "__main__":
    report = process_folder("/data/docs", workers=8) 
    print(report.processed, report.skipped, report.failed)
```

## Ошибки

```python
from pdfdoc import process_pdf, PdfOpenError, TesseractNotAvailableError

try:
    doc = process_pdf(path)
except PdfOpenError:
    ...
except TesseractNotAvailableError:
    ...
```
