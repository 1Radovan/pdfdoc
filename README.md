# pdfdoc

```bash
sudo apt-get install -y tesseract-ocr tesseract-ocr-rus tesseract-ocr-eng tesseract-ocr-osd
pip install -r requirements.txt
pip install -e /путь/к/pdfdoc
```

```python
from pdfdoc import process_pdf, process_bytes, process_folder, PipelineConfig, Document, PdfDocError
```

```python
from pdfdoc import process_pdf, PipelineConfig

doc = process_pdf("docs/file.pdf")                                   
doc = process_pdf("docs/file.pdf", save=False)                       
doc = process_pdf("docs/file.pdf", PipelineConfig(output_dir="out")) 
```

```python
from pdfdoc import Document

doc = Document.load("docs/file.json")           
json_str = doc.to_json()                         
doc = Document.model_validate_json(json_str)     
```

```python
from pdfdoc import process_bytes

doc = process_bytes(uploaded_bytes, "upload.pdf")
```

```python
from pdfdoc import process_folder

if __name__ == "__main__":
    report = process_folder("/data/docs", workers=8) 
    print(report.processed, report.skipped, report.failed)
```
