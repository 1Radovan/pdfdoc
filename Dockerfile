# Пример образа с pdfdoc. Для встраивания в своё приложение возьмите блок apt-get
# (Tesseract + языки) и pip install, остальное — по вкусу.
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_THREAD_LIMIT=1

# Tesseract 5.x + русский, английский и OSD (определение поворота страницы)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-rus \
        tesseract-ocr-eng \
        tesseract-ocr-osd \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY pdfdoc ./pdfdoc

# Пример: распознать всё в смонтированной папке /docs, JSON — рядом с PDF
#   docker run --rm -v /path/to/docs:/docs pdfdoc-image /docs --workers 4
ENTRYPOINT ["python", "-m", "pdfdoc"]
