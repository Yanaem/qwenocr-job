FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FINAL_MD_LAYOUT=page_interleaved \
    INCLUDE_OCR_ANNEX=true \
    INCLUDE_THINKING_ANNEX=false \
    CAPTURE_REASONING_CONTENT=true \
    PARALLEL_INDEPENDENT_PASSES=true \
    PAGE_WORKERS=4

# Poppler est requis par pdf2image. Aucun composant graphique n'est nécessaire.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY ocr_qwenVL.py qwenocr_runner.py VERSION.txt /app/

CMD ["python", "-u", "qwenocr_runner.py"]
