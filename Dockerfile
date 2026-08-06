FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CAPTURE_REASONING_CONTENT=true \
    PARALLEL_INDEPENDENT_PASSES=true \
    PAGE_WORKERS=4

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ocr_qwenVL.py qwenocr_runner.py ./
RUN python -m py_compile ocr_qwenVL.py qwenocr_runner.py

CMD ["python", "-u", "qwenocr_runner.py"]
