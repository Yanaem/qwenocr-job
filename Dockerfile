FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Dépendances système :
# - poppler-utils : pour pdf2image
# - python3-tk : pour que l'import tkinter de ocr_qwenVL.py ne casse pas
RUN apt-get update && apt-get install -y --no-install-recommends \
        poppler-utils \
        python3-tk \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# requirements.txt est à la RACINE du repo
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# On copie tout le code Python à la racine
COPY . /app/

# Entrée du job : le runner batch
CMD ["python", "-u", "qwenocr_runner.py"]
