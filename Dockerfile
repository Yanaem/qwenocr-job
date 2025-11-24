FROM python:3.11-slim

# Options Python pour un runtime plus propre
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Installation des dépendances système nécessaires (notamment poppler-utils pour pdf2image)
RUN apt-get update && apt-get install -y --no-install-recommends \
        poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Dossier de travail dans le conteneur
WORKDIR /app

# Copier les dépendances Python spécifiques à qwenocr
COPY qwenocr/requirements.txt /app/requirements.txt

# Installer les dépendances Python
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copier le code qwenocr (runner + ton script ocr_qwenVL.py + éventuels modules annexes)
COPY qwenocr/ /app/

# Commande par défaut : lancer le runner qwenocr
CMD ["python", "-u", "qwenocr_runner.py"]
