FROM python:3.11-slim

# On réduit un peu la taille de l'image
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Si tu as un requirements spécifique qwenocr
COPY qwenocr/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copier uniquement le code nécessaire pour qwenocr
COPY qwenocr /app/qwenocr

# (Optionnel) si tu utilises des modules communs à la racine
# COPY common /app/common

# Script d'entrée du job
CMD ["python", "-u", "qwenocr/qwenocr_runner.py"]
