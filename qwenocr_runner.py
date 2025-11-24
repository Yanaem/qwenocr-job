#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Runner pour le job Cloud Run `qwenocr`.

Ce fichier sert juste de point d'entrée :
- lit éventuellement des variables d'environnement / arguments
- appelle la fonction principale de ton script OCR Qwen (`ocr_qwenVL.py`)
"""

import sys
import traceback

try:
    # On importe ton script principal Qwen
    import ocr_qwenVL
except ImportError:
    print("❌ Impossible d'importer ocr_qwenVL.py. Vérifie qu'il est bien dans le même dossier.")
    sys.exit(1)


def main():
    """
    Point d'entrée logique du job.
    Ici tu peux ajouter de la logique spécifique Cloud Run :
      - lecture d'arguments (sys.argv)
      - lecture de variables d'env (GCS_INPUT_URI, GCS_OUTPUT_URI, etc.)
      - logging
    Pour l'instant, on délègue tout à ocr_qwenVL.main().
    """
    if not hasattr(ocr_qwenVL, "main"):
        raise RuntimeError(
            "Le module ocr_qwenVL ne contient pas de fonction main(). "
            "Ajoute une fonction main() ou adapte qwenocr_runner.py."
        )

    # Appel direct de ta fonction principale
    ocr_qwenVL.main()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("❌ Erreur dans qwenocr_runner.py :", e, file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
