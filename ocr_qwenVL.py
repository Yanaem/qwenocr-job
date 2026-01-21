#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ocr_qwenVL.py — module compatible qwenocr_runner.py (sans modifier le runner)

Fonctions/constantes attendues (d'après logs) :
- MODEL (str)
- INTER_REQUEST_DELAY (float)
- get_pdf_info(pdf_path) -> dict {page_count,...}
- load_progress(pdf_path) -> Dict[str, Dict]
- save_progress(pdf_path, completed_pages) -> None
- clear_progress(pdf_path) -> None
- process_page_with_cache(pdf_path, page_num, api_key, is_first_page=False) -> (markdown, stats)

Implémentation fidèle :
- 2 étapes : OCR brut (image->texte) puis Markdown (texte->md)
- Annexe OCR brut ajoutée dans le markdown (ajout code, 0 token).
"""

import base64
import os
import re
import time
import json
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple, Any

import requests
from pdf2image import convert_from_path

# PDF reader (pour page_count)
PdfReader = None
try:
    from pypdf import PdfReader as _PdfReader
    PdfReader = _PdfReader
except Exception:
    try:
        from PyPDF2 import PdfReader as _PdfReader
        PdfReader = _PdfReader
    except Exception:
        PdfReader = None


# =====================
# Configuration
# =====================

API_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

MODEL_OCR = "qwen-vl-max"
MODEL_MD = "qwen-vl-max"
MODEL = MODEL_OCR  # attendu par le runner

# Attendu par qwenocr_runner.py (pause entre pages/requêtes)
INTER_REQUEST_DELAY = 1.0  # secondes (0.0 si tu veux aucune pause)

RENDER_DPI = 300

MAX_TOKENS_OCR = 20000
MAX_TOKENS_MD = 20000
TEMPERATURE = 0.0

REQUEST_TIMEOUT = 600
MAX_RETRIES = 5
BACKOFF_BASE = 2
BACKOFF_MAX = 120


# =====================
# Prompts
# =====================

OCR_PROMPT = """Tu es un moteur OCR.

Tache : transcrire TOUT le texte visible sur l'image (une page de facture).

Regles :
- Sortie = TEXTE BRUT uniquement. Pas de Markdown. Pas de JSON. Pas d'explication.
- Ne corrige pas. Ne reformule pas. Ne normalise pas.
- Respecte autant que possible l'ordre de lecture visuel (haut->bas, gauche->droite).
- Preserve les sauts de ligne.
- Inclus aussi : totaux, echeances, mentions legales, pied de page, texte vertical, annotations manuscrites.
- Si une portion est vraiment illisible : ecris [ILLISIBLE] a l'endroit correspondant.
"""

SYSTEM_PROMPT_MD = """Vous êtes un assistant spécialisé dans le traitement de documents comptables.
Votre tâche est de convertir un texte brut issu d'un OCR d'une facture PDF (en français) en un document Markdown strictement fidèle au contenu original, sans aucune modification ni interprétation.

IMPORTANT:
- L'entrée fournie est déjà le TEXTE OCR BRUT.
- Vous NE DEVEZ PAS inclure la section "## Annexe - OCR brut" dans votre sortie. Elle sera ajoutée automatiquement après coup.
- Vous NE DEVEZ PAS recopier l'OCR brut complet en fin de réponse.

⚠️ Règles absolues :
- Ne jamais deviner ou supposer l'identité des parties.
- Ne jamais remplacer un champ manquant par une hypothèse.
- Respectez exactement les libellés, dates, montants, unités, abréviations, majuscules, tirets, espaces, symboles (€, %, etc.).
- Ne reformulez aucun mot : copiez tel quel, même si le texte contient des fautes d'OCR.
- Conservez les structures visuelles : tableaux, colonnes, lignes, séparateurs, barres verticales, etc.
- Ne fusionnez jamais des colonnes ni ne réorganisez les données.
- Utilisez [CHAMP MANQUANT] uniquement si une information attendue est illisible ou absente.
- Dans le tableau des lignes, ne générez aucune ligne vide : ne conservez que les lignes réellement présentes et arrêtez au dernier article.
- Interdiction absolue d'utiliser des infos d'une autre page pour remplir la page courante.

⚠️ RÈGLE ANTI-PADDING (priorité maximale)
- Interdiction de "remplir" un tableau Markdown pour reproduire la hauteur/espacement du document.
- Interdiction d’émettre une ligne de tableau où toutes les cellules sont vides.

⚠️ RÈGLE ANTI-COUPURE (priorité maximale)
Les consignes "arrêtez au dernier article" et "fin du tableau" s'appliquent uniquement au tableau des lignes.
Après le tableau, continuez la transcription du reste de la page (totaux, échéances, paiement, mentions, pied de page).

Structure de sortie (Markdown uniquement, sans commentaire) :

## Informations Émetteur (Fournisseur)
[Données exactes présentes dans la zone d'en-tête uniquement (avant le tableau des lignes de facturation)]

## Informations Client
[Données du destinataire présentes dans la zone d'en-tête uniquement ou [CHAMP MANQUANT]]

## Détails de la Facture
[Informations de facturation en en-tête : numéro, dates, références, objet, etc.]

## Tableau des Lignes de Facturation
[Reproduisez le tableau original avec ses colonnes, sans lignes vides.]

## Montants Récapitulatifs
[Reprenez tous les blocs de totaux et récapitulatifs présents sur la page. Gardez la forme d'origine.]

## Informations de Paiement
[Modalités, échéances, paiements, etc.]

## Mentions Légales et Notes Complémentaires
[Toute information supplémentaire / mentions / pied de page / annotations non classées ailleurs.]

➡️ Sortie finale : uniquement le document Markdown structuré, sans explication.
"""


# =====================
# Progress cache (attendu par le runner)
# =====================

def _progress_path(pdf_path: str) -> str:
    return str(Path(pdf_path).with_suffix(".progress.json"))

def load_progress(pdf_path: str) -> Dict[str, Dict]:
    p = _progress_path(pdf_path)
    if not os.path.exists(p):
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "pages" in data and isinstance(data["pages"], dict):
            return data["pages"]
        if isinstance(data, dict):
            return data
        return {}
    except Exception:
        return {}

def save_progress(pdf_path: str, completed_pages: Dict[str, Dict]) -> None:
    p = _progress_path(pdf_path)
    tmp = p + ".tmp"
    payload = {"pages": completed_pages}
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, p)

def clear_progress(pdf_path: str) -> None:
    p = _progress_path(pdf_path)
    try:
        if os.path.exists(p):
            os.remove(p)
    except Exception:
        pass


# =====================
# get_pdf_info (attendu par le runner)
# =====================

def get_pdf_info(pdf_path: str) -> Dict[str, Any]:
    pdf_path = str(pdf_path)
    file_size = os.path.getsize(pdf_path)

    if PdfReader is None:
        raise RuntimeError("pypdf/PyPDF2 indisponible : impossible de compter les pages. Ajoute pypdf au requirements.")

    with open(pdf_path, "rb") as f:
        reader = PdfReader(f)
        page_count = len(reader.pages)

    return {
        "page_count": int(page_count),
        "file_size_bytes": int(file_size),
        "file_size_mb": file_size / (1024 * 1024),
    }


# =====================
# Helpers API
# =====================

def _calculate_backoff_delay(attempt: int) -> int:
    return min(BACKOFF_BASE ** attempt, BACKOFF_MAX)

def _handle_api_error(error: Exception, attempt: int, context: str) -> Tuple[bool, int]:
    err = str(error).lower()
    non_retryable = ["invalid api key", "authentication failed", "permission denied"]
    if any(x in err for x in non_retryable):
        return False, 0
    if attempt >= MAX_RETRIES:
        return False, 0
    wait_time = _calculate_backoff_delay(attempt)
    if "429" in err or "rate limit" in err:
        wait_time = max(wait_time, 60)
    elif "overloaded" in err:
        wait_time = max(wait_time, 30)
    return True, wait_time

def _extract_text_from_response_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("text"):
                parts.append(part["text"])
        return "\n\n".join(parts)
    return str(content)

def _strip_triple_backticks(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t)
    return t.strip("\n")

def _remove_empty_md_table_rows(md: str) -> str:
    out: List[str] = []
    for line in md.splitlines():
        if re.match(r'^\|.*\|\s*$', line):
            cells = [c.strip() for c in line.strip().strip('|').split('|')]
            if cells and all(re.fullmatch(r':?-{3,}:?', c) for c in cells):
                out.append(line)
                continue
            if all(c == "" for c in cells):
                continue
        out.append(line)
    return "\n".join(out)

def _strip_existing_ocr_appendix(md: str) -> str:
    m = re.search(r"^##\s+Annexe\s*-\s*OCR\s+brut\s*$", md, flags=re.IGNORECASE | re.MULTILINE)
    if not m:
        return md.strip()
    return md[:m.start()].rstrip()


# =====================
# Rendering PDF -> image base64
# =====================

def render_single_page_to_base64(pdf_path: str, page_num: int, dpi: int = RENDER_DPI) -> Tuple[str, float]:
    images = convert_from_path(pdf_path, dpi=dpi, first_page=page_num, last_page=page_num)
    if not images:
        raise ValueError(f"Aucune image générée pour la page {page_num}")
    img = images[0]
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    b = buf.read()
    b64 = base64.b64encode(b).decode("utf-8")
    return b64, len(b) / 1024


# =====================
# Calls Qwen
# =====================

def _call_chat(api_key: str, model: str, messages: List[Dict], max_tokens: int, context: str) -> Tuple[str, Dict]:
    url = f"{API_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {"model": model, "max_tokens": max_tokens, "temperature": TEMPERATURE, "messages": messages}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.post(url, headers=headers, json=body, timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                js = r.json()
                usage = js.get("usage", {})
                input_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
                output_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
                choices = js.get("choices", [])
                content = choices[0].get("message", {}).get("content", "") if choices else ""
                text = _extract_text_from_response_content(content).strip()
                stats = {
                    "input_tokens": int(input_tokens),
                    "output_tokens": int(output_tokens),
                    "total_tokens": int(input_tokens + output_tokens),
                }
                return text, stats

            try:
                err_json = r.json()
            except Exception:
                err_json = r.text[:300]
            msg = f"HTTP {r.status_code}: {str(err_json)[:300]}"
            should_retry, wait_time = _handle_api_error(Exception(msg), attempt, context)
            if not should_retry:
                raise Exception(msg)
            time.sleep(wait_time)

        except requests.exceptions.Timeout as e:
            should_retry, wait_time = _handle_api_error(e, attempt, f"{context} timeout")
            if not should_retry:
                raise
            time.sleep(wait_time)
        except requests.exceptions.RequestException as e:
            should_retry, wait_time = _handle_api_error(e, attempt, f"{context} réseau")
            if not should_retry:
                raise
            time.sleep(wait_time)

    raise Exception(f"Échec {context} après {MAX_RETRIES} tentatives")


def ocr_page_with_vl(api_key: str, pdf_path: str, page_num: int) -> Tuple[str, Dict]:
    image_b64, _ = render_single_page_to_base64(pdf_path, page_num)
    data_url = f"data:image/png;base64,{image_b64}"

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": OCR_PROMPT},
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": f"OCR de la page {page_num}. Retourne uniquement le texte OCR brut."},
        ],
    }]

    text, stats = _call_chat(api_key, MODEL_OCR, messages, MAX_TOKENS_OCR, f"OCR page {page_num}")
    text = _strip_triple_backticks(text)
    if len(text.strip()) < 20:
        raise Exception("OCR trop court / vide (suspect)")
    return text, stats


def markdown_from_ocr(api_key: str, ocr_text: str, page_num: int) -> Tuple[str, Dict]:
    user_block = (
        f"Voici le texte OCR brut de la page {page_num} :\n\n"
        "```text\n"
        f"{ocr_text}\n"
        "```\n\n"
        "Génère uniquement le Markdown structuré pour cette page. "
        "N'inclus PAS la section '## Annexe - OCR brut'."
    )

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": SYSTEM_PROMPT_MD},
            {"type": "text", "text": user_block},
        ],
    }]

    md, stats = _call_chat(api_key, MODEL_MD, messages, MAX_TOKENS_MD, f"Markdown page {page_num}")
    md = _strip_triple_backticks(md)
    md = _strip_existing_ocr_appendix(md)
    md = _remove_empty_md_table_rows(md)
    return md.strip(), stats


# =====================
# Fonction attendue: process_page_with_cache
# =====================

def process_page_with_cache(pdf_path: str, page_num: int, api_key: str, is_first_page: bool = False) -> Tuple[str, Dict]:
    """
    Retourne (markdown_page, stats) comme attendu.
    Le runner gère le cache via load_progress/save_progress/clear_progress.
    """
    page_num = int(page_num)

    # 1) OCR brut
    ocr_text, ocr_stats = ocr_page_with_vl(api_key=api_key, pdf_path=pdf_path, page_num=page_num)

    # 2) Markdown structuré depuis OCR brut
    md_core, md_stats = markdown_from_ocr(api_key=api_key, ocr_text=ocr_text, page_num=page_num)

    page_md = (
        f"<!-- PAGE {page_num} -->\n\n"
        f"{md_core}\n\n"
        "## Annexe - OCR brut\n"
        "```text\n"
        f"{ocr_text.rstrip()}\n"
        "```\n\n"
        "---"
    )

    stats = {
        "input_tokens": int(ocr_stats.get("input_tokens", 0)) + int(md_stats.get("input_tokens", 0)),
        "output_tokens": int(ocr_stats.get("output_tokens", 0)) + int(md_stats.get("output_tokens", 0)),
        "total_tokens": int(ocr_stats.get("total_tokens", 0)) + int(md_stats.get("total_tokens", 0)),
        "details": {"ocr": ocr_stats, "md": md_stats},
    }

    return page_md, stats


__all__ = [
    "MODEL",
    "INTER_REQUEST_DELAY",
    "MODEL_OCR",
    "MODEL_MD",
    "get_pdf_info",
    "load_progress",
    "save_progress",
    "clear_progress",
    "process_page_with_cache",
    "ocr_page_with_vl",
    "markdown_from_ocr",
]
