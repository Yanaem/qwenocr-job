#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module OCR QwenVL compatible avec qwenocr_runner.py (SANS modifier le runner).

Interface attendue (comme ton script "TOPDUTOP") :
- MODEL (str)
- process_page_with_cache(pdf_path, page_num, api_key, is_first_page=False) -> (markdown, stats)

Implémentation :
- 2 étapes pour fidélité maximale :
  1) Vision OCR -> texte brut
  2) Texte -> Markdown structuré
- Le .md retourné inclut une annexe "OCR brut" (ajoutée par le code, 0 token).
"""

import base64
import re
import time
from io import BytesIO
from typing import Dict, List, Tuple

import requests
from pdf2image import convert_from_path  # nécessite poppler dans l'image

# =====================
# Configuration
# =====================

API_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

MODEL_OCR = "qwen-vl-max"   # vision (image -> OCR texte)
MODEL_MD = "qwen-vl-max"    # texte  (OCR texte -> Markdown)

# Le runner affiche ocr.MODEL
MODEL = MODEL_OCR

MAX_TOKENS_OCR = 20000
MAX_TOKENS_MD = 20000

TEMPERATURE = 0.0
REQUEST_TIMEOUT = 600
MAX_RETRIES = 5
BACKOFF_BASE = 2
BACKOFF_MAX = 120

RENDER_DPI = 300

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
# Helpers
# =====================

def calculate_backoff_delay(attempt: int) -> int:
    return min(BACKOFF_BASE ** attempt, BACKOFF_MAX)


def handle_api_error(error: Exception, attempt: int, context: str) -> Tuple[bool, int]:
    err = str(error).lower()

    non_retryable = ["invalid api key", "authentication failed", "permission denied"]
    if any(x in err for x in non_retryable):
        return False, 0

    if attempt >= MAX_RETRIES:
        return False, 0

    wait_time = calculate_backoff_delay(attempt)

    if "429" in err or "rate limit" in err:
        wait_time = max(wait_time, 60)
    elif "overloaded" in err:
        wait_time = max(wait_time, 30)

    return True, wait_time


def extract_text_from_response_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, dict):
                txt = part.get("text")
                if txt:
                    parts.append(txt)
        return "\n\n".join(parts)
    return str(content)


def strip_triple_backticks(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t)
    return t.strip("\n")


def remove_empty_md_table_rows(md: str) -> str:
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


def strip_existing_ocr_appendix(md: str) -> str:
    m = re.search(r"^##\s+Annexe\s*-\s*OCR\s+brut\s*$", md, flags=re.IGNORECASE | re.MULTILINE)
    if not m:
        return md.strip()
    return md[:m.start()].rstrip()


def render_single_page_to_base64(pdf_path: str, page_num: int, dpi: int = RENDER_DPI) -> Tuple[str, float]:
    images = convert_from_path(pdf_path, dpi=dpi, first_page=page_num, last_page=page_num)
    if not images:
        raise ValueError(f"Aucune image générée pour la page {page_num}")

    image = images[0]
    buf = BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    image_bytes = buf.read()

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    size_kb = len(image_bytes) / 1024
    return image_b64, size_kb


def call_chat_completions(
    api_key: str,
    model: str,
    messages: List[Dict],
    max_tokens: int,
    context: str,
    temperature: float = TEMPERATURE,
) -> Tuple[str, Dict]:
    url = f"{API_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {"model": model, "max_tokens": max_tokens, "temperature": temperature, "messages": messages}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.post(url, headers=headers, json=body, timeout=REQUEST_TIMEOUT)

            if r.status_code == 200:
                js = r.json()
                usage = js.get("usage", {})
                input_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0))
                output_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0))

                choices = js.get("choices", [])
                content = ""
                if choices:
                    content = choices[0].get("message", {}).get("content", "")

                text = extract_text_from_response_content(content).strip()
                stats = {
                    "input_tokens": int(input_tokens or 0),
                    "output_tokens": int(output_tokens or 0),
                    "total_tokens": int((input_tokens or 0) + (output_tokens or 0)),
                }
                return text, stats

            # Erreur HTTP -> retry selon règles
            try:
                err_json = r.json()
            except Exception:
                err_json = r.text[:300]
            error_msg = f"HTTP {r.status_code}: {str(err_json)[:300]}"
            should_retry, wait_time = handle_api_error(Exception(error_msg), attempt, context)
            if not should_retry:
                raise Exception(error_msg)
            time.sleep(wait_time)

        except requests.exceptions.Timeout as e:
            should_retry, wait_time = handle_api_error(e, attempt, f"{context} timeout")
            if not should_retry:
                raise
            time.sleep(wait_time)

        except requests.exceptions.RequestException as e:
            should_retry, wait_time = handle_api_error(e, attempt, f"{context} réseau")
            if not should_retry:
                raise
            time.sleep(wait_time)

    raise Exception(f"Échec {context} après {MAX_RETRIES} tentatives")


# =====================
# 2 étapes : OCR puis Markdown
# =====================

def ocr_page_with_vl(api_key: str, pdf_path: str, page_num: int) -> Tuple[str, Dict]:
    image_b64, _ = render_single_page_to_base64(pdf_path, page_num)
    data_url = f"data:image/png;base64,{image_b64}"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": OCR_PROMPT},
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": f"OCR de la page {page_num}. Retourne uniquement le texte OCR brut."},
            ],
        }
    ]

    text, stats = call_chat_completions(
        api_key=api_key,
        model=MODEL_OCR,
        messages=messages,
        max_tokens=MAX_TOKENS_OCR,
        context=f"OCR page {page_num}",
    )

    text = strip_triple_backticks(text)
    if len(text.strip()) < 20:
        raise Exception("OCR trop court / vide (suspect)")
    return text, stats


def markdown_from_ocr(api_key: str, ocr_text: str, page_num: int) -> Tuple[str, Dict]:
    user_ocr_block = (
        f"Voici le texte OCR brut de la page {page_num} :\n\n"
        "```text\n"
        f"{ocr_text}\n"
        "```\n\n"
        "Génère uniquement le Markdown structuré pour cette page. "
        "N'inclus PAS la section '## Annexe - OCR brut'."
    )

    messages = [
        {"role": "user", "content": [{"type": "text", "text": SYSTEM_PROMPT_MD}, {"type": "text", "text": user_ocr_block}]}
    ]

    md, stats = call_chat_completions(
        api_key=api_key,
        model=MODEL_MD,
        messages=messages,
        max_tokens=MAX_TOKENS_MD,
        context=f"Markdown page {page_num}",
    )

    md = strip_triple_backticks(md)
    md = strip_existing_ocr_appendix(md)
    md = remove_empty_md_table_rows(md)
    return md.strip(), stats


# =====================
# Fonction attendue par le runner
# =====================

def process_page_with_cache(
    pdf_path: str,
    page_num: int,
    api_key: str,
    is_first_page: bool = False
) -> Tuple[str, Dict]:
    """
    Fonction attendue par qwenocr_runner.py (interface TOPDUTOP).

    Retour:
      - markdown (str) : inclut marqueur <!-- PAGE n -->, Markdown structuré, annexe OCR brut, puis '---'
      - stats (dict)   : tokens agrégés des 2 appels (OCR + MD) + détail
    """
    # 1) OCR brut
    ocr_text, ocr_stats = ocr_page_with_vl(api_key=api_key, pdf_path=pdf_path, page_num=page_num)

    # 2) Markdown structuré à partir de l'OCR brut
    md_core, md_stats = markdown_from_ocr(api_key=api_key, ocr_text=ocr_text, page_num=page_num)

    # 3) Assemblage page (avec annexe OCR)
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
        "details": {
            "ocr": ocr_stats,
            "md": md_stats,
        },
    }

    return page_md, stats


__all__ = [
    "MODEL",
    "MODEL_OCR",
    "MODEL_MD",
    "process_page_with_cache",
    "ocr_page_with_vl",
    "markdown_from_ocr",
]
