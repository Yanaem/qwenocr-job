#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ocr_qwenVL.py — module compatible qwenocr_runner.py (sans modifier le runner)

Expose (d'après tes logs) :
- MODEL (str)
- INTER_REQUEST_DELAY (float)
- get_pdf_info(pdf_path) -> dict {page_count,...}
- load_progress(pdf_path) -> Dict[str, Dict]
- save_progress(pdf_path, completed_pages) -> None
- clear_progress(pdf_path) -> None
- process_page_with_cache(pdf_path, page_num, api_key, is_first_page=False) -> (markdown_page, stats_payload)
- calculate_costs(stats_list) -> dict
- validate_markdown_quality(final_markdown, page_count) -> dict

Implémentation :
- 2 étapes (fidélité) :
  1) OCR brut (image -> texte brut)
  2) Markdown (texte brut -> markdown structuré)
- Ajout d'une annexe "OCR brut" dans le markdown (côté code, 0 token).

Robustesse Cloud Run :
- Conversion PDF->PNG low-memory (pdf2image paths_only + fichier temp)
- DPI par défaut 200 (configurable)
- Backoff court + logs (évite les sleeps 60s silencieux)
- Payload stats retourne à la fois des clés "flat" et une sous-clé "stats" pour compat runner
  (évite KeyError 'stats').
"""

from __future__ import annotations

import base64
import gc
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests
from pdf2image import convert_from_path, pdfinfo_from_path

# --- Lecture PDF (optionnelle) ---
PdfReader = None
try:
    from pypdf import PdfReader as _PdfReader  # type: ignore
    PdfReader = _PdfReader
except Exception:
    try:
        from PyPDF2 import PdfReader as _PdfReader  # type: ignore
        PdfReader = _PdfReader
    except Exception:
        PdfReader = None


# =====================
# Helpers ENV
# =====================

def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    try:
        return int(v)
    except Exception:
        return default

def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    try:
        return float(v)
    except Exception:
        return default

def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    return v.strip().lower() not in ("0", "false", "no", "off")


# =====================
# Configuration
# =====================

API_URL = os.getenv("QWEN_API_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")

MODEL_OCR = os.getenv("QWEN_MODEL_OCR", "qwen-vl-max")
MODEL_MD = os.getenv("QWEN_MODEL_MD", "qwen-vl-max")

# Attendu par le runner
MODEL = MODEL_OCR

# Pause entre pages (le runner l'utilise)
INTER_REQUEST_DELAY = _env_float("INTER_REQUEST_DELAY", 1.0)

# DPI : 300 est plus lourd. 200 suffit souvent.
RENDER_DPI = _env_int("RENDER_DPI", 200)

MAX_TOKENS_OCR = _env_int("MAX_TOKENS_OCR", 12000)
MAX_TOKENS_MD = _env_int("MAX_TOKENS_MD", 12000)

TEMPERATURE = _env_float("TEMPERATURE", 0.0)

# Timeouts/retries
REQUEST_TIMEOUT_SECONDS = _env_int("REQUEST_TIMEOUT_SECONDS", 120)
CONNECT_TIMEOUT_SECONDS = _env_int("CONNECT_TIMEOUT_SECONDS", 10)

MAX_RETRIES = _env_int("MAX_RETRIES", 3)
BACKOFF_BASE = _env_float("BACKOFF_BASE", 2.0)
BACKOFF_MAX = _env_float("BACKOFF_MAX", 15.0)  # volontairement bas

# Logs
VERBOSE = _env_bool("VERBOSE", True)

# Rate-limit behavior
FAIL_FAST_ON_429 = _env_bool("FAIL_FAST_ON_429", False)  # False = retry court; True = fail vite


def _log(msg: str) -> None:
    if VERBOSE:
        print(msg, flush=True)


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
# Progress (attendu par le runner)
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
# PDF info (attendu par le runner)
# =====================

def get_pdf_info(pdf_path: str) -> Dict[str, Any]:
    pdf_path = str(pdf_path)
    file_size = os.path.getsize(pdf_path)

    page_count = None

    # 1) pypdf/PyPDF2
    if PdfReader is not None:
        try:
            with open(pdf_path, "rb") as f:
                reader = PdfReader(f)
                page_count = len(reader.pages)
        except Exception:
            page_count = None

    # 2) poppler pdfinfo
    if page_count is None:
        try:
            info = pdfinfo_from_path(pdf_path)
            page_count = int(info.get("Pages"))
        except Exception:
            page_count = None

    if page_count is None:
        raise RuntimeError("Impossible de déterminer le nombre de pages (pypdf/PyPDF2 + pdfinfo indisponibles).")

    return {
        "page_count": int(page_count),
        "file_size_bytes": int(file_size),
        "file_size_mb": file_size / (1024 * 1024),
    }


# =====================
# Helpers (texte / markdown)
# =====================

def _extract_text_from_response_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("text"):
                parts.append(str(part["text"]))
        return "\n\n".join(parts)
    return str(content)

def _strip_triple_backticks(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t)
    return t.strip("\n")

def _strip_existing_ocr_appendix(md: str) -> str:
    m = re.search(r"^##\s+Annexe\s*-\s*OCR\s+brut\s*$", md, flags=re.IGNORECASE | re.MULTILINE)
    if not m:
        return md.strip()
    return md[:m.start()].rstrip()

def _remove_empty_md_table_rows(md: str) -> str:
    out: List[str] = []
    for line in md.splitlines():
        if re.match(r'^\|.*\|\s*$', line):
            cells = [c.strip() for c in line.strip().strip('|').split('|')]
            # garder séparateur
            if cells and all(re.fullmatch(r':?-{3,}:?', c) for c in cells):
                out.append(line)
                continue
            # supprimer ligne vide
            if all(c == "" for c in cells):
                continue
        out.append(line)
    return "\n".join(out)


# =====================
# Rendu PDF -> PNG base64 (low memory)
# =====================

def render_single_page_to_base64(pdf_path: str, page_num: int, dpi: int = RENDER_DPI) -> Tuple[str, float]:
    """
    Utilise paths_only=True pour réduire la RAM.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = convert_from_path(
            pdf_path,
            dpi=dpi,
            first_page=page_num,
            last_page=page_num,
            fmt="png",
            output_folder=tmpdir,
            paths_only=True,
            thread_count=1,
        )
        if not paths:
            raise ValueError(f"Aucune image générée pour la page {page_num}")
        png_path = paths[0]
        with open(png_path, "rb") as f:
            b = f.read()

    b64 = base64.b64encode(b).decode("utf-8")
    return b64, (len(b) / 1024.0)


# =====================
# Appels API Qwen
# =====================

def _backoff(attempt: int) -> float:
    delay = min((BACKOFF_BASE ** attempt), BACKOFF_MAX)
    return float(delay)

def _compute_retry_delay(http_status: int | None, err_msg: str, attempt: int) -> Tuple[bool, float]:
    """
    Renvoie (retry?, delay_sec). Backoff court pour éviter les kills de job.
    """
    if attempt >= MAX_RETRIES:
        return False, 0.0

    msg = (err_msg or "").lower()

    non_retryable = ["invalid api key", "authentication failed", "permission denied"]
    if any(x in msg for x in non_retryable):
        return False, 0.0

    if http_status == 429 or "rate limit" in msg:
        if FAIL_FAST_ON_429:
            return False, 0.0
        return True, min(10.0 * attempt, 15.0)

    if "overloaded" in msg:
        return True, min(5.0 * attempt, 10.0)

    return True, _backoff(attempt)

def _call_chat(api_key: str, model: str, messages: List[Dict[str, Any]], max_tokens: int, context: str) -> Tuple[str, Dict[str, int]]:
    url = f"{API_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {"model": model, "max_tokens": max_tokens, "temperature": TEMPERATURE, "messages": messages}

    for attempt in range(1, MAX_RETRIES + 1):
        t0 = time.time()
        try:
            r = requests.post(
                url,
                headers=headers,
                json=body,
                timeout=(CONNECT_TIMEOUT_SECONDS, REQUEST_TIMEOUT_SECONDS),
            )

            if r.status_code == 200:
                js = r.json()
                usage = js.get("usage", {}) or {}

                input_tokens = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
                output_tokens = int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)

                choices = js.get("choices", []) or []
                content = ""
                if choices:
                    content = choices[0].get("message", {}).get("content", "")

                text = _extract_text_from_response_content(content).strip()
                stats = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                }

                _log(f"✅ {context}: OK en {time.time()-t0:.2f}s (in={input_tokens} out={output_tokens})")
                return text, stats

            # non-200
            try:
                err_json = r.json()
                err_msg = json.dumps(err_json, ensure_ascii=False)[:800]
            except Exception:
                err_msg = (r.text or "")[:800]

            retry, delay = _compute_retry_delay(r.status_code, err_msg, attempt)
            _log(f"⚠️ {context}: HTTP {r.status_code} retry={retry} dans {delay:.1f}s | {err_msg[:200]}")
            if not retry:
                raise RuntimeError(f"{context}: HTTP {r.status_code} {err_msg}")

            time.sleep(delay)

        except requests.exceptions.Timeout as e:
            retry, delay = _compute_retry_delay(None, str(e), attempt)
            _log(f"⚠️ {context}: timeout retry={retry} dans {delay:.1f}s | {e}")
            if not retry:
                raise
            time.sleep(delay)

        except requests.exceptions.RequestException as e:
            retry, delay = _compute_retry_delay(None, str(e), attempt)
            _log(f"⚠️ {context}: réseau retry={retry} dans {delay:.1f}s | {e}")
            if not retry:
                raise
            time.sleep(delay)

    raise RuntimeError(f"Échec {context} après {MAX_RETRIES} tentatives")


# =====================
# 2 étapes : OCR puis Markdown
# =====================

def ocr_page_with_vl(api_key: str, pdf_path: str, page_num: int) -> Tuple[str, Dict[str, int]]:
    _log(f"➡️ Page {page_num}: rendu image (dpi={RENDER_DPI})")
    image_b64, size_kb = render_single_page_to_base64(pdf_path, page_num, dpi=RENDER_DPI)
    _log(f"➡️ Page {page_num}: image prête ({size_kb:.0f} KB), appel OCR")

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

    text, stats = _call_chat(api_key, MODEL_OCR, messages, MAX_TOKENS_OCR, f"OCR page {page_num}")
    text = _strip_triple_backticks(text)

    if len(text.strip()) < 20:
        raise RuntimeError("OCR trop court / vide (suspect)")

    # libère mémoire
    del image_b64
    gc.collect()

    _log(f"✅ Page {page_num}: OCR OK ({stats.get('total_tokens', 0)} tokens)")
    return text, stats


def markdown_from_ocr(api_key: str, ocr_text: str, page_num: int) -> Tuple[str, Dict[str, int]]:
    _log(f"➡️ Page {page_num}: appel Markdown (depuis OCR brut)")

    user_block = (
        f"Voici le texte OCR brut de la page {page_num} :\n\n"
        "```text\n"
        f"{ocr_text}\n"
        "```\n\n"
        "Génère uniquement le Markdown structuré pour cette page. "
        "N'inclus PAS la section '## Annexe - OCR brut'."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": SYSTEM_PROMPT_MD},
                {"type": "text", "text": user_block},
            ],
        }
    ]

    md, stats = _call_chat(api_key, MODEL_MD, messages, MAX_TOKENS_MD, f"Markdown page {page_num}")
    md = _strip_triple_backticks(md)
    md = _strip_existing_ocr_appendix(md)
    md = _remove_empty_md_table_rows(md)

    _log(f"✅ Page {page_num}: Markdown OK ({stats.get('total_tokens', 0)} tokens)")
    return md.strip(), stats


# =====================
# Fonction attendue par le runner
# =====================

def process_page_with_cache(pdf_path: str, page_num: int, api_key: str, is_first_page: bool = False) -> Tuple[str, Dict[str, Any]]:
    """
    Doit retourner (markdown_page, stats_payload).
    stats_payload contient volontairement :
      - des champs flat (input_tokens/output_tokens/total_tokens)
      - ET une sous-clé "stats" qui répète ces champs
    -> ça évite les KeyError 'stats' côté runner si le runner attend payload['stats'].
    """
    page_num = int(page_num)

    # 1) OCR brut
    ocr_text, ocr_stats = ocr_page_with_vl(api_key=api_key, pdf_path=pdf_path, page_num=page_num)

    # 2) Markdown structuré
    md_core, md_stats = markdown_from_ocr(api_key=api_key, ocr_text=ocr_text, page_num=page_num)

    # 3) Assemblage (inclut OCR brut en annexe)
    page_md = (
        f"<!-- PAGE {page_num} -->\n\n"
        f"{md_core}\n\n"
        "## Annexe - OCR brut\n"
        "```text\n"
        f"{ocr_text.rstrip()}\n"
        "```\n\n"
        "---"
    )

    input_tokens = int(ocr_stats.get("input_tokens", 0)) + int(md_stats.get("input_tokens", 0))
    output_tokens = int(ocr_stats.get("output_tokens", 0)) + int(md_stats.get("output_tokens", 0))
    total_tokens = int(ocr_stats.get("total_tokens", 0)) + int(md_stats.get("total_tokens", 0))

    stats_core = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "details": {"ocr": ocr_stats, "md": md_stats},
        "models": {"ocr": MODEL_OCR, "md": MODEL_MD},
        "render_dpi": RENDER_DPI,
    }

    # payload "compat": keys flat + nested 'stats'
    stats_payload: Dict[str, Any] = dict(stats_core)
    stats_payload["stats"] = dict(stats_core)  # <-- clé attendue par certains runners

    # libère mémoire
    del ocr_text
    gc.collect()

    # Pause optionnelle (si tu veux 0, mets INTER_REQUEST_DELAY=0)
    if INTER_REQUEST_DELAY and INTER_REQUEST_DELAY > 0:
        time.sleep(INTER_REQUEST_DELAY)

    return page_md, stats_payload


# =====================
# Attendu: calculate_costs
# =====================

def calculate_costs(stats_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compat runner : renvoie des coûts à 0.0 mais garde les totaux tokens.
    Supporte deux formats :
      - dict flat avec input_tokens/output_tokens/total_tokens
      - dict wrapper avec une sous-clé 'stats' contenant ces champs
    """
    stats_list = stats_list or []

    total_input = 0
    total_output = 0
    total_tokens = 0

    for s in stats_list:
        if not isinstance(s, dict):
            continue
        core = s.get("stats") if isinstance(s.get("stats"), dict) else s
        total_input += int(core.get("input_tokens", 0) or 0)
        total_output += int(core.get("output_tokens", 0) or 0)
        tt = core.get("total_tokens")
        if tt is None:
            tt = (int(core.get("input_tokens", 0) or 0) + int(core.get("output_tokens", 0) or 0))
        total_tokens += int(tt or 0)

    pages = max(len(stats_list), 1)

    return {
        "total_input": total_input,
        "total_output": total_output,
        "total_tokens": total_tokens,
        "cost_input": 0.0,
        "cost_output": 0.0,
        "cost_total": 0.0,
        "cost_per_page": 0.0,
        "pages": pages,
        "stats": {  # compat éventuelle
            "total_input": total_input,
            "total_output": total_output,
            "total_tokens": total_tokens,
            "pages": pages,
        },
    }


# =====================
# Attendu: validate_markdown_quality
# =====================

def validate_markdown_quality(final_markdown: str, page_count: int) -> Dict[str, Any]:
    """
    Compat runner : validation légère, non bloquante.
    Renvoie un dict avec plusieurs clés compatibles, et inclut une sous-clé 'stats'.
    """
    errors: List[str] = []
    warnings: List[str] = []

    if not isinstance(final_markdown, str) or final_markdown.strip() == "":
        errors.append("Markdown final vide.")
        pages_found: List[int] = []
    else:
        pages_found = []
        for m in re.finditer(r"<!--\s*PAGE\s+(\d+)\s*-->", final_markdown):
            try:
                pages_found.append(int(m.group(1)))
            except Exception:
                continue

        # Vérifs simples
        expected = int(page_count or 0)
        got_pages = len(set(pages_found))

        if expected and got_pages != expected:
            warnings.append(f"Pages détectées: {got_pages}, attendu: {expected}.")

        annexes = len(
            re.findall(
                r"^##\s+Annexe\s*-\s*OCR\s+brut\s*$",
                final_markdown,
                flags=re.MULTILINE | re.IGNORECASE,
            )
        )
        if expected and annexes < expected:
            warnings.append(f"Annexes OCR: {annexes}, attendu >= {expected}.")

    ok = (len(errors) == 0)
    score = 1.0 if ok else 0.0

    stats = {
        "page_count": int(page_count or 0),
        "pages_found": sorted(set(pages_found)) if isinstance(final_markdown, str) else [],
        "warnings_count": len(warnings),
        "errors_count": len(errors),
        "score": score,
    }

    return {
        "ok": ok,
        "is_valid": ok,
        "valid": ok,
        "passed": ok,
        "score": score,
        "errors": errors,
        "warnings": warnings,
        "stats": stats,  # <-- clé que certains runners attendent
        "summary": ("OK" if ok else "KO") + ("" if not warnings else f" (warnings={len(warnings)})"),
    }


__all__ = [
    "MODEL",
    "MODEL_OCR",
    "MODEL_MD",
    "INTER_REQUEST_DELAY",
    "RENDER_DPI",
    "get_pdf_info",
    "load_progress",
    "save_progress",
    "clear_progress",
    "process_page_with_cache",
    "calculate_costs",
    "validate_markdown_quality",
    "ocr_page_with_vl",
    "markdown_from_ocr",
]

