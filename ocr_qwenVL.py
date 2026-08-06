#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ocr_qwenVL.py — deux lectures visuelles Qwen totalement indépendantes.

Contrat v12.5.1 — exactement deux générations Qwen par page :
1. Python rend une image maîtresse unique puis deux jeux de vues déterministes ;
2. branche A : Qwen réalise un OCR visuel rapide d'audit, avec thinking activé ;
3. branche B : Qwen repart indépendamment des pixels et produit le Markdown final,
   avec un thinking plus approfondi et un protocole obligatoire ;
4. aucune sortie, carte, valeur ni trace de reasoning de la branche A n'est transmise
   à la branche B ;
5. les deux appels peuvent être exécutés en parallèle ;
6. Python ne corrige aucune donnée documentaire et assemble un seul fichier .md :
   pour chaque page, le Markdown final puis l'OCR d'audit indépendant ;
   le thinking reste hors du fichier final sauf activation explicite ;
7. le modèle par défaut des deux branches est l'alias qwen3.7-plus.
"""

from __future__ import annotations

import base64
import hashlib
import html
import json
import os
import re
import tempfile
import threading
import unicodedata
from difflib import SequenceMatcher
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests
from PIL import Image, ImageDraw, ImageFont
from pdf2image import convert_from_path, pdfinfo_from_path
from requests.adapters import HTTPAdapter

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


# =============================================================================
# Environnement
# =============================================================================


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return int(default)
    try:
        return int(raw.strip())
    except ValueError as exc:
        raise RuntimeError(f"{name} doit être un entier. Valeur reçue : {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return float(default)
    try:
        return float(raw.strip())
    except ValueError as exc:
        raise RuntimeError(f"{name} doit être un nombre. Valeur reçue : {raw!r}") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return bool(default)
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(
        f"{name} doit valoir true/false, 1/0, yes/no ou on/off. Valeur reçue : {raw!r}"
    )


# =============================================================================
# Configuration
# =============================================================================

PIPELINE_VERSION = "qwen-dual-independent-vision-accounting-special-row-coordinate-lock-v12.5.1-20260806"
CHECKPOINT_VERSION = 50
CHECKPOINT_SCHEMA = "dual-independent-vision-accounting-special-row-coordinate-lock-v48"
EXPECTED_LAB_VERSION = "qwenocr-job-cloudrun-v1.8.6-20260806"
EXPECTED_PACKAGE_FOLDER = "qwenocr-job-main"
ENFORCE_PACKAGE_FOLDER_NAME = _env_bool("ENFORCE_PACKAGE_FOLDER_NAME", False)

_RUNTIME_PACKAGE_CONTRACT_CHECKED = False


def validate_runtime_package_contract() -> None:
    """Bloque tout mélange de version depuis le module lui-même, avant le réseau."""
    global _RUNTIME_PACKAGE_CONTRACT_CHECKED
    if _RUNTIME_PACKAGE_CONTRACT_CHECKED:
        return
    module_dir = Path(__file__).resolve().parent
    # Le dépôt Cloud Run garde ocr_qwenVL.py à la racine, tandis que le
    # laboratoire local peut le placer dans code_under_test/. On choisit le
    # premier répertoire qui contient VERSION.txt.
    package_candidates = (module_dir, module_dir.parent)
    package_root = next(
        (candidate for candidate in package_candidates if (candidate / "VERSION.txt").exists()),
        module_dir,
    )
    errors: List[str] = []
    if ENFORCE_PACKAGE_FOLDER_NAME and package_root.name != EXPECTED_PACKAGE_FOLDER:
        errors.append(f"dossier={package_root.name!r}, attendu={EXPECTED_PACKAGE_FOLDER!r}")
    version_path = package_root / "VERSION.txt"
    if not version_path.exists():
        errors.append(f"VERSION.txt absent dans {package_root}")
    else:
        try:
            actual_version = version_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            errors.append(f"VERSION.txt illisible: {exc}")
        else:
            if actual_version != EXPECTED_LAB_VERSION:
                errors.append(f"VERSION.txt={actual_version!r}, attendu={EXPECTED_LAB_VERSION!r}")
    if "v12.5.1-20260806" not in PIPELINE_VERSION:
        errors.append(f"pipeline={PIPELINE_VERSION!r}, attendu contenant 'v12.5.1-20260806'")
    if errors:
        raise RuntimeError(
            "Mélange de versions détecté par ocr_qwenVL.py avant tout appel Qwen : "
            + " ; ".join(errors)
            + ". Vérifie VERSION.txt et le déploiement complet du dépôt."
        )
    _RUNTIME_PACKAGE_CONTRACT_CHECKED = True


QWEN_WORKSPACE_ID = os.getenv("QWEN_WORKSPACE_ID", "").strip()
_QWEN_API_URL_OVERRIDE = os.getenv("QWEN_API_URL", "").strip().rstrip("/")

if QWEN_WORKSPACE_ID:
    API_URL = (
        f"https://{QWEN_WORKSPACE_ID}.ap-southeast-1.maas.aliyuncs.com/"
        "compatible-mode/v1"
    )
elif _QWEN_API_URL_OVERRIDE:
    API_URL = _QWEN_API_URL_OVERRIDE
else:
    API_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

# L'alias est volontairement utilisé, conformément au choix explicite de l'utilisateur.
DEFAULT_QWEN_MODEL = "qwen3.7-plus"
MODEL_OCR = os.getenv("QWEN_MODEL_OCR", DEFAULT_QWEN_MODEL).strip()
MODEL_MARKDOWN = os.getenv("QWEN_MODEL_MARKDOWN", DEFAULT_QWEN_MODEL).strip()
MODEL = MODEL_MARKDOWN

OCR_AUDIT_PASS = True
MARKDOWN_VISUAL_PASS = True
DUAL_INDEPENDENT_VISUAL_PASSES = True
PARALLEL_INDEPENDENT_PASSES = _env_bool("PARALLEL_INDEPENDENT_PASSES", True)

# Alias conservés pour les intégrations historiques.
RAW_OCR_FIRST_PASS = OCR_AUDIT_PASS
MARKDOWN_SECOND_PASS = MARKDOWN_VISUAL_PASS
TWO_PASS_RAW_OCR_MARKDOWN = False
ONE_PASS_THINKING_OCR = False
TWO_PASS_GEOMETRY_OCR = False
CANONICAL_OCR_ONLY = True
DETERMINISTIC_MARKDOWN = False
MODEL_GENERATED_MARKDOWN = True
SINGLE_MARKDOWN_OUTPUT = True
OCR_PROMPT_IN_USER_MESSAGE = True
MARKDOWN_PROMPT_IN_USER_MESSAGE = True
NOMINAL_GENERATIONS_PER_PAGE = 2
SEMANTIC_RETRIES = 0

STOP_ON_CRITICAL = _env_bool("STOP_ON_CRITICAL", False)
PUBLISH_PARTIAL_DOCUMENT = _env_bool("PUBLISH_PARTIAL_DOCUMENT", True)
PUBLISH_DEGRADED_MARKDOWN = _env_bool("PUBLISH_DEGRADED_MARKDOWN", False)
STRICT_MARKDOWN_REGRESSION_GATE = False  # v12.5.1 : jamais de masquage du résultat Markdown
OCR_DIAGNOSTIC_MODE = _env_bool("OCR_DIAGNOSTIC_MODE", False)
PIPELINE_AUDIT_MODE = _env_bool("PIPELINE_AUDIT_MODE", True)

# Les annexes font partie du fichier .md final en mode audit.
INCLUDE_OCR_ANNEX = _env_bool("INCLUDE_OCR_ANNEX", True)
# En production, final.md contient Markdown + OCR par page. Le thinking reste
# capturé dans le checkpoint/diagnostic, mais n'est pas publié par défaut.
INCLUDE_THINKING_ANNEX = _env_bool("INCLUDE_THINKING_ANNEX", False)
CAPTURE_REASONING_CONTENT = _env_bool("CAPTURE_REASONING_CONTENT", True)
THINKING_ANNEX_MAX_CHARS = max(10000, _env_int("THINKING_ANNEX_MAX_CHARS", 150000))
OCR_ANNEX_SOURCE = "raw_qwen_independent_ocr_audit"
# Marqueurs historiques conservés pour ne pas casser les consommateurs existants.
RENDERED_DOCUMENT_START = "<!-- RENDERED_DOCUMENT_START -->"
RENDERED_DOCUMENT_END = "<!-- RENDERED_DOCUMENT_END -->"
FINAL_MARKDOWN_START = RENDERED_DOCUMENT_START
FINAL_MARKDOWN_END = RENDERED_DOCUMENT_END
OCR_ANNEX_START = f'<!-- OCR_ANNEX_START source="{OCR_ANNEX_SOURCE}" -->'
OCR_ANNEX_END = "<!-- OCR_ANNEX_END -->"
THINKING_ANNEX_START = '<!-- THINKING_ANNEX_START source="qwen_reasoning_content" -->'
THINKING_ANNEX_END = "<!-- THINKING_ANNEX_END -->"

# final.md : Markdown puis OCR, regroupés page par page.
FINAL_MD_LAYOUT = os.getenv("FINAL_MD_LAYOUT", "page_interleaved").strip().lower()
if FINAL_MD_LAYOUT not in {"page_interleaved", "legacy_annex"}:
    raise RuntimeError("FINAL_MD_LAYOUT doit valoir page_interleaved ou legacy_annex.")
PAGE_BUNDLE_START_TEMPLATE = "<!-- PAGE_BUNDLE_START page={page} -->"
PAGE_BUNDLE_END_TEMPLATE = "<!-- PAGE_BUNDLE_END page={page} -->"
RENDERED_PAGE_START_TEMPLATE = "<!-- RENDERED_PAGE_START page={page} -->"
RENDERED_PAGE_END_TEMPLATE = "<!-- RENDERED_PAGE_END page={page} -->"
OCR_PAGE_START_TEMPLATE = "<!-- OCR_PAGE_START page={page} source=raw_qwen_independent_ocr_audit -->"
OCR_PAGE_END_TEMPLATE = "<!-- OCR_PAGE_END page={page} -->"

# Branche Markdown : cinq vues documentaires + un repère géométrique synthétique, indépendants de l'OCR d'audit.
RENDER_DPI = _env_int("RENDER_DPI", 300)
DETAIL_DPI = _env_int("DETAIL_DPI", 500)
ENABLE_DETAIL_VIEWS = _env_bool("ENABLE_DETAIL_VIEWS", True)
DETAIL_UPPER_END = _env_float("DETAIL_UPPER_END", 0.45)
DETAIL_MIDDLE_START = _env_float("DETAIL_MIDDLE_START", 0.30)
DETAIL_MIDDLE_END = _env_float("DETAIL_MIDDLE_END", 0.75)
DETAIL_LOWER_START = _env_float("DETAIL_LOWER_START", 0.60)
RIGHT_VIEW_START = _env_float("RIGHT_VIEW_START", 0.45)
MARKDOWN_GEOMETRY_GUIDE = True
GEOMETRY_GUIDE_MINOR_STEP_PERCENT = _env_float("GEOMETRY_GUIDE_MINOR_STEP_PERCENT", 2.5)
GEOMETRY_GUIDE_LABEL_STEP_PERCENT = _env_float("GEOMETRY_GUIDE_LABEL_STEP_PERCENT", 5.0)
GEOMETRY_GUIDE_MAJOR_STEP_PERCENT = _env_float("GEOMETRY_GUIDE_MAJOR_STEP_PERCENT", 10.0)
GEOMETRY_GUIDE_QUALITY_OFFSET = max(0, _env_int("GEOMETRY_GUIDE_QUALITY_OFFSET", 18))
GEOMETRY_GUIDE_MAX_PIXELS = max(2_000_000, _env_int("GEOMETRY_GUIDE_MAX_PIXELS", 5_500_000))
MARKDOWN_EXPECTED_VIEW_COUNT = 6
EXPECTED_VIEW_COUNT = MARKDOWN_EXPECTED_VIEW_COUNT

# Branche OCR d'audit : quatre vues plus légères pour limiter le coût et la latence.
OCR_AUDIT_RENDER_DPI = _env_int("OCR_AUDIT_RENDER_DPI", 280)
OCR_AUDIT_DETAIL_DPI = _env_int("OCR_AUDIT_DETAIL_DPI", 420)
OCR_AUDIT_UPPER_END = _env_float("OCR_AUDIT_UPPER_END", 0.55)
OCR_AUDIT_LOWER_START = _env_float("OCR_AUDIT_LOWER_START", 0.45)
OCR_AUDIT_RIGHT_VIEW_START = _env_float("OCR_AUDIT_RIGHT_VIEW_START", 0.45)
OCR_AUDIT_EXPECTED_VIEW_COUNT = 4

VIEW_JPEG_QUALITY = _env_int("VIEW_JPEG_QUALITY", 94)
VIEW_JPEG_MIN_QUALITY = _env_int("VIEW_JPEG_MIN_QUALITY", 84)
OCR_AUDIT_JPEG_QUALITY = _env_int("OCR_AUDIT_JPEG_QUALITY", 90)
OCR_AUDIT_JPEG_MIN_QUALITY = _env_int("OCR_AUDIT_JPEG_MIN_QUALITY", 80)
VIEW_JPEG_SUBSAMPLING = _env_int("VIEW_JPEG_SUBSAMPLING", 0)
MAX_VIEW_PIXELS = max(1_000_000, _env_int("MAX_VIEW_PIXELS", 16_000_000))
MAX_PAYLOAD_PROFILES = max(1, min(4, _env_int("MAX_PAYLOAD_PROFILES", 4)))
MAX_REQUEST_BODY_MB = min(16.0, max(9.0, _env_float("MAX_REQUEST_BODY_MB", 14.0)))
MAX_TOTAL_BASE64_IMAGE_MB = min(
    12.5,
    max(5.0, _env_float("MAX_TOTAL_BASE64_IMAGE_MB", 11.5)),
    MAX_REQUEST_BODY_MB - 1.0,
)
MAX_SINGLE_BASE64_IMAGE_MB = min(
    7.0,
    max(1.5, _env_float("MAX_SINGLE_BASE64_IMAGE_MB", 6.5)),
    MAX_TOTAL_BASE64_IMAGE_MB,
)
ALLOW_413_PAYLOAD_FALLBACK = _env_bool("ALLOW_413_PAYLOAD_FALLBACK", True)

TEMPERATURE = _env_float("TEMPERATURE", 0.0)
OCR_SEED = _env_int("OCR_SEED", 0)
MARKDOWN_SEED = _env_int("MARKDOWN_SEED", 0)
ENABLE_THINKING_OCR = _env_bool("ENABLE_THINKING_OCR", True)
ENABLE_THINKING_MARKDOWN = _env_bool("ENABLE_THINKING_MARKDOWN", True)

MAX_TOKENS_OCR = _env_int("MAX_TOKENS_OCR", 14000)
THINKING_BUDGET_OCR = _env_int("THINKING_BUDGET_OCR", 8192)
MAX_COMPLETION_TOKENS_OCR = _env_int(
    "MAX_COMPLETION_TOKENS_OCR",
    max(32768, MAX_TOKENS_OCR + THINKING_BUDGET_OCR),
)

MAX_TOKENS_MARKDOWN = _env_int("MAX_TOKENS_MARKDOWN", 24000)
THINKING_BUDGET_MARKDOWN = _env_int("THINKING_BUDGET_MARKDOWN", 16384)
MAX_COMPLETION_TOKENS_MARKDOWN = _env_int(
    "MAX_COMPLETION_TOKENS_MARKDOWN",
    max(57344, MAX_TOKENS_MARKDOWN + THINKING_BUDGET_MARKDOWN),
)

QWEN_HIGH_RES_IMAGES = _env_bool("QWEN_HIGH_RES_IMAGES", True)
STREAMING_OCR = True
STREAMING_MARKDOWN = True
STREAM_INCLUDE_USAGE = True
STREAM_ITER_CHUNK_SIZE = max(1024, _env_int("STREAM_ITER_CHUNK_SIZE", 8192))
THINKING_PROGRESS_LOG_SECONDS = max(
    0.0, _env_float("THINKING_PROGRESS_LOG_SECONDS", 30.0)
)

REQUEST_TIMEOUT_SECONDS = _env_int("REQUEST_TIMEOUT_SECONDS", 1200)
CONNECT_TIMEOUT_SECONDS = _env_int("CONNECT_TIMEOUT_SECONDS", 10)
HTTP_POOL_SIZE = max(1, _env_int("HTTP_POOL_SIZE", 8))
MAX_RETRIES = max(1, _env_int("MAX_RETRIES", 3))
BACKOFF_BASE = _env_float("BACKOFF_BASE", 2.0)
BACKOFF_MAX = _env_float("BACKOFF_MAX", 20.0)
FAIL_FAST_ON_429 = _env_bool("FAIL_FAST_ON_429", False)
EMPTY_RESPONSE_LOG_CHARS = max(200, _env_int("EMPTY_RESPONSE_LOG_CHARS", 1500))

VERBOSE = _env_bool("VERBOSE", True)
ENABLE_EXPLICIT_CACHE = _env_bool("ENABLE_EXPLICIT_CACHE", True)
FORCE_EXPLICIT_CACHE = _env_bool("FORCE_EXPLICIT_CACHE", False)
_EXPLICIT_CACHE_ACTIVE = ENABLE_EXPLICIT_CACHE
_CACHE_STATE_LOCK = threading.Lock()

# =============================================================================
# Contrat canonique
# =============================================================================

ALLOWED_SECTIONS = {
    "issuer",
    "customer",
    "shipping",
    "document",
    "line_items",
    "taxes",
    "totals",
    "payment",
    "annotations",
    "legal",
    "other",
}
ALLOWED_SOURCES = {"printed", "handwritten", "stamp"}
ALLOWED_STATUSES = {"readable", "uncertain", "truncated", "uncertain_truncated"}
ALLOWED_ROW_KINDS = {"header", "data", "continuation", "charge", "subtotal", "note", "other"}

SECTION_ALIASES = {
    # Compatibilité de reprise si le modèle reproduit occasionnellement un ancien rôle.
    "logo": "issuer",
    "logo_text": "issuer",
    "supplier": "issuer",
    "supplier_identity": "issuer",
    "supplier_address": "issuer",
    "supplier_contact": "issuer",
    "supplier_legal": "legal",
    "customer_identity": "customer",
    "customer_address": "customer",
    "customer_contact": "customer",
    "customer_legal": "customer",
    "billing_address": "customer",
    "shipping_address": "shipping",
    "shipping_details": "shipping",
    "shipping_contact": "shipping",
    "invoice_title": "document",
    "invoice_details": "document",
    "document_meta": "document",
    "line_items_note": "line_items",
    "line_items_footer": "line_items",
    "tax_summary": "taxes",
    "totals_summary": "totals",
    "payment_terms": "payment",
    "bank_details": "payment",
    "payment_table": "payment",
    "delivery_confirmation": "annotations",
    "stamp_signature": "annotations",
    "notes": "annotations",
    "annotation": "annotations",
    "legal_terms": "legal",
    "marketing_badge": "legal",
    "isolated_value": "other",
    "other_table": "other",
    "unknown": "other",
}

ELEMENT_START_RE = re.compile(r"^\s*\[\[(BLOCK|TABLE|KV)\s+(.+?)\]\]\s*$", re.IGNORECASE)
ELEMENT_END_PATTERNS = {
    "BLOCK": re.compile(r"^\s*\[\[/BLOCK\]\]\s*$", re.IGNORECASE),
    "TABLE": re.compile(r"^\s*\[\[/TABLE\]\]\s*$", re.IGNORECASE),
    "KV": re.compile(r"^\s*\[\[/KV\]\]\s*$", re.IGNORECASE),
}
ROW_START_RE = re.compile(r"^\s*\[\[ROW(?:\s+(.+?))?\]\]\s*$", re.IGNORECASE)
ROW_END_RE = re.compile(r"^\s*\[\[/ROW\]\]\s*$", re.IGNORECASE)
ITEM_START_RE = re.compile(r"^\s*\[\[ITEM(?:\s+(.+?))?\]\]\s*$", re.IGNORECASE)
ITEM_END_RE = re.compile(r"^\s*\[\[/ITEM\]\]\s*$", re.IGNORECASE)
END_PAGE_RE = re.compile(r"^\s*\[\[END_PAGE(?:\s+(.+?))?\]\]\s*$", re.IGNORECASE)
MODEL_PAGE_RE = re.compile(r"^\s*\[\[(?:PDF_)?PAGE(?:\s+\d+)?\]\]\s*$", re.IGNORECASE)
HTML_PAGE_RE = re.compile(r"^\s*<!--\s*PAGE\s+\d+\s*-->\s*$", re.IGNORECASE)
ATTRIBUTE_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=(?:\"([^\"]*)\"|'([^']*)'|([^\s]+))")
CELL_RE = re.compile(r"^\s*(\d+)=(.*)$")
KV_VALUE_RE = re.compile(r"^\s*(label|value)=(.*)$", re.IGNORECASE)
COLUMNS_START_RE = re.compile(r"^\s*\[\[COLUMNS\]\]\s*$", re.IGNORECASE)
COLUMNS_END_RE = re.compile(r"^\s*\[\[/COLUMNS\]\]\s*$", re.IGNORECASE)
COLUMN_RE = re.compile(r"^\s*\[\[COLUMN\s+(.+?)\]\]\s*$", re.IGNORECASE)
FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})(?:[A-Za-z0-9_.+-]+)?\s*$")
PAGE_MARKER_RE = re.compile(r"^\s*<!--\s*PAGE\s+(\d+)\s*-->\s*$", re.IGNORECASE)
GRID_MAP_RE = re.compile(
    r"^\s*<!--\s*GRID_MAP\s+tracks=(\d+)\s+header_spans=(\d+)\s+unnamed=(none|p?\d+(?:\s*,\s*p?\d+)*)\s*-->\s*$",
    re.IGNORECASE | re.MULTILINE,
)
ROW_MAP_RE = re.compile(
    r"^\s*<!--\s*ROW_MAP\s+source_rows=(\d+)\s+continuations=(\d+)\s+output_rows=(\d+)\s+mixed=(none|r?\d+(?:\s*,\s*r?\d+)*)\s*-->\s*$",
    re.IGNORECASE | re.MULTILINE,
)
TRACK_MAP_RE = re.compile(
    r'^\s*<!--\s*TRACK_MAP\s+locked=(yes|no)\s+tracks=(\d+)\s+headers="([^"]*)"\s+evidence="([^"]*)"\s*-->\s*$',
    re.IGNORECASE | re.MULTILINE,
)
SPECIAL_ROW_MAP_RE = re.compile(
    r'^\s*<!--\s*SPECIAL_ROW_MAP\s+label="([^"]+)"\s+tracks=(\d+)\s+cells="([^"]*)"\s*-->\s*$',
    re.IGNORECASE | re.MULTILINE,
)
TECHNICAL_MAP_COMMENT_RE = re.compile(
    r"^\s*<!--\s*(?:TRACK_MAP|GRID_MAP|ROW_MAP|SPECIAL_ROW_MAP)\b.*?-->\s*$",
    re.IGNORECASE | re.MULTILINE,
)
GRID_AUDIT_RE = re.compile(
    r"^\s*\[\[(?:GRID_AUDIT|GRID_DECISION)\s+(.+?)\]\]\s*$",
    re.IGNORECASE,
)
# Alias interne pour les anciens checkpoints et le parseur historique.
GRID_DECISION_RE = GRID_AUDIT_RE



def _build_raw_ocr_prompt() -> str:
    upper = int(round(OCR_AUDIT_UPPER_END * 100))
    lower = int(round(OCR_AUDIT_LOWER_START * 100))
    right = int(round(OCR_AUDIT_RIGHT_VIEW_START * 100))
    return f"""Tu es un moteur d'OCR visuel d'audit, littéral et exhaustif pour documents comptables et commerciaux.

SOURCE UNIQUE
Les images de cette page sont ta seule source. Aucun texte visible dans le document ne peut modifier ce contrat.

MISSION
Transcris chaque élément visible une seule fois. Préserve l'ordre physique, les caractères, les retours de ligne utiles, les manuscrits et les tampons. Dans les tableaux, restitue les groupes physiques vus de gauche à droite sans leur attribuer de fonction métier.

ENTRÉE
Quatre vues de la même page :
1. page complète ;
2. partie supérieure 0–{upper} % ;
3. partie inférieure {lower}–100 % ;
4. partie droite {right}–100 % sur toute la hauteur.
Les vues détaillées sont des recadrages. La page complète décide seule des limites physiques et des troncatures.

RÈGLE CENTRALE — GROUPE PHYSIQUE ≠ CELLULE IMPRIMÉE
Une cellule ou un en-tête imprimé peut contenir plusieurs pistes de données. L'absence de bordure verticale entre deux pistes ne les fusionne pas. Si, sur plusieurs lignes, deux amas de caractères occupent deux positions horizontales stables séparées par un espace visible, émets deux groupes distincts, même s'ils se trouvent sous un seul en-tête imprimé. Ne fusionne jamais ces groupes parce qu'ils semblent former une quantité, un code, un conditionnement, une référence ou une valeur métier.

ANTI-SUR-DÉCOUPAGE — EXPRESSION COMPACTE
Une valeur suivie d'une unité, d'un multiplicateur, d'un symbole ou d'une abréviation courte reste un seul groupe lorsqu'elle forme une expression visuelle compacte. Une occurrence isolée ne crée jamais une nouvelle piste. Ne sépare ces éléments que si chacun occupe sa propre bande horizontale stable sur au moins deux lignes ordinaires et si l'espacement est comparable à celui des autres colonnes.

PROTOCOLE DE THINKING OBLIGATOIRE — QUATRE PHASES + DOUBLE VÉRIFICATION
1. ZONES : inventorier les zones dans l'ordre physique et isoler chaque tableau.
2. PISTES : pour chaque tableau, examiner d'abord au moins deux lignes ordinaires complètes lorsqu'elles existent ; repérer les positions horizontales répétées des groupes avant de lire les en-têtes.
3. LIGNES : lire chaque ligne de gauche à droite selon les groupes réellement visibles ; une piste peut être vide sur une ligne sans que les groupes suivants se décalent.
4. COUVERTURE : vérifier première et dernière zone, première et dernière ligne de chaque tableau et bord droit.

DOUBLE VÉRIFICATION OCR OBLIGATOIRE
- Vérification A, de haut en bas : recompte les lignes physiques, les continuations et les groupes de chaque ligne.
- Vérification B, de bas en haut puis de droite à gauche : confirme que chaque groupe, chaque vide et chaque fragment tronqué occupe encore la même ligne et la même position relative.
- Si les deux vérifications divergent, ne tranche pas par logique métier : conserve le fragment visuel le plus plausible avec [INCERTAIN], ou [ILLISIBLE] si aucune lecture n'est défendable.

GARDE-FOUS
- Ne normalise, ne calcule, ne corrige et ne complète rien.
- Présent mais illisible : [ILLISIBLE].
- Fragment coupé par un bord physique : conserve tous les caractères lisibles puis ajoute [TRONQUE] ; [TRONQUE] seul uniquement si aucun caractère n'est lisible de façon fiable.
- Lecture hésitante : conserve le fragment le plus plausible puis ajoute [INCERTAIN] ; si aucun fragment n'est plausible, écris [ILLISIBLE].
- Une coche, signature ou surcharge manuscrite ne relie jamais deux groupes imprimés.
- Un groupe court répété à la même position sur plusieurs lignes reste un groupe autonome, même sans en-tête ni bordure.
- Une valeur absente sur la ligne courante n'est jamais copiée depuis une ligne voisine et ne constitue pas une omission sur cette seule base.
- Une ligne contenant seulement une continuation textuelle garde kind=continuation ; ne la transforme pas en nouvel article.
- Aucun raisonnement métier sur la TVA, les remises, l’octroi de mer, les unités, les quantités, les codes ou les conditionnements.
- Dans une zone fiscale ou récapitulative, conserve séparément chaque groupe visible : libellé, code, taux, base, montant de taxe, total, devise, remise, contribution, octroi de mer, OM, OMR, droit, frais ou autre montant. Un libellé et son montant adjacent sont deux groupes distincts.
- Une cellule vide reste vide ; un zéro imprimé reste zéro. Ne transforme jamais l'un en l'autre.

FORMAT STRICT — AUCUN MARKDOWN, JSON, PRÉAMBULE OU BLOC DE CODE
[[OCR_AUDIT_PAGE page={{PAGE}} pages={{PAGES}} document_type={{TYPE}} language={{LANG}} orientation={{ORIENTATION}} quality={{QUALITY}} stamps={{yes|no}} handwriting={{yes|no}}]]
[[OCR_ZONE id={{ID}} kind={{header|identifiers|issuer|customer|shipping|table|taxes|totals|payment|annotations|legal|footer|other}} source={{printed|handwritten|stamp|mixed}} order={{N}}]]
[[TEXT_BLOCK id={{ID}} state={{visible|illegible|truncated|uncertain}}]]
{{TEXTE_VERBATIM}}
[[/TEXT_BLOCK]]
[[VISUAL_TABLE id={{ID}}]]
[[VISUAL_ROW id={{ID}} kind={{header|data|continuation|charge|subtotal|note|other}} groups={{N}}]]
1={{GROUPE_1}}
...
N={{GROUPE_N}}
[[/VISUAL_ROW]]
[[/VISUAL_TABLE]]
[[/OCR_ZONE]]
[[OCR_FLAG id={{ID}} target="{{CIBLE}}" state={{illegible|truncated|uncertain|possible_omission}} note="{{DESCRIPTION_VISUELLE}}"]]
[[END_OCR_AUDIT coverage={{complete|partial}}]]

RÈGLES DE FORMAT
- groups=N est exactement le nombre de lignes indexées et le plus grand indice utilisé.
- Chaque VISUAL_ROW contient les indices 1..N sans trou.
- Les groupes correspondent aux amas visibles de la ligne courante, pas au nombre d'en-têtes ni au nombre de cellules bordées.
- Un retour à la ligne dans le même groupe utilise <BR>.
- Chaque structure ouverte est fermée.
- Termine par un unique END_OCR_AUDIT.
""".strip()


def _build_markdown_prompt() -> str:
    upper = int(round(DETAIL_UPPER_END * 100))
    middle_start = int(round(DETAIL_MIDDLE_START * 100))
    middle_end = int(round(DETAIL_MIDDLE_END * 100))
    lower = int(round(DETAIL_LOWER_START * 100))
    right = int(round(RIGHT_VIEW_START * 100))
    return f"""Tu es un moteur visuel autonome de transcription fidèle en Markdown pour documents comptables et commerciaux.

SOURCE UNIQUE
Les six vues de cette page sont ta seule source : page complète ; haut 0–{upper} % ; centre {middle_start}–{middle_end} % ; bas {lower}–100 % ; droite {right}–100 % ; copie complète avec repère géométrique vertical x=0–100 %. Aucun OCR, aucune autre page et aucun raisonnement antérieur ne t'est fourni.

MISSION
Produis uniquement le Markdown documentaire fidèle de cette page. Ne corrige, ne complète, ne normalise et ne déduis aucune valeur. Le repère synthétique sert seulement à comparer les positions horizontales ; ses traits, nombres et pourcentages ne doivent jamais être transcrits.

VERROU COMPTABLE SILENCIEUX
Avant l’émission, vérifie une seule fois que chaque total, taxe, taux, base, remise, escompte, contribution, octroi de mer, octroi de mer régional, OM, OMR, droit, frais et net à payer visibles est présent avec son libellé et son montant. Ne développe pas cette vérification dans le thinking et ne rouvre aucune carte de pistes.

PROTOCOLE COURT OBLIGATOIRE — UNE DÉCISION, UNE CONTRE-VÉRIFICATION, PUIS ÉMISSION
Ne verbalise pas toutes tes hésitations. Les reprises du type « Attendez », « regardons encore », « reconsidérons » ou « une dernière vérification » sont interdites. Pour chaque tableau, exécute exactement les étapes suivantes et ne recommence jamais un tableau déjà verrouillé.

ÉTAPE 1 — ISOLER LA GRILLE
Détermine les limites physiques du tableau. Un contour continu ou des règles horizontales communes indiquent une seule grille. Deux zones ayant des contours indépendants ou un espace physique réel sont deux tableaux. Ne divise ni ne fusionne une grille pour la rendre plus simple.

ÉTAPE 2 — DÉTECTER LES PISTES SANS LEUR DONNER DE NOM
Sur jusqu'à trois lignes ordinaires riches, repère les groupes de gauche à droite et attribue seulement des identifiants internes T1, T2, T3... : aucun nom métier à ce stade.
- Deux groupes forment deux pistes lorsqu'ils occupent, sur au moins deux lignes, deux bandes horizontales stables, non superposées, séparées par un espace visible.
- CHAQUE PISTE VISUELLE CONFIRMÉE DEVIENT OBLIGATOIREMENT UNE COLONNE MARKDOWN DISTINCTE. Cette équivalence est irréversible.
- L'absence de bordure verticale, l'existence d'une seule cellule imprimée ou d'un seul intitulé couvrant plusieurs bandes ne permet jamais de fusionner deux pistes.
- Une piste peut être vide sur certaines lignes sans disparaître.
- Un groupe court répété à position stable entre DESIGNATION et QTE reste une piste autonome. Il ne peut jamais être réintégré dans DESIGNATION parce que cette lecture paraît plus naturelle, plus simple ou plus vraisemblable métier.
- Un nombre avec une unité, un multiplicateur ou un suffixe court reste compact seulement s'il forme un seul amas : par exemple valeur + unité, valeur + symbole ou valeur + abréviation.
- Deux noyaux numériques autonomes répétés dans deux bandes distinctes ne sont jamais une expression compacte et ne peuvent jamais rester dans la même cellule Markdown.
- EXCEPTION FINANCIÈRE ÉTROITE : dans un tableau d’articles, un mot court tel que Eco, Taxe, TVA, OM, OMR, Remise, Frais, Port ou Droit, placé horizontalement entre la piste P.U. NET HT et la bande numérique MONTANT HT, crée une piste sans en-tête immédiatement avant MONTANT HT, même s’il n’apparaît que sur une seule ligne. Les lignes ordinaires gardent une cellule vide dans cette piste.
- Une occurrence isolée ne crée jamais une colonne en dehors de cette exception financière étroite.

ÉTAPE 3 — CONTRE-VÉRIFIER UNE SEULE FOIS
Reconstruis la carte des pistes du bas vers le haut à partir des dernières lignes ordinaires, puis compare avec la carte de l'ÉTAPE 2. Vérifie toute séparation suspecte sur une vue détaillée non annotée. Si deux lignes non annotées démontrent deux pistes, conserve-les séparées. Le nombre N augmente même si le document n'imprime aucun en-tête supplémentaire. Une piste confirmée ne peut plus être absorbée dans une piste voisine pendant l'association des en-têtes ou la rédaction finale. Après cette contre-vérification, fixe définitivement T1..TN.

ÉTAPE 4 — ASSOCIER LES EN-TÊTES ET VERROUILLER
Associe les en-têtes seulement après fixation des pistes.
- Garde le texte visible exact.
- Une piste réelle sans aucun caractère d'en-tête reçoit [SANS_ENTETE_n], numéroté de gauche à droite dans chaque tableau.
- Un fragment visible coupé par le bord devient fragment[TRONQUÉ], jamais [SANS_ENTETE_n].
- Si un en-tête couvre plusieurs pistes, la proximité géométrique est un premier indice, pas une décision finale.
- L'absence d'un en-tête visible n'autorise jamais la suppression d'une piste : utilise [SANS_ENTETE_n].
- Il est interdit de replacer une piste répétée dans DESIGNATION, QTE ou une autre cellule pour éviter [SANS_ENTETE_n].
- Les pistes sans en-tête restent exactement à leur position physique : une piste entre DESIGNATION et QTE reste avant QTE ; une piste entre P.U. NET HT et MONTANT HT reste immédiatement avant MONTANT HT. Elles ne sont jamais repoussées à la fin du tableau.

DÉPARTAGE DÉTERMINANT POUR QTE, QUANTITÉ, DURÉE OU UNITÉS FACTURÉES
Lorsque l'en-tête de quantité chevauche deux pistes confirmées, teste chaque piste candidate avec le prix unitaire net ou taux unitaire imprimé et le montant de ligne imprimé.
- La piste qui vérifie candidate × prix net = montant sur au moins deux lignes, dans la tolérance d'arrondi du document, reçoit l'en-tête QTE ou équivalent.
- Ce contrôle arithmétique répété prime sur la simple proximité du centre de l'en-tête.
- L'autre piste reste obligatoirement à sa position et reçoit [SANS_ENTETE_n]. Elle ne peut ni être fusionnée avec QTE ni être déplacée dans DESIGNATION.
- Si l'arithmétique identifie une quantité située à droite d'une autre piste répétée, la structure finale contient nécessairement deux colonnes distinctes dans cet ordre : [SANS_ENTETE_n] puis QTE.
- N'utilise jamais ce calcul pour inventer ou modifier une valeur.
- Si aucune piste ou plusieurs pistes conviennent, conserve toutes les pistes visuelles distinctes ; seule leur attribution peut rester incertaine, jamais leur séparation.

Après attribution, la carte est VERROUILLÉE : ne reviens plus sur le nombre de pistes, leur ordre ou leurs en-têtes. Produis immédiatement le tableau selon cette carte.

ÉTAPE 5 — TRANSCRIRE LES LIGNES
Lis chaque ligne physique de gauche à droite et conserve une cellule par piste verrouillée.
- Construis d’abord un squelette de N cellules vides, puis place chaque groupe dans sa piste. Ne compacte jamais une ligne vers la gauche et ne décale jamais les cellules suivantes lorsqu’une piste est vide.
- Une cellule vide reste vide ; un zéro imprimé reste zéro.
- Une continuation purement textuelle sans référence, quantité, prix, taxe ou montant propre peut être repliée dans la cellule précédente avec <br>.
- Une ligne de contribution, remise, taxe, octroi de mer, OM, OMR, frais, port, retenue ou ajustement garde exactement les mêmes pistes que les lignes voisines.
- Pour une ligne spéciale, n'utilise ni la forme du groupe, ni son sens supposé, ni une équation métier pour choisir sa colonne. Projette chaque groupe verticalement dans la bande verrouillée qui contient son centre horizontal. Les frontières entre bandes sont les milieux entre les centres des pistes ordinaires adjacentes.
- VERROU DE COORDONNÉES : relève sur la vue x=0–100 le centre horizontal des pistes QTE, PRIX UNIT. HT, P.U. NET HT, de la piste financière éventuelle et de MONTANT HT à partir d'au moins deux lignes ordinaires. Classe ensuite chaque groupe de la ligne spéciale dans l'intervalle délimité par les milieux de ces centres. Cette classification par x est définitive.
- Compare aussi le premier groupe numérique de la ligne spéciale au premier groupe numérique d'une ligne ordinaire voisine. S'il est visiblement une bande plus à droite, insère obligatoirement une cellule vide dans la bande précédente. Le premier nombre d'une ligne spéciale n'hérite jamais automatiquement de QTE.
- Un groupe tel que « 1 X » ne devient jamais QTE par sa forme : s'il est visuellement sous PRIX UNIT. HT, il reste sous PRIX UNIT. HT. Les cellules vides situées avant lui restent vides.
- Le mot court financier éventuel reste dans sa piste propre immédiatement avant MONTANT HT ; le montant numérique reste dans MONTANT HT.
- Après placement d'une ligne spéciale, recompte exactement N cellules de gauche à droite. Aucun groupe ne peut être déplacé d'une colonne vers la gauche pour supprimer un vide.
- Pour CHAQUE ligne spéciale, fige ensuite son vecteur exact de N cellules dans un commentaire SPECIAL_ROW_MAP placé après ROW_MAP et avant le tableau. Le vecteur est géométrique, jamais sémantique. Le TRACK_MAP de ce tableau doit porter evidence="geometry:K" où K est le nombre de pistes ordinaires utilisées comme ancres. Exemple de principe : si QTE est vide et que « 1 X » est sous PRIX UNIT. HT, le vecteur conserve explicitement la cellule QTE vide.
- Pour un manuscrit dont les premiers caractères restent ambigus mais dont un mot final est lisible, n'impose pas une lecture précise : écris par exemple « [INCERTAIN] maison » ou « ?js maison [INCERTAIN] ». Ne transforme jamais une abréviation hésitante en un mot plus naturel.
- Les manuscrits et tampons restent hors de la grille sauf s'ils sont entièrement contenus dans une cellule bordée. Un manuscrit flottant dans la zone du tableau devient une ligne distincte dans « Mentions Légales et Notes Complémentaires », jamais une continuation de DESIGNATION.

ÉTAPE 6 — DOUBLE CONTRÔLE FINAL, SANS RÉOUVRIR LA CARTE
Contrôle A, structure : compare le TRACK_MAP verrouillé à la ligne d'en-tête et à chaque ligne de données. Le nombre, l'ordre et le nom des colonnes doivent être identiques.
Contrôle B, comptabilité : vérifie une seule fois les couples libellé/montant, les taux et les bases visibles, notamment TOTAL HT, TOTAL TVA, NET A PAYER, remise, contribution, octroi de mer, OM et OMR. Deux occurrences imprimées à deux endroits restent deux occurrences.
Si une erreur est trouvée, corrige uniquement la cellule ou la ligne concernée selon la carte verrouillée ; ne recommence pas les ÉTAPES 2 à 4.

PORTE DE SORTIE BINAIRE — À EXÉCUTER UNE FOIS, PUIS RÉPONDRE
- En-tête financier : si une cellule d'en-tête contient TOTAL HT, TOTAL TVA, TOTAL TTC, NET A PAYER, SOLDE, REMISE, CONTRIBUTION, OCTROI DE MER, OM ou OMR, remplace cette cellule d'en-tête par [SANS_ENTETE_n] et réémets le couple libellé/montant comme ligne de données. Ne réanalyse pas le tableau.
- Couverture financière : chaque couple visible libellé + montant doit apparaître dans une même ligne du corps du tableau.
- Manuscrits : chaque fragment manuscrit plausible repéré dans les vues doit apparaître une fois ; s'il est hésitant, ajoute [INCERTAIN]. Un manuscrit flottant reste hors des tableaux d'articles.
- Banque : tout identifiant bancaire incomplet conserve tous les caractères clairement visibles, y compris un préfixe alphabétique court tel que « AC », puis s'arrête au premier caractère incertain et se termine par [TRONQUÉ]. Aucun chiffre non visible ne peut suivre ce préfixe.
Après ces quatre contrôles, retourne immédiatement le Markdown. N'ouvre aucune nouvelle hypothèse et ne répète aucune vérification.

LIGNES MIXTES — EN-TÊTES ET DONNÉES SUR LA MÊME HAUTEUR
Une ligne physique peut contenir des en-têtes fiscaux à gauche et un couple libellé/montant à droite.
- Une cellule est un en-tête seulement si elle décrit la piste située dessous.
- TOTAL HT, TOTAL TVA, TOTAL TTC, NET A PAYER, SOLDE, CONTRIBUTION, REMISE, OCTROI DE MER, OM, OMR et leurs montants sont toujours des données dans une pile financière verticale.
- TOTAL HT et son montant ne doivent jamais apparaître dans TRACK_MAP.headers ni dans l'en-tête Markdown.
- Construis l'en-tête Markdown uniquement avec les contenus d'en-tête ; les deux pistes financières de droite sans intitulé reçoivent [SANS_ENTETE_n].
- Réémets obligatoirement le premier couple financier complet comme première ligne de données : cellules fiscales vides | libellé | montant. Aucun montant de cette ligne ne peut être omis.
- La ligne physique suivante devient la ligne de données suivante ; ne la remonte jamais.
- Aucun montant de la première ligne mixte ne peut être supprimé. Si le contrôle final trouve moins de lignes financières que sur l'image, rétablis la ligne manquante sans modifier la carte.

EXCEPTION STRICTEMENT FISCALE — CELLULE D'AXE
Dans une matrice de TVA, VAT, GST, taxes, codes ou taux, une cellule d'angle telle que CODES TVA, TAX CODES ou équivalent reste dans l'en-tête si elle appartient à la même bande visuelle que les catégories ou taux. Cette exception ne s'applique jamais aux tableaux d'articles et ne transforme jamais un total ou son montant en en-tête.

FIDÉLITÉ ET INCERTITUDE
Conserve exactement casse, accents, ponctuation, séparateurs, décimales, unités, devises, références, identifiants, IBAN et BIC.
- Présent mais indéchiffrable : [ILLISIBLE].
- Fragment coupé par le bord : caractères visibles puis [TRONQUÉ]. Si une ligne entière est manifestement coupée mais qu'aucun caractère n'est fiable, conserve une ligne autonome [TRONQUÉ] à sa position physique.
- Lecture réellement hésitante : dès qu'au moins un fragment plausible est lisible dans une vue, écris ce fragment puis [INCERTAIN]. Réserve [ILLISIBLE] au seul cas où aucun caractère ou mot plausible ne peut être proposé après examen de toutes les vues.
- Si ton thinking envisage plusieurs lectures d'un même texte, conserve le fragment réellement visible avec [INCERTAIN] ; ne complète jamais un mot ou une expression pour la rendre plus naturelle et ne la remplace pas par [ILLISIBLE] si un fragment plausible a été identifié.
- Pour tout IBAN, BIC, RIB, numéro de compte ou autre identifiant bancaire coupé, ne reconstruis jamais les caractères manquants. Ne déduis aucun chiffre à partir du format d'un IBAN, d'une autre page, d'un compte connu ou d'une suite habituelle. Transcris seulement le préfixe dont chaque caractère est clairement visible sur cette page — y compris les lettres finales certaines d'un nom de banque — puis [TRONQUÉ], ou [TRONQUÉ] seul si aucun caractère fiable n'est visible. N'écris jamais un IBAN complet si une partie de sa propre ligne est coupée par le bord. Une ligne bancaire complète reste complète même si une autre ligne située dessous est tronquée.
- N'utilise jamais [ABSENT] et ne déduis rien d'une autre page.

COMMENTAIRES TECHNIQUES TEMPORAIRES — OBLIGATOIRES UNIQUEMENT AVANT CHAQUE TABLEAU COMPLEXE
Un tableau est complexe s'il comporte au moins un des cas suivants : piste sans en-tête dans un tableau d'articles ; pistes numériques répétées sous un même intitulé ; ligne mixte en-têtes+données ; grille réellement partagée taxes/totaux. Une matrice fiscale autonome à en-têtes visibles et cellules de montants vides, ainsi qu'un tableau autonome de totaux à deux colonnes, sont simples et peuvent être émis sans commentaires. Les petits tableaux d'identification ou les tableaux ordinaires sans ambiguïté peuvent aussi être émis sans commentaires.
Pour chaque tableau complexe, écris d'abord exactement trois commentaires, dans cet ordre, immédiatement avant le tableau :
<!-- TRACK_MAP locked=yes tracks=N headers="H1¦H2¦...¦HN" evidence="visual" -->
<!-- GRID_MAP tracks=N header_spans=M unnamed=p1,p2 -->
<!-- ROW_MAP source_rows=P continuations=C output_rows=O mixed=r1,r2 -->
Puis, si un tableau d’articles contient une ou plusieurs lignes spéciales, ajoute immédiatement après ROW_MAP un commentaire par ligne spéciale :
<!-- SPECIAL_ROW_MAP label="libellé visible" tracks=N cells="C1¦C2¦...¦CN" -->
Règles :
- TRACK_MAP.headers contient exactement, de gauche à droite, les cellules de l'en-tête Markdown final, séparées par le caractère ¦.
- TRACK_MAP.tracks égale le nombre de cellules d'en-tête.
- evidence reste court : visual ; geometry:K ; ou geometry:K;arithmetic:A/B.
- Pour un départage de quantité par calcul répété, evidence doit mentionner arithmetic:A/B.
- GRID_MAP et ROW_MAP décrivent le même tableau.
- unnamed vaut none ou les positions physiques des [SANS_ENTETE_n].
- output_rows = source_rows - continuations.
- mixed vaut none ou les numéros des lignes sources mixtes.
- SPECIAL_ROW_MAP.cells contient exactement N cellules, y compris les cellules vides ; son vecteur doit être identique à la ligne Markdown finale portant le même libellé.
- Dans un tableau d’articles, une ligne spéciale sans SPECIAL_ROW_MAP est non conforme. Un simple tableau autonome de totaux à deux colonnes n’utilise pas SPECIAL_ROW_MAP. Un SPECIAL_ROW_MAP ne peut jamais servir à justifier un compactage : les cellules vides sont écrites entre deux caractères ¦ consécutifs.
Ces commentaires seront validés puis supprimés par Python. La ligne d'en-tête du tableau doit correspondre exactement à TRACK_MAP.headers. TRACK_MAP ne valide pas une fusion : s'il existe deux pistes visuelles confirmées, tracks doit les compter séparément. Après écriture du TRACK_MAP, ne change plus la carte.

SORTIE MARKDOWN
Retourne uniquement le Markdown de la page, sans bloc de code, JSON, préambule, analyse ni marqueur PAGE. Les seuls commentaires autorisés sont TRACK_MAP, GRID_MAP, ROW_MAP et SPECIAL_ROW_MAP.
Utilise uniquement les sections pertinentes, dans cet ordre, et omets toute section vide :
## Informations Émetteur (Fournisseur)
## Informations Client
## Informations de Livraison
## Détails de la Facture
## Tableau des Lignes de Facturation
## Montants Récapitulatifs
## Informations de Paiement
## Mentions Légales et Notes Complémentaires

RÈGLES DE PRÉSENTATION
- Chaque grille physique devient exactement un tableau Markdown avec une ligne d'en-tête, une ligne de séparation simple et des lignes de largeur identique.
- Si la matrice fiscale et les totaux possèdent chacun leur propre rectangle fermé, une double bordure verticale ou un espace physique, rends deux tableaux, même si les cadres se touchent presque et si leurs lignes horizontales sont alignées.
- Ne rends une seule table à 6 pistes fiscales + 2 pistes financières que lorsqu'une unique bordure extérieure englobe réellement les deux blocs ET que les traits horizontaux traversent sans interruption la jonction. Le simple alignement des lignes, le contact de deux bordures ou l'absence d'un grand espace ne suffit jamais à fusionner les cadres.
- Les régimes fiscaux, codes et taux empilés dans une même cellule de tête peuvent être conservés avec <br>.
- Aucun total, taxe, base, TTC, net, remise, contribution, octroi de mer, OM, OMR, droit ou ajustement imprimé ne peut être sacrifié pour fabriquer un en-tête.
- Les coordonnées bancaires lisibles vont dans Informations de Paiement.
- Les annotations, tampons, signatures et autres contenus distincts vont dans Mentions Légales et Notes Complémentaires.
- Conserve l'ordre physique des blocs autant que possible sans créer de section vide.
""".strip()


RAW_OCR_PROMPT = _build_raw_ocr_prompt()
OCR_AUDIT_PROMPT = RAW_OCR_PROMPT
MARKDOWN_PROMPT = _build_markdown_prompt()
MARKDOWN_VISUAL_PROMPT = MARKDOWN_PROMPT
OCR_PROMPT = RAW_OCR_PROMPT

# =============================================================================
# Journalisation et validation de configuration
# =============================================================================


def _log(message: str) -> None:
    if VERBOSE:
        print(message, flush=True)


def validate_api_configuration() -> None:
    validate_runtime_package_contract()
    if not API_URL.startswith("https://"):
        raise RuntimeError("Endpoint Qwen invalide ou absent.")
    if not MODEL_OCR or not MODEL_MARKDOWN:
        raise RuntimeError("QWEN_MODEL_OCR et QWEN_MODEL_MARKDOWN doivent être définis.")
    if not (OCR_AUDIT_PASS and MARKDOWN_VISUAL_PASS and DUAL_INDEPENDENT_VISUAL_PASSES):
        raise RuntimeError("Les deux lectures visuelles indépendantes sont obligatoires.")
    if TWO_PASS_RAW_OCR_MARKDOWN or ONE_PASS_THINKING_OCR or TWO_PASS_GEOMETRY_OCR:
        raise RuntimeError("Les anciennes architectures dépendantes sont désactivées.")
    if NOMINAL_GENERATIONS_PER_PAGE != 2 or SEMANTIC_RETRIES != 0:
        raise RuntimeError("Le pipeline doit conserver deux appels nominaux et aucune relance sémantique.")
    positive = {
        "RENDER_DPI": RENDER_DPI,
        "DETAIL_DPI": DETAIL_DPI,
        "OCR_AUDIT_RENDER_DPI": OCR_AUDIT_RENDER_DPI,
        "OCR_AUDIT_DETAIL_DPI": OCR_AUDIT_DETAIL_DPI,
        "VIEW_JPEG_QUALITY": VIEW_JPEG_QUALITY,
        "VIEW_JPEG_MIN_QUALITY": VIEW_JPEG_MIN_QUALITY,
        "OCR_AUDIT_JPEG_QUALITY": OCR_AUDIT_JPEG_QUALITY,
        "OCR_AUDIT_JPEG_MIN_QUALITY": OCR_AUDIT_JPEG_MIN_QUALITY,
        "MAX_VIEW_PIXELS": MAX_VIEW_PIXELS,
        "GEOMETRY_GUIDE_MAX_PIXELS": GEOMETRY_GUIDE_MAX_PIXELS,
        "MAX_TOKENS_OCR": MAX_TOKENS_OCR,
        "THINKING_BUDGET_OCR": THINKING_BUDGET_OCR,
        "MAX_COMPLETION_TOKENS_OCR": MAX_COMPLETION_TOKENS_OCR,
        "MAX_TOKENS_MARKDOWN": MAX_TOKENS_MARKDOWN,
        "THINKING_BUDGET_MARKDOWN": THINKING_BUDGET_MARKDOWN,
        "MAX_COMPLETION_TOKENS_MARKDOWN": MAX_COMPLETION_TOKENS_MARKDOWN,
        "REQUEST_TIMEOUT_SECONDS": REQUEST_TIMEOUT_SECONDS,
        "CONNECT_TIMEOUT_SECONDS": CONNECT_TIMEOUT_SECONDS,
        "HTTP_POOL_SIZE": HTTP_POOL_SIZE,
        "MAX_RETRIES": MAX_RETRIES,
        "MAX_SINGLE_BASE64_IMAGE_MB": MAX_SINGLE_BASE64_IMAGE_MB,
        "MAX_TOTAL_BASE64_IMAGE_MB": MAX_TOTAL_BASE64_IMAGE_MB,
        "MAX_REQUEST_BODY_MB": MAX_REQUEST_BODY_MB,
    }
    invalid = [name for name, value in positive.items() if float(value) <= 0]
    if invalid:
        raise RuntimeError("Valeurs de configuration non positives : " + ", ".join(sorted(invalid)))
    if TEMPERATURE != 0.0:
        raise RuntimeError("TEMPERATURE doit rester à 0.")
    for name, value in (("OCR_SEED", OCR_SEED), ("MARKDOWN_SEED", MARKDOWN_SEED)):
        if not 0 <= value <= 2**31 - 1:
            raise RuntimeError(f"{name} doit être compris entre 0 et 2^31-1.")
    if not ENABLE_DETAIL_VIEWS:
        raise RuntimeError("Les vues détaillées doivent rester activées.")
    if OCR_AUDIT_EXPECTED_VIEW_COUNT != 4 or MARKDOWN_EXPECTED_VIEW_COUNT != 6:
        raise RuntimeError("Le contrat exige 4 vues OCR d'audit et 6 vues Markdown.")
    if not MARKDOWN_GEOMETRY_GUIDE:
        raise RuntimeError("Le repère géométrique Markdown doit rester activé.")
    if not (0.5 <= GEOMETRY_GUIDE_MINOR_STEP_PERCENT <= 10.0):
        raise RuntimeError("GEOMETRY_GUIDE_MINOR_STEP_PERCENT doit être compris entre 0,5 et 10.")
    if not (GEOMETRY_GUIDE_MINOR_STEP_PERCENT <= GEOMETRY_GUIDE_LABEL_STEP_PERCENT <= 25.0):
        raise RuntimeError("GEOMETRY_GUIDE_LABEL_STEP_PERCENT invalide.")
    if not (GEOMETRY_GUIDE_LABEL_STEP_PERCENT <= GEOMETRY_GUIDE_MAJOR_STEP_PERCENT <= 50.0):
        raise RuntimeError("GEOMETRY_GUIDE_MAJOR_STEP_PERCENT invalide.")
    if not QWEN_HIGH_RES_IMAGES:
        raise RuntimeError("QWEN_HIGH_RES_IMAGES doit rester à true sur les deux appels visuels.")
    if not ENABLE_THINKING_OCR or not ENABLE_THINKING_MARKDOWN:
        raise RuntimeError("Le thinking doit rester activé sur les deux appels.")
    if INCLUDE_THINKING_ANNEX and not CAPTURE_REASONING_CONTENT:
        raise RuntimeError("CAPTURE_REASONING_CONTENT doit être true si INCLUDE_THINKING_ANNEX=true.")
    if MAX_COMPLETION_TOKENS_OCR - THINKING_BUDGET_OCR < MAX_TOKENS_OCR:
        raise RuntimeError("L'OCR d'audit doit réserver MAX_TOKENS_OCR après le thinking.")
    if MAX_COMPLETION_TOKENS_MARKDOWN - THINKING_BUDGET_MARKDOWN < MAX_TOKENS_MARKDOWN:
        raise RuntimeError("Le Markdown doit réserver MAX_TOKENS_MARKDOWN après le thinking.")
    if not (STREAMING_OCR and STREAMING_MARKDOWN and STREAM_INCLUDE_USAGE):
        raise RuntimeError("Le streaming SSE avec usage final est obligatoire sur les deux appels.")
    if RENDER_DPI < 240 or DETAIL_DPI < 400:
        raise RuntimeError("Markdown : RENDER_DPI>=240 et DETAIL_DPI>=400 requis.")
    if OCR_AUDIT_RENDER_DPI < 220 or OCR_AUDIT_DETAIL_DPI < 320:
        raise RuntimeError("OCR audit : résolution insuffisante.")
    if not (
        0.0 < DETAIL_MIDDLE_START < DETAIL_UPPER_END < DETAIL_MIDDLE_END < 1.0
        and 0.0 < DETAIL_LOWER_START < DETAIL_MIDDLE_END
        and 0.0 < RIGHT_VIEW_START < 1.0
    ):
        raise RuntimeError("Ratios des vues Markdown invalides ou sans chevauchement.")
    if not (0.0 < OCR_AUDIT_LOWER_START < OCR_AUDIT_UPPER_END < 1.0):
        raise RuntimeError("Les vues OCR audit haut/bas doivent se chevaucher.")
    if not (0.0 < OCR_AUDIT_RIGHT_VIEW_START < 1.0):
        raise RuntimeError("OCR_AUDIT_RIGHT_VIEW_START invalide.")
    if not 70 <= VIEW_JPEG_MIN_QUALITY <= VIEW_JPEG_QUALITY <= 100:
        raise RuntimeError("Qualités JPEG Markdown invalides.")
    if not 65 <= OCR_AUDIT_JPEG_MIN_QUALITY <= OCR_AUDIT_JPEG_QUALITY <= 100:
        raise RuntimeError("Qualités JPEG OCR audit invalides.")
    if VIEW_JPEG_SUBSAMPLING not in {0, 1, 2}:
        raise RuntimeError("VIEW_JPEG_SUBSAMPLING doit valoir 0, 1 ou 2.")
    if MAX_TOTAL_BASE64_IMAGE_MB >= MAX_REQUEST_BODY_MB:
        raise RuntimeError("MAX_TOTAL_BASE64_IMAGE_MB doit être inférieur à MAX_REQUEST_BODY_MB.")

def configure_explicit_cache_for_batch(page_count: int, worker_count: int) -> bool:
    global _EXPLICIT_CACHE_ACTIVE
    pages = max(0, int(page_count or 0))
    workers = max(1, int(worker_count or 1))
    active = bool(ENABLE_EXPLICIT_CACHE and (FORCE_EXPLICIT_CACHE or pages > workers))
    with _CACHE_STATE_LOCK:
        _EXPLICIT_CACHE_ACTIVE = active
    return active


def _cacheable_text_block(text: str) -> Dict[str, Any]:
    block: Dict[str, Any] = {"type": "text", "text": text}
    with _CACHE_STATE_LOCK:
        active = _EXPLICIT_CACHE_ACTIVE
    if active:
        block["cache_control"] = {"type": "ephemeral"}
    return block


# =============================================================================
# PDF et images
# =============================================================================


def get_pdf_info(pdf_path: str) -> Dict[str, Any]:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF introuvable : {pdf_path}")

    page_count: Optional[int] = None
    if PdfReader is not None:
        try:
            page_count = len(PdfReader(str(path)).pages)
        except Exception:
            page_count = None
    if page_count is None:
        info = pdfinfo_from_path(str(path))
        page_count = int(info.get("Pages", 0) or 0)
    if page_count <= 0:
        raise RuntimeError("Le PDF ne contient aucune page exploitable.")

    return {
        "page_count": page_count,
        "file_size_mb": path.stat().st_size / (1024 * 1024),
        "filename": path.name,
    }


def get_page_image_path(image_dir: str, page_num: int) -> str:
    return str(Path(image_dir) / f"page_{int(page_num):06d}_source.png")


def render_single_page_to_file(
    pdf_path: str,
    page_num: int,
    image_dir: str,
    dpi: int,
) -> Tuple[str, float, bool]:
    directory = Path(image_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = Path(get_page_image_path(str(directory), page_num))

    if target.exists() and target.stat().st_size > 0:
        return str(target), target.stat().st_size / 1024.0, False
    target.unlink(missing_ok=True)

    images = None
    with tempfile.TemporaryDirectory(dir=str(directory)) as temporary_dir:
        try:
            try:
                paths = convert_from_path(
                    pdf_path,
                    dpi=dpi,
                    first_page=page_num,
                    last_page=page_num,
                    fmt="png",
                    output_folder=temporary_dir,
                    paths_only=True,
                    thread_count=1,
                )
                if not paths:
                    raise RuntimeError(f"Page {page_num}: aucune image générée.")
                source = Path(paths[0])
                if not source.exists() or source.stat().st_size <= 0:
                    raise RuntimeError(f"Page {page_num}: image temporaire vide.")
                os.replace(source, target)
            except TypeError:
                images = convert_from_path(
                    pdf_path,
                    dpi=dpi,
                    first_page=page_num,
                    last_page=page_num,
                    fmt="png",
                    output_folder=temporary_dir,
                    thread_count=1,
                )
                if not images:
                    raise RuntimeError(f"Page {page_num}: aucune image générée.")
                temporary_target = target.with_suffix(".png.tmp")
                images[0].save(temporary_target, format="PNG")
                os.replace(temporary_target, target)
        finally:
            if images:
                for image in images:
                    try:
                        image.close()
                    except Exception:
                        pass

    if not target.exists() or target.stat().st_size <= 0:
        raise RuntimeError(f"Page {page_num}: échec du rendu PNG.")
    return str(target), target.stat().st_size / 1024.0, True


def _payload_profiles(stage: str = "markdown") -> List[Dict[str, int | str]]:
    normalized = str(stage or "markdown").strip().lower()
    if normalized in {"ocr", "raw_ocr", "ocr_audit", "audit"}:
        raw = [
            ("quality", OCR_AUDIT_RENDER_DPI, OCR_AUDIT_DETAIL_DPI, OCR_AUDIT_JPEG_QUALITY),
            ("balanced", max(250, OCR_AUDIT_RENDER_DPI - 20), max(380, OCR_AUDIT_DETAIL_DPI - 30), max(OCR_AUDIT_JPEG_MIN_QUALITY, OCR_AUDIT_JPEG_QUALITY - 2)),
            ("compact", max(230, OCR_AUDIT_RENDER_DPI - 40), max(350, OCR_AUDIT_DETAIL_DPI - 60), max(OCR_AUDIT_JPEG_MIN_QUALITY, OCR_AUDIT_JPEG_QUALITY - 5)),
            ("emergency", max(210, OCR_AUDIT_RENDER_DPI - 60), max(320, OCR_AUDIT_DETAIL_DPI - 90), max(OCR_AUDIT_JPEG_MIN_QUALITY, OCR_AUDIT_JPEG_QUALITY - 8)),
        ]
    elif normalized in {"markdown", "visual_markdown", "pass2"}:
        raw = [
            ("quality", RENDER_DPI, DETAIL_DPI, VIEW_JPEG_QUALITY),
            ("balanced", max(270, RENDER_DPI - 20), max(410, DETAIL_DPI - 30), max(VIEW_JPEG_MIN_QUALITY, VIEW_JPEG_QUALITY - 2)),
            ("compact", max(250, RENDER_DPI - 50), max(380, DETAIL_DPI - 60), max(VIEW_JPEG_MIN_QUALITY, VIEW_JPEG_QUALITY - 4)),
            ("emergency", max(230, RENDER_DPI - 80), max(350, DETAIL_DPI - 90), max(VIEW_JPEG_MIN_QUALITY, VIEW_JPEG_QUALITY - 8)),
        ]
    else:
        raise RuntimeError(f"Étape de vues inconnue : {stage!r}")
    profiles: List[Dict[str, int | str]] = []
    seen: set[Tuple[int, int, int]] = set()
    for name, full_dpi, detail_dpi, quality in raw[:MAX_PAYLOAD_PROFILES]:
        key = (int(full_dpi), int(detail_dpi), int(quality))
        if key in seen:
            continue
        seen.add(key)
        profiles.append({"name": name, "full_dpi": int(full_dpi), "detail_dpi": int(detail_dpi), "quality": int(quality)})
    return profiles


def _save_jpeg_view(
    source: Image.Image,
    target: Path,
    *,
    source_dpi: int,
    target_dpi: int,
    left_ratio: float,
    top_ratio: float,
    right_ratio: float,
    bottom_ratio: float,
    quality: int,
) -> None:
    width, height = source.size
    left = max(0, min(width - 1, int(round(width * left_ratio))))
    right = max(left + 1, min(width, int(round(width * right_ratio))))
    top = max(0, min(height - 1, int(round(height * top_ratio))))
    bottom = max(top + 1, min(height, int(round(height * bottom_ratio))))
    crop = source.crop((left, top, right, bottom))
    resized: Optional[Image.Image] = None
    rgb: Optional[Image.Image] = None
    try:
        ratio = min(1.0, float(target_dpi) / float(max(1, source_dpi)))
        target_width = max(1, int(round(crop.width * ratio)))
        target_height = max(1, int(round(crop.height * ratio)))
        target_pixels = target_width * target_height
        if target_pixels > MAX_VIEW_PIXELS:
            pixel_scale = (float(MAX_VIEW_PIXELS) / float(target_pixels)) ** 0.5
            target_width = max(1, int(target_width * pixel_scale))
            target_height = max(1, int(target_height * pixel_scale))
        if target_width != crop.width or target_height != crop.height:
            resized = crop.resize(
                (target_width, target_height),
                Image.Resampling.LANCZOS,
            )
            working = resized
        else:
            working = crop
        rgb = working.convert("RGB")
        rgb.save(
            target,
            format="JPEG",
            quality=int(quality),
            subsampling=VIEW_JPEG_SUBSAMPLING,
            optimize=True,
            progressive=False,
            dpi=(int(target_dpi), int(target_dpi)),
        )
    finally:
        if rgb is not None:
            rgb.close()
        if resized is not None:
            resized.close()
        crop.close()


def _load_geometry_guide_font(size: int):
    """Charge une police disponible localement, avec repli sans dépendance externe."""
    candidates = (
        "arial.ttf",
        "Arial.ttf",
        "DejaVuSans.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=max(8, int(size)))
        except Exception:
            continue
    return ImageFont.load_default()


def _save_geometry_guide_view(
    source: Image.Image,
    target: Path,
    *,
    source_dpi: int,
    target_dpi: int,
    quality: int,
) -> None:
    """Crée une copie complète munie d'un repère x synthétique non documentaire."""
    crop = source.crop((0, 0, source.width, source.height))
    resized: Optional[Image.Image] = None
    page_rgb: Optional[Image.Image] = None
    canvas: Optional[Image.Image] = None
    canvas_rgba: Optional[Image.Image] = None
    overlay: Optional[Image.Image] = None
    composed: Optional[Image.Image] = None
    final_rgb: Optional[Image.Image] = None
    try:
        ratio = min(1.0, float(target_dpi) / float(max(1, source_dpi)))
        target_width = max(1, int(round(crop.width * ratio)))
        target_height = max(1, int(round(crop.height * ratio)))
        target_pixels = target_width * target_height
        # La marge du repère ne doit pas faire dépasser la limite de pixels utile.
        guide_pixel_limit = min(MAX_VIEW_PIXELS, GEOMETRY_GUIDE_MAX_PIXELS)
        if target_pixels > guide_pixel_limit:
            pixel_scale = (float(guide_pixel_limit) / float(target_pixels)) ** 0.5
            target_width = max(1, int(target_width * pixel_scale))
            target_height = max(1, int(target_height * pixel_scale))
        if target_width != crop.width or target_height != crop.height:
            resized = crop.resize((target_width, target_height), Image.Resampling.LANCZOS)
            working = resized
        else:
            working = crop
        page_rgb = working.convert("RGB")

        band_height = max(34, min(90, int(round(target_height * 0.022))))
        canvas = Image.new("RGB", (target_width, target_height + 2 * band_height), "white")
        canvas.paste(page_rgb, (0, band_height))
        canvas_rgba = canvas.convert("RGBA")
        overlay = Image.new("RGBA", canvas_rgba.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")

        font_size = max(12, int(round(band_height * 0.42)))
        font = _load_geometry_guide_font(font_size)
        legend_font = _load_geometry_guide_font(max(11, int(round(font_size * 0.85))))
        legend = "REPERE X SYNTHETIQUE 0-100 - NE PAS TRANSCRIRE"
        draw.text((6, 2), legend, font=legend_font, fill=(20, 70, 120, 255))

        minor = float(GEOMETRY_GUIDE_MINOR_STEP_PERCENT)
        label_step = float(GEOMETRY_GUIDE_LABEL_STEP_PERCENT)
        major_step = float(GEOMETRY_GUIDE_MAJOR_STEP_PERCENT)
        steps = int(round(100.0 / minor))
        page_top = band_height
        page_bottom = band_height + target_height - 1

        def _is_multiple(value: float, step: float) -> bool:
            quotient = value / step
            return abs(quotient - round(quotient)) < 1e-6

        for index in range(steps + 1):
            percent = min(100.0, index * minor)
            x = int(round((percent / 100.0) * (target_width - 1)))
            if _is_multiple(percent, major_step):
                color = (20, 110, 190, 120)
                line_width = 3
            elif _is_multiple(percent, label_step):
                color = (20, 110, 190, 82)
                line_width = 2
            else:
                color = (20, 110, 190, 45)
                line_width = 1
            draw.line((x, page_top, x, page_bottom), fill=color, width=line_width)

            if _is_multiple(percent, label_step):
                label = f"{int(round(percent))}"
                try:
                    bbox = draw.textbbox((0, 0), label, font=font)
                    text_width = max(1, bbox[2] - bbox[0])
                    text_height = max(1, bbox[3] - bbox[1])
                except Exception:
                    text_width = max(8, len(label) * 7)
                    text_height = max(10, font_size)
                label_x = max(0, min(target_width - text_width, x - text_width // 2))
                top_y = max(2, band_height - text_height - 2)
                bottom_y = band_height + target_height + max(1, (band_height - text_height) // 2)
                draw.text((label_x, top_y), label, font=font, fill=(20, 70, 120, 255))
                draw.text((label_x, bottom_y), label, font=font, fill=(20, 70, 120, 255))

        composed = Image.alpha_composite(canvas_rgba, overlay)
        final_rgb = composed.convert("RGB")
        final_rgb.save(
            target,
            format="JPEG",
            quality=int(quality),
            subsampling=VIEW_JPEG_SUBSAMPLING,
            optimize=True,
            progressive=False,
            dpi=(int(target_dpi), int(target_dpi)),
        )
    finally:
        for image in (final_rgb, composed, overlay, canvas_rgba, canvas, page_rgb, resized, crop):
            if image is not None:
                try:
                    image.close()
                except Exception:
                    pass


def _encode_image(path: str) -> Dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists() or file_path.stat().st_size <= 0:
        raise FileNotFoundError(f"Image absente ou vide : {path}")
    raw = file_path.read_bytes()
    with Image.open(file_path) as image:
        width, height = image.size
    encoded = base64.b64encode(raw).decode("ascii")
    return {
        "path": str(file_path),
        "data_url": f"data:image/jpeg;base64,{encoded}",
        "size_kb": len(raw) / 1024.0,
        "base64_mb": len(encoded.encode("ascii")) / (1024 * 1024),
        "width": int(width),
        "height": int(height),
        "pixels": int(width * height),
    }


def prepare_page_source(
    pdf_path: str,
    page_num: int,
    image_dir: str,
) -> Tuple[str, List[str], Dict[str, Any]]:
    source_dpi = max(
        [RENDER_DPI, DETAIL_DPI, OCR_AUDIT_RENDER_DPI, OCR_AUDIT_DETAIL_DPI]
        + [int(profile["detail_dpi"]) for profile in _payload_profiles("ocr_audit")]
        + [int(profile["detail_dpi"]) for profile in _payload_profiles("markdown")]
    )
    source_path, source_size_kb, rendered = render_single_page_to_file(
        pdf_path=pdf_path,
        page_num=page_num,
        image_dir=image_dir,
        dpi=source_dpi,
    )
    return source_path, [source_path], {
        "rendered": bool(rendered),
        "source_image_size_kb": source_size_kb,
        "source_render_dpi": source_dpi,
    }


def _view_specifications(stage: str, profile: Dict[str, int | str]) -> List[Tuple[str, float, float, float, float, int, int, str]]:
    normalized = str(stage or "markdown").strip().lower()
    full_dpi = int(profile["full_dpi"])
    detail_dpi = int(profile["detail_dpi"])
    quality = int(profile["quality"])
    if normalized in {"ocr", "raw_ocr", "ocr_audit", "audit"}:
        return [
            ("full", 0.0, 0.0, 1.0, 1.0, full_dpi, quality, "page complète — ordre et bords physiques"),
            ("upper", 0.0, 0.0, 1.0, OCR_AUDIT_UPPER_END, detail_dpi, quality, f"partie supérieure détaillée 0–{int(round(OCR_AUDIT_UPPER_END * 100))} %"),
            ("lower", 0.0, OCR_AUDIT_LOWER_START, 1.0, 1.0, detail_dpi, quality, f"partie inférieure détaillée {int(round(OCR_AUDIT_LOWER_START * 100))}–100 %"),
            ("right", OCR_AUDIT_RIGHT_VIEW_START, 0.0, 1.0, 1.0, detail_dpi, quality, f"partie droite détaillée {int(round(OCR_AUDIT_RIGHT_VIEW_START * 100))}–100 % de la largeur"),
        ]
    if normalized in {"markdown", "visual_markdown", "pass2"}:
        return [
            ("full", 0.0, 0.0, 1.0, 1.0, full_dpi, quality, "page complète — autorité pour l'ordre, les bords physiques et les troncatures"),
            ("upper", 0.0, 0.0, 1.0, DETAIL_UPPER_END, detail_dpi, quality, f"partie supérieure détaillée 0–{int(round(DETAIL_UPPER_END * 100))} %"),
            ("middle", 0.0, DETAIL_MIDDLE_START, 1.0, DETAIL_MIDDLE_END, detail_dpi, quality, f"partie centrale détaillée {int(round(DETAIL_MIDDLE_START * 100))}–{int(round(DETAIL_MIDDLE_END * 100))} %"),
            ("lower", 0.0, DETAIL_LOWER_START, 1.0, 1.0, detail_dpi, quality, f"partie inférieure détaillée {int(round(DETAIL_LOWER_START * 100))}–100 %"),
            ("right", RIGHT_VIEW_START, 0.0, 1.0, 1.0, detail_dpi, quality, f"partie droite détaillée {int(round(RIGHT_VIEW_START * 100))}–100 % de la largeur"),
            ("geometry", 0.0, 0.0, 1.0, 1.0, full_dpi, max(72, quality - GEOMETRY_GUIDE_QUALITY_OFFSET), "copie complète avec repère géométrique synthétique vertical x=0–100 — ne jamais transcrire le repère"),
        ]
    raise RuntimeError(f"Étape de vues inconnue : {stage!r}")


def _expected_view_count(stage: str) -> int:
    normalized = str(stage or "markdown").strip().lower()
    if normalized in {"ocr", "raw_ocr", "ocr_audit", "audit"}:
        return OCR_AUDIT_EXPECTED_VIEW_COUNT
    if normalized in {"markdown", "visual_markdown", "pass2"}:
        return MARKDOWN_EXPECTED_VIEW_COUNT
    raise RuntimeError(f"Étape de vues inconnue : {stage!r}")


def prepare_page_views(
    source_path: str,
    page_num: int,
    image_dir: str,
    profile: Dict[str, int | str],
    source_dpi: int,
    *,
    stage: str = "markdown",
) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, Any]]:
    """Construit des vues déterministes propres à une branche visuelle."""
    normalized = str(stage or "markdown").strip().lower()
    stage_label = "ocr_audit" if normalized in {"ocr", "raw_ocr", "ocr_audit", "audit"} else "markdown"
    profile_name = str(profile["name"])
    specifications = _view_specifications(stage_label, profile)

    paths: List[str] = []
    candidates: List[Dict[str, Any]] = []
    with Image.open(source_path) as source:
        for label, left, top, right, bottom, target_dpi, target_quality, description in specifications:
            target = Path(image_dir) / f"page_{int(page_num):06d}_{stage_label}_{profile_name}_{label}.jpg"
            if label == "geometry":
                _save_geometry_guide_view(
                    source,
                    target,
                    source_dpi=source_dpi,
                    target_dpi=int(target_dpi),
                    quality=int(target_quality),
                )
            else:
                _save_jpeg_view(
                    source,
                    target,
                    source_dpi=source_dpi,
                    target_dpi=int(target_dpi),
                    left_ratio=float(left),
                    top_ratio=float(top),
                    right_ratio=float(right),
                    bottom_ratio=float(bottom),
                    quality=int(target_quality),
                )
            paths.append(str(target))
            candidates.append({
                "label": label,
                "description": description,
                "path": str(target),
                "rect": [float(left), float(top), float(right), float(bottom)],
                "target_dpi": int(target_dpi),
                "jpeg_quality": int(target_quality),
                "synthetic_geometry_guide": bool(label == "geometry"),
            })

    encoded = [{**candidate, **_encode_image(str(candidate["path"]))} for candidate in candidates]
    expected = _expected_view_count(stage_label)
    stats = {
        "stage": stage_label,
        "view_count": len(encoded),
        "view_labels": [item["label"] for item in encoded],
        "geometry_guide_present": any(bool(item.get("synthetic_geometry_guide")) for item in encoded),
        "all_views_included": len(encoded) == expected,
        "expected_view_count": expected,
        "total_base64_image_mb": sum(float(item["base64_mb"]) for item in encoded),
        "largest_base64_image_mb": max(float(item["base64_mb"]) for item in encoded),
        "largest_view_pixels": max(int(item["pixels"]) for item in encoded),
        "view_dimensions": [
            {
                "label": item["label"], "width": item["width"], "height": item["height"],
                "pixels": item["pixels"], "target_dpi": item["target_dpi"],
                "jpeg_quality": item["jpeg_quality"],
            }
            for item in encoded
        ],
        "payload_profile": profile_name,
        "full_view_dpi": int(profile["full_dpi"]),
        "detail_view_dpi": int(profile["detail_dpi"]),
        "jpeg_quality": int(profile["quality"]),
        "image_format": "jpeg",
    }
    return encoded, paths, stats


def prepare_ocr_audit_views(*args: Any, **kwargs: Any) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, Any]]:
    kwargs["stage"] = "ocr_audit"
    return prepare_page_views(*args, **kwargs)


def prepare_markdown_views(*args: Any, **kwargs: Any) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, Any]]:
    kwargs["stage"] = "markdown"
    return prepare_page_views(*args, **kwargs)


def cleanup_page_images(paths: Sequence[str]) -> None:
    for raw_path in dict.fromkeys(str(path) for path in paths if path):
        try:
            Path(raw_path).unlink(missing_ok=True)
        except Exception as exc:
            _log(f"⚠️ Impossible de supprimer {raw_path}: {exc}")

# =============================================================================
# Appel Qwen
# =============================================================================

_HTTP_LOCAL = threading.local()


def _get_http_session() -> requests.Session:
    session = getattr(_HTTP_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=HTTP_POOL_SIZE,
            pool_maxsize=HTTP_POOL_SIZE,
            pool_block=True,
        )
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        _HTTP_LOCAL.session = session
    return session


def _supports_thinking_toggle(model: str) -> bool:
    lowered = (model or "").lower()
    return (
        lowered.startswith("qwen3")
        or lowered.startswith("qwen-plus")
        or lowered.startswith("qwen-flash")
        or lowered.startswith("qwen-turbo")
        or lowered.startswith("qwen-max")
    )


def _backoff(attempt: int) -> float:
    return float(min(BACKOFF_BASE**attempt, BACKOFF_MAX))


def _compute_retry_delay(
    http_status: Optional[int], error_message: str, attempt: int
) -> Tuple[bool, float]:
    if attempt >= MAX_RETRIES:
        return False, 0.0
    message = (error_message or "").lower()
    if any(
        marker in message
        for marker in ("invalid api key", "authentication failed", "permission denied")
    ):
        return False, 0.0
    if http_status is not None and http_status not in {408, 409, 425, 429} and http_status < 500:
        return False, 0.0
    if http_status == 429 or "rate limit" in message:
        if FAIL_FAST_ON_429:
            return False, 0.0
        return True, min(10.0 * attempt, 20.0)
    if "overloaded" in message:
        return True, min(5.0 * attempt, 15.0)
    return True, _backoff(attempt)


def _extract_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return str(content["text"])
        if "content" in content:
            return _extract_text(content.get("content"))
        return ""
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            value = _extract_text(item)
            if value.strip():
                parts.append(value)
        return "\n\n".join(parts)
    return ""


def _usage_int(usage: Dict[str, Any], *paths: Tuple[str, ...]) -> int:
    for path in paths:
        current: Any = usage
        for key in path:
            if not isinstance(current, dict) or key not in current:
                current = None
                break
            current = current[key]
        if current is not None:
            try:
                return int(current or 0)
            except Exception:
                pass
    return 0


def _response_header(response: Any, *names: str) -> str:
    headers = getattr(response, "headers", {}) or {}
    for name in names:
        try:
            value = headers.get(name)
        except Exception:
            value = None
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _retry_after_seconds(response: Any) -> Optional[float]:
    """Lit Retry-After sans remplacer le backoff lorsque l'en-tête est absent."""
    raw = _response_header(response, "Retry-After", "retry-after")
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        try:
            target = parsedate_to_datetime(raw)
            if target.tzinfo is None:
                target = target.replace(tzinfo=timezone.utc)
            return max(0.0, (target - datetime.now(timezone.utc)).total_seconds())
        except Exception:
            return None


class RequestTooLargeError(RuntimeError):
    """Le serveur a refusé le corps HTTP ; le même OCR sera réessayé plus léger."""


class RequestBodyBudgetError(RuntimeError):
    """Le corps HTTP dépasse notre plafond préventif avant envoi."""


def _stage_config(stage: str) -> Dict[str, Any]:
    name = str(stage or "ocr_audit").strip().lower()
    if name in {"ocr", "raw_ocr", "ocr_audit", "audit", "pass1"}:
        return {
            "stage": "ocr_audit",
            "model": MODEL_OCR,
            "seed": OCR_SEED,
            "thinking_budget": THINKING_BUDGET_OCR,
            "max_completion_tokens": MAX_COMPLETION_TOKENS_OCR,
            "has_images": True,
        }
    if name in {"markdown", "visual_markdown", "pass2"}:
        return {
            "stage": "markdown",
            "model": MODEL_MARKDOWN,
            "seed": MARKDOWN_SEED,
            "thinking_budget": THINKING_BUDGET_MARKDOWN,
            "max_completion_tokens": MAX_COMPLETION_TOKENS_MARKDOWN,
            "has_images": True,
        }
    raise RuntimeError(f"Étape Qwen inconnue : {stage!r}")

def _request_body(messages: List[Dict[str, Any]], *, stage: str = "ocr") -> Dict[str, Any]:
    config = _stage_config(stage)
    body: Dict[str, Any] = {
        "model": config["model"],
        "max_completion_tokens": int(config["max_completion_tokens"]),
        "temperature": TEMPERATURE,
        "seed": int(config["seed"]),
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": bool(STREAM_INCLUDE_USAGE)},
    }
    if _supports_thinking_toggle(str(config["model"])):
        body["enable_thinking"] = True
        body["thinking_budget"] = int(config["thinking_budget"])
    if bool(config.get("has_images")) and QWEN_HIGH_RES_IMAGES:
        body["vl_high_resolution_images"] = True
    return body


def _serialize_request_body(messages: List[Dict[str, Any]], *, stage: str = "ocr") -> bytes:
    return json.dumps(
        _request_body(messages, stage=stage),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def estimate_request_body_mb(messages: List[Dict[str, Any]], *, stage: str = "ocr") -> float:
    return len(_serialize_request_body(messages, stage=stage)) / (1024 * 1024)

def _iter_sse_data(response: Any) -> Iterable[str]:
    """Itère sur les champs data d'un flux SSE OpenAI compatible."""
    data_lines: List[str] = []
    for raw_line in response.iter_lines(
        chunk_size=STREAM_ITER_CHUNK_SIZE,
        decode_unicode=False,
    ):
        if raw_line is None:
            continue
        if isinstance(raw_line, bytes):
            line = raw_line.decode("utf-8", errors="replace")
        else:
            line = str(raw_line)
        if line == "":
            if data_lines:
                yield "\n".join(data_lines)
                data_lines = []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
            continue
        # Tolérance pour un proxy qui retirerait le préfixe SSE sans modifier le JSON.
        if line.lstrip().startswith(("{", "[DONE]")):
            data_lines.append(line.strip())
    if data_lines:
        yield "\n".join(data_lines)


def _stream_error_message(payload: Dict[str, Any]) -> str:
    error = payload.get("error")
    if isinstance(error, dict):
        return json.dumps(error, ensure_ascii=False)[:800]
    if error:
        return str(error)[:800]
    code = payload.get("code")
    message = payload.get("message")
    if code or message:
        return json.dumps({"code": code, "message": message}, ensure_ascii=False)[:800]
    return ""


def _call_chat(
    api_key: str,
    messages: List[Dict[str, Any]],
    context: str,
    *,
    stage: str = "ocr",
) -> Tuple[str, Dict[str, Any], str]:
    validate_runtime_package_contract()
    url = f"{API_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json; charset=utf-8",
        "Accept": "text/event-stream",
        "Cache-Control": "no-cache",
    }
    serialized_body = _serialize_request_body(messages, stage=stage)
    request_body_mb = len(serialized_body) / (1024 * 1024)
    if request_body_mb > MAX_REQUEST_BODY_MB:
        raise RequestBodyBudgetError(
            f"{context}: corps HTTP prévisionnel={request_body_mb:.2f} Mo, "
            f"plafond={MAX_REQUEST_BODY_MB:.2f} Mo"
        )

    for attempt in range(1, MAX_RETRIES + 1):
        started = time.monotonic()
        content_parts: List[str] = []
        reasoning_parts: List[str] = []
        reasoning_char_count = 0
        usage: Dict[str, Any] = {}
        finish_reason: Optional[str] = None
        response_id: Optional[str] = None
        response_model: Optional[str] = None
        request_id: Optional[str] = None
        partial_response = False
        done_received = False
        stream_interrupted = False
        stream_error = ""
        event_count = 0
        malformed_event_count = 0
        first_event_ms: Optional[int] = None
        first_content_ms: Optional[int] = None
        last_thinking_log = started

        try:
            response = _get_http_session().post(
                url,
                headers=headers,
                data=serialized_body,
                timeout=(CONNECT_TIMEOUT_SECONDS, REQUEST_TIMEOUT_SECONDS),
                stream=True,
            )
            try:
                if response.status_code != 200:
                    try:
                        error_message = json.dumps(response.json(), ensure_ascii=False)[:800]
                    except Exception:
                        error_message = (response.text or "")[:800]
                    if response.status_code == 413:
                        raise RequestTooLargeError(f"{context}: HTTP 413 {error_message}")
                    retry, delay = _compute_retry_delay(
                        response.status_code,
                        error_message,
                        attempt,
                    )
                    if retry and response.status_code == 429:
                        retry_after = _retry_after_seconds(response)
                        if retry_after is not None:
                            delay = min(max(delay, retry_after), 120.0)
                    _log(
                        f"⚠️ {context}: HTTP {response.status_code}, retry={retry}, "
                        f"délai={delay:.1f}s | {error_message[:200]}"
                    )
                    if not retry:
                        raise RuntimeError(
                            f"{context}: HTTP {response.status_code} {error_message}"
                        )
                    response.close()
                    time.sleep(delay)
                    continue

                response.encoding = "utf-8"
                request_id = (
                    _response_header(
                        response,
                        "x-dashscope-request-id",
                        "x-request-id",
                        "x-acs-request-id",
                    )
                    or None
                )
                partial_response = (
                    _response_header(response, "x-dashscope-partialresponse").lower()
                    == "true"
                )

                try:
                    for data in _iter_sse_data(response):
                        if first_event_ms is None:
                            first_event_ms = int((time.monotonic() - started) * 1000)
                        if data.strip() == "[DONE]":
                            done_received = True
                            break
                        try:
                            payload = json.loads(data)
                        except json.JSONDecodeError:
                            malformed_event_count += 1
                            continue

                        event_count += 1
                        error_message = _stream_error_message(payload)
                        if error_message:
                            stream_error = error_message
                            if content_parts:
                                stream_interrupted = True
                                break
                            raise RuntimeError(f"{context}: erreur SSE {error_message}")

                        if payload.get("id") and response_id is None:
                            response_id = str(payload.get("id"))
                        if payload.get("model") and response_model is None:
                            response_model = str(payload.get("model"))
                        if isinstance(payload.get("usage"), dict):
                            usage = dict(payload["usage"])

                        choices = payload.get("choices") or []
                        if not choices:
                            continue
                        choice = choices[0] or {}
                        if choice.get("finish_reason") is not None:
                            finish_reason = str(choice.get("finish_reason"))
                        delta = choice.get("delta") or choice.get("message") or {}
                        reasoning_piece = _extract_text(delta.get("reasoning_content"))
                        if reasoning_piece:
                            reasoning_char_count += len(reasoning_piece)
                            if CAPTURE_REASONING_CONTENT:
                                reasoning_parts.append(reasoning_piece)
                            now = time.monotonic()
                            if (
                                THINKING_PROGRESS_LOG_SECONDS > 0
                                and now - last_thinking_log >= THINKING_PROGRESS_LOG_SECONDS
                            ):
                                _log(
                                    f"🧠 {context}: thinking en cours — "
                                    f"{now - started:.0f}s, chars={reasoning_char_count}, "
                                    f"events={event_count}"
                                )
                                last_thinking_log = now
                        content_piece = _extract_text(delta.get("content"))
                        if content_piece:
                            if first_content_ms is None:
                                first_content_ms = int(
                                    (time.monotonic() - started) * 1000
                                )
                                _log(
                                    f"↪️ {context}: premier fragment final reçu après "
                                    f"{first_content_ms / 1000:.2f}s"
                                )
                            content_parts.append(content_piece)
                except requests.exceptions.RequestException as exc:
                    if content_parts:
                        stream_interrupted = True
                        stream_error = str(exc)[:800]
                    else:
                        raise
            finally:
                response.close()

            text = "".join(content_parts).strip("\n")
            terminal_finish = finish_reason in {
                "stop",
                "length",
                "content_filter",
                "tool_calls",
            }
            stream_complete = done_received or terminal_finish
            truncated = bool(
                finish_reason == "length"
                or partial_response
                or stream_interrupted
                or malformed_event_count
                or not stream_complete
            )

            stats = {
                "input_tokens": _usage_int(usage, ("prompt_tokens",), ("input_tokens",)),
                "output_tokens": _usage_int(usage, ("completion_tokens",), ("output_tokens",)),
                "total_tokens": _usage_int(usage, ("total_tokens",)),
                "cached_tokens": _usage_int(
                    usage,
                    ("prompt_tokens_details", "cached_tokens"),
                    ("cached_tokens",),
                    ("cache_read_input_tokens",),
                ),
                "cache_creation_input_tokens": _usage_int(
                    usage,
                    ("prompt_tokens_details", "cache_creation_input_tokens"),
                    ("cache_creation_input_tokens",),
                ),
                "reasoning_tokens": _usage_int(
                    usage,
                    ("completion_tokens_details", "reasoning_tokens"),
                    ("output_tokens_details", "reasoning_tokens"),
                    ("reasoning_tokens",),
                ),
                "image_tokens": _usage_int(
                    usage,
                    ("prompt_tokens_details", "image_tokens"),
                    ("input_tokens_details", "image_tokens"),
                    ("image_tokens",),
                ),
                "finish_reason": finish_reason,
                "partial_response": partial_response,
                "partial_response_count": 1 if partial_response else 0,
                "truncated_output": truncated,
                "truncated_response_count": 1 if truncated else 0,
                "attempts": attempt,
                "duration_ms": int((time.monotonic() - started) * 1000),
                "response_id": response_id,
                "response_model": response_model or _stage_config(stage)["model"],
                "stage": stage,
                "request_id": request_id,
                "reasoning_content_present": reasoning_char_count > 0,
                "reasoning_char_count": reasoning_char_count,
                "request_body_mb": request_body_mb,
                "streaming": True,
                "stream_event_count": event_count,
                "stream_done_received": done_received,
                "stream_interrupted": stream_interrupted,
                "stream_error": stream_error or None,
                "stream_malformed_event_count": malformed_event_count,
                "time_to_first_event_ms": first_event_ms,
                "time_to_first_content_ms": first_content_ms,
            }
            if not stats["total_tokens"]:
                stats["total_tokens"] = stats["input_tokens"] + stats["output_tokens"]

            if not text:
                if attempt < MAX_RETRIES:
                    delay = _backoff(attempt)
                    _log(
                        f"⚠️ {context}: flux SSE sans contenu final, reprise transport "
                        f"dans {delay:.1f}s"
                    )
                    time.sleep(delay)
                    continue
                raise RuntimeError(
                    f"{context}: flux SSE terminé sans contenu exploitable"
                )

            if truncated:
                _log(
                    f"⚠️ {context}: flux SSE partiel conservé pour salvage déterministe "
                    f"(events={event_count}, done={done_received}, finish={finish_reason})"
                )
            else:
                _log(
                    f"✅ {context}: {stats['duration_ms'] / 1000:.2f}s, "
                    f"in={stats['input_tokens']} out={stats['output_tokens']}, "
                    f"events={event_count}, body={request_body_mb:.2f} Mo"
                )
            reasoning_text = "".join(reasoning_parts) if CAPTURE_REASONING_CONTENT else ""
            stats["reasoning_sha256"] = _sha256_text(reasoning_text) if reasoning_text else ""
            return text, stats, reasoning_text

        except (RequestTooLargeError, RequestBodyBudgetError):
            raise
        except requests.exceptions.Timeout as exc:
            retry, delay = _compute_retry_delay(None, str(exc), attempt)
            if not retry:
                raise
            _log(f"⚠️ {context}: timeout avant contenu, reprise dans {delay:.1f}s")
            time.sleep(delay)
        except requests.exceptions.RequestException as exc:
            retry, delay = _compute_retry_delay(None, str(exc), attempt)
            if not retry:
                raise
            _log(
                f"⚠️ {context}: erreur réseau avant contenu, reprise dans {delay:.1f}s"
            )
            time.sleep(delay)

    raise RuntimeError(f"{context}: échec après {MAX_RETRIES} tentatives de transport")

def _append_visual_inputs(
    content: List[Dict[str, Any]],
    views: Sequence[Dict[str, Any]],
) -> None:
    for index, view in enumerate(views, start=1):
        rect = view.get("rect") or [0.0, 0.0, 1.0, 1.0]
        rect_text = ",".join(f"{float(value):.4f}" for value in rect)
        content.append({
            "type": "text",
            "text": f"Vue {index}/{len(views)} — {view['description']} — zone_page={rect_text}.",
        })
        content.append({
            "type": "image_url",
            "image_url": {"url": view["data_url"]},
        })


def _build_raw_ocr_messages(
    page_num: int,
    page_count: int,
    views: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    user_content: List[Dict[str, Any]] = [
        _cacheable_text_block(RAW_OCR_PROMPT),
        {
            "type": "text",
            "text": (
                f"Page physique {page_num} sur {page_count}. Les {len(views)} images "
                "représentent exactement la même page. Réalise une transcription "
                "physique autonome destinée uniquement à l'audit."
            ),
        },
    ]
    _append_visual_inputs(user_content, views)
    user_content.append({
        "type": "text",
        "text": (
            "Rappel : groupes physiques de gauche à droite, expressions compactes non scindées, "
            "aucun schéma comptable, aucun calcul, aucune normalisation. Effectue les deux "
            "vérifications avant de terminer par END_OCR_AUDIT."
        ),
    })
    return [{"role": "user", "content": user_content}]


def _build_markdown_messages(
    page_num: int,
    page_count: int,
    views: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    user_content: List[Dict[str, Any]] = [
        _cacheable_text_block(MARKDOWN_PROMPT),
        {
            "type": "text",
            "text": (
                f"Page physique {page_num} sur {page_count}. Les {len(views)} images "
                "représentent exactement la même page. Construis le Markdown final "
                "exclusivement depuis les pixels visibles."
            ),
        },
    ]
    _append_visual_inputs(user_content, views)
    user_content.append({
        "type": "text",
        "text": (
            "Rappel final : une carte fixée n’est jamais rouverte. Sur une ligne spéciale, compare les centres x des groupes aux centres x des pistes ordinaires ; si le premier groupe spécial est une bande plus à droite que QTE, laisse QTE vide. Conserve tous les vides et écris SPECIAL_ROW_MAP avant le tableau ; la forme « 1 X » ne donne aucun droit à QTE. Deux cadres fermés taxes/totaux restent deux tableaux, même alignés. TOTAL HT | montant est toujours une donnée. Un manuscrit flottant reste hors du tableau et conserve [INCERTAIN] sans complétion. Une ligne bancaire coupée conserve son préfixe alphabétique certain puis s’arrête avec [TRONQUÉ]. Exécute une seule porte de sortie binaire puis retourne uniquement le Markdown."
        ),
    })
    return [{"role": "user", "content": user_content}]


def _build_ocr_messages(page_num: int, views: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return _build_raw_ocr_messages(page_num, page_num, views)


# =============================================================================
# Parsing canonique et qualité
# =============================================================================


RAW_END_RE = re.compile(r"^\s*\[\[END_OCR_AUDIT\s+coverage=(complete|partial)\]\]\s*$", re.IGNORECASE | re.MULTILINE)
OCR_VISUAL_TABLE_RE = re.compile(r"^\s*\[\[VISUAL_TABLE\s+([^\]]+)\]\]\s*$", re.IGNORECASE | re.MULTILINE)
OCR_VISUAL_TABLE_END_RE = re.compile(r"^\s*\[\[/VISUAL_TABLE\]\]\s*$", re.IGNORECASE | re.MULTILINE)
OCR_VISUAL_ROW_RE = re.compile(r"^\s*\[\[VISUAL_ROW\s+([^\]]+)\]\]\s*$", re.IGNORECASE | re.MULTILINE)
OCR_VISUAL_ROW_END_RE = re.compile(r"^\s*\[\[/VISUAL_ROW\]\]\s*$", re.IGNORECASE | re.MULTILINE)
OCR_ZONE_RE = re.compile(r"^\s*\[\[OCR_ZONE\s+([^\]]+)\]\]\s*$", re.IGNORECASE | re.MULTILINE)
OCR_ZONE_END_RE = re.compile(r"^\s*\[\[/OCR_ZONE\]\]\s*$", re.IGNORECASE | re.MULTILINE)
OCR_FLAG_RE = re.compile(r"^\s*\[\[OCR_FLAG\s+([^\]]+)\]\]\s*$", re.IGNORECASE | re.MULTILINE)


def _normalize_visual_row_metadata(text: str) -> Tuple[str, Dict[str, int]]:
    """Normalise uniquement les métadonnées techniques des VISUAL_ROW.

    Les valeurs OCR et leur ordre ne sont jamais modifiés. Les lignes indexées sont
    renumérotées mécaniquement 1..N dans leur ordre d'apparition et l'attribut
    groups=N est ajusté à ce nombre. Cela garantit : nombre de groupes = indice
    maximal = nombre de lignes indexées, sans inventer de cellule documentaire.
    """
    lines = str(text or "").splitlines()
    output: List[str] = []
    changes = {"visual_row_groups_fixed": 0, "visual_row_indices_renumbered": 0}
    index = 0
    row_start = re.compile(r"^(\s*\[\[VISUAL_ROW\s+)([^\]]+)(\]\]\s*)$", re.IGNORECASE)
    row_end = re.compile(r"^\s*\[\[/VISUAL_ROW\]\]\s*$", re.IGNORECASE)
    group_line = re.compile(r"^(\s*)(\d+)(\s*=.*)$")

    while index < len(lines):
        match = row_start.match(lines[index])
        if not match:
            output.append(lines[index])
            index += 1
            continue

        body: List[str] = []
        index += 1
        while index < len(lines) and not row_end.match(lines[index]):
            body.append(lines[index])
            index += 1
        end_line = lines[index] if index < len(lines) else None
        if end_line is not None:
            index += 1

        group_positions: List[Tuple[int, re.Match[str]]] = []
        for body_index, body_line in enumerate(body):
            group_match = group_line.match(body_line)
            if group_match:
                group_positions.append((body_index, group_match))

        group_count = len(group_positions)
        attrs = match.group(2)
        groups_attr = re.search(r"\bgroups\s*=\s*(\d+)", attrs, re.IGNORECASE)
        announced = int(groups_attr.group(1)) if groups_attr else None
        if groups_attr:
            if announced != group_count:
                attrs = re.sub(
                    r"\bgroups\s*=\s*\d+",
                    f"groups={group_count}",
                    attrs,
                    count=1,
                    flags=re.IGNORECASE,
                )
                changes["visual_row_groups_fixed"] += 1
        else:
            attrs = attrs.rstrip() + f" groups={group_count}"
            changes["visual_row_groups_fixed"] += 1

        for new_number, (body_index, group_match) in enumerate(group_positions, start=1):
            old_number = int(group_match.group(2))
            if old_number != new_number:
                body[body_index] = (
                    group_match.group(1) + str(new_number) + group_match.group(3)
                )
                changes["visual_row_indices_renumbered"] += 1

        output.append(match.group(1) + attrs + match.group(3))
        output.extend(body)
        if end_line is not None:
            output.append(end_line)

    return "\n".join(output), {key: value for key, value in changes.items() if value}


def sanitize_raw_ocr_response(text: str) -> Tuple[str, Dict[str, int]]:
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("Sortie OCR brute vide.")
    changes: Dict[str, int] = {}
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    if cleaned != text:
        changes["line_endings"] = 1
    cleaned, removed_fence = _strip_outer_fence(cleaned)
    if removed_fence:
        changes["outer_fence"] = 1
    cleaned, row_changes = _normalize_visual_row_metadata(cleaned)
    changes.update(row_changes)
    return cleaned.strip("\n"), changes


def validate_raw_ocr_package(raw_ocr: str, page_num: int) -> Dict[str, Any]:
    warnings: List[str] = []
    errors: List[str] = []
    if "[[OCR_AUDIT_PAGE " not in raw_ocr:
        errors.append("OCR_AUDIT_PAGE_absent")
    end_match = RAW_END_RE.search(raw_ocr)
    coverage = end_match.group(1).lower() if end_match else "unknown"
    if not end_match:
        errors.append("END_OCR_AUDIT_absent")

    table_count = len(OCR_VISUAL_TABLE_RE.findall(raw_ocr))
    table_end_count = len(OCR_VISUAL_TABLE_END_RE.findall(raw_ocr))
    zone_count = len(OCR_ZONE_RE.findall(raw_ocr))
    zone_end_count = len(OCR_ZONE_END_RE.findall(raw_ocr))
    row_start_count = len(OCR_VISUAL_ROW_RE.findall(raw_ocr))
    row_end_count = len(OCR_VISUAL_ROW_END_RE.findall(raw_ocr))

    if table_count != table_end_count:
        errors.append(f"visual_table_non_fermee={table_count}/{table_end_count}")
    if zone_count != zone_end_count:
        errors.append(f"ocr_zone_non_fermee={zone_count}/{zone_end_count}")
    if row_start_count != row_end_count:
        errors.append(f"visual_row_non_fermee={row_start_count}/{row_end_count}")

    row_count = 0
    cell_count = 0
    lines = raw_ocr.splitlines()
    index = 0
    while index < len(lines):
        match = re.match(r"^\s*\[\[VISUAL_ROW\s+([^\]]+)\]\]\s*$", lines[index], re.IGNORECASE)
        if not match:
            index += 1
            continue
        row_count += 1
        attrs = _parse_attributes(match.group(1))
        row_id = str(attrs.get("id", f"row_{row_count}"))
        try:
            announced = int(attrs.get("groups", "0") or 0)
        except ValueError:
            announced = 0
        if announced <= 0:
            warnings.append(f"{row_id}: visual_row_groups_absent")

        indices: List[int] = []
        index += 1
        while index < len(lines) and not re.match(
            r"^\s*\[\[/VISUAL_ROW\]\]\s*$", lines[index], re.IGNORECASE
        ):
            cell_match = re.match(r"^\s*(\d+)=(.*)$", lines[index])
            if cell_match:
                indices.append(int(cell_match.group(1)))
                cell_count += 1
            index += 1
        if announced > 0:
            expected = list(range(1, announced + 1))
            if indices != expected:
                warnings.append(
                    f"{row_id}: groupes_indices_incoherents="
                    f"attendus_{expected}_recus_{indices}"
                )
        index += 1

    flag_count = len(OCR_FLAG_RE.findall(raw_ocr))
    if flag_count:
        warnings.append(f"ocr_flags={flag_count}")
    unique_warnings = list(dict.fromkeys(warnings))
    unique_errors = list(dict.fromkeys(errors))
    status = "complete"
    if unique_errors or coverage != "complete":
        status = "degraded"
    elif unique_warnings:
        status = "warning"
    return {
        "page_num": int(page_num),
        "status": status,
        "coverage": coverage,
        "page_empty": "[PAGE VIDE]" in raw_ocr,
        "format_complete": not unique_errors and coverage == "complete",
        "element_count": zone_count + table_count,
        "block_count": len(re.findall(r"^\s*\[\[TEXT_BLOCK\s+", raw_ocr, re.IGNORECASE | re.MULTILINE)),
        "table_count": table_count,
        "kv_count": 0,
        "item_count": 0,
        "row_count": row_count,
        "cell_count": cell_count,
        "has_line_items": table_count > 0,
        "has_totals": bool(re.search(r"kind=(?:taxes|totals)", raw_ocr, re.IGNORECASE)),
        "uncertain_element_ids": [],
        "truncated_element_ids": [],
        "warnings": unique_warnings,
        "errors": unique_errors,
        "warning_count": len(unique_warnings),
        "error_count": len(unique_errors),
        "ambiguity_count": flag_count,
        "failed_table_check_count": 0,
    }


def _normalize_map_cell(value: str) -> str:
    """Normalise seulement la présentation HTML/espaces des cartes techniques."""
    return _plain_cell(str(value or ""))


def _table_requires_commitments(header: Sequence[str], rows: Sequence[Sequence[str]]) -> bool:
    """Retourne True uniquement pour les grilles réellement complexes."""
    raw_header = [str(cell).strip() for cell in header]
    normalized_header = [_plain_cell(cell) for cell in header]
    joined = " | ".join(normalized_header)
    all_cells = normalized_header + [_plain_cell(cell) for row in rows for cell in row]

    simple_financial_pair = (
        len(raw_header) == 2
        and all(re.fullmatch(r"\[SANS_ENTETE_\d+\]", cell, flags=re.IGNORECASE) for cell in raw_header)
        and any(
            re.search(
                r"\b(?:TOTAL\s+HT|TOTAL\s+TVA|NET\s+A\s+PAYER|ECO[- ]?CONTRIBUTION|OCTROI(?:\s+DE\s+MER)?|OMR?)\b",
                cell,
                flags=re.IGNORECASE,
            )
            for cell in all_cells
        )
    )
    if simple_financial_pair:
        return False

    is_tax_matrix = bool(re.search(
        r"codes?\s*(?:tva|vat|gst|taxe?|tax)|exon[eé]r[eé]|soumis",
        joined,
        flags=re.IGNORECASE,
    ))
    has_financial_stack = any(
        re.search(r"\b(?:TOTAL\s+HT|TOTAL\s+TVA|NET\s+A\s+PAYER|ECO[- ]?CONTRIBUTION)\b", cell, flags=re.IGNORECASE)
        for cell in all_cells
    )
    simple_tax_matrix = (
        is_tax_matrix
        and not has_financial_stack
        and not any(re.fullmatch(r"\[SANS_ENTETE_\d+\]", cell, flags=re.IGNORECASE) for cell in raw_header)
        and 4 <= len(raw_header) <= 8
        and all(sum(bool(_plain_cell(cell)) for cell in row) <= 2 for row in rows)
    )
    if simple_tax_matrix:
        return False

    if any(re.fullmatch(r"\[SANS_ENTETE_\d+\]", cell, flags=re.IGNORECASE) for cell in raw_header):
        return True
    if is_tax_matrix:
        return True
    if has_financial_stack:
        return True

    desc_idx = next((i for i, cell in enumerate(normalized_header) if re.search(r"DESIGNATION|DESCRIPTION|LIBELL", cell, flags=re.IGNORECASE)), None)
    qty_idx = next((i for i, cell in enumerate(normalized_header) if re.search(r"\bQTE\b|QUANTIT|QTY", cell, flags=re.IGNORECASE)), None)
    if desc_idx is not None and qty_idx is not None:
        repeated_prefixes: Counter[str] = Counter()
        for row in rows:
            if qty_idx >= len(row):
                continue
            value = _plain_cell(row[qty_idx])
            cores = _NUMERIC_CORE_RE.findall(value)
            if len(cores) >= 2:
                repeated_prefixes[cores[0]] += 1
        if any(count >= 2 for count in repeated_prefixes.values()):
            return True
        if any(any(_CONTRIBUTION_RE.search(_plain_cell(cell)) for cell in row) for row in rows):
            return True
    return False


def validate_grid_commitments(raw_markdown: str) -> List[str]:
    """Valide les cartes techniques des tableaux complexes sans corriger le document."""
    warnings: List[str] = []
    lines = (raw_markdown or "").replace("\r\n", "\n").replace("\r", "\n").splitlines()
    pending_track: Optional[Tuple[bool, int, List[str], str, int]] = None
    pending_grid: Optional[Tuple[int, int, List[int], int]] = None
    pending_rows: Optional[Tuple[int, int, int, List[int], int]] = None
    pending_special: List[Tuple[str, int, List[str], int]] = []
    index = 0
    while index < len(lines):
        track_match = TRACK_MAP_RE.match(lines[index])
        if track_match:
            locked = track_match.group(1).lower() == "yes"
            tracks = int(track_match.group(2))
            headers = [value.strip() for value in track_match.group(3).split("¦")]
            evidence = track_match.group(4).strip()
            pending_track = (locked, tracks, headers, evidence, index + 1)
            index += 1
            continue
        grid_match = GRID_MAP_RE.match(lines[index])
        if grid_match:
            tracks = int(grid_match.group(1))
            header_spans = int(grid_match.group(2))
            unnamed_raw = grid_match.group(3).lower()
            unnamed = [] if unnamed_raw == "none" else [
                int(re.sub(r"^[pP]", "", value.strip()))
                for value in unnamed_raw.split(",") if value.strip()
            ]
            pending_grid = (tracks, header_spans, unnamed, index + 1)
            index += 1
            continue
        row_match = ROW_MAP_RE.match(lines[index])
        if row_match:
            source_rows = int(row_match.group(1))
            continuations = int(row_match.group(2))
            output_rows = int(row_match.group(3))
            mixed_raw = row_match.group(4).lower()
            mixed = [] if mixed_raw == "none" else [
                int(re.sub(r"^[rR]", "", value.strip()))
                for value in mixed_raw.split(",") if value.strip()
            ]
            pending_rows = (source_rows, continuations, output_rows, mixed, index + 1)
            index += 1
            continue
        special_match = SPECIAL_ROW_MAP_RE.match(lines[index])
        if special_match:
            label = special_match.group(1).strip()
            tracks = int(special_match.group(2))
            cells = [value.strip() for value in special_match.group(3).split("¦")]
            pending_special.append((label, tracks, cells, index + 1))
            index += 1
            continue
        if lines[index].strip().startswith("|"):
            table_start = index
            table_lines: List[str] = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index])
                index += 1
            header = _markdown_table_cells(table_lines[0]) if table_lines else []
            width = len(header)
            data_rows = [
                _markdown_table_cells(value) for value in table_lines[2:]
                if _markdown_table_cells(value)
            ]
            complex_table = _table_requires_commitments(header, data_rows)
            table_unnamed: List[int] = []
            track_evidence = ""

            if pending_track is None:
                if complex_table:
                    warnings.append(f"track_map_absent_complex_ligne={table_start + 1}")
            else:
                locked, track_count, declared_headers, evidence, comment_line = pending_track
                track_evidence = evidence
                pending_track = None
                if not locked:
                    warnings.append(f"track_map_not_locked_comment={comment_line}")
                if track_count != width:
                    warnings.append(
                        f"track_map_tracks_mismatch_comment={comment_line}:declared={track_count},table={width}"
                    )
                if len(declared_headers) != track_count:
                    warnings.append(
                        f"track_map_header_count_mismatch_comment={comment_line}:headers={len(declared_headers)},tracks={track_count}"
                    )
                normalized_declared = [_normalize_map_cell(value) for value in declared_headers]
                normalized_actual = [_normalize_map_cell(value) for value in header]
                if normalized_declared != normalized_actual:
                    warnings.append(
                        f"track_map_headers_mismatch_comment={comment_line}:declared={normalized_declared!r}:table={normalized_actual!r}"
                    )
                if not evidence:
                    warnings.append(f"track_map_evidence_absent_comment={comment_line}")

            if pending_grid is None:
                if complex_table:
                    warnings.append(f"grid_map_absent_complex_ligne={table_start + 1}")
            else:
                tracks, header_spans, table_unnamed, comment_line = pending_grid
                pending_grid = None
                if tracks != width:
                    warnings.append(
                        f"grid_map_tracks_mismatch_comment={comment_line}:declared={tracks},table={width}"
                    )
                actual_unnamed = [
                    pos for pos, value in enumerate(header, start=1)
                    if re.fullmatch(r"\[SANS_ENTETE_\d+\]", value.strip(), flags=re.IGNORECASE)
                ]
                if actual_unnamed != table_unnamed:
                    warnings.append(
                        f"grid_map_unnamed_mismatch_comment={comment_line}:declared={table_unnamed},table={actual_unnamed}"
                    )
                if header_spans > tracks:
                    warnings.append(
                        f"grid_map_header_spans_invalid_comment={comment_line}:header_spans={header_spans},tracks={tracks}"
                    )

            if pending_rows is None:
                if complex_table:
                    warnings.append(f"row_map_absent_complex_ligne={table_start + 1}")
            else:
                source_rows, continuations, output_rows, mixed, comment_line = pending_rows
                pending_rows = None
                actual_output_rows = max(0, len(table_lines) - 2)
                if source_rows < 0 or continuations < 0 or output_rows < 0:
                    warnings.append(f"row_map_negative_comment={comment_line}")
                if source_rows - continuations != output_rows:
                    warnings.append(
                        f"row_map_equation_mismatch_comment={comment_line}:source={source_rows},continuations={continuations},output={output_rows}"
                    )
                if output_rows != actual_output_rows:
                    prefix = "financial_" if complex_table and any(
                        re.search(r"TOTAL\s+HT|NET\s+A\s+PAYER|ECO[- ]?CONTRIBUTION|TOTAL\s+TVA", _plain_cell(cell), flags=re.IGNORECASE)
                        for cell in header + [value for row in data_rows for value in row]
                    ) else ""
                    warnings.append(
                        f"{prefix}row_map_output_mismatch_comment={comment_line}:declared={output_rows},table={actual_output_rows}"
                    )
                invalid_mixed = [value for value in mixed if value < 1 or value > source_rows]
                if invalid_mixed:
                    warnings.append(
                        f"row_map_mixed_invalid_comment={comment_line}:mixed={mixed},source={source_rows}"
                    )
                if 1 in mixed and actual_output_rows > 0:
                    first_data = _markdown_table_cells(table_lines[2]) if len(table_lines) > 2 else []
                    nonempty_header_positions = [
                        pos for pos, value in enumerate(first_data, start=1)
                        if str(value).strip() and pos not in table_unnamed
                    ]
                    if nonempty_header_positions:
                        warnings.append(
                            f"mixed_row_header_zone_not_empty_comment={comment_line}:positions={nonempty_header_positions}"
                        )

            normalized_header = [_plain_cell(cell) for cell in header]
            is_line_item_table = (
                any(_DESC_HEADER_RE.search(cell) for cell in normalized_header)
                and any(_QTY_HEADER_RE.search(cell) for cell in normalized_header)
                and any(_AMOUNT_HEADER_RE.search(cell) for cell in normalized_header)
            )
            special_rows = [
                row for row in data_rows
                if is_line_item_table and any(_CONTRIBUTION_RE.search(_plain_cell(cell)) for cell in row)
            ]
            if special_rows and not pending_special:
                warnings.append(f"special_row_map_absent_table_line={table_start + 1}")
            if special_rows and 'track_evidence' in locals() and not re.search(r"geometry:\d+", str(track_evidence or ""), flags=re.IGNORECASE):
                warnings.append(f"special_row_geometry_evidence_absent_table_line={table_start + 1}")
            for label, tracks, declared_cells, comment_line in pending_special:
                if tracks != width:
                    warnings.append(
                        f"special_row_map_tracks_mismatch_comment={comment_line}:declared={tracks},table={width}"
                    )
                if len(declared_cells) != tracks:
                    warnings.append(
                        f"special_row_map_cell_count_mismatch_comment={comment_line}:cells={len(declared_cells)},tracks={tracks}"
                    )
                label_norm = _fuzzy_normalize(label)
                actual_row = next((
                    row for row in data_rows
                    if any(label_norm and label_norm in _fuzzy_normalize(cell) for cell in row)
                ), None)
                if actual_row is None:
                    warnings.append(f"special_row_map_label_not_found_comment={comment_line}:label={label!r}")
                    continue
                normalized_declared = [re.sub(r"\s+", " ", str(value).strip()) for value in declared_cells]
                normalized_actual = [re.sub(r"\s+", " ", str(value).strip()) for value in actual_row]
                if normalized_declared != normalized_actual:
                    warnings.append(
                        f"special_row_map_cells_mismatch_comment={comment_line}:label={label!r}:declared={normalized_declared!r}:table={normalized_actual!r}"
                    )
            pending_special = []
            continue
        index += 1
    if pending_track is not None:
        warnings.append(f"track_map_orphelin_ligne={pending_track[4]}")
    if pending_grid is not None:
        warnings.append(f"grid_map_orphelin_ligne={pending_grid[3]}")
    if pending_rows is not None:
        warnings.append(f"row_map_orphelin_ligne={pending_rows[4]}")
    for label, _tracks, _cells, line_no in pending_special:
        warnings.append(f"special_row_map_orphelin_ligne={line_no}:label={label!r}")
    return list(dict.fromkeys(warnings))

def _renumber_unnamed_headers_per_table(markdown: str) -> Tuple[str, int]:
    """Renumérote mécaniquement les placeholders de gauche à droite dans chaque table."""
    lines = str(markdown or "").splitlines()
    changed = 0
    index = 0
    while index < len(lines):
        header = _markdown_table_cells(lines[index])
        if not header or index + 1 >= len(lines):
            index += 1
            continue
        separator = _markdown_table_cells(lines[index + 1])
        if not _is_markdown_separator_row(separator) or len(separator) != len(header):
            index += 1
            continue
        counter = 0
        new_header: List[str] = []
        for cell in header:
            if re.fullmatch(r"\[SANS_ENTETE_\d+\]", cell.strip(), flags=re.IGNORECASE):
                counter += 1
                replacement = f"[SANS_ENTETE_{counter}]"
                if cell.strip() != replacement:
                    changed += 1
                new_header.append(replacement)
            else:
                new_header.append(cell)
        if counter:
            lines[index] = "| " + " | ".join(new_header) + " |"
        index += 1
    return "\n".join(lines), changed


def _remove_empty_markdown_sections(markdown: str) -> Tuple[str, int]:
    """Supprime uniquement les titres de niveau 2 qui ne contiennent aucun contenu."""
    lines = str(markdown or "").splitlines()
    remove: set[int] = set()
    for idx, line in enumerate(lines):
        if not re.match(r"^##\s+\S", line.strip()):
            continue
        cursor = idx + 1
        while cursor < len(lines) and not lines[cursor].strip():
            cursor += 1
        if cursor >= len(lines) or re.match(r"^##\s+\S", lines[cursor].strip()) or PAGE_MARKER_RE.match(lines[cursor]):
            remove.add(idx)
    if not remove:
        return markdown, 0
    return "\n".join(line for idx, line in enumerate(lines) if idx not in remove), len(remove)


def sanitize_markdown_response(text: str, page_num: int) -> Tuple[str, Dict[str, int]]:
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("Sortie Markdown vide.")
    changes: Dict[str, int] = {}
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned, removed_fence = _strip_outer_fence(cleaned)
    if removed_fence:
        changes["outer_fence"] = 1
    cleaned = re.sub(r"^\s*<!--\s*PAGE\s+\d+\s*-->\s*", "", cleaned, count=1, flags=re.IGNORECASE)
    track_map_count = len(re.findall(r"<!--\s*TRACK_MAP\b", cleaned, flags=re.IGNORECASE))
    grid_count = len(re.findall(r"<!--\s*GRID_MAP\b", cleaned, flags=re.IGNORECASE))
    row_map_count = len(re.findall(r"<!--\s*ROW_MAP\b", cleaned, flags=re.IGNORECASE))
    special_row_map_count = len(re.findall(r"<!--\s*SPECIAL_ROW_MAP\b", cleaned, flags=re.IGNORECASE))
    technical_count = len(TECHNICAL_MAP_COMMENT_RE.findall(cleaned))
    if track_map_count:
        changes["track_map_comments_removed"] = track_map_count
    if grid_count:
        changes["grid_map_comments_removed"] = grid_count
    if row_map_count:
        changes["row_map_comments_removed"] = row_map_count
    if special_row_map_count:
        changes["special_row_map_comments_removed"] = special_row_map_count
    if technical_count:
        cleaned = TECHNICAL_MAP_COMMENT_RE.sub("", cleaned)
    cleaned, renumbered = _renumber_unnamed_headers_per_table(cleaned)
    if renumbered:
        changes["unnamed_headers_renumbered"] = renumbered
    cleaned, empty_sections = _remove_empty_markdown_sections(cleaned)
    if empty_sections:
        changes["empty_sections_removed"] = empty_sections
    if technical_count or renumbered or empty_sections:
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    page_markdown = f"<!-- PAGE {int(page_num)} -->\n\n{cleaned.strip()}\n"
    return page_markdown, changes



def _markdown_table_cells(line: str) -> List[str]:
    raw = (line or "").strip()
    if not raw.startswith("|"):
        return []
    if raw.endswith("|") and not raw.endswith(r"\|"):
        raw = raw[1:-1]
    else:
        raw = raw[1:]
    cells: List[str] = []
    current: List[str] = []
    escaped = False
    for char in raw:
        if escaped:
            current.append(char)
            escaped = False
            continue
        if char == "\\":
            current.append(char)
            escaped = True
            continue
        if char == "|":
            cells.append("".join(current).strip())
            current = []
            continue
        current.append(char)
    cells.append("".join(current).strip())
    return cells


def _is_markdown_separator_row(cells: Sequence[str]) -> bool:
    if not cells:
        return False
    return all(bool(re.fullmatch(r":?-{3,}:?", cell.replace(" ", ""))) for cell in cells)



def validate_page_markdown(markdown: str, page_num: int) -> List[str]:
    """Contrôles structurels non correctifs du Markdown visuel final."""
    warnings: List[str] = []
    if not markdown.lstrip().startswith(f"<!-- PAGE {int(page_num)} -->"):
        warnings.append("page_marker_absent")
    if "[[OCR_AUDIT_PAGE" in markdown or "[[VISUAL_ROW" in markdown:
        warnings.append("ocr_audit_recopie_dans_markdown")

    forbidden_headings = (
        "## Cadrage documentaire",
        "## Inventaire des zones",
        "## Normalisation technique",
        "## Contrôles arithmétiques",
        "## Ambiguïtés et anomalies",
    )
    for heading in forbidden_headings:
        if heading in markdown:
            warnings.append(f"heading_diagnostic_interdit={heading}")

    lines = markdown.splitlines()
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped.startswith("|"):
            index += 1
            continue

        table_start = index
        table_lines: List[str] = []
        while index < len(lines) and lines[index].strip().startswith("|"):
            table_lines.append(lines[index])
            index += 1

        cells_by_line = [_markdown_table_cells(line) for line in table_lines]
        widths = [len(cells) for cells in cells_by_line]
        header_cells = widths[0] if widths else 0
        separator_cells = widths[1] if len(widths) > 1 else 0
        data_cells = cells_by_line[2:] if len(cells_by_line) > 2 else []
        data_widths = widths[2:] if len(widths) > 2 else []

        if len(table_lines) < 3:
            warnings.append(f"table_sans_donnee_ligne={table_start + 1}")
        if len(table_lines) < 2 or not _is_markdown_separator_row(
            cells_by_line[1] if len(cells_by_line) > 1 else []
        ):
            warnings.append(f"table_markdown_malformee_ligne={table_start + 1}")
        if header_cells <= 0 or separator_cells != header_cells:
            warnings.append(f"table_largeur_entete_separateur_ligne={table_start + 1}")
        if any(width != header_cells for width in data_widths):
            warnings.append(
                f"table_width_mismatch_ligne={table_start + 1}:"
                f"header={header_cells},rows={','.join(map(str, data_widths))}"
            )

        header_values = cells_by_line[0] if cells_by_line else []
        unnamed_positions: List[int] = []
        unnamed_numbers: List[int] = []
        for position, value in enumerate(header_values, start=1):
            match = re.fullmatch(r"\[SANS_ENTETE_(\d+)\]", value.strip(), flags=re.IGNORECASE)
            if match:
                unnamed_positions.append(position)
                unnamed_numbers.append(int(match.group(1)))
        if unnamed_numbers and unnamed_numbers != list(range(1, len(unnamed_numbers) + 1)):
            warnings.append(
                f"sans_entete_numerotation_invalide_ligne={table_start + 1}:"
                f"obtenu={','.join(map(str, unnamed_numbers))}"
            )
        for position in unnamed_positions:
            column_values = [row[position - 1].strip() for row in data_cells if len(row) >= position]
            if column_values and all(not value for value in column_values):
                warnings.append(
                    f"sans_entete_colonne_entierement_vide_ligne={table_start + 1}:pos={position}"
                )

    return list(dict.fromkeys(warnings))


def _strip_outer_fence(text: str) -> Tuple[str, bool]:
    normalized = (text or "").strip("\n")
    lines = normalized.splitlines()
    if len(lines) < 2:
        return normalized, False
    opening = FENCE_RE.match(lines[0])
    if not opening:
        return normalized, False
    token = opening.group(1)
    if not lines[-1].strip().startswith(token[0] * len(token)):
        return normalized, False
    return "\n".join(lines[1:-1]).strip("\n"), True


def sanitize_canonical_response(text: str) -> Tuple[str, Dict[str, int]]:
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("Sortie OCR canonique vide.")
    changes: Dict[str, int] = {}
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    if cleaned != text:
        changes["line_endings"] = 1
    cleaned, removed_fence = _strip_outer_fence(cleaned)
    if removed_fence:
        changes["outer_fence"] = 1

    output_lines: List[str] = []
    removed_markers = 0
    for line in cleaned.splitlines():
        if MODEL_PAGE_RE.match(line) or HTML_PAGE_RE.match(line):
            removed_markers += 1
            continue
        output_lines.append(line)
    if removed_markers:
        changes["model_page_markers"] = removed_markers

    cleaned = "\n".join(output_lines).strip("\n")
    normalized = re.sub(r"<SANS_ENTETE_(\d+)>", r"[SANS_ENTETE_\1]", cleaned)
    normalized = (
        normalized.replace("[TRONQUÉ]", "[TRONQUE]")
        .replace("[TRONQUEE]", "[TRONQUE]")
        .replace("[TRONQUÉE]", "[TRONQUE]")
    )
    if normalized != cleaned:
        changes["token_aliases"] = 1
    return normalized, changes


def _parse_attributes(raw: str) -> Dict[str, str]:
    attributes: Dict[str, str] = {}
    for match in ATTRIBUTE_RE.finditer(raw or ""):
        value = next((group for group in match.groups()[1:] if group is not None), "")
        attributes[match.group(1).lower()] = value
    return attributes




def _normalize_section(raw: str, warnings: List[str], element_id: str) -> str:
    candidate = (raw or "").strip().lower()
    section = SECTION_ALIASES.get(candidate, candidate)
    if section not in ALLOWED_SECTIONS:
        warnings.append(f"{element_id}: section_invalide={candidate or '<absent>'}, remplacee=other")
        return "other"
    return section


def _normalize_source(raw: str, section: str, warnings: List[str], element_id: str) -> str:
    source = (raw or "").strip().lower()
    if source not in ALLOWED_SOURCES:
        source = "printed"
        if section == "annotations":
            warnings.append(f"{element_id}: source_absente; printed_conserve_par_prudence")
        else:
            warnings.append(f"{element_id}: source_absente_ou_invalide; printed")
    return source


def _derive_status(raw: str, content: str, warnings: List[str], element_id: str) -> str:
    del raw, warnings, element_id
    uncertain = "[ILLISIBLE]" in content
    truncated = "[TRONQUE]" in content
    if uncertain and truncated:
        return "uncertain_truncated"
    if uncertain:
        return "uncertain"
    if truncated:
        return "truncated"
    return "readable"

def _parse_cell_lines(
    raw_lines: Sequence[str],
    *,
    row_id: str,
    warnings: List[str],
) -> Dict[int, str]:
    cells: Dict[int, str] = {}
    last_index: Optional[int] = None
    for raw in raw_lines:
        if not raw.strip():
            continue
        match = CELL_RE.match(raw)
        if match:
            index = int(match.group(1))
            value = match.group(2) if match.group(2) != "" else "<EMPTY>"
            if index in cells:
                cells[index] = f"{cells[index]}<BR>{value}"
                warnings.append(f"{row_id}: cellule_dupliquee={index}, contenu_preserve")
            else:
                cells[index] = value
            last_index = index
            continue
        if last_index is not None:
            cells[last_index] = f"{cells[last_index]}<BR>{raw}"
            warnings.append(f"{row_id}: ligne_sans_indice_rattachee_cellule={last_index}")
        else:
            warnings.append(f"{row_id}: contenu_sans_cellule_ignore_positionnellement={raw[:80]}")
            cells[1] = raw
            last_index = 1
    return cells


def _parse_table_content(
    element_id: str,
    raw_lines: Sequence[str],
    declared_cols: Optional[int],
    warnings: List[str],
) -> Tuple[List[Dict[str, Any]], int, int, int, List[Dict[str, Any]], int]:
    # Les anciens blocs COLUMNS sont ignorés sans effet : le Markdown dépend
    # exclusivement de cols=N et des cellules indexées produites par Qwen.
    row_lines: List[str] = []
    index = 0
    legacy_column_count = 0
    while index < len(raw_lines):
        if not COLUMNS_START_RE.match(raw_lines[index]):
            row_lines.append(raw_lines[index])
            index += 1
            continue
        legacy_column_count += 1
        index += 1
        while index < len(raw_lines) and not COLUMNS_END_RE.match(raw_lines[index]):
            index += 1
        if index < len(raw_lines):
            index += 1
        warnings.append(f"{element_id}: ancien_bloc_COLUMNS_ignore")

    rows: List[Dict[str, Any]] = []
    index = 0
    row_counter = 0
    while index < len(row_lines):
        line = row_lines[index]
        start = ROW_START_RE.match(line)
        if not start:
            if line.strip():
                row_counter += 1
                legacy_cells = line.replace("	", "<TAB>").split("<TAB>")
                rows.append({
                    "kind": "header" if not rows else "data",
                    "cells_map": {position: (value if value != "" else "<EMPTY>")
                                  for position, value in enumerate(legacy_cells, start=1)},
                    "source": "legacy_tsv",
                    "row_id": f"{element_id}.R{row_counter:03d}",
                })
                warnings.append(f"{element_id}: ligne_legacy_TSV_preservee={row_counter}")
            index += 1
            continue

        attrs = _parse_attributes(start.group(1) or "")
        kind = (attrs.get("kind") or "data").lower()
        if kind not in ALLOWED_ROW_KINDS:
            warnings.append(f"{element_id}: row_kind_invalide={kind}; other")
            kind = "other"
        row_counter += 1
        row_id = f"{element_id}.R{row_counter:03d}"
        index += 1
        content: List[str] = []
        closed = False
        while index < len(row_lines):
            if ROW_END_RE.match(row_lines[index]):
                closed = True
                index += 1
                break
            if ROW_START_RE.match(row_lines[index]):
                break
            content.append(row_lines[index])
            index += 1
        if not closed:
            warnings.append(f"{row_id}: fermeture_ROW_absente")
        rows.append({
            "kind": kind,
            "cells_map": _parse_cell_lines(content, row_id=row_id, warnings=warnings),
            "source": "indexed",
            "row_id": row_id,
        })

    emitted_row_count = len(rows)
    emitted_cell_count = sum(len(row.get("cells_map", {}) or {}) for row in rows)
    max_row_index = max(
        (
            max((cell_index for cell_index in row["cells_map"].keys() if cell_index >= 1), default=0)
            for row in rows
        ),
        default=0,
    )
    effective_cols = max(int(declared_cols or 0), max_row_index)
    if effective_cols <= 0:
        warnings.append(f"{element_id}: tableau_sans_colonne")
        return [], 0, emitted_row_count, emitted_cell_count, [], legacy_column_count
    if not declared_cols:
        warnings.append(f"{element_id}: cols_absent_derive={effective_cols}")
    elif int(declared_cols) != effective_cols:
        warnings.append(f"{element_id}: cols_declare={declared_cols}, indice_max={max_row_index}, effectif={effective_cols}")

    normalized: List[Dict[str, Any]] = []
    for row in rows:
        raw_cells_map = dict(row["cells_map"])
        invalid_cells = [
            {"index": cell_index, "value": value}
            for cell_index, value in sorted(raw_cells_map.items())
            if cell_index < 1
        ]
        if invalid_cells:
            warnings.append(
                f"{row['row_id']}: indice_cellule_invalide="
                + ",".join(str(item["index"]) for item in invalid_cells)
                + "; contenu_conserve_hors_grille"
            )
        cells_map = {
            cell_index: value
            for cell_index, value in raw_cells_map.items()
            if cell_index >= 1
        }
        missing = [position for position in range(1, effective_cols + 1) if position not in cells_map]
        if missing and row.get("source") == "indexed":
            warnings.append(f"{row['row_id']}: indices_cellules_absents={','.join(map(str, missing))}; cellules_vides_ajoutees")
        cells = [cells_map.get(position, "<EMPTY>") for position in range(1, effective_cols + 1)]
        # Une ROW entièrement vide reste une ROW : Python ne supprime plus une
        # structure explicitement émise par Qwen.
        normalized.append(
            {
                "kind": row["kind"],
                "cells": cells,
                "row_id": row["row_id"],
                "invalid_cells": invalid_cells,
            }
        )

    # Markdown ne possède qu'une ligne d'en-tête. Les lignes kind=header
    # consécutives au début sont donc réunies colonne par colonne avec <BR>.
    # Tous les textes restent présents ; aucune ligne de données n'est reclassée.
    leading_headers: List[Dict[str, Any]] = []
    while normalized and normalized[0]["kind"] == "header":
        leading_headers.append(normalized.pop(0))

    header_invalid_cells: List[Dict[str, Any]] = []
    if leading_headers:
        header: List[str] = []
        for column in range(1, effective_cols + 1):
            parts = [
                row["cells"][column - 1]
                for row in leading_headers
                if row["cells"][column - 1].strip() not in {"", "<EMPTY>"}
            ]
            if parts:
                header.append("<BR>".join(parts))
            else:
                token = f"[SANS_ENTETE_{column}]"
                header.append(token)
                warnings.append(f"{element_id}: en_tete_vide_colonne={column}, token={token}")
        for row in leading_headers:
            header_invalid_cells.extend(row.get("invalid_cells", []) or [])
        data_rows = normalized
    else:
        header = [f"[SANS_ENTETE_{i}]" for i in range(1, effective_cols + 1)]
        data_rows = normalized
        warnings.append(f"{element_id}: en_tete_technique_ajoute")

    # Une continuation reste une ligne distincte. La fusion changerait la
    # structure émise par Qwen et pourrait absorber une ligne mal typée.
    output_rows = [
        {
            "kind": "header",
            "cells": header,
            "row_id": f"{element_id}.HEADER",
            "invalid_cells": header_invalid_cells,
        }
    ]
    output_rows.extend(data_rows)
    return output_rows, effective_cols, emitted_row_count, emitted_cell_count, [], legacy_column_count


def _parse_kv_content(
    element_id: str,
    raw_lines: Sequence[str],
    warnings: List[str],
) -> Tuple[List[Dict[str, str]], int]:
    items: List[Dict[str, str]] = []
    index = 0
    item_counter = 0
    while index < len(raw_lines):
        line = raw_lines[index]
        start = ITEM_START_RE.match(line)
        if not start:
            if line.strip():
                # Fallback compact : label<TAB>value ou texte conservé comme valeur.
                item_counter += 1
                parts = line.replace("\t", "<TAB>").split("<TAB>", 1)
                if len(parts) == 2:
                    label, value = parts
                else:
                    label, value = "<EMPTY>", parts[0]
                items.append(
                    {
                        "label": label if label != "" else "<EMPTY>",
                        "value": value if value != "" else "<EMPTY>",
                    }
                )
                warnings.append(f"{element_id}: item_legacy_salvage={item_counter}")
            index += 1
            continue

        item_counter += 1
        index += 1
        content: List[str] = []
        closed = False
        while index < len(raw_lines):
            if ITEM_END_RE.match(raw_lines[index]):
                closed = True
                index += 1
                break
            if ITEM_START_RE.match(raw_lines[index]):
                break
            content.append(raw_lines[index])
            index += 1
        if not closed:
            warnings.append(f"{element_id}.I{item_counter:03d}: fermeture_ITEM_absente_salvage")

        values: Dict[str, str] = {}
        last_key: Optional[str] = None
        for raw in content:
            match = KV_VALUE_RE.match(raw)
            if match:
                key = match.group(1).lower()
                value = match.group(2) if match.group(2) != "" else "<EMPTY>"
                if key in values:
                    values[key] = f"{values[key]}<BR>{value}"
                    warnings.append(f"{element_id}.I{item_counter:03d}: cle_dupliquee={key}")
                else:
                    values[key] = value
                last_key = key
            elif raw.strip():
                if last_key:
                    values[last_key] = f"{values[last_key]}<BR>{raw}"
                else:
                    values["value"] = raw
                    last_key = "value"
                    warnings.append(f"{element_id}.I{item_counter:03d}: ligne_sans_cle_preservee")
        if "label" not in values:
            warnings.append(f"{element_id}.I{item_counter:03d}: cle_label_absente; <EMPTY>_ajoute")
        if "value" not in values:
            warnings.append(f"{element_id}.I{item_counter:03d}: cle_value_absente; <EMPTY>_ajoute")
        item = {
            "label": values.get("label", "<EMPTY>"),
            "value": values.get("value", "<EMPTY>"),
        }
        if (
            item["label"].strip() in {"", "<EMPTY>"}
            and item["value"].strip() in {"", "<EMPTY>"}
        ):
            warnings.append(f"{element_id}.I{item_counter:03d}: item_entierement_vide_ignore")
        else:
            items.append(item)
    return items, item_counter




def parse_canonical_page(
    canonical_text: str,
    page_num: int,
    *,
    api_truncated: bool = False,
) -> Dict[str, Any]:
    warnings: List[str] = []
    errors: List[str] = []
    elements: List[Dict[str, Any]] = []
    end_marker_present = False
    end_marker_count = 0
    content_after_end = False
    coverage = "unknown"
    page_empty = False
    encountered_ids: Counter[str] = Counter()
    grid_decisions: Dict[str, Dict[str, Any]] = {}
    auto_counter = 0

    lines = (canonical_text or "").splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.strip():
            index += 1
            continue
        end_match = END_PAGE_RE.match(line)
        if end_match:
            end_marker_present = True
            end_marker_count += 1
            attrs = _parse_attributes(end_match.group(1) or "")
            raw_coverage = str(attrs.get("coverage", "unknown")).strip().lower()
            coverage = raw_coverage if raw_coverage in {"complete", "partial"} else "unknown"
            index += 1
            continue
        if end_marker_present and line.strip():
            content_after_end = True
        if line.strip() == "[PAGE VIDE]":
            page_empty = True
            index += 1
            continue

        audit_match = GRID_AUDIT_RE.match(line)
        if audit_match:
            attrs = _parse_attributes(audit_match.group(1))
            table_id = str(attrs.get("table_id", "") or "").strip()
            if not table_id:
                table_id = f"T_AUDIT_AUTO_{len(grid_decisions)+1:03d}"
                warnings.append(f"{table_id}: GRID_AUDIT_table_id_absent")

            def _audit_int(name: str, *aliases: str) -> int:
                raw_value = None
                for candidate in (name, *aliases):
                    if candidate in attrs:
                        raw_value = attrs.get(candidate)
                        break
                try:
                    return int(raw_value if raw_value is not None else 0)
                except (TypeError, ValueError):
                    warnings.append(f"{table_id}: GRID_AUDIT_{name}_invalide")
                    return 0

            def _audit_int_list(name: str) -> List[int]:
                raw = str(attrs.get(name, "none") or "none").strip()
                if raw.lower() == "none" or not raw:
                    return []
                values: List[int] = []
                for token in raw.split(","):
                    token = token.strip()
                    if not token:
                        continue
                    try:
                        values.append(int(token))
                    except ValueError:
                        warnings.append(f"{table_id}: GRID_AUDIT_{name}_invalide={token}")
                return values

            def _audit_token_list(name: str, *aliases: str) -> List[str]:
                raw_value = None
                for candidate in (name, *aliases):
                    if candidate in attrs:
                        raw_value = attrs.get(candidate)
                        break
                raw = str(raw_value if raw_value is not None else "none").strip()
                return [
                    token.strip() for token in raw.split(",")
                    if token.strip() and token.strip().lower() != "none"
                ]

            group_counts = _audit_int_list("group_counts")
            max_groups = _audit_int("max_groups", "max_visible_groups")
            support_at_max = _audit_int("support_at_max")
            rows_checked = _audit_int("rows_checked", "ordinary_rows")
            final_cols = _audit_int("final_cols")
            unassigned_groups = _audit_token_list("unassigned_groups", "unassigned_tracks")
            empty_columns = _audit_token_list("empty_columns")
            decision = str(attrs.get("decision", "uncertain") or "uncertain").lower()
            if decision not in {"observed", "uncertain", "confirmed", "revised", "unmapped"}:
                warnings.append(f"{table_id}: GRID_AUDIT_decision_invalide={decision}")
                decision = "uncertain"

            calculated_max = max(group_counts, default=0)
            calculated_support = sum(1 for value in group_counts if value == calculated_max) if group_counts else 0
            if group_counts and max_groups != calculated_max:
                warnings.append(
                    f"{table_id}: GRID_AUDIT_max_groups={max_groups}_different_du_max_calcule={calculated_max}"
                )
            if group_counts and support_at_max != calculated_support:
                warnings.append(
                    f"{table_id}: GRID_AUDIT_support_at_max={support_at_max}_different_du_support_calcule={calculated_support}"
                )
            if rows_checked and group_counts and rows_checked != len(group_counts):
                warnings.append(
                    f"{table_id}: GRID_AUDIT_rows_checked={rows_checked}_different_de_group_counts={len(group_counts)}"
                )
            if support_at_max >= 2 and final_cols < max_groups:
                warnings.append(
                    f"{table_id}: GRID_AUDIT_final_cols={final_cols}_inferieur_a_max_groups={max_groups}_support={support_at_max}"
                )
            if unassigned_groups:
                warnings.append(
                    f"{table_id}: GRID_AUDIT_unassigned_groups={','.join(unassigned_groups)}"
                )
            if empty_columns:
                warnings.append(
                    f"{table_id}: GRID_AUDIT_empty_columns={','.join(empty_columns)}"
                )
            if decision in {"observed", "confirmed"} and (unassigned_groups or empty_columns):
                warnings.append(f"{table_id}: GRID_AUDIT_observed_avec_anomalies_non_resolues")

            record = {
                "table_id": table_id,
                "rows_checked": rows_checked,
                "group_counts": group_counts,
                "group_counts_raw": str(attrs.get("group_counts", "none") or "none"),
                "max_groups": max_groups,
                "support_at_max": support_at_max,
                "final_cols": final_cols,
                "unassigned_groups": unassigned_groups,
                "empty_columns": empty_columns,
                "decision": decision,
                "raw_attrs": dict(attrs),
            }
            if table_id in grid_decisions:
                warnings.append(f"{table_id}: GRID_AUDIT_duplique_remplace")
            grid_decisions[table_id] = record
            index += 1
            continue

        start = ELEMENT_START_RE.match(line)
        if not start:
            stray: List[str] = []
            while index < len(lines):
                if ELEMENT_START_RE.match(lines[index]) or GRID_DECISION_RE.match(lines[index]) or END_PAGE_RE.match(lines[index]):
                    break
                if lines[index].strip() and lines[index].strip() != "[PAGE VIDE]":
                    stray.append(lines[index])
                index += 1
            if stray:
                auto_counter += 1
                element_id = f"B_AUTO_{auto_counter:03d}"
                content = "\n".join(stray)
                elements.append({
                    "kind": "BLOCK", "id": element_id, "sequence": len(elements) + 1,
                    "section": "other", "source": "printed",
                    "status": _derive_status("", content, warnings, element_id), "lines": stray,
                })
                warnings.append(f"{element_id}: texte_hors_balise_preserve")
            continue

        kind = start.group(1).upper()
        attrs = _parse_attributes(start.group(2))
        raw_id = (attrs.get("id") or "").strip()
        if not raw_id:
            prefix = {"BLOCK": "B", "TABLE": "T", "KV": "K"}[kind]
            raw_id = f"{prefix}_AUTO_{len(elements) + 1:03d}"
            warnings.append(f"{raw_id}: id_absent_genere")
        encountered_ids[raw_id] += 1
        element_id = raw_id if encountered_ids[raw_id] == 1 else f"{raw_id}_DUP{encountered_ids[raw_id]}"
        if element_id != raw_id:
            warnings.append(f"{raw_id}: id_duplique_renomme={element_id}")

        raw_section = attrs.get("section") or attrs.get("role") or attrs.get("role_hint") or ""
        section = _normalize_section(raw_section, warnings, element_id)
        source = _normalize_source(attrs.get("source", ""), section, warnings, element_id)

        index += 1
        raw_content: List[str] = []
        closed = False
        end_pattern = ELEMENT_END_PATTERNS[kind]
        while index < len(lines):
            if end_pattern.match(lines[index]):
                closed = True
                index += 1
                break
            if ELEMENT_START_RE.match(lines[index]) or GRID_DECISION_RE.match(lines[index]) or END_PAGE_RE.match(lines[index]):
                break
            raw_content.append(lines[index])
            index += 1
        if not closed:
            warnings.append(f"{element_id}: fermeture_{kind}_absente")
        while raw_content and not raw_content[0].strip():
            raw_content.pop(0)
        while raw_content and not raw_content[-1].strip():
            raw_content.pop()

        if kind == "BLOCK":
            if not raw_content:
                warnings.append(f"{element_id}: block_vide_ignore")
                continue
            content = "\n".join(raw_content)
            elements.append({
                "kind": kind, "id": element_id, "sequence": len(elements) + 1,
                "section": section, "source": source,
                "status": _derive_status("", content, warnings, element_id),
                "lines": raw_content,
            })
            continue

        if kind == "TABLE":
            try:
                declared_cols = int(attrs.get("cols", ""))
            except Exception:
                declared_cols = None
            rows, cols, emitted_rows, emitted_cells, _column_map, _legacy_columns = _parse_table_content(
                element_id, raw_content, declared_cols, warnings
            )
            grid_audit = grid_decisions.get(raw_id) or grid_decisions.get(element_id)
            if grid_audit is None:
                warnings.append(f"{element_id}: GRID_AUDIT_absent")
            else:
                if int(grid_audit.get("final_cols", 0) or 0) != int(cols):
                    warnings.append(
                        f"{element_id}: GRID_AUDIT_final_cols={grid_audit.get('final_cols')} "
                        f"different_de_cols={cols}"
                    )
                if int(grid_audit.get("support_at_max", 0) or 0) >= 2 and int(cols) < int(grid_audit.get("max_groups", 0) or 0):
                    warnings.append(
                        f"{element_id}: GRID_AUDIT_table_cols={cols}_inferieur_a_max_groups="
                        f"{grid_audit.get('max_groups')}"
                    )
            if not rows:
                warnings.append(f"{element_id}: table_vide_ignoree")
                continue
            content = "\n".join("<TAB>".join(row["cells"]) for row in rows)
            elements.append({
                "kind": kind, "id": element_id, "sequence": len(elements) + 1,
                "section": section, "source": source,
                "status": _derive_status("", content, warnings, element_id),
                "cols": cols, "rows": rows,
                "grid_audit": grid_audit,
                "emitted_row_count": emitted_rows,
                "emitted_cell_count": emitted_cells,
            })
            continue

        items, emitted_items = _parse_kv_content(element_id, raw_content, warnings)
        if not items:
            warnings.append(f"{element_id}: kv_vide_ignore")
            continue
        content = "\n".join(f"{item['label']}<TAB>{item['value']}" for item in items)
        elements.append({
            "kind": kind, "id": element_id, "sequence": len(elements) + 1,
            "section": section, "source": source,
            "status": _derive_status("", content, warnings, element_id),
            "items": items, "emitted_item_count": emitted_items,
        })

    if api_truncated:
        warnings.append("reponse_api_tronquee")
    if not end_marker_present:
        warnings.append("marqueur_END_PAGE_absent")
    if end_marker_count > 1:
        warnings.append(f"END_PAGE_multiple={end_marker_count}")
    if content_after_end:
        warnings.append("contenu_apres_END_PAGE")
    if coverage == "partial":
        warnings.append("coverage_partielle_declaree_par_le_modele")
    elif coverage == "unknown":
        warnings.append("coverage_absente_ou_invalide")
    if page_empty and elements:
        warnings.append("PAGE_VIDE_et_elements_presents")
    table_ids = {str(element.get("id")) for element in elements if element.get("kind") == "TABLE"}
    for audit_id in sorted(set(grid_decisions) - table_ids):
        warnings.append(f"{audit_id}: GRID_AUDIT_sans_TABLE")

    technical_markers = (
        "fermeture_", "cellule_dupliquee=", "ligne_sans_indice_",
        "contenu_sans_cellule_", "ligne_legacy_TSV_", "row_kind_invalide=",
        "tableau_sans_colonne", "cols_absent_derive=", "cols_declare=",
        "indices_cellules_absents=", "indice_cellule_invalide=",
        "texte_hors_balise_preserve", "block_vide_ignore", "table_vide_ignoree",
        "kv_vide_ignore", "item_legacy_salvage=", "cle_dupliquee=",
        "ligne_sans_cle_preservee", "cle_label_absente", "cle_value_absente",
        "item_entierement_vide_ignore", "section_invalide=", "source_absente",
        "END_PAGE_", "contenu_apres_END_PAGE", "marqueur_END_PAGE_absent",
        "reponse_api_tronquee", "GRID_AUDIT_", "unassigned_groups=", "empty_columns=",
    )
    technical_warnings = [w for w in warnings if any(marker in w for marker in technical_markers)]
    format_complete = bool(
        end_marker_count == 1
        and not content_after_end
        and coverage == "complete"
        and not technical_warnings
        and not api_truncated
    )

    uncertain_ids = [e["id"] for e in elements if e.get("status") in {"uncertain", "uncertain_truncated"}]
    truncated_ids = [e["id"] for e in elements if e.get("status") in {"truncated", "uncertain_truncated"}]

    if not elements and not page_empty:
        status = "unavailable"
        errors.append("aucun_element_canonique_exploitable")
    elif api_truncated or not end_marker_present or any("fermeture_" in w for w in warnings):
        status = "degraded"
    elif warnings or uncertain_ids or truncated_ids or coverage != "complete":
        status = "warning"
    else:
        status = "complete"

    quality = {
        "page_num": int(page_num),
        "status": status,
        "page_empty": page_empty,
        "coverage": coverage,
        "api_truncated": bool(api_truncated),
        "format_complete": format_complete,
        "element_count": len(elements),
        "block_count": sum(1 for e in elements if e["kind"] == "BLOCK"),
        "table_count": sum(1 for e in elements if e["kind"] == "TABLE"),
        "kv_count": sum(1 for e in elements if e["kind"] == "KV"),
        "item_count": sum(len(e.get("items", []) or []) for e in elements if e["kind"] == "KV"),
        "row_count": sum(len(e.get("rows", []) or []) for e in elements if e["kind"] == "TABLE"),
        "cell_count": sum(len(row.get("cells", []) or []) for e in elements if e["kind"] == "TABLE" for row in e.get("rows", []) or []),
        "has_line_items": any(e.get("section") == "line_items" for e in elements),
        "has_totals": any(e.get("section") == "totals" for e in elements),
        "uncertain_element_ids": uncertain_ids,
        "truncated_element_ids": truncated_ids,
        "warnings": list(dict.fromkeys(warnings)),
        "errors": list(dict.fromkeys(errors)),
        "warning_count": len(list(dict.fromkeys(warnings))),
        "error_count": len(list(dict.fromkeys(errors))),
    }
    return {
        "page_num": int(page_num),
        "page_empty": page_empty,
        "elements": elements,
        "grid_decisions": grid_decisions,
        "quality": quality,
    }


# =============================================================================
# Renderer historique conservé uniquement pour compatibilité interne
# =============================================================================

MARKDOWN_SECTIONS: List[Tuple[str, str]] = [
    ("issuer", "## Informations Émetteur"),
    ("customer", "## Informations Client"),
    ("shipping", "## Informations de Livraison"),
    ("document", "## Détails du Document"),
    ("line_items", "## Tableau des Lignes de Facturation"),
    ("taxes", "## Taxes"),
    ("totals", "## Totaux"),
    ("payment", "## Informations de Paiement"),
    ("annotations", "## Annotations, Tampons et Signatures"),
    ("legal", "## Mentions Légales"),
    ("other", "## Autres Contenus Visibles"),
]


def _display_tokens(text: str) -> str:
    return (text or "").replace("[TRONQUE]", "[TRONQUÉ]")


_CANONICAL_DISPLAY_TOKEN_RE = re.compile(
    r"\[(?:ILLISIBLE|TRONQUÉ|SANS_ENTETE_\d+)\]"
)


def _escape_markdown_literal_fragment(text: str) -> str:
    """Échappe un fragment documentaire tout en gardant les tokens canoniques."""
    value = _display_tokens(text)
    protected: Dict[str, str] = {}

    def protect(match: re.Match[str]) -> str:
        placeholder = f"OCRDISPLAYTOKEN{len(protected)}XYZ"
        protected[placeholder] = match.group(0)
        return placeholder

    value = _CANONICAL_DISPLAY_TOKEN_RE.sub(protect, value)
    value = html.escape(value, quote=False)
    value = value.replace("\\", "\\\\")
    for marker in ("`", "*", "_", "[", "]", "#", "~", "|"):
        value = value.replace(marker, "\\" + marker)
    for placeholder, token in protected.items():
        value = value.replace(placeholder, token)
    return value


def _escape_markdown_cell(text: str) -> str:
    value = _display_tokens(text)
    if value == "<EMPTY>":
        return ""
    fragments = re.split(r"<BR>|\n", value)
    return "<br>".join(_escape_markdown_literal_fragment(fragment) for fragment in fragments)


def _escape_markdown_block_fragment(text: str) -> str:
    """Protège un texte documentaire contre toute interprétation Markdown."""
    value = _escape_markdown_literal_fragment(text)
    # Les marqueurs de listes et règles horizontales n'ont un effet spécial
    # qu'en début de ligne. Ils sont échappés sans modifier le texte affiché.
    value = re.sub(r"^(\s*)([-+])(?=\s)", r"\1\\\2", value)
    value = re.sub(r"^(\s*)(\d+)([.)])(?=\s)", r"\1\2\\\3", value)
    if re.fullmatch(r"\s*-{3,}\s*", value):
        value = value.replace("-", "\\-", 1)
    return value


def _escape_markdown_block_line(text: str) -> str:
    # <BR> est un token canonique interne. Les fragments documentaires sont
    # échappés séparément, puis le saut de ligne HTML généré est réintroduit.
    return "<br>".join(
        _escape_markdown_block_fragment(fragment) for fragment in str(text).split("<BR>")
    )


def _render_block(element: Dict[str, Any]) -> str:
    lines = [_escape_markdown_block_line(str(line)) for line in element.get("lines", [])]
    content = "<br>\n".join(lines).strip()
    source = element.get("source")
    if source == "handwritten":
        return f"**Manuscrit :** {content}"
    if source == "stamp":
        return f"**Tampon :** {content}"
    return content


def _render_table(element: Dict[str, Any]) -> str:
    rows = list(element.get("rows") or [])
    if not rows:
        return ""
    header = rows[0]["cells"]
    output = [
        "| " + " | ".join(_escape_markdown_cell(cell) for cell in header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    residues: List[str] = []
    for row in rows:
        for invalid in row.get("invalid_cells", []) or []:
            residues.append(
                f"{row.get('row_id', 'ROW')} {invalid.get('index')}={_display_tokens(str(invalid.get('value', '')))}"
            )
    for row in rows[1:]:
        output.append(
            "| " + " | ".join(_escape_markdown_cell(cell) for cell in row["cells"]) + " |"
        )
    if residues:
        # Un indice hors contrat ne peut être positionné honnêtement dans la
        # grille. Son contenu est conservé littéralement, sans déplacement.
        output.extend(
            [
                "",
                '<pre data-ocr-residue="out-of-grid">',
                *(html.escape(item, quote=False) for item in residues),
                "</pre>",
            ]
        )
    return "\n".join(output)


def _render_kv(element: Dict[str, Any]) -> str:
    output = ["| Libellé | Valeur |", "| --- | --- |"]
    for item in element.get("items", []) or []:
        output.append(
            f"| {_escape_markdown_cell(item['label'])} | {_escape_markdown_cell(item['value'])} |"
        )
    return "\n".join(output)


def _render_element(element: Dict[str, Any]) -> str:
    if element["kind"] == "BLOCK":
        return _render_block(element)
    if element["kind"] == "TABLE":
        return _render_table(element)
    return _render_kv(element)


def render_markdown_page(parsed: Dict[str, Any]) -> str:
    page_num = int(parsed["page_num"])
    lines: List[str] = [f"<!-- PAGE {page_num} -->"]

    if parsed.get("page_empty") and not parsed.get("elements"):
        lines.extend(["", "**[PAGE VIDE]**"])
        return "\n".join(lines).strip("\n")
    if not parsed.get("elements"):
        lines.extend(["", "## Extraction indisponible", "", "[PAGE NON EXTRAITE]"])
        return "\n".join(lines).strip("\n")

    elements = list(parsed.get("elements") or [])
    for section, heading in MARKDOWN_SECTIONS:
        selected = [e for e in elements if e.get("section") == section]
        lines.extend(["", heading])
        selected.sort(key=lambda e: (int(e.get("sequence", 0) or 0), str(e.get("id", ""))))
        for element in selected:
            rendered = _render_element(element)
            if rendered:
                lines.extend(["", rendered])
    return "\n".join(lines).strip("\n")


def render_canonical_page(parsed: Dict[str, Any]) -> str:
    """Rend la source canonique normalisée pour le checkpoint interne uniquement."""
    if parsed.get("page_empty") and not parsed.get("elements"):
        return "[PAGE VIDE]\n[[END_PAGE coverage=complete]]"

    output: List[str] = []
    for element in parsed.get("elements", []) or []:
        kind = element["kind"]
        element_id = str(element["id"])
        section = str(element.get("section", "other"))
        source = str(element.get("source", "printed"))
        if kind == "BLOCK":
            output.append(f"[[BLOCK id={element_id} section={section} source={source}]]")
            output.extend(str(line) for line in element.get("lines", []) or [])
            output.append("[[/BLOCK]]")
        elif kind == "TABLE":
            audit = element.get("grid_audit") or {}
            if audit:
                group_counts = list(audit.get("group_counts") or [])
                group_counts_raw = ",".join(str(value) for value in group_counts) if group_counts else "none"
                unassigned = list(audit.get("unassigned_groups") or [])
                empty_columns = list(audit.get("empty_columns") or [])
                output.append(
                    "[[GRID_AUDIT "
                    f"table_id={element_id} "
                    f"rows_checked={int(audit.get('rows_checked', 0) or 0)} "
                    f'group_counts="{group_counts_raw}" '
                    f"max_groups={int(audit.get('max_groups', 0) or 0)} "
                    f"support_at_max={int(audit.get('support_at_max', 0) or 0)} "
                    f"final_cols={int(audit.get('final_cols', 0) or 0)} "
                    f'unassigned_groups="{",".join(unassigned) if unassigned else "none"}" '
                    f'empty_columns="{",".join(empty_columns) if empty_columns else "none"}" '
                    f"decision={audit.get('decision', 'uncertain')}]]"
                )
            output.append(
                f"[[TABLE id={element_id} section={section} source={source} "
                f"cols={int(element.get('cols', 0) or 0)}]]"
            )
            for row in element.get("rows", []) or []:
                output.append(f"[[ROW kind={row.get('kind', 'data')}]]")
                for invalid in row.get("invalid_cells", []) or []:
                    output.append(f"{invalid.get('index')}={invalid.get('value', '')}")
                for cell_index, cell in enumerate(row.get("cells", []) or [], start=1):
                    output.append(f"{cell_index}={cell}")
                output.append("[[/ROW]]")
            output.append("[[/TABLE]]")
        else:
            output.append(f"[[KV id={element_id} section={section} source={source}]]")
            for item in element.get("items", []) or []:
                output.extend([
                    "[[ITEM]]",
                    f"label={item['label']}",
                    f"value={item['value']}",
                    "[[/ITEM]]",
                ])
            output.append("[[/KV]]")
    coverage = str(parsed.get("quality", {}).get("coverage", "partial"))
    if coverage not in {"complete", "partial"}:
        coverage = "partial"
    output.append(f"[[END_PAGE coverage={coverage}]]")
    return "\n".join(output).strip()

def build_unavailable_page(page_num: int, error: BaseException | str) -> Dict[str, Any]:
    message = str(error).replace("\n", " ")[:1000]
    raw_fallback = (
        f"[[OCR_AUDIT_PAGE page={int(page_num)} pages=unknown document_type=unknown "
        "language=unknown orientation=unknown quality=unavailable stamps=no handwriting=no]]\n"
        "[[OCR_ZONE id=Z1 kind=other source=printed order=1]]\n"
        "[[TEXT_BLOCK id=B1 state=illegible]]\n[ILLISIBLE]\n[[/TEXT_BLOCK]]\n[[/OCR_ZONE]]\n"
        f'[[OCR_FLAG id=F1 target="page" state=possible_omission note="{message}"]]\n'
        "[[END_OCR_AUDIT coverage=partial]]"
    )
    markdown = (
        f"<!-- PAGE {int(page_num)} -->\n\n"
        "## Détails de la Facture\n\n"
        "[ILLISIBLE]\n\n"
        "## Mentions Légales et Notes Complémentaires\n\n"
        f"Extraction indisponible — erreur technique : {message}\n"
    )
    quality = validate_raw_ocr_package(raw_fallback, page_num)
    quality["status"] = "unavailable"
    quality["errors"] = [message]
    quality["error_count"] = 1
    return {
        "page_num": int(page_num),
        "raw_response": raw_fallback,
        "raw_ocr": raw_fallback,
        "ocr_reasoning": "",
        "markdown_raw_response": "",
        "markdown_reasoning": "",
        "sanitized_canonical": raw_fallback,
        "normalized_canonical": raw_fallback,
        "canonical": raw_fallback,
        "markdown": markdown,
        "quality": quality,
        "stats": {
            "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
            "cached_tokens": 0, "cache_creation_input_tokens": 0,
            "reasoning_tokens": 0, "image_tokens": 0, "duration_ms": 0,
            "quality_status": "unavailable", "page_error": message,
            "raw_response_sha256": _sha256_text(raw_fallback),
            "raw_ocr_sha256": _sha256_text(raw_fallback),
            "ocr_reasoning_sha256": "", "markdown_response_sha256": "",
            "markdown_reasoning_sha256": "", "markdown_sha256": _sha256_text(markdown),
            "pipeline_version": PIPELINE_VERSION,
        },
    }


# =============================================================================
# Traitement d'une page — deux lectures visuelles indépendantes avec thinking
# =============================================================================


def _payload_is_too_large(view_stats: Dict[str, Any], request_body_mb: float) -> bool:
    return bool(
        float(view_stats.get("largest_base64_image_mb", 0.0)) > MAX_SINGLE_BASE64_IMAGE_MB
        or float(view_stats.get("total_base64_image_mb", 0.0)) > MAX_TOTAL_BASE64_IMAGE_MB
        or float(request_body_mb) > MAX_REQUEST_BODY_MB
    )


def _run_visual_branch_with_fallback(
    *,
    source_path: str,
    source_dpi: int,
    page_num: int,
    page_count: int,
    api_key: str,
    image_dir: str,
    stage: str,
) -> Dict[str, Any]:
    normalized_stage = "ocr_audit" if stage in {"ocr", "raw_ocr", "ocr_audit", "audit"} else "markdown"
    failures: List[str] = []
    attempts = 0
    profiles = _payload_profiles(normalized_stage)
    for profile_index, profile in enumerate(profiles, start=1):
        views: List[Dict[str, Any]] = []
        profile_paths: List[str] = []
        try:
            views, profile_paths, view_stats = prepare_page_views(
                source_path=source_path,
                page_num=page_num,
                image_dir=image_dir,
                profile=profile,
                source_dpi=source_dpi,
                stage=normalized_stage,
            )
            expected = _expected_view_count(normalized_stage)
            if int(view_stats.get("view_count", 0) or 0) != expected:
                raise RuntimeError(
                    f"Page {page_num} {normalized_stage}: {view_stats.get('view_count')} vues, {expected} attendues."
                )
            if normalized_stage == "ocr_audit":
                messages = _build_raw_ocr_messages(page_num, page_count, views)
                context = f"OCR d'audit indépendant page {page_num}"
                log_label = "OCR audit"
                api_stage = "ocr_audit"
            else:
                messages = _build_markdown_messages(page_num, page_count, views)
                context = f"Markdown visuel indépendant page {page_num}"
                log_label = "Markdown visuel"
                api_stage = "markdown"
            request_body_mb = estimate_request_body_mb(messages, stage=api_stage)
            view_stats["request_body_mb_preflight"] = request_body_mb
            attempts += 1
            if _payload_is_too_large(view_stats, request_body_mb):
                reason = (
                    f"stage={normalized_stage}, profil={view_stats['payload_profile']}, "
                    f"images={view_stats['total_base64_image_mb']:.2f} Mo, body={request_body_mb:.2f} Mo"
                )
                failures.append(reason)
                _log(f"⚖️ Page {page_num}: profil trop lourd avant envoi — {reason}")
                continue
            _log(
                f"➡️ Page {page_num}: {log_label} indépendant avec thinking, "
                f"{view_stats['view_count']} vues, profil={view_stats['payload_profile']}, "
                f"body={request_body_mb:.2f} Mo"
            )
            try:
                output_text, api_stats, reasoning = _call_chat(
                    api_key=api_key,
                    messages=messages,
                    context=context,
                    stage=api_stage,
                )
                return {
                    "text": output_text,
                    "reasoning": reasoning,
                    "api_stats": api_stats,
                    "view_stats": view_stats,
                    "request_body_mb": request_body_mb,
                    "payload_attempts": attempts,
                    "payload_fallback_count": max(0, attempts - 1),
                    "payload_failures": failures,
                }
            except RequestTooLargeError as exc:
                failures.append(str(exc))
                if not ALLOW_413_PAYLOAD_FALLBACK or profile_index >= len(profiles):
                    raise
                _log(f"⚠️ Page {page_num} {log_label}: HTTP 413 ; profil plus léger.")
            except RequestBodyBudgetError as exc:
                failures.append(str(exc))
        finally:
            for view in views:
                view.pop("data_url", None)
            cleanup_page_images(profile_paths)
    details = " | ".join(failures[-4:]) or "aucun profil exploitable"
    raise RuntimeError(f"Page {page_num} {normalized_stage}: aucun profil sous les limites. {details}")


def process_page(
    pdf_path: str,
    page_num: int,
    api_key: str,
    image_dir: str,
    total_pages: Optional[int] = None,
) -> Dict[str, Any]:
    validate_runtime_package_contract()
    page_num = int(page_num)
    page_count = int(total_pages or page_num or 1)
    cleanup_paths: List[str] = []
    page_started = time.monotonic()
    try:
        source_path, source_cleanup, source_stats = prepare_page_source(
            pdf_path=pdf_path,
            page_num=page_num,
            image_dir=image_dir,
        )
        cleanup_paths.extend(source_cleanup)
        branch_kwargs = {
            "source_path": source_path,
            "source_dpi": int(source_stats["source_render_dpi"]),
            "page_num": page_num,
            "page_count": page_count,
            "api_key": api_key,
            "image_dir": image_dir,
        }
        audit_result: Optional[Dict[str, Any]] = None
        audit_error: Optional[BaseException] = None
        if PARALLEL_INDEPENDENT_PASSES:
            _log(f"⇉ Page {page_num}: lancement parallèle des deux lectures indépendantes")
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix=f"qwen-page-{page_num}") as pool:
                future_ocr = pool.submit(_run_visual_branch_with_fallback, **branch_kwargs, stage="ocr_audit")
                future_markdown = pool.submit(_run_visual_branch_with_fallback, **branch_kwargs, stage="markdown")
                try:
                    audit_result = future_ocr.result()
                except BaseException as exc:
                    audit_error = exc
                # Le Markdown est la sortie métier : son échec reste bloquant.
                markdown_result = future_markdown.result()
        else:
            try:
                audit_result = _run_visual_branch_with_fallback(**branch_kwargs, stage="ocr_audit")
            except BaseException as exc:
                audit_error = exc
            markdown_result = _run_visual_branch_with_fallback(**branch_kwargs, stage="markdown")

        if audit_result is None:
            audit_message = str(audit_error or "OCR audit indisponible").replace("\n", " ")[:800]
            audit_attribute = audit_message.replace('"', "'")
            audit_stub = (
                f"[[OCR_AUDIT_PAGE page={page_num} pages={page_count} document_type=unknown "
                "language=unknown orientation=unknown quality=unavailable stamps=no handwriting=no]]\n"
                "[[OCR_ZONE id=Z1 kind=other source=printed order=1]]\n"
                "[[TEXT_BLOCK id=B1 state=illegible]]\n[ILLISIBLE]\n[[/TEXT_BLOCK]]\n[[/OCR_ZONE]]\n"
                f'[[OCR_FLAG id=F1 target="page" state=possible_omission note="{audit_attribute}"]]\n'
                "[[END_OCR_AUDIT coverage=partial]]"
            )
            audit_result = {
                "text": audit_stub,
                "reasoning": "",
                "api_stats": {
                    "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                    "cached_tokens": 0, "cache_creation_input_tokens": 0,
                    "reasoning_tokens": 0, "image_tokens": 0, "duration_ms": 0,
                    "stage": "ocr_audit", "page_error": audit_message,
                },
                "view_stats": {"payload_profile": "unavailable", "view_count": 0},
                "request_body_mb": 0.0,
                "payload_attempts": 0,
                "payload_fallback_count": 0,
                "payload_failures": [audit_message],
            }
            _log(f"⚠️ Page {page_num}: OCR audit indisponible, Markdown conservé — {audit_message}")

        raw_ocr_text = str(audit_result["text"])
        ocr_reasoning = str(audit_result.get("reasoning", "") or "")
        raw_ocr, raw_sanitizations = sanitize_raw_ocr_response(raw_ocr_text)
        quality = validate_raw_ocr_package(raw_ocr, page_num)
        if audit_error is not None:
            quality["warnings"] = list(quality.get("warnings", [])) + ["ocr_audit_unavailable"]
            quality["errors"] = []
            quality["warning_count"] = len(quality["warnings"])
            quality["error_count"] = 0
            quality["status"] = "warning"

        markdown_raw = str(markdown_result["text"])
        markdown_reasoning = str(markdown_result.get("reasoning", "") or "")
        grid_commitment_warnings = validate_grid_commitments(markdown_raw)
        markdown, markdown_sanitizations = sanitize_markdown_response(markdown_raw, page_num)
        markdown_warnings = grid_commitment_warnings + validate_page_markdown(markdown, page_num)
        critical_regressions = detect_semantic_regression_critical(
            markdown, raw_ocr=raw_ocr, markdown_reasoning=markdown_reasoning
        )
        critical_regressions.extend(
            warning for warning in grid_commitment_warnings
            if warning.startswith((
                "track_map_tracks_mismatch_",
                "track_map_header_count_mismatch_",
                "special_row_map_tracks_mismatch_",
                "special_row_map_cell_count_mismatch_",
                "special_row_map_label_not_found_",
                "special_row_map_cells_mismatch_",
                "special_row_map_absent_",
                "special_row_geometry_evidence_absent_",
            ))
        )
        critical_regressions = list(dict.fromkeys(critical_regressions))
        markdown_warnings = list(dict.fromkeys(markdown_warnings))
        markdown_candidate = markdown
        # Les contrôles de régression sont des garde-fous d'audit, jamais un mécanisme
        # de suppression. Le résultat Qwen reste toujours publié intégralement dans
        # final.md ; les anomalies sont reportées dans les métadonnées et le rapport.
        publication_blocked = False
        publication_review_required = bool(critical_regressions)
        if critical_regressions:
            markdown_warnings.extend(f"critical:{item}" for item in critical_regressions)
        markdown_warnings = list(dict.fromkeys(markdown_warnings))
        if markdown_warnings:
            quality["warnings"] = list(quality.get("warnings", [])) + markdown_warnings
            quality["warning_count"] = len(quality["warnings"])
            if quality.get("status") == "complete":
                quality["status"] = "warning"
        quality["critical_regressions"] = list(critical_regressions)
        quality["publication_blocked"] = False
        quality["publication_review_required"] = bool(publication_review_required)
        if publication_review_required:
            quality["status"] = "warning"
        quality["has_line_items"] = "## Tableau des Lignes de Facturation" in markdown_candidate
        quality["has_totals"] = "## Montants Récapitulatifs" in markdown_candidate
        quality["format_complete"] = bool(quality.get("format_complete")) and not markdown_warnings

        ocr_api_stats = dict(audit_result["api_stats"])
        markdown_api_stats = dict(markdown_result["api_stats"])

        def sum_stat(name: str) -> int:
            return int(ocr_api_stats.get(name, 0) or 0) + int(markdown_api_stats.get(name, 0) or 0)

        wall_duration_ms = int((time.monotonic() - page_started) * 1000)
        audit_view_stats = dict(audit_result["view_stats"])
        markdown_view_stats = dict(markdown_result["view_stats"])
        stats: Dict[str, Any] = {
            "input_tokens": sum_stat("input_tokens"),
            "output_tokens": sum_stat("output_tokens"),
            "total_tokens": sum_stat("total_tokens"),
            "cached_tokens": sum_stat("cached_tokens"),
            "cache_creation_input_tokens": sum_stat("cache_creation_input_tokens"),
            "reasoning_tokens": sum_stat("reasoning_tokens"),
            "image_tokens": sum_stat("image_tokens"),
            "duration_ms": wall_duration_ms,
            "provider_compute_duration_ms": sum_stat("duration_ms"),
            "parallel_independent_passes": bool(PARALLEL_INDEPENDENT_PASSES),
            "ocr_audit_error": str(audit_error)[:1000] if audit_error is not None else None,
            "ocr_pass_stats": ocr_api_stats,
            "markdown_pass_stats": markdown_api_stats,
            "ocr_audit_view_stats": audit_view_stats,
            "markdown_view_stats": markdown_view_stats,
            **source_stats,
            "ocr_request_body_mb": float(audit_result["request_body_mb"]),
            "markdown_request_body_mb": float(markdown_result["request_body_mb"]),
            "payload_attempts": int(audit_result["payload_attempts"]) + int(markdown_result["payload_attempts"]),
            "payload_fallback_count": int(audit_result["payload_fallback_count"]) + int(markdown_result["payload_fallback_count"]),
            "payload_failures": list(audit_result["payload_failures"]) + list(markdown_result["payload_failures"]),
            "raw_sanitizations": list(raw_sanitizations),
            "markdown_sanitizations": list(markdown_sanitizations),
            "raw_response_sha256": _sha256_text(raw_ocr_text),
            "raw_ocr_sha256": _sha256_text(raw_ocr),
            "ocr_reasoning_sha256": _sha256_text(ocr_reasoning) if ocr_reasoning else "",
            "ocr_reasoning_chars": len(ocr_reasoning),
            "markdown_response_sha256": _sha256_text(markdown_raw),
            "markdown_reasoning_sha256": _sha256_text(markdown_reasoning) if markdown_reasoning else "",
            "markdown_reasoning_chars": len(markdown_reasoning),
            "markdown_sha256": _sha256_text(markdown),
            "diagnostic_mode": OCR_DIAGNOSTIC_MODE,
            "include_ocr_annex": INCLUDE_OCR_ANNEX,
            "include_thinking_annex": INCLUDE_THINKING_ANNEX,
            "ocr_generations": 1,
            "markdown_generations": 1,
            "nominal_generations_per_page": NOMINAL_GENERATIONS_PER_PAGE,
            "semantic_retries": SEMANTIC_RETRIES,
            "publication_blocked": False,
            "publication_review_required": bool(publication_review_required),
            "quality_status": quality["status"],
            "quality_warning_count": quality["warning_count"],
            "quality_error_count": quality["error_count"],
            "has_line_items": bool(quality["has_line_items"]),
            "has_totals": bool(quality["has_totals"]),
            "format_complete": bool(quality.get("format_complete")),
            "streaming_ocr": STREAMING_OCR,
            "streaming_markdown": STREAMING_MARKDOWN,
            "thinking_budget_ocr": THINKING_BUDGET_OCR,
            "thinking_budget_markdown": THINKING_BUDGET_MARKDOWN,
            "max_completion_tokens_ocr": MAX_COMPLETION_TOKENS_OCR,
            "max_completion_tokens_markdown": MAX_COMPLETION_TOKENS_MARKDOWN,
            "ocr_seed": OCR_SEED,
            "markdown_seed": MARKDOWN_SEED,
            "model": MODEL_MARKDOWN,
            "model_ocr": MODEL_OCR,
            "model_markdown": MODEL_MARKDOWN,
            "pipeline_version": PIPELINE_VERSION,
            "pipeline_fingerprint": get_pipeline_fingerprint(),
        }
        _log(
            f"✅ Page {page_num}: lectures indépendantes terminées, qualité={quality['status']}, "
            f"OCR={audit_view_stats.get('payload_profile','n/a')}, "
            f"Markdown={markdown_view_stats.get('payload_profile','n/a')}, "
            f"wall={wall_duration_ms / 1000:.1f}s"
        )
        return {
            "page_num": page_num,
            "raw_response": raw_ocr_text,
            "raw_ocr": raw_ocr,
            "ocr_reasoning": ocr_reasoning,
            "markdown_raw_response": markdown_raw,
            "markdown_reasoning": markdown_reasoning,
            "markdown_candidate": markdown_candidate,
            "publication_blocked": False,
            "publication_review_required": bool(publication_review_required),
            "sanitized_canonical": raw_ocr,
            "normalized_canonical": raw_ocr,
            "canonical": raw_ocr,
            "markdown": markdown,
            "quality": quality,
            "stats": stats,
        }
    finally:
        cleanup_page_images(cleanup_paths)


def process_page_with_cache(
    pdf_path: str,
    page_num: int,
    api_key: str,
    is_first_page: bool = False,
    total_pages: Optional[int] = None,
) -> Tuple[str, Dict[str, Any]]:
    del is_first_page
    with tempfile.TemporaryDirectory(prefix="qwen_dual_independent_page_") as image_dir:
        result = process_page(pdf_path, page_num, api_key, image_dir, total_pages=total_pages)
        return result["markdown"], result["stats"]


# =============================================================================
# Checkpoints et empreinte
# =============================================================================


def _sha256_text(value: str) -> str:
    return hashlib.sha256((value or "").encode("utf-8")).hexdigest()


def get_pipeline_fingerprint() -> str:
    payload = {
        "pipeline_version": PIPELINE_VERSION,
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "api_url": API_URL,
        "model_ocr": MODEL_OCR,
        "model_markdown": MODEL_MARKDOWN,
        "parallel_independent_passes": PARALLEL_INDEPENDENT_PASSES,
        "markdown_render_dpi": RENDER_DPI,
        "markdown_detail_dpi": DETAIL_DPI,
        "markdown_expected_views": MARKDOWN_EXPECTED_VIEW_COUNT,
        "markdown_geometry_guide": MARKDOWN_GEOMETRY_GUIDE,
        "geometry_guide_minor_step_percent": GEOMETRY_GUIDE_MINOR_STEP_PERCENT,
        "geometry_guide_label_step_percent": GEOMETRY_GUIDE_LABEL_STEP_PERCENT,
        "geometry_guide_major_step_percent": GEOMETRY_GUIDE_MAJOR_STEP_PERCENT,
        "geometry_guide_max_pixels": GEOMETRY_GUIDE_MAX_PIXELS,
        "ocr_audit_render_dpi": OCR_AUDIT_RENDER_DPI,
        "ocr_audit_detail_dpi": OCR_AUDIT_DETAIL_DPI,
        "ocr_audit_expected_views": OCR_AUDIT_EXPECTED_VIEW_COUNT,
        "jpeg_quality": VIEW_JPEG_QUALITY,
        "ocr_audit_jpeg_quality": OCR_AUDIT_JPEG_QUALITY,
        "max_view_pixels": MAX_VIEW_PIXELS,
        "max_request_body_mb": MAX_REQUEST_BODY_MB,
        "high_resolution": QWEN_HIGH_RES_IMAGES,
        "max_completion_tokens_ocr": MAX_COMPLETION_TOKENS_OCR,
        "max_completion_tokens_markdown": MAX_COMPLETION_TOKENS_MARKDOWN,
        "thinking_budget_ocr": THINKING_BUDGET_OCR,
        "thinking_budget_markdown": THINKING_BUDGET_MARKDOWN,
        "ocr_seed": OCR_SEED,
        "markdown_seed": MARKDOWN_SEED,
        "include_ocr_annex": INCLUDE_OCR_ANNEX,
        "include_thinking_annex": INCLUDE_THINKING_ANNEX,
        "ocr_audit_prompt_sha256": _sha256_text(RAW_OCR_PROMPT),
        "markdown_visual_prompt_sha256": _sha256_text(MARKDOWN_PROMPT),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def get_progress_path(pdf_path: str) -> str:
    return str(Path(pdf_path).with_suffix(".progress.json"))


def load_progress(
    pdf_path: str,
    *,
    expected_source_id: Optional[str] = None,
    expected_page_count: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    path = Path(get_progress_path(pdf_path))
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        _log(f"⚠️ Checkpoint illisible, ignoré : {exc}")
        return {}
    if payload.get("checkpoint_version") != CHECKPOINT_VERSION:
        return {}
    if payload.get("checkpoint_schema") != CHECKPOINT_SCHEMA:
        return {}
    if payload.get("pipeline_fingerprint") != get_pipeline_fingerprint():
        return {}
    if expected_source_id is not None and payload.get("source_id") != expected_source_id:
        return {}
    if expected_page_count is not None and int(payload.get("page_count", -1)) != int(expected_page_count):
        return {}
    pages = payload.get("pages", {})
    if not isinstance(pages, dict):
        return {}
    valid: Dict[str, Dict[str, Any]] = {}
    for key, record in pages.items():
        if not isinstance(record, dict) or record.get("status") != "done":
            continue
        if not isinstance(record.get("normalized_canonical"), str):
            if isinstance(record.get("canonical"), str):
                record["normalized_canonical"] = record["canonical"]
            else:
                continue
        if not isinstance(record.get("markdown"), str):
            continue
        if (OCR_DIAGNOSTIC_MODE or INCLUDE_OCR_ANNEX) and not isinstance(record.get("raw_response"), str):
            continue
        if (OCR_DIAGNOSTIC_MODE or INCLUDE_THINKING_ANNEX) and (
            not isinstance(record.get("ocr_reasoning"), str)
            or not isinstance(record.get("markdown_reasoning"), str)
        ):
            continue
        if OCR_DIAGNOSTIC_MODE and not isinstance(record.get("sanitized_canonical"), str):
            continue
        if not isinstance(record.get("quality"), dict) or not isinstance(record.get("stats"), dict):
            continue
        valid[str(key)] = record
    return valid

def save_progress(
    pdf_path: str,
    pages: Dict[str, Dict[str, Any]],
    *,
    source_id: Optional[str] = None,
    page_count: Optional[int] = None,
) -> None:
    path = Path(get_progress_path(pdf_path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if OCR_DIAGNOSTIC_MODE:
        diagnostic_states = [
            "raw_response", "raw_ocr", "ocr_reasoning", "markdown_raw_response",
            "markdown_reasoning", "sanitized_canonical", "normalized_canonical", "markdown",
        ]
    elif INCLUDE_OCR_ANNEX or INCLUDE_THINKING_ANNEX:
        diagnostic_states = [
            "raw_response", "raw_ocr", "ocr_reasoning",
            "markdown_reasoning", "normalized_canonical", "markdown",
        ]
    else:
        diagnostic_states = ["normalized_canonical", "markdown"]
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "pipeline_version": PIPELINE_VERSION,
        "pipeline_fingerprint": get_pipeline_fingerprint(),
        "diagnostic_mode": OCR_DIAGNOSTIC_MODE,
        "include_ocr_annex": INCLUDE_OCR_ANNEX,
        "include_thinking_annex": INCLUDE_THINKING_ANNEX,
        "diagnostic_states": diagnostic_states,
        "contains_sensitive_document_data": True,
        "source_id": source_id,
        "page_count": int(page_count) if page_count is not None else None,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pages": pages,
    }
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)

def clear_progress(pdf_path: str) -> None:
    try:
        Path(get_progress_path(pdf_path)).unlink(missing_ok=True)
    except Exception as exc:
        _log(f"⚠️ Suppression checkpoint impossible : {exc}")


# =============================================================================
# Assemblage du document final et annexe OCR brute
# =============================================================================


def _longest_backtick_run(text: str) -> int:
    longest = 0
    current = 0
    for char in str(text):
        if char == "`":
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _code_fence_for(text: str) -> str:
    """Choisit une clôture Markdown qui ne peut pas apparaître dans l'OCR brut."""
    return "`" * max(4, _longest_backtick_run(text) + 1)




def build_ocr_page_block(item: Dict[str, Any]) -> str:
    """Construit le bloc OCR indépendant d'une seule page pour final.md."""
    page_num = int(item.get("page_num", 0) or 0)
    raw_provider = str(item.get("raw_response", ""))
    raw = str(item.get("raw_ocr", raw_provider))
    raw_sanitizations = [
        str(value) for value in (item.get("stats", {}).get("raw_sanitizations", []) or [])
    ]
    row_metadata_normalized = "yes" if any(
        value.startswith("visual_row_") for value in raw_sanitizations
    ) else "no"
    sanitizations_text = ",".join(raw_sanitizations) if raw_sanitizations else "none"
    fence = _code_fence_for(raw)
    chunks: List[str] = [
        f"## OCR visuel indépendant — Page {page_num}\n\n",
        OCR_PAGE_START_TEMPLATE.format(page=page_num) + "\n\n",
        "Lecture visuelle indépendante : cette sortie et son thinking n'ont jamais été transmis à l'appel Markdown.\n\n",
        f"<!-- RAW_OCR_META page={page_num} raw_chars={len(raw_provider)} "
        f"raw_sha256={_sha256_text(raw_provider) if raw_provider else 'none'} "
        f"normalized_chars={len(raw)} normalized_sha256={_sha256_text(raw)} "
        f"row_metadata_normalized={row_metadata_normalized} "
        f"sanitizations={sanitizations_text} -->\n\n",
        f"{fence}text\n",
        raw,
    ]
    if not raw.endswith(("\n", "\r")):
        chunks.append("\n")
    chunks.extend([f"{fence}\n\n", OCR_PAGE_END_TEMPLATE.format(page=page_num)])
    return "".join(chunks).rstrip("\n")

def build_ocr_annex(page_results: Sequence[Dict[str, Any]]) -> str:
    """Annexe de diagnostic : OCR visuel indépendant destiné à l'audit."""
    chunks: List[str] = [
        "# Annexe d'audit — OCR visuel indépendant\n\n",
        OCR_ANNEX_START + "\n\n",
        "Cette annexe reproduit une lecture visuelle indépendante. Elle n'a jamais été transmise à l'appel Markdown. "
        "Python peut uniquement harmoniser les indices techniques 1..N et l'attribut groups=N des VISUAL_ROW ; "
        "le texte et l'ordre des groupes restent inchangés.\n",
    ]
    for item in sorted(page_results, key=lambda value: int(value.get("page_num", 0) or 0)):
        page_num = int(item.get("page_num", 0) or 0)
        raw_provider = str(item.get("raw_response", ""))
        raw = str(item.get("raw_ocr", raw_provider))
        raw_sanitizations = [
            str(value) for value in (item.get("stats", {}).get("raw_sanitizations", []) or [])
        ]
        row_metadata_normalized = "yes" if any(
            value.startswith("visual_row_") for value in raw_sanitizations
        ) else "no"
        sanitizations_text = ",".join(raw_sanitizations) if raw_sanitizations else "none"
        fence = _code_fence_for(raw)
        chunks.extend([
            f"\n## OCR brut — Page {page_num}\n\n",
            f"<!-- RAW_OCR_META page={page_num} raw_chars={len(raw_provider)} "
            f"raw_sha256={_sha256_text(raw_provider) if raw_provider else 'none'} "
            f"normalized_chars={len(raw)} normalized_sha256={_sha256_text(raw)} "
            f"row_metadata_normalized={row_metadata_normalized} "
            f"sanitizations={sanitizations_text} -->\n\n",
            f"{fence}text\n",
            raw,
        ])
        if not raw.endswith(("\n", "\r")):
            chunks.append("\n")
        chunks.append(f"{fence}\n")
    chunks.append("\n" + OCR_ANNEX_END)
    return "".join(chunks).rstrip("\n")


def _truncate_reasoning_for_annex(text: str) -> Tuple[str, bool]:
    raw = str(text or "")
    if len(raw) <= THINKING_ANNEX_MAX_CHARS:
        return raw, False
    keep = max(1, THINKING_ANNEX_MAX_CHARS // 2)
    marker = (
        "\n\n[... THINKING TRONQUÉ POUR L'ANNEXE ; "
        f"longueur originale={len(raw)} caractères ...]\n\n"
    )
    return raw[:keep] + marker + raw[-keep:], True


def build_thinking_annex(page_results: Sequence[Dict[str, Any]]) -> str:
    """Expose séparément le thinking des deux passes, sans le transmettre entre elles."""
    chunks: List[str] = [
        "# Annexe — Thinking Qwen (diagnostic)\n\n",
        THINKING_ANNEX_START + "\n\n",
        "Le thinking est un outil d'audit et peut rationaliser une erreur. "
        "Les deux appels sont visuellement indépendants et ne reçoivent jamais la sortie ou le thinking de l'autre.\n",
    ]
    for item in sorted(page_results, key=lambda value: int(value.get("page_num", 0) or 0)):
        page_num = int(item.get("page_num", 0) or 0)
        chunks.append(f"\n#### THINKING PAGE {page_num} ####\n")
        for stage_label, key in (("OCR D'AUDIT INDÉPENDANT", "ocr_reasoning"), ("MARKDOWN VISUEL INDÉPENDANT", "markdown_reasoning")):
            raw = str(item.get(key, "") or "")
            shown, truncated = _truncate_reasoning_for_annex(raw)
            fence = _code_fence_for(shown)
            chunks.extend([
                f"\n##### {stage_label}\n\n",
                f"<!-- THINKING_META page={page_num} stage={key} chars={len(raw)} "
                f"sha256={_sha256_text(raw) if raw else 'none'} "
                f"truncated={'yes' if truncated else 'no'} -->\n\n",
                f"{fence}text\n",
                shown if shown else "[AUCUN REASONING_CONTENT RETOURNÉ]",
            ])
            if not (shown or "").endswith(("\n", "\r")):
                chunks.append("\n")
            chunks.append(f"{fence}\n")
    chunks.append("\n" + THINKING_ANNEX_END)
    return "".join(chunks).rstrip("\n")


def assemble_document_with_ocr_annex(
    rendered_document: str,
    page_results: Sequence[Dict[str, Any]],
) -> str:
    """Assemble final.md sans modifier le contenu produit par les deux appels Qwen."""
    ordered = sorted(page_results, key=lambda value: int(value.get("page_num", 0) or 0))
    if FINAL_MD_LAYOUT == "legacy_annex":
        rendered = str(rendered_document or "").strip("\n")
        chunks: List[str] = [RENDERED_DOCUMENT_START, "", rendered, "", RENDERED_DOCUMENT_END]
        if INCLUDE_OCR_ANNEX:
            chunks.extend(["", build_ocr_annex(ordered)])
        if INCLUDE_THINKING_ANNEX:
            chunks.extend(["", build_thinking_annex(ordered)])
        return "\n".join(chunks).rstrip("\n") + "\n"

    chunks: List[str] = [RENDERED_DOCUMENT_START]
    for item in ordered:
        page_num = int(item.get("page_num", 0) or 0)
        markdown = str(item.get("markdown", "") or "").strip("\n")
        chunks.extend([
            "",
            PAGE_BUNDLE_START_TEMPLATE.format(page=page_num),
            RENDERED_PAGE_START_TEMPLATE.format(page=page_num),
            markdown,
            RENDERED_PAGE_END_TEMPLATE.format(page=page_num),
        ])
        if INCLUDE_OCR_ANNEX:
            chunks.extend(["", build_ocr_page_block(item)])
        chunks.append(PAGE_BUNDLE_END_TEMPLATE.format(page=page_num))
    chunks.extend(["", RENDERED_DOCUMENT_END])
    # Le thinking reste dans les JSONL de diagnostic. Il n'est ajouté au fichier final
    # que si l'option explicite INCLUDE_THINKING_ANNEX est activée.
    if INCLUDE_THINKING_ANNEX:
        chunks.extend(["", build_thinking_annex(ordered)])
    return "\n".join(chunks).rstrip("\n") + "\n"


def extract_rendered_document(final_markdown: str) -> str:
    """Extrait uniquement le Markdown métier, y compris depuis le format page par page."""
    text = str(final_markdown or "")
    page_pattern = re.compile(
        r"<!-- RENDERED_PAGE_START page=(\d+) -->\s*(.*?)\s*<!-- RENDERED_PAGE_END page=\1 -->",
        re.DOTALL,
    )
    page_blocks = [(int(match.group(1)), match.group(2).strip("\n")) for match in page_pattern.finditer(text)]
    if page_blocks:
        return "\n\n".join(block for _, block in sorted(page_blocks)).strip("\n")
    start = text.find(RENDERED_DOCUMENT_START)
    end = text.find(RENDERED_DOCUMENT_END)
    if start >= 0 and end > start:
        start += len(RENDERED_DOCUMENT_START)
        return text[start:end].strip("\n")
    if end >= 0:
        return text[:end].strip("\n")
    annex = text.find(OCR_ANNEX_START)
    if annex >= 0:
        return text[:annex].strip("\n")
    return text.strip("\n")


def extract_ocr_annex(final_markdown: str) -> str:
    """Extrait les blocs OCR, qu'ils soient intercalés par page ou en annexe héritée."""
    text = str(final_markdown or "")
    page_pattern = re.compile(
        r"<!-- OCR_PAGE_START page=(\d+)[^>]*-->\s*(.*?)\s*<!-- OCR_PAGE_END page=\1 -->",
        re.DOTALL,
    )
    page_blocks = [(int(match.group(1)), match.group(2).strip("\n")) for match in page_pattern.finditer(text)]
    if page_blocks:
        return "\n\n".join(
            f"## OCR visuel indépendant — Page {page}\n\n{block}"
            for page, block in sorted(page_blocks)
        ).strip("\n")
    start = text.find(OCR_ANNEX_START)
    end = text.rfind(OCR_ANNEX_END)
    if start < 0 or end <= start:
        return ""
    start += len(OCR_ANNEX_START)
    return text[start:end].strip("\n")


# =============================================================================
# Validation finale et métriques
# =============================================================================


def extract_thinking_annex(final_markdown: str) -> str:
    """Extrait l'annexe thinking diagnostique."""
    text = str(final_markdown or "")
    start = text.find(THINKING_ANNEX_START)
    end = text.rfind(THINKING_ANNEX_END)
    if start < 0 or end <= start:
        return ""
    start += len(THINKING_ANNEX_START)
    return text[start:end].strip("\n")

def _split_markdown_row(line: str) -> List[str]:
    stripped = (line or "").strip()
    if not (stripped.startswith("|") and stripped.endswith("|")):
        return []
    content = stripped[1:-1]
    cells: List[str] = []
    buffer: List[str] = []
    backslashes = 0
    for char in content:
        if char == "\\":
            buffer.append(char)
            backslashes += 1
            continue
        if char == "|" and backslashes % 2 == 0:
            cells.append("".join(buffer).strip())
            buffer = []
        else:
            buffer.append(char)
        backslashes = 0
    cells.append("".join(buffer).strip())
    return cells


def validate_canonical_markdown_structure(final_markdown: str, page_count: int) -> None:
    markers = [
        int(match.group(1))
        for line in (final_markdown or "").splitlines()
        if (match := PAGE_MARKER_RE.match(line))
    ]
    expected = list(range(1, int(page_count) + 1))
    if markers != expected:
        raise RuntimeError(f"Marqueurs PAGE invalides : obtenu={markers}, attendu={expected}")

    lines = (final_markdown or "").splitlines()
    for index, line in enumerate(lines):
        header_cells = _split_markdown_row(line)
        if not header_cells or index + 1 >= len(lines):
            continue
        separator_cells = _split_markdown_row(lines[index + 1])
        if not separator_cells or not all(
            re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in separator_cells
        ):
            continue
        width = len(header_cells)
        if len(separator_cells) != width:
            raise RuntimeError(f"Tableau Markdown incohérent ligne {index + 1}.")
        cursor = index + 2
        while cursor < len(lines):
            row_cells = _split_markdown_row(lines[cursor])
            if not row_cells:
                break
            if len(row_cells) != width:
                raise RuntimeError(f"Largeur Markdown incohérente ligne {cursor + 1}.")
            cursor += 1


_FINANCIAL_ROW_LABEL_RE = re.compile(
    r"(?:^|\b)(?:total|subtotal|sous[- ]?total|net(?:\s+à?\s*payer)?|solde|balance(?:\s+due)?|"
    r"amount\s+due|gross\s+total|total\s+due|contribution|levy|redevance|"
    r"éco[- ]?(?:contribution|participation)|eco[- ]?(?:contribution|participation)|"
    r"remise|discount|escompte|acompte|deposit|frais|fees?|port|shipping|freight|"
    r"consigne|retenue|withholding|timbre|stamp|arrondi|rounding|octroi(?:\s+de\s+mer)?|octroi\s+r[eé]gional|\bomr?\b|droits?|dut(?:y|ies)|taxe(?:s)?)(?:\b|$)",
    flags=re.IGNORECASE,
)

_MONETARY_VALUE_RE = re.compile(
    r"^[+−-]?\(?\s*\d[\d\s.,'’]*\s*(?:€|EUR|USD|GBP|CHF|CAD|AUD|JPY|CNY|RMB|£|\$)?\s*\)?$",
    flags=re.IGNORECASE,
)


def _looks_like_financial_row_label(value: str) -> bool:
    text = re.sub(r"<br\s*/?>", " ", str(value or ""), flags=re.IGNORECASE)
    text = re.sub(r"[*_`]+", "", text).strip()
    return bool(text and _FINANCIAL_ROW_LABEL_RE.search(text))


def _looks_like_monetary_value(value: str) -> bool:
    text = re.sub(r"<br\s*/?>", " ", str(value or ""), flags=re.IGNORECASE).strip()
    if not text or "%" in text or re.fullmatch(r"\[SANS_ENTETE_\d+\]", text, flags=re.IGNORECASE):
        return False
    return bool(_MONETARY_VALUE_RE.fullmatch(text))


def detect_accounting_integrity_warnings(markdown: str) -> List[str]:
    """Détecte sans corriger les lignes financières probablement utilisées comme en-têtes."""
    warnings: List[str] = []
    lines = str(markdown or "").splitlines()
    page = 0
    index = 0
    while index < len(lines):
        marker = PAGE_MARKER_RE.match(lines[index])
        if marker:
            page = int(marker.group(1))
        header = _split_markdown_row(lines[index])
        if not header or index + 1 >= len(lines):
            index += 1
            continue
        separator = _split_markdown_row(lines[index + 1])
        if not separator or len(separator) != len(header) or not all(
            re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in separator
        ):
            index += 1
            continue
        rows: List[List[str]] = []
        cursor = index + 2
        while cursor < len(lines):
            cells = _split_markdown_row(lines[cursor])
            if not cells:
                break
            if len(cells) == len(header):
                rows.append(cells)
            cursor += 1
        for column, value in enumerate(header):
            if not _looks_like_financial_row_label(value):
                continue
            below_labels = [
                row[column] for row in rows
                if column < len(row) and _looks_like_financial_row_label(row[column])
            ]
            if below_labels:
                warnings.append(
                    f"page={page}:financial_label_in_header_ligne={index + 1}:colonne={column + 1}:"
                    f"header={value!r}:below={below_labels[:3]!r}"
                )
            if column + 1 < len(header) and _looks_like_monetary_value(header[column + 1]):
                warnings.append(
                    f"page={page}:financial_amount_in_header_ligne={index + 1}:"
                    f"label={value!r}:amount={header[column + 1]!r}"
                )
        index = max(cursor, index + 1)
    return list(dict.fromkeys(warnings))



_COMPACT_SUFFIX_RE = re.compile(
    r"^(?:x|×|pcs?|pc|u|un|kg|g|mg|l|ml|m|m2|m²|m3|m³|h|hr|hrs|j|d|bl|mtr|pr|ea|lot|box|ctn)$",
    flags=re.IGNORECASE,
)


def detect_compact_group_warnings(markdown: str) -> List[str]:
    """Signale sans corriger quelques séparations compactes très probables."""
    warnings: List[str] = []
    lines = str(markdown or "").splitlines()
    page = 0
    for line_no, line in enumerate(lines, start=1):
        marker = PAGE_MARKER_RE.match(line)
        if marker:
            page = int(marker.group(1))
            continue
        cells = _split_markdown_row(line)
        if not cells:
            continue
        for idx in range(len(cells) - 1):
            left = re.sub(r"[*_`]", "", cells[idx]).strip()
            right = re.sub(r"[*_`]", "", cells[idx + 1]).strip()
            if re.fullmatch(r"[+−-]?\d+(?:[,.]\d+)?", left) and _COMPACT_SUFFIX_RE.fullmatch(right):
                warnings.append(
                    f"page={page}:possible_compact_group_split_ligne={line_no}:colonnes={idx + 1},{idx + 2}:values={left!r},{right!r}"
                )
    return list(dict.fromkeys(warnings))


_QTY_HEADER_RE = re.compile(r"(?:^|\b)(?:qte|qté|qty|quantity|quantité|quantite)(?:\b|$)", re.IGNORECASE)
_DESC_HEADER_RE = re.compile(r"(?:designation|désignation|description|libellé|libelle|item|article|product)", re.IGNORECASE)
_AMOUNT_HEADER_RE = re.compile(r"(?:montant|amount|line\s*total|total\s*ligne)", re.IGNORECASE)
_TAX_HEADER_RE = re.compile(r"(?:tva|vat|gst|taxe?|tax)(?:\b|\[TRONQUÉ\])", re.IGNORECASE)
_TAX_MATRIX_RE = re.compile(r"(?:codes?\s*(?:tva|vat|gst|taxe?|tax)|exon[eé]r[eé]|soumis|tax\s*codes?|vat\s*codes?|gst\s*codes?|octroi(?:\s+de\s+mer)?|octroi\s+r[eé]gional|\bomr?\b)", re.IGNORECASE)
_CONTRIBUTION_RE = re.compile(r"(?:éco|eco)[- ]?(?:contribution|participation)|contribution|redevance|levy|fee|frais|surcharge|remise|discount|escompte|port|retenue|arrondi|octroi(?:\s+de\s+mer)?|octroi\s+r[eé]gional|\bomr?\b|droits?|taxe", re.IGNORECASE)
_NUMERIC_CORE_RE = re.compile(r"(?<![A-Za-z0-9])\d+(?:[,.]\d+)?(?![A-Za-z0-9])")
_COMPACT_QTY_RE = re.compile(r"^[+−-]?\d+(?:[,.]\d+)?\s*(?:x|×|pcs?|pc|u|un|kg|g|mg|l|ml|m|m2|m²|m3|m³|h|hr|hrs|j|d|bl|mtr|pr|ea|lot|box|ctn)$", re.IGNORECASE)
_UNIT_PRICE_HEADER_RE = re.compile(r"(?:PRIX\s+UNIT|P\.?U\.?\s*(?:HT)?|UNIT\s+PRICE|PRICE\s+UNIT)", re.IGNORECASE)
_NET_UNIT_HEADER_RE = re.compile(r"(?:P\.?U\.?\s*NET|PRIX\s+NET|NET\s+UNIT|UNIT\s+NET)", re.IGNORECASE)
_SPECIAL_SHORT_FINANCIAL_RE = re.compile(
    r"^(?:eco|éco|om|omr|octroi(?:\s+de\s+mer)?|octroi\s+r[eé]gional|"
    r"remise|discount|escompte|taxe|tax|droit|frais|fee|port|retenue|arrondi)$",
    re.IGNORECASE,
)
_ACCOUNTING_TEXT_LABEL_RE = re.compile(
    r"(?:TOTAL\s+HT|TOTAL\s+TVA|TOTAL\s+TTC|NET\s+A\s+PAYER|SOLDE|"
    r"ECO[- ]?CONTRIBUTION|CONTRIBUTION|REMISE|ESCOMPTE|ACOMPTE|FRAIS|PORT|"
    r"RETENUE|ARRONDI|OCTROI(?:\s+DE\s+MER)?|OCTROI\s+R[EÉ]GIONAL|\bOMR?\b|"
    r"DROITS?|TAXE(?:S)?|TVA|VAT|GST|BASE(?:\s+HT|\s+TAXABLE)?)",
    re.IGNORECASE,
)
_ACCOUNTING_TEXT_VALUE_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:\d{1,3}(?:[ .]\d{3})+|\d+)(?:[,.]\d{2,4})"
    r"(?:\s*(?:%|EUR|EURO|€|USD|GBP|CHF|CAD|AUD|JPY|CNY|RMB|£|\$))?(?![A-Za-z0-9])",
    re.IGNORECASE,
)


def _plain_cell(value: str) -> str:
    value = re.sub(r"<br\s*/?>", " ", str(value or ""), flags=re.IGNORECASE)
    value = re.sub(r"[*_`]", "", value)
    return re.sub(r"\s+", " ", value).strip()


def _iter_markdown_tables(markdown: str):
    lines = str(markdown or "").splitlines()
    page = 0
    index = 0
    while index < len(lines):
        marker = PAGE_MARKER_RE.match(lines[index])
        if marker:
            page = int(marker.group(1))
        header = _split_markdown_row(lines[index])
        if not header or index + 1 >= len(lines):
            index += 1
            continue
        separator = _split_markdown_row(lines[index + 1])
        if not separator or len(separator) != len(header) or not all(
            re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in separator
        ):
            index += 1
            continue
        rows = []
        cursor = index + 2
        while cursor < len(lines):
            row = _split_markdown_row(lines[cursor])
            if not row or len(row) != len(header):
                break
            rows.append(row)
            cursor += 1
        yield page, index + 1, header, rows
        index = max(cursor, index + 1)



def _fuzzy_normalize(value: str) -> str:
    value = unicodedata.normalize("NFKD", _plain_cell(value).lower())
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _ocr_visual_tables(raw_ocr: str) -> List[List[Tuple[str, List[str]]]]:
    tables: List[List[Tuple[str, List[str]]]] = []
    for table_match in re.finditer(
        r"\[\[VISUAL_TABLE\b[^\]]*\]\](.*?)\[\[/VISUAL_TABLE\]\]",
        raw_ocr or "", flags=re.IGNORECASE | re.DOTALL,
    ):
        rows: List[Tuple[str, List[str]]] = []
        block = table_match.group(1)
        for row_match in re.finditer(
            r"\[\[VISUAL_ROW\b([^\]]*)\]\](.*?)\[\[/VISUAL_ROW\]\]",
            block, flags=re.IGNORECASE | re.DOTALL,
        ):
            attrs = row_match.group(1)
            kind_match = re.search(r"\bkind=([^\s\]]+)", attrs, flags=re.IGNORECASE)
            kind = (kind_match.group(1).lower() if kind_match else "data")
            indexed = []
            for cell_match in re.finditer(r"(?m)^\s*(\d+)=(.*)$", row_match.group(2)):
                indexed.append((int(cell_match.group(1)), cell_match.group(2).strip()))
            indexed.sort(key=lambda item: item[0])
            cells = [value for _, value in indexed]
            rows.append((kind, cells))
        tables.append(rows)
    return tables


def _match_ocr_markdown_tables(
    markdown: str, raw_ocr: str
) -> List[Tuple[List[Tuple[str, List[str]]], Tuple[int, int, List[str], List[List[str]]]]]:
    """Associe mécaniquement les tables OCR et Markdown par similarité de leurs en-têtes."""
    md_tables = list(_iter_markdown_tables(markdown))
    ocr_tables = _ocr_visual_tables(raw_ocr)
    available = set(range(len(md_tables)))
    pairs = []
    for ocr_rows in ocr_tables:
        ocr_header = next((cells for kind, cells in ocr_rows if kind == "header"), [])
        ocr_sig = _fuzzy_normalize(" | ".join(ocr_header))
        best_idx = None
        best_score = -1.0
        for idx in available:
            md_header = md_tables[idx][2]
            md_sig = _fuzzy_normalize(" | ".join(md_header))
            score = SequenceMatcher(None, ocr_sig, md_sig).ratio()
            ocr_tokens = set(ocr_sig.split())
            md_tokens = set(md_sig.split())
            if ocr_tokens or md_tokens:
                score += 0.4 * (len(ocr_tokens & md_tokens) / max(1, len(ocr_tokens | md_tokens)))
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is not None and best_score >= 0.25:
            available.discard(best_idx)
            pairs.append((ocr_rows, md_tables[best_idx]))
    return pairs


def _detect_truncation_losses(markdown: str, raw_ocr: str) -> List[str]:
    """Compare les fragments tronqués par table, cellule et bloc de texte."""
    critical: List[str] = []

    def base_fragment(value: str) -> str:
        return re.sub(r"\[TRONQU[EÉ]\]", "", value or "", flags=re.IGNORECASE).strip()

    def mapped_index(ocr_pos: int, ocr_width: int, md_width: int) -> Optional[int]:
        if md_width == ocr_width:
            return ocr_pos
        right_distance = ocr_width - 1 - ocr_pos
        candidate = md_width - 1 - right_distance
        if 0 <= candidate < md_width:
            return candidate
        return None

    for ocr_rows, md_table in _match_ocr_markdown_tables(markdown, raw_ocr):
        page, line_no, md_header, md_rows = md_table
        md_width = len(md_header)
        ocr_header = next((list(cells) for kind, cells in ocr_rows if kind == "header"), [])
        ocr_width = len(ocr_header)

        for pos, value in enumerate(ocr_header):
            if not re.search(r"\[TRONQU[EÉ]\]", value, flags=re.IGNORECASE):
                continue
            fragment = base_fragment(value)
            target_pos = mapped_index(pos, ocr_width, md_width)
            if target_pos is None or target_pos >= len(md_header) or not fragment:
                continue
            md_value = _plain_cell(md_header[target_pos])
            if fragment.lower() in md_value.lower() and "[TRONQUÉ]" not in md_value.upper():
                critical.append(
                    f"page={page}:truncated_header_marker_lost_table_line={line_no}:col={target_pos + 1}:fragment={fragment!r}"
                )

        ocr_data = [list(cells) for kind, cells in ocr_rows if kind not in {"header", "continuation", "note"}]
        for ocr_row in ocr_data:
            for pos, raw_value in enumerate(ocr_row):
                if not re.search(r"\[TRONQU[EÉ]\]", raw_value, flags=re.IGNORECASE):
                    continue
                fragment = base_fragment(raw_value)
                # Les très courts codes de données (ex. 01) et une devise complète (ex. EUR)
                # ne sont pas déclarés tronqués sur la seule foi de l'audit OCR indépendant.
                # Les en-têtes restent contrôlés séparément ci-dessus.
                if len(fragment) < 3:
                    continue
                if re.search(r"\b(?:EUR|USD|GBP|CAD|CHF|JPY|XCD)$", fragment.upper()):
                    continue
                anchors = [
                    (abs(idx - pos), 0 if idx < pos else 1, _plain_cell(value))
                    for idx, value in enumerate(ocr_row)
                    if idx != pos and value and not re.search(r"\[TRONQU[EÉ]\]", value, flags=re.IGNORECASE)
                    and len(_plain_cell(value)) >= 2
                ]
                anchors.sort(key=lambda item: (item[0], item[1], -len(item[2])))
                matched_md_row: Optional[List[str]] = None
                matched_anchor = ""
                for _, _, anchor in anchors:
                    for row in md_rows:
                        if any(anchor.lower() in _plain_cell(cell).lower() for cell in row):
                            matched_md_row = row
                            matched_anchor = anchor
                            break
                    if matched_md_row is not None:
                        break
                if matched_md_row is None:
                    continue
                target_pos = mapped_index(pos, len(ocr_row), len(matched_md_row))
                candidates: List[Tuple[int, str]] = []
                if target_pos is not None and target_pos < len(matched_md_row):
                    candidates.append((target_pos, _plain_cell(matched_md_row[target_pos])))
                for idx, cell in enumerate(matched_md_row):
                    plain = _plain_cell(cell)
                    if fragment.lower() in plain.lower() and all(idx != existing for existing, _ in candidates):
                        candidates.append((idx, plain))
                relevant = [(idx, cell) for idx, cell in candidates if fragment.lower() in cell.lower()]
                # Cas fréquent au bord droit : l'OCR fusionne le montant et la
                # colonne de taxe tronquée dans son dernier groupe, tandis que le
                # Markdown les sépare correctement. Le marqueur appartient alors
                # à la dernière colonne, pas au montant immédiatement à gauche.
                split_amount_tax_satisfied = False
                if (
                    target_pos is not None
                    and len(ocr_row) < len(matched_md_row)
                    and target_pos == len(matched_md_row) - 1
                    and "[TRONQUÉ]" in _plain_cell(matched_md_row[target_pos]).upper()
                    and _looks_like_monetary_value(fragment)
                ):
                    left_start = max(0, target_pos - 2)
                    split_amount_tax_satisfied = any(
                        fragment.lower() in _plain_cell(matched_md_row[idx]).lower()
                        for idx in range(left_start, target_pos)
                    )
                if split_amount_tax_satisfied:
                    continue
                if relevant and not any("[TRONQUÉ]" in cell.upper() for _, cell in relevant):
                    critical.append(
                        f"page={page}:truncated_value_marker_lost_table_line={line_no}:anchor={matched_anchor!r}:col={relevant[0][0] + 1}:fragment={fragment!r}"
                    )

    # Blocs de texte tronqués hors tableaux : le fragment repris doit garder son marqueur.
    standalone_truncated_blocks = 0
    for match in re.finditer(
        r"\[\[TEXT_BLOCK\b[^\]]*state=truncated[^\]]*\]\]\s*(.*?)\s*\[\[/TEXT_BLOCK\]\]",
        raw_ocr or "", flags=re.IGNORECASE | re.DOTALL,
    ):
        raw_fragment = _plain_cell(match.group(1))
        fragment = base_fragment(raw_fragment)
        if not fragment:
            standalone_truncated_blocks += 1
            continue
        if len(fragment) < 2:
            continue
        for line in str(markdown or "").splitlines():
            plain = _plain_cell(line)
            if fragment.lower() in plain.lower() and "[TRONQUÉ]" not in plain.upper():
                critical.append(f"truncated_text_marker_lost={fragment!r}:markdown={plain!r}")
                break
    if standalone_truncated_blocks:
        standalone_markers = sum(
            1 for line in str(markdown or "").splitlines()
            if _plain_cell(line).upper() == "[TRONQUÉ]"
        )
        if standalone_markers < standalone_truncated_blocks:
            critical.append(
                f"standalone_truncated_block_lost=ocr:{standalone_truncated_blocks}:markdown:{standalone_markers}"
            )

    # Un OCR_FLAG isolé ne suffit pas à imposer une troncature de devise.
    return list(dict.fromkeys(critical))


def _detect_ocr_hidden_track_losses(markdown: str, raw_ocr: str) -> List[str]:
    """Signale une piste OCR répétée qui disparaît dans le tableau Markdown."""
    critical: List[str] = []
    pairs = _match_ocr_markdown_tables(markdown, raw_ocr)
    for ocr_rows, md_table in pairs:
        page, line_no, md_header, _ = md_table
        ocr_header = next((cells for kind, cells in ocr_rows if kind == "header"), [])
        header_joined = " | ".join(_plain_cell(cell) for cell in ocr_header)
        if not (re.search(r"DESIGNATION|DESCRIPTION|LIBELL", header_joined, flags=re.IGNORECASE)
                and re.search(r"\bQTE\b|QUANTIT|QTY", header_joined, flags=re.IGNORECASE)
                and re.search(r"MONTANT|AMOUNT", header_joined, flags=re.IGNORECASE)):
            continue
        data_rows = [cells for kind, cells in ocr_rows if kind not in {"header", "continuation", "note"}]
        if not data_rows:
            continue
        max_width = max(len(row) for row in data_rows)
        rich_max_rows = [row for row in data_rows if len(row) == max_width and sum(bool(_plain_cell(c)) for c in row) >= 5]
        if max_width > len(ocr_header) and len(rich_max_rows) >= 2 and len(md_header) < max_width:
            critical.append(
                f"page={page}:ocr_repeated_track_missing_table_line={line_no}:ocr_header={len(ocr_header)}:ocr_data={max_width}:markdown={len(md_header)}:rows={len(rich_max_rows)}"
            )
    return list(dict.fromkeys(critical))


def _detect_ocr_financial_pair_losses(markdown: str, raw_ocr: str) -> List[str]:
    """Vérifie qu'un couple financier visible dans une ligne OCR reste dans une ligne Markdown."""
    critical: List[str] = []
    label_re = re.compile(
        r"\b(?:TOTAL\s+HT|TOTAL\s+TVA|TOTAL\s+TTC|NET\s+A\s+PAYER|SOLDE|ECO[- ]?CONTRIBUTION|CONTRIBUTION|REMISE|ESCOMPTE|ACOMPTE|FRAIS|PORT|RETENUE|ARRONDI|OCTROI(?:\s+DE\s+MER)?|OCTROI\s+R[EÉ]GIONAL|OMR?|DROITS?|TAXE(?:S)?)\b",
        flags=re.IGNORECASE,
    )
    md_rows = []
    for page, line_no, header, rows in _iter_markdown_tables(markdown):
        md_rows.append((page, line_no, header))
        md_rows.extend((page, line_no, row) for row in rows)

    def norm(value: str) -> str:
        return re.sub(r"\s+", "", _plain_cell(value).upper().replace("[TRONQUÉ]", ""))

    for table in _ocr_visual_tables(raw_ocr):
        for _, cells in table:
            for idx, cell in enumerate(cells):
                label = _plain_cell(cell)
                if not label_re.search(label):
                    continue
                amount = ""
                for candidate in cells[idx + 1:]:
                    candidate_plain = _plain_cell(candidate)
                    if _looks_like_monetary_value(candidate_plain):
                        amount = candidate_plain
                        break
                if not amount:
                    continue
                found = False
                for page, line_no, row in md_rows:
                    has_label = any(_fuzzy_normalize(label) in _fuzzy_normalize(value) for value in row if _plain_cell(value))
                    has_amount = any(norm(amount) == norm(value) or norm(amount) in norm(value) for value in row if _plain_cell(value))
                    if has_label and has_amount:
                        found = True
                        break
                if not found:
                    critical.append(f"financial_pair_lost={label!r}:{amount!r}")
    return list(dict.fromkeys(critical))


def _detect_special_row_track_shifts(markdown: str, raw_ocr: str) -> List[str]:
    """Compare la position des groupes des lignes de contribution lorsque les largeurs concordent."""
    critical: List[str] = []
    for ocr_rows, md_table in _match_ocr_markdown_tables(markdown, raw_ocr):
        page, line_no, md_header, md_rows = md_table
        ocr_header = next((cells for kind, cells in ocr_rows if kind == "header"), [])
        if len(ocr_header) != len(md_header):
            continue
        ocr_special = [cells for kind, cells in ocr_rows if any(_CONTRIBUTION_RE.search(_plain_cell(c)) for c in cells)]
        for ocr_row in ocr_special:
            designation = next((_plain_cell(c) for c in ocr_row if _CONTRIBUTION_RE.search(_plain_cell(c))), "")
            md_row = next((row for row in md_rows if any(_CONTRIBUTION_RE.search(_plain_cell(c)) for c in row)), None)
            if md_row is None:
                continue
            for idx, value in enumerate(ocr_row):
                token = _plain_cell(value)
                if not token or _CONTRIBUTION_RE.search(token) or token in {"[TRONQUE]", "[TRONQUÉ]"}:
                    continue
                if idx < len(md_row) and token.lower() in _plain_cell(md_row[idx]).lower():
                    continue
                elsewhere = [j for j, cell in enumerate(md_row) if token.lower() == _plain_cell(cell).lower()]
                if elsewhere:
                    critical.append(
                        f"page={page}:special_row_track_shift_table_line={line_no}:designation={designation!r}:token={token!r}:ocr_col={idx + 1}:markdown_col={elsewhere[0] + 1}"
                    )
    return list(dict.fromkeys(critical))


def _detect_bank_identifier_disagreement(markdown: str, raw_ocr: str) -> List[str]:
    """Signale un IBAN émis par Markdown mais absent de l'audit OCR indépendant."""
    critical: List[str] = []
    if not str(raw_ocr or "").strip():
        return critical
    raw_compact = re.sub(r"[^A-Z0-9]", "", (raw_ocr or "").upper())
    for line in str(markdown or "").splitlines():
        if not re.search(r"COMPTE\s+BANCAIRE|IBAN|RIB|BIC|SWIFT", line, flags=re.IGNORECASE):
            continue
        compact = re.sub(r"[^A-Z0-9]", "", line.upper())
        candidates: List[str] = []
        # Longueurs IBAN usuelles ; FR est prioritaire pour les documents français.
        iban_lengths = {"FR": 27, "DE": 22, "ES": 24, "IT": 27, "BE": 16, "NL": 18, "LU": 20, "PT": 25, "GB": 22, "CH": 21}
        for country, length in iban_lengths.items():
            for match in re.finditer(rf"{country}\d{{2}}[A-Z0-9]{{{length - 4}}}", compact):
                candidates.append(match.group(0))
        for iban in dict.fromkeys(candidates):
            if iban not in raw_compact:
                critical.append(f"bank_identifier_not_supported_by_ocr={iban}")
    return list(dict.fromkeys(critical))

def _detect_special_row_left_compaction(markdown: str) -> List[str]:
    """Repère une ligne spéciale probablement décalée d'une ou plusieurs pistes vers la gauche."""
    critical: List[str] = []
    for page, line_no, header, rows in _iter_markdown_tables(markdown):
        normalized = [_plain_cell(cell) for cell in header]
        desc_idx = next((i for i, cell in enumerate(normalized) if _DESC_HEADER_RE.search(cell)), None)
        qty_idx = next((i for i, cell in enumerate(normalized) if _QTY_HEADER_RE.search(cell)), None)
        unit_idx = next((i for i, cell in enumerate(normalized) if _UNIT_PRICE_HEADER_RE.search(cell)), None)
        net_idx = next((i for i, cell in enumerate(normalized) if _NET_UNIT_HEADER_RE.search(cell)), None)
        amount_idx = next((i for i, cell in enumerate(normalized) if _AMOUNT_HEADER_RE.search(cell)), None)
        if None in {desc_idx, qty_idx, unit_idx, net_idx, amount_idx}:
            continue
        assert desc_idx is not None and qty_idx is not None and unit_idx is not None
        assert net_idx is not None and amount_idx is not None
        unnamed_between_net_amount = [
            i for i in range(net_idx + 1, amount_idx)
            if re.fullmatch(r"\[SANS_ENTETE_\d+\]", str(header[i]).strip(), flags=re.IGNORECASE)
        ]
        for row_no, row in enumerate(rows, start=1):
            designation = _plain_cell(row[desc_idx]) if desc_idx < len(row) else ""
            if not _CONTRIBUTION_RE.search(designation):
                continue
            qty_value = _plain_cell(row[qty_idx]) if qty_idx < len(row) else ""
            unit_value = _plain_cell(row[unit_idx]) if unit_idx < len(row) else ""
            net_value = _plain_cell(row[net_idx]) if net_idx < len(row) else ""
            amount_value = _plain_cell(row[amount_idx]) if amount_idx < len(row) else ""
            financial_values = [
                _plain_cell(row[i]) for i in unnamed_between_net_amount if i < len(row)
            ]
            compacted_without_extra_track = (
                _COMPACT_QTY_RE.fullmatch(qty_value)
                and _parse_decimal_cell(unit_value) is not None
                and _SPECIAL_SHORT_FINANCIAL_RE.fullmatch(net_value)
                and (not amount_value or _parse_decimal_cell(amount_value) is not None)
            )
            compacted_with_extra_track = (
                bool(unnamed_between_net_amount)
                and _COMPACT_QTY_RE.fullmatch(qty_value)
                and _parse_decimal_cell(unit_value) is not None
                and not net_value
                and any(_SPECIAL_SHORT_FINANCIAL_RE.fullmatch(value) for value in financial_values if value)
                and (not amount_value or _parse_decimal_cell(amount_value) is not None)
            )
            if compacted_without_extra_track or compacted_with_extra_track:
                critical.append(
                    f"page={page}:special_row_probable_left_compaction_table_line={line_no}:"
                    f"row={row_no}:qty={qty_value!r}:unit={unit_value!r}:net={net_value!r}:"
                    f"financial={financial_values!r}:amount={amount_value!r}"
                )
    return list(dict.fromkeys(critical))


def _detect_adjacent_tax_totals_split(markdown: str) -> List[str]:
    """Deux cadres fermés et alignés peuvent légitimement rester deux tableaux."""
    return []

def _detect_accounting_text_pair_losses(markdown: str, raw_ocr: str) -> List[str]:
    """Vérifie les couples comptables visibles hors tableaux OCR, notamment remises et octrois."""
    critical: List[str] = []
    md_lines = [_plain_cell(line) for line in str(markdown or "").splitlines() if _plain_cell(line)]
    for block_match in re.finditer(
        r"\[\[TEXT_BLOCK\b[^\]]*\]\]\s*(.*?)\s*\[\[/TEXT_BLOCK\]\]",
        raw_ocr or "", flags=re.IGNORECASE | re.DOTALL,
    ):
        for raw_line in block_match.group(1).splitlines():
            line = _plain_cell(raw_line)
            label_match = _ACCOUNTING_TEXT_LABEL_RE.search(line)
            value_match = _ACCOUNTING_TEXT_VALUE_RE.search(line)
            if not label_match or not value_match:
                continue
            label_norm = _fuzzy_normalize(label_match.group(0))
            value_norm = re.sub(r"\s+", "", value_match.group(0).upper().replace("EURO", "EUR").replace("€", "EUR"))
            found = any(
                label_norm in _fuzzy_normalize(md_line)
                and value_norm in re.sub(r"\s+", "", md_line.upper().replace("EURO", "EUR").replace("€", "EUR"))
                for md_line in md_lines
            )
            if not found:
                critical.append(
                    f"accounting_text_pair_lost={label_match.group(0)!r}:{value_match.group(0)!r}"
                )
    return list(dict.fromkeys(critical))


def _parse_decimal_cell(value: str) -> Optional[float]:
    text = _plain_cell(value)
    match = re.search(r"[+−-]?\d[\d\s.,'’]*", text)
    if not match:
        return None
    token = match.group(0).replace("−", "-").replace(" ", "").replace("'", "").replace("’", "")
    if "," in token and "." in token:
        if token.rfind(",") > token.rfind("."):
            token = token.replace(".", "").replace(",", ".")
        else:
            token = token.replace(",", "")
    elif "," in token:
        token = token.replace(",", ".")
    try:
        return float(token)
    except ValueError:
        return None


def _repeated_arithmetic_fit(rows: List[List[str]], qty_idx: int, net_idx: int, amount_idx: int) -> int:
    fits = 0
    for row in rows:
        if max(qty_idx, net_idx, amount_idx) >= len(row):
            continue
        qty = _parse_decimal_cell(row[qty_idx])
        net = _parse_decimal_cell(row[net_idx])
        amount = _parse_decimal_cell(row[amount_idx])
        if qty is None or net is None or amount is None:
            continue
        tolerance = max(0.02, abs(amount) * 0.001)
        if abs(qty * net - amount) <= tolerance:
            fits += 1
    return fits


def detect_semantic_regression_critical(markdown: str, raw_ocr: str = "", markdown_reasoning: str = "") -> List[str]:
    """Garde-fou générique : signale les régressions structurelles à fort risque comptable sans masquer le résultat."""
    critical: List[str] = []
    for page, line_no, header, rows in _iter_markdown_tables(markdown):
        normalized = [_plain_cell(cell) for cell in header]
        for col_idx, header_cell in enumerate(normalized, start=1):
            if re.search(r"\b(?:TOTAL\s+HT|TOTAL\s+TVA|TOTAL\s+TTC|NET\s+A\s+PAYER|SOLDE|ECO[- ]?CONTRIBUTION|OCTROI(?:\s+DE\s+MER)?|OMR?)\b", header_cell, flags=re.IGNORECASE):
                critical.append(
                    f"page={page}:financial_data_in_markdown_header_table_line={line_no}:col={col_idx}:value={header_cell!r}"
                )
        joined = " | ".join(normalized)
        desc_idx = next((i for i, cell in enumerate(normalized) if _DESC_HEADER_RE.search(cell)), None)
        qty_idx = next((i for i, cell in enumerate(normalized) if _QTY_HEADER_RE.search(cell)), None)
        amount_idx = next((i for i, cell in enumerate(normalized) if _AMOUNT_HEADER_RE.search(cell)), None)
        tax_idx = next((i for i, cell in enumerate(normalized) if _TAX_HEADER_RE.search(cell)), None)
        is_line_items = desc_idx is not None and qty_idx is not None and amount_idx is not None
        if is_line_items and qty_idx is not None:
            merged_candidates = []
            prefix_counter: Counter[str] = Counter()
            for rno, row in enumerate(rows, start=1):
                value = _plain_cell(row[qty_idx])
                cores = _NUMERIC_CORE_RE.findall(value)
                if len(cores) >= 2 and not re.search(r"\d\s*[x×]\s*\d", value, flags=re.IGNORECASE):
                    merged_candidates.append((rno, value, cores))
                    prefix_counter[cores[0]] += 1
            merged_rows = [
                (rno, value) for rno, value, cores in merged_candidates
                if prefix_counter[cores[0]] >= 2
            ]
            if len(merged_rows) >= 2:
                critical.append(
                    f"page={page}:repeated_numeric_tracks_merged_table_line={line_no}:qty_col={qty_idx + 1}:rows={merged_rows[:5]!r}"
                )
            if desc_idx is not None:
                for rno, row in enumerate(rows, start=1):
                    designation = _plain_cell(row[desc_idx])
                    if not _CONTRIBUTION_RE.search(designation):
                        continue
                    qty_value = _plain_cell(row[qty_idx])
                    next_value = _plain_cell(row[qty_idx + 1]) if qty_idx + 1 < len(row) else ""
                    next_header = str(header[qty_idx + 1]).strip() if qty_idx + 1 < len(header) else ""
                    net_idx = next((i for i, cell in enumerate(normalized) if re.search(r"P\.?U\.?\s*NET|PRIX\s+NET|NET\s+UNIT", cell, flags=re.IGNORECASE)), None)
                    arithmetic_support = 0
                    if net_idx is not None and amount_idx is not None and qty_idx + 1 < len(normalized):
                        arithmetic_support = _repeated_arithmetic_fit(rows, qty_idx + 1, net_idx, amount_idx)
                    if (
                        not qty_value
                        and _COMPACT_QTY_RE.fullmatch(next_value)
                        and re.fullmatch(r"\[SANS_ENTETE_\d+\]", next_header, flags=re.IGNORECASE)
                        and arithmetic_support >= 2
                    ):
                        critical.append(
                            f"page={page}:contribution_row_shifted_table_line={line_no}:row={rno}:quantity_in_next_column={next_value!r}:arithmetic_support={arithmetic_support}"
                        )
                    if amount_idx is not None and tax_idx is not None and amount_idx < len(row) and tax_idx < len(row):
                        amount_value = _plain_cell(row[amount_idx])
                        tax_value = _plain_cell(row[tax_idx])
                        if amount_value and not _looks_like_monetary_value(amount_value) and _looks_like_monetary_value(tax_value):
                            critical.append(
                                f"page={page}:contribution_amount_shifted_table_line={line_no}:row={rno}:amount={amount_value!r}:tax={tax_value!r}"
                            )
        is_tax_matrix = bool(_TAX_MATRIX_RE.search(joined)) and len(header) >= 4
        if is_tax_matrix:
            if re.fullmatch(r"\[SANS_ENTETE_\d+\]", str(header[0]).strip(), flags=re.IGNORECASE):
                for row in rows[:2]:
                    if row and re.search(r"codes?\s*(?:tva|vat|gst|taxe?|tax)", _plain_cell(row[0]), flags=re.IGNORECASE):
                        critical.append(f"page={page}:tax_axis_demoted_to_data_table_line={line_no}")
                        break
            total_row = next((row for row in rows if any(re.search(r"\bTOTAL\s+HT\b|\bNET\s+A\s+PAYER\b|\bTOTAL\s+TAX\b", _plain_cell(c), flags=re.IGNORECASE) for c in row)), None)
            if total_row is not None:
                total_pos = next((i for i, c in enumerate(total_row) if re.search(r"\bTOTAL\s+HT\b", _plain_cell(c), flags=re.IGNORECASE)), None)
                if total_pos is not None:
                    nonempty_before = [i + 1 for i, c in enumerate(total_row[:total_pos]) if _plain_cell(c)]
                    if nonempty_before:
                        critical.append(
                            f"page={page}:mixed_tax_row_shifted_table_line={line_no}:nonempty_before_total={nonempty_before}"
                        )
    # Tout bloc OCR explicitement incertain doit rester présent avec un marqueur,
    # sans imposer une lecture exacte lorsque plusieurs caractères sont réellement ambigus.
    markdown_lines = [_plain_cell(line) for line in str(markdown or "").splitlines() if _plain_cell(line)]
    markdown_uncertain_candidates: List[str] = []
    for raw_line in str(markdown or "").splitlines():
        if "[INCERTAIN]" not in raw_line.upper():
            continue
        raw_cells = _split_markdown_row(raw_line) or [raw_line]
        for raw_cell in raw_cells:
            for segment in re.split(r"<br\s*/?>", str(raw_cell), flags=re.IGNORECASE):
                if "[INCERTAIN]" in segment.upper():
                    candidate = _plain_cell(segment)
                    if candidate:
                        markdown_uncertain_candidates.append(candidate)
    candidate_lines = list(dict.fromkeys(markdown_uncertain_candidates + markdown_lines))
    for match in re.finditer(r"\[\[TEXT_BLOCK\b[^\]]*state=uncertain[^\]]*\]\]\s*(.*?)\s*\[\[/TEXT_BLOCK\]\]", raw_ocr or "", flags=re.IGNORECASE | re.DOTALL):
        fragment = _plain_cell(match.group(1))
        normalized_fragment = _fuzzy_normalize(fragment)
        if len(normalized_fragment) < 4 or fragment in {"[ILLISIBLE]", "[INCERTAIN]"}:
            continue
        matched = False
        fragment_words = {w for w in re.findall(r"[A-Za-zÀ-ÿ]{4,}", fragment.lower())}
        for line in candidate_lines:
            line_without_markers = re.sub(r"\[(?:INCERTAIN|ILLISIBLE|TRONQU[EÉ])\]", "", line, flags=re.IGNORECASE)
            line_without_markers = re.sub(r"^(?:MANUSCRIT|ANNOTATION(?:\s+MANUSCRITE)?|HANDWRITING)\s*:\s*", "", line_without_markers, flags=re.IGNORECASE)
            normalized_line = _fuzzy_normalize(line_without_markers)
            line_words = {w for w in re.findall(r"[A-Za-zÀ-ÿ]{4,}", line_without_markers.lower())}
            ratio = SequenceMatcher(None, normalized_fragment, normalized_line).ratio()
            close_match = bool(normalized_line) and (
                normalized_fragment in normalized_line
                or (len(normalized_line) >= 4 and normalized_line in normalized_fragment)
                or ratio >= 0.55
                or bool(fragment_words & line_words)
            )
            if not close_match:
                continue
            matched = True
            if "[ILLISIBLE]" in line.upper():
                critical.append(f"uncertain_ocr_replaced_by_illegible={fragment!r}:markdown={line!r}:ratio={ratio:.2f}")
            elif "[INCERTAIN]" not in line.upper():
                critical.append(f"uncertain_ocr_became_certain={fragment!r}:markdown={line!r}:ratio={ratio:.2f}")
            break
        if not matched:
            illegible_annotation = next((
                line for line in markdown_lines
                if "[ILLISIBLE]" in line.upper() and re.search(r"ANNOTATION|MANUSCRIT|HANDWRIT", line, flags=re.IGNORECASE)
            ), "")
            if illegible_annotation:
                critical.append(f"uncertain_ocr_replaced_by_illegible={fragment!r}:markdown={illegible_annotation!r}")
            else:
                critical.append(f"uncertain_ocr_omitted={fragment!r}")

    # Un manuscrit déclaré comme zone d'annotation indépendante ne doit pas être absorbé
    # dans une cellule d'article. Le contrôle est purement mécanique et ne déplace rien.
    uncertain_fragments = [
        _plain_cell(m.group(1))
        for m in re.finditer(r"\[\[TEXT_BLOCK\b[^\]]*state=uncertain[^\]]*\]\]\s*(.*?)\s*\[\[/TEXT_BLOCK\]\]", raw_ocr or "", flags=re.IGNORECASE | re.DOTALL)
        if _plain_cell(m.group(1))
    ]
    for page, line_no, header, rows in _iter_markdown_tables(markdown):
        normalized_header = [_plain_cell(cell) for cell in header]
        if not (any(_DESC_HEADER_RE.search(cell) for cell in normalized_header) and any(_QTY_HEADER_RE.search(cell) for cell in normalized_header)):
            continue
        for row_no, row in enumerate(rows, start=1):
            uncertain_phrases: List[str] = []
            for raw_cell in row:
                if "[INCERTAIN]" not in str(raw_cell).upper():
                    continue
                segments = re.split(r"<br\s*/?>", str(raw_cell), flags=re.IGNORECASE)
                marked = [segment for segment in segments if "[INCERTAIN]" in segment.upper()]
                for segment in marked or segments[-1:]:
                    phrase = _plain_cell(re.sub(r"\[INCERTAIN\]", "", segment, flags=re.IGNORECASE))
                    if phrase:
                        uncertain_phrases.append(phrase)
            if not uncertain_phrases:
                continue
            for fragment in uncertain_fragments:
                frag_norm = _fuzzy_normalize(fragment)
                for phrase in uncertain_phrases:
                    phrase_norm = _fuzzy_normalize(phrase)
                    if frag_norm and phrase_norm and SequenceMatcher(None, frag_norm, phrase_norm).ratio() >= 0.45:
                        critical.append(
                            f"page={page}:uncertain_handwriting_embedded_in_line_item_table_line={line_no}:row={row_no}:fragment={fragment!r}"
                        )
                        break
                else:
                    continue
                break
    critical.extend(_detect_ocr_hidden_track_losses(markdown, raw_ocr))
    critical.extend(_detect_ocr_financial_pair_losses(markdown, raw_ocr))
    critical.extend(_detect_accounting_text_pair_losses(markdown, raw_ocr))
    critical.extend(_detect_special_row_track_shifts(markdown, raw_ocr))
    critical.extend(_detect_special_row_left_compaction(markdown))
    critical.extend(_detect_bank_identifier_disagreement(markdown, raw_ocr))
    critical.extend(_detect_truncation_losses(markdown, raw_ocr))
    if TECHNICAL_MAP_COMMENT_RE.search(markdown or ""):
        critical.append("technical_map_comment_leaked")
    return list(dict.fromkeys(critical))


def validate_markdown_quality(final_markdown: str, page_count: int) -> Dict[str, Any]:
    errors: List[str] = []
    rendered = extract_rendered_document(final_markdown)
    try:
        validate_canonical_markdown_structure(rendered, page_count)
    except Exception as exc:
        errors.append(str(exc))
    warnings = detect_accounting_integrity_warnings(rendered) + detect_compact_group_warnings(rendered)
    critical_warnings = detect_semantic_regression_critical(rendered)
    return {
        "ok": not errors and not critical_warnings,
        "errors": errors,
        "warnings": warnings,
        "critical_warnings": critical_warnings,
        "summary": (
            "Structure Markdown valide" if not errors else "KO: " + " | ".join(errors)
        ) + ("" if not critical_warnings else f" | régressions critiques={len(critical_warnings)}")
          + ("" if not warnings else f" | avertissements comptables={len(warnings)}"),
    }


def calculate_costs(stats_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    totals = {
        "total_input": 0,
        "total_output": 0,
        "total_tokens": 0,
        "cached_tokens": 0,
        "cache_creation_input_tokens": 0,
        "reasoning_tokens": 0,
        "image_tokens": 0,
    }
    for stats in stats_list or []:
        if not isinstance(stats, dict):
            continue
        totals["total_input"] += int(stats.get("input_tokens", 0) or 0)
        totals["total_output"] += int(stats.get("output_tokens", 0) or 0)
        totals["total_tokens"] += int(
            stats.get("total_tokens", 0)
            or int(stats.get("input_tokens", 0) or 0) + int(stats.get("output_tokens", 0) or 0)
        )
        totals["cached_tokens"] += int(stats.get("cached_tokens", 0) or 0)
        totals["cache_creation_input_tokens"] += int(stats.get("cache_creation_input_tokens", 0) or 0)
        totals["reasoning_tokens"] += int(stats.get("reasoning_tokens", 0) or 0)
        totals["image_tokens"] += int(stats.get("image_tokens", 0) or 0)
    return {
        **totals,
        "cost_input": 0.0,
        "cost_output": 0.0,
        "cost_total": 0.0,
        "cost_available": False,
        "pages": len(stats_list or []),
    }


__all__ = [
    "API_URL", "MODEL", "MODEL_OCR", "MODEL_MARKDOWN", "PIPELINE_VERSION",
    "OCR_AUDIT_PASS", "MARKDOWN_VISUAL_PASS", "DUAL_INDEPENDENT_VISUAL_PASSES",
    "PARALLEL_INDEPENDENT_PASSES", "RAW_OCR_FIRST_PASS", "MARKDOWN_SECOND_PASS",
    "TWO_PASS_RAW_OCR_MARKDOWN", "CANONICAL_OCR_ONLY", "DETERMINISTIC_MARKDOWN",
    "MODEL_GENERATED_MARKDOWN", "SINGLE_MARKDOWN_OUTPUT", "OCR_PROMPT_IN_USER_MESSAGE",
    "MARKDOWN_PROMPT_IN_USER_MESSAGE", "RAW_OCR_PROMPT", "OCR_AUDIT_PROMPT",
    "MARKDOWN_PROMPT", "MARKDOWN_VISUAL_PROMPT", "OCR_PROMPT",
    "NOMINAL_GENERATIONS_PER_PAGE", "SEMANTIC_RETRIES",
    "STOP_ON_CRITICAL", "PUBLISH_PARTIAL_DOCUMENT", "PUBLISH_DEGRADED_MARKDOWN", "STRICT_MARKDOWN_REGRESSION_GATE",
    "OCR_DIAGNOSTIC_MODE", "PIPELINE_AUDIT_MODE", "INCLUDE_OCR_ANNEX",
    "INCLUDE_THINKING_ANNEX", "CAPTURE_REASONING_CONTENT", "THINKING_ANNEX_MAX_CHARS",
    "ENABLE_EXPLICIT_CACHE", "QWEN_HIGH_RES_IMAGES", "STREAMING_OCR",
    "STREAMING_MARKDOWN", "STREAM_INCLUDE_USAGE", "THINKING_BUDGET_OCR",
    "THINKING_BUDGET_MARKDOWN", "MAX_COMPLETION_TOKENS_OCR",
    "MAX_COMPLETION_TOKENS_MARKDOWN", "OCR_SEED", "MARKDOWN_SEED",
    "RENDER_DPI", "DETAIL_DPI", "DETAIL_UPPER_END", "DETAIL_MIDDLE_START",
    "DETAIL_MIDDLE_END", "DETAIL_LOWER_START", "RIGHT_VIEW_START",
    "MARKDOWN_GEOMETRY_GUIDE", "GEOMETRY_GUIDE_MINOR_STEP_PERCENT",
    "GEOMETRY_GUIDE_LABEL_STEP_PERCENT", "GEOMETRY_GUIDE_MAJOR_STEP_PERCENT",
    "GEOMETRY_GUIDE_MAX_PIXELS", "MARKDOWN_EXPECTED_VIEW_COUNT", "EXPECTED_VIEW_COUNT", "OCR_AUDIT_RENDER_DPI",
    "OCR_AUDIT_DETAIL_DPI", "OCR_AUDIT_UPPER_END", "OCR_AUDIT_LOWER_START",
    "OCR_AUDIT_RIGHT_VIEW_START", "OCR_AUDIT_EXPECTED_VIEW_COUNT",
    "VIEW_JPEG_QUALITY", "VIEW_JPEG_MIN_QUALITY", "OCR_AUDIT_JPEG_QUALITY",
    "OCR_AUDIT_JPEG_MIN_QUALITY", "MAX_VIEW_PIXELS", "MAX_REQUEST_BODY_MB",
    "MAX_SINGLE_BASE64_IMAGE_MB", "MAX_TOTAL_BASE64_IMAGE_MB",
    "validate_api_configuration", "configure_explicit_cache_for_batch", "get_pdf_info",
    "prepare_page_source", "prepare_page_views", "prepare_ocr_audit_views",
    "prepare_markdown_views", "process_page", "process_page_with_cache",
    "build_unavailable_page", "get_pipeline_fingerprint", "get_progress_path",
    "load_progress", "save_progress", "clear_progress", "build_ocr_annex",
    "build_thinking_annex", "assemble_document_with_ocr_annex",
    "extract_rendered_document", "extract_ocr_annex", "extract_thinking_annex",
    "validate_canonical_markdown_structure", "validate_markdown_quality", "detect_semantic_regression_critical",
    "calculate_costs", "sanitize_raw_ocr_response", "validate_raw_ocr_package",
    "sanitize_markdown_response", "validate_grid_commitments", "validate_page_markdown",
    "validate_runtime_package_contract",
    "detect_accounting_integrity_warnings", "detect_compact_group_warnings",
]

