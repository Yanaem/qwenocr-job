#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ocr_qwenVL.py — extraction visuelle canonique Qwen, un seul appel par page.

Architecture définitive :
1. rendu unique de chaque page PDF à 300 DPI ;
2. création locale de vues détaillées de la même page ;
3. un seul appel multimodal Qwen par page ;
4. sortie canonique balisée, exhaustive et audit-able ;
5. parsing et validation structurelle non bloquante ;
6. génération Markdown déterministe par Python, sans second LLM ;
7. production facultative d'un Markdown synthétique et d'un rapport qualité.

Python ne corrige aucune donnée documentaire. Il ne choisit jamais entre deux
lectures, ne recalcule aucun montant et ne déplace aucune valeur. Les seules
normalisations réalisées sont techniques : fermeture/salvage des balises,
largeur constante des tableaux et échappement Markdown.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import tempfile
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests
from PIL import Image
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

PIPELINE_VERSION = "qwen-canonical-ocr-deterministic-md-v5.0.0-20260731"
CHECKPOINT_VERSION = 5
CHECKPOINT_SCHEMA = "canonical-ocr-one-call-page-state-v1"

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

DEFAULT_QWEN_MODEL = "qwen3.7-plus"
MODEL_OCR = os.getenv("QWEN_MODEL_OCR", DEFAULT_QWEN_MODEL).strip()
MODEL = MODEL_OCR  # compatibilité avec les métriques et runners historiques

CANONICAL_OCR_ONLY = True
DETERMINISTIC_MARKDOWN = True
NOMINAL_GENERATIONS_PER_PAGE = 1
SEMANTIC_RETRIES = 0

STOP_ON_CRITICAL = _env_bool("STOP_ON_CRITICAL", True)
PUBLISH_PARTIAL_DOCUMENT = _env_bool("PUBLISH_PARTIAL_DOCUMENT", True)
PUBLISH_DEGRADED_MARKDOWN = _env_bool("PUBLISH_DEGRADED_MARKDOWN", True)

RENDER_DPI = _env_int("RENDER_DPI", 300)
DETAIL_DPI = _env_int("DETAIL_DPI", 360)
ENABLE_DETAIL_VIEWS = _env_bool("ENABLE_DETAIL_VIEWS", True)
DETAIL_UPPER_END = _env_float("DETAIL_UPPER_END", 0.60)
DETAIL_LOWER_START = _env_float("DETAIL_LOWER_START", 0.40)
MAX_SINGLE_BASE64_IMAGE_MB = max(1.0, _env_float("MAX_SINGLE_BASE64_IMAGE_MB", 10.0))
MAX_TOTAL_BASE64_IMAGE_MB = max(1.0, _env_float("MAX_TOTAL_BASE64_IMAGE_MB", 24.0))

MAX_TOKENS_OCR = _env_int("MAX_TOKENS_OCR", 12000)
TEMPERATURE = _env_float("TEMPERATURE", 0.0)
ENABLE_THINKING_OCR = _env_bool("ENABLE_THINKING_OCR", True)
QWEN_HIGH_RES_IMAGES = _env_bool("QWEN_HIGH_RES_IMAGES", True)

REQUEST_TIMEOUT_SECONDS = _env_int("REQUEST_TIMEOUT_SECONDS", 600)
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

BLOCK_ROLES = {
    "logo",
    "supplier",
    "customer",
    "shipping",
    "document",
    "line_note",
    "payment",
    "bank",
    "legal",
    "marketing",
    "annotation",
    "stamp",
    "signature_label",
    "other",
}
TABLE_ROLES = {
    "document_meta",
    "line_items",
    "tax_summary",
    "totals_summary",
    "payment_table",
    "other_table",
}
ALLOWED_SOURCES = {"printed", "handwritten", "stamp", "mixed"}
ALLOWED_STATUSES = {"readable", "uncertain", "truncated", "uncertain_truncated"}

ROLE_ALIASES = {
    "logo_text": "logo",
    "supplier_identity": "supplier",
    "supplier_address": "supplier",
    "supplier_contact": "supplier",
    "supplier_legal": "legal",
    "customer_identity": "customer",
    "customer_address": "customer",
    "customer_contact": "customer",
    "customer_legal": "customer",
    "billing_address": "customer",
    "shipping_address": "shipping",
    "shipping_details": "shipping",
    "shipping_contact": "shipping",
    "delivery_confirmation": "stamp",
    "invoice_title": "document",
    "invoice_details": "document_meta",
    "line_items_note": "line_note",
    "line_items_footer": "line_note",
    "payment_terms": "payment",
    "bank_details": "bank",
    "payment": "payment",
    "legal_terms": "legal",
    "marketing_badge": "marketing",
    "stamp_signature": "stamp",
    "qr_barcode_text": "other",
    "notes": "annotation",
    "isolated_value": "other",
    "unknown": "other",
}

ELEMENT_START_RE = re.compile(r"^\s*\[\[(BLOCK|TABLE)\s+(.+?)\]\]\s*$", re.IGNORECASE)
BLOCK_END_RE = re.compile(r"^\s*\[\[/BLOCK\]\]\s*$", re.IGNORECASE)
TABLE_END_RE = re.compile(r"^\s*\[\[/TABLE\]\]\s*$", re.IGNORECASE)
END_PAGE_RE = re.compile(r"^\s*\[\[END_PAGE\]\]\s*$", re.IGNORECASE)
MODEL_PAGE_RE = re.compile(r"^\s*\[\[(?:PDF_)?PAGE(?:\s+\d+)?\]\]\s*$", re.IGNORECASE)
HTML_PAGE_RE = re.compile(r"^\s*<!--\s*PAGE\s+\d+\s*-->\s*$", re.IGNORECASE)
ATTRIBUTE_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=(?:\"([^\"]*)\"|'([^']*)'|([^\s]+))")
FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})(?:[A-Za-z0-9_.+-]+)?\s*$")
PAGE_MARKER_RE = re.compile(r"^\s*<!--\s*PAGE\s+(\d+)\s*-->\s*$", re.IGNORECASE)


OCR_PROMPT = r"""Tu es un moteur d'extraction visuelle canonique spécialisé dans les documents comptables et commerciaux.

MISSION
Transcrire une seule page physique de manière exhaustive, fidèle et structurée. Toutes les images de la requête sont des vues de la même page : une vue complète et, éventuellement, des recadrages détaillés. Les recadrages ne sont jamais des pages supplémentaires et ne doivent créer aucun doublon.

PRIORITÉS ABSOLUES
1. Exhaustivité : tout texte visible et lisible doit apparaître exactement une fois, sauf répétition réellement imprimée sur la page.
2. Fidélité : aucun caractère, mot, nombre, signe, unité, devise, taux, montant ou libellé ne doit être inventé, corrigé, normalisé ou complété.
3. Géométrie : ne fusionne jamais deux zones, deux lignes ou deux colonnes visuellement distinctes.
4. Incertitude honnête : utilise [ILLISIBLE] à l'emplacement exact d'un caractère ou segment indéterminable. Si la source est physiquement coupée au bord de la page, ajoute [TRONQUE] à la fin du fragment incomplet.
5. Aucune interprétation linguistique : une chaîne plausible n'est pas une preuve visuelle.

MÉTHODE INTERNE OBLIGATOIRE — NE L'EXPOSE PAS
PASSAGE 1 — CARTOGRAPHIE
- Examine d'abord la page complète.
- Balaye les neuf zones : haut-gauche, haut-centre, haut-droite, milieu-gauche, milieu-centre, milieu-droite, bas-gauche, bas-centre, bas-droite.
- Repère tous les blocs, tableaux, en-têtes, pieds de page, taxes, totaux, coordonnées bancaires, mentions légales, annotations et tampons.
- Détermine les limites de chaque tableau avant toute transcription.

PASSAGE 2 — TRANSCRIPTION
- Lis chaque contenu directement dans la vue la plus détaillée disponible.
- Conserve exactement la casse, les accents, les espaces significatifs, les tirets, les barres, les points, les virgules, les parenthèses, les signes, les pourcentages et les devises.
- Ne traduis pas, ne reformule pas et ne corrige pas l'orthographe réellement imprimée.
- Ne déduis rien d'une autre page, d'un fournisseur connu ou de connaissances externes.

PASSAGE 3 — AUDIT D'EXHAUSTIVITÉ
- Relis la page du bas vers le haut pour contrôler les totaux, banques, mentions légales et pieds de page.
- Relis chaque tableau de droite à gauche pour contrôler les dernières colonnes, codes, taxes et montants.
- Pour chaque ligne tabulaire, compte silencieusement tous les groupes alphanumériques et numériques visibles, puis vérifie que chacun apparaît une fois dans une cellule.
- Vérifie la première et la dernière ligne de chaque tableau.
- Vérifie que les vues qui se chevauchent n'ont créé aucun doublon.

IDENTIFIANTS ET CHAÎNES OPAQUES
Les références d'articles, numéros de facture, commandes, clients, séries, identifiants fiscaux, IBAN, BIC et autres codes sont des chaînes opaques, jamais des mots à corriger.
- Lis chaque caractère de gauche à droite, puis vérifie de droite à gauche.
- Contrôle silencieusement le nombre et la position des caractères.
- N'ajoute jamais une lettre pour former un mot ou une marque connue.
- Ne supprime jamais un caractère parce qu'il paraît inhabituel.
- Résous O/0, I/1/l, B/8, S/5, G/6, Z/2 et toute autre ambiguïté uniquement depuis l'image.
- Si un caractère reste incertain, remplace uniquement ce caractère par [ILLISIBLE].

SÉPARATION DES SOURCES
- Sépare systématiquement texte imprimé, manuscrit et texte de tampon.
- Un manuscrit superposé à un tableau devient un BLOCK distinct source=handwritten ; il n'entre jamais dans une désignation, une référence, une quantité, un prix ou un montant imprimé.
- Un texte de tampon devient un BLOCK distinct source=stamp.
- Pour un manuscrit partiellement lisible, transcris seulement les caractères certains et utilise [ILLISIBLE] pour les autres. Ne transforme jamais des lettres ambiguës en mot plausible et ne décris ni couleur, ni coche, ni intention.
- Ignore les traits, paraphes et signatures purement graphiques sans texte lisible.

TABLEAUX — CARTE DES COLONNES
Avant de transcrire un tableau, construis silencieusement une carte horizontale unique de ses colonnes à partir de toutes les lignes disponibles.
- Le nombre de colonnes est déterminé par les lignes de données les plus complètes, pas par l'en-tête seul.
- Chaque groupe de valeurs répété au même emplacement horizontal constitue une colonne distincte, même sans bordure ou sans en-tête visible.
- Une valeur placée entre la désignation et la quantité ne doit jamais être absorbée dans la désignation ni fusionnée avec la quantité.
- Une ligne clairsemée conserve exactement la même carte de colonnes ; les cellules absentes sont <EMPTY> et aucune valeur n'est décalée.
- Toute colonne alimentée mais sans en-tête visible reçoit [SANS_ENTETE_1], [SANS_ENTETE_2], etc., de gauche à droite.
- N'invente jamais un nom comme remise, taxe, code, unité ou prix si cet intitulé n'est pas imprimé.
- Une continuation de désignation, référence ou caractéristique reste dans la même cellule avec <BR> seulement si elle ne possède aucune quantité, prix, montant, taux ou code propre.
- Ne crée aucune ligne pour reproduire une zone blanche.
- Deux tableaux ayant des en-têtes, bordures ou alignements distincts restent séparés. Les taxes et les totaux côte à côte doivent rester deux TABLE distinctes s'ils forment deux groupes visuels.

CONTRÔLES ARITHMÉTIQUES DE STRUCTURE
Les calculs silencieux sont autorisés et obligatoires uniquement pour détecter une mauvaise carte de colonnes ou un décalage de ligne :
- quantité × prix unitaire net ≈ montant de ligne ;
- prix brut × (1 - taux/100) ≈ prix net ;
- base taxable × taux ≈ montant de taxe ;
- somme des lignes ± remises, frais ou contributions ≈ sous-total ou total.
Une incohérence impose de relire l'image et la carte des colonnes. Elle n'autorise jamais à modifier, arrondir, compléter ou inventer une valeur visible. L'alignement répété sur plusieurs lignes prime sur une coïncidence arithmétique isolée.

CONTENU À COUVRIR LORSQU'IL EST VISIBLE
Logo textuel, fournisseur, établissements, client, livraison, titre du document, numéro, dates, échéance, commandes, références, lignes de biens ou services, quantités, unités, prix, remises, contributions, taxes, sous-totaux, acomptes, totaux, solde, statut de paiement, banque, RIB, IBAN, BIC, mentions légales, texte de tampon, manuscrits lisibles, pagination imprimée, pied de page et texte lisible autour d'un QR code ou code-barres. Ne décode jamais le contenu invisible d'un QR code ou code-barres.

FORMAT DE SORTIE STRICT
Retourne uniquement les éléments ci-dessous, sans Markdown, JSON, explication, commentaire, bloc de code ni marqueur PAGE.

BLOCK :
[[BLOCK id=B001 order=001 role=supplier source=printed status=readable]]
texte visible
[[/BLOCK]]

TABLE :
[[TABLE id=T001 order=002 role=line_items source=printed status=readable cols=N]]
cellule<TAB>cellule<TAB>cellule
cellule<TAB>cellule<TAB>cellule
[[/TABLE]]

Termine toujours par :
[[END_PAGE]]

RÔLES BLOCK AUTORISÉS
logo, supplier, customer, shipping, document, line_note, payment, bank, legal, marketing, annotation, stamp, signature_label, other

RÔLES TABLE AUTORISÉS
document_meta, line_items, tax_summary, totals_summary, payment_table, other_table

SOURCE OBLIGATOIRE
printed, handwritten, stamp, mixed

STATUS OBLIGATOIRE
readable, uncertain, truncated, uncertain_truncated

TOKENS AUTORISÉS DANS LE CONTENU
<TAB> : séparation de cellules
<EMPTY> : cellule réellement vide
<BR> : retour à la ligne dans une même cellule
[ILLISIBLE] : caractère ou segment réellement indéterminable
[TRONQUE] : fin physiquement coupée par le cadrage de la source
[SANS_ENTETE_n] : en-tête technique d'une colonne réelle sans intitulé visible

RÈGLES DE FORMAT
- Les id BLOCK sont B001, B002... et les id TABLE T001, T002...
- order est global à la page, unique et strictement croissant selon l'ordre de lecture.
- Aucun élément ne peut être imbriqué dans un autre.
- Chaque TABLE commence par une ligne d'en-têtes. Si aucun en-tête n'est visible, crée une ligne composée uniquement de [SANS_ENTETE_n].
- cols=N est le nombre réel de colonnes de la carte.
- Chaque ligne d'une TABLE contient exactement N cellules et N-1 tokens <TAB>.
- Toute cellule vide, y compris en fin de ligne, est explicitement <EMPTY>.
- Un TABLE contient au minimum une ligne d'en-têtes et une ligne de données.
- Aucun <TAB> dans un BLOCK.
- Aucun texte visible ne doit rester hors d'un BLOCK ou d'une TABLE.
- Si le rôle est incertain, utilise other ou other_table ; n'omets jamais le contenu.
- Si la page est réellement vide, retourne exactement [PAGE VIDE] puis [[END_PAGE]].

CONTRÔLE FINAL SILENCIEUX
Avant de répondre, vérifie : couverture des neuf zones, absence d'omission, absence de doublon, identifiants relus caractère par caractère, carte de colonnes stable, cellules vides explicites, imprimé/manuscrit/tampon séparés, taxes et totaux séparés lorsqu'ils sont visuellement distincts, toutes les balises fermées et présence finale de [[END_PAGE]]."""


# =============================================================================
# Journalisation et validation de configuration
# =============================================================================


def _log(message: str) -> None:
    if VERBOSE:
        print(message, flush=True)


def validate_api_configuration() -> None:
    if not API_URL.startswith("https://"):
        raise RuntimeError("Endpoint Qwen invalide ou absent.")
    if not MODEL_OCR:
        raise RuntimeError("QWEN_MODEL_OCR doit être défini.")

    positive = {
        "RENDER_DPI": RENDER_DPI,
        "DETAIL_DPI": DETAIL_DPI,
        "MAX_TOKENS_OCR": MAX_TOKENS_OCR,
        "REQUEST_TIMEOUT_SECONDS": REQUEST_TIMEOUT_SECONDS,
        "CONNECT_TIMEOUT_SECONDS": CONNECT_TIMEOUT_SECONDS,
        "HTTP_POOL_SIZE": HTTP_POOL_SIZE,
        "MAX_RETRIES": MAX_RETRIES,
        "BACKOFF_BASE": BACKOFF_BASE,
        "BACKOFF_MAX": BACKOFF_MAX,
        "MAX_SINGLE_BASE64_IMAGE_MB": MAX_SINGLE_BASE64_IMAGE_MB,
        "MAX_TOTAL_BASE64_IMAGE_MB": MAX_TOTAL_BASE64_IMAGE_MB,
    }
    invalid = [name for name, value in positive.items() if float(value) <= 0]
    if invalid:
        raise RuntimeError(
            "Valeurs de configuration non positives : " + ", ".join(sorted(invalid))
        )
    if not 0.0 <= TEMPERATURE <= 2.0:
        raise RuntimeError("TEMPERATURE doit être comprise entre 0 et 2.")
    if not 0.0 < DETAIL_LOWER_START < DETAIL_UPPER_END < 1.0:
        raise RuntimeError(
            "Les vues détaillées doivent vérifier 0 < DETAIL_LOWER_START "
            "< DETAIL_UPPER_END < 1."
        )
    if QWEN_WORKSPACE_ID and (
        any(char.isspace() for char in QWEN_WORKSPACE_ID) or "/" in QWEN_WORKSPACE_ID
    ):
        raise RuntimeError("QWEN_WORKSPACE_ID invalide.")


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
    return str(Path(image_dir) / f"page_{int(page_num):06d}_full.png")


def render_single_page_to_file(
    pdf_path: str,
    page_num: int,
    image_dir: str,
    dpi: int = RENDER_DPI,
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


def _create_full_view(
    source_path: str,
    image_dir: str,
    page_num: int,
    source_dpi: int,
) -> str:
    """Crée la vue complète au DPI demandé sans refaire le rendu PDF."""
    if source_dpi <= RENDER_DPI:
        return source_path

    target = Path(image_dir) / f"page_{int(page_num):06d}_full_view.png"
    with Image.open(source_path) as image:
        ratio = float(RENDER_DPI) / float(source_dpi)
        width = max(1, int(round(image.width * ratio)))
        height = max(1, int(round(image.height * ratio)))
        resized = image.resize((width, height), Image.Resampling.LANCZOS)
        try:
            resized.save(target, format="PNG")
        finally:
            resized.close()
    return str(target)


def _create_detail_views(full_path: str, image_dir: str, page_num: int) -> List[Dict[str, Any]]:
    views: List[Dict[str, Any]] = []
    if not ENABLE_DETAIL_VIEWS:
        return views

    ranges = (
        ("upper", 0.0, DETAIL_UPPER_END, "partie supérieure détaillée"),
        ("lower", DETAIL_LOWER_START, 1.0, "partie inférieure détaillée"),
    )
    with Image.open(full_path) as image:
        width, height = image.size
        for label, start_ratio, end_ratio, description in ranges:
            top = max(0, min(height - 1, int(round(height * start_ratio))))
            bottom = max(top + 1, min(height, int(round(height * end_ratio))))
            target = Path(image_dir) / f"page_{int(page_num):06d}_{label}.png"
            crop = image.crop((0, top, width, bottom))
            try:
                crop.save(target, format="PNG")
            finally:
                crop.close()
            views.append(
                {
                    "label": label,
                    "description": description,
                    "path": str(target),
                    "range": [start_ratio, end_ratio],
                }
            )
    return views


def _encode_image(path: str) -> Dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists() or file_path.stat().st_size <= 0:
        raise FileNotFoundError(f"Image absente ou vide : {path}")
    raw = file_path.read_bytes()
    encoded = base64.b64encode(raw).decode("ascii")
    base64_mb = len(encoded.encode("ascii")) / (1024 * 1024)
    if base64_mb > MAX_SINGLE_BASE64_IMAGE_MB:
        raise RuntimeError(
            f"Image {file_path.name} trop volumineuse ({base64_mb:.2f} Mo Base64), "
            f"limite={MAX_SINGLE_BASE64_IMAGE_MB:.2f} Mo."
        )
    return {
        "path": str(file_path),
        "data_url": f"data:image/png;base64,{encoded}",
        "size_kb": len(raw) / 1024.0,
        "base64_mb": base64_mb,
    }


def prepare_page_views(
    pdf_path: str,
    page_num: int,
    image_dir: str,
) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, Any]]:
    source_dpi = max(RENDER_DPI, DETAIL_DPI if ENABLE_DETAIL_VIEWS else RENDER_DPI)
    source_path, source_size_kb, rendered = render_single_page_to_file(
        pdf_path=pdf_path,
        page_num=page_num,
        image_dir=image_dir,
        dpi=source_dpi,
    )
    full_path = _create_full_view(
        source_path=source_path,
        image_dir=image_dir,
        page_num=page_num,
        source_dpi=source_dpi,
    )
    full_size_kb = Path(full_path).stat().st_size / 1024.0

    candidates: List[Dict[str, Any]] = [
        {
            "label": "full",
            "description": "page complète",
            "path": full_path,
            "range": [0.0, 1.0],
        }
    ]
    candidates.extend(_create_detail_views(source_path, image_dir, page_num))

    accepted: List[Dict[str, Any]] = []
    omitted: List[str] = []
    total_base64_mb = 0.0
    for candidate in candidates:
        encoded = _encode_image(candidate["path"])
        if accepted and total_base64_mb + encoded["base64_mb"] > MAX_TOTAL_BASE64_IMAGE_MB:
            omitted.append(str(candidate["label"]))
            continue
        accepted.append({**candidate, **encoded})
        total_base64_mb += float(encoded["base64_mb"])

    if not accepted:
        raise RuntimeError(f"Page {page_num}: aucune vue image n'a pu être préparée.")

    stats = {
        "rendered": bool(rendered),
        "source_image_size_kb": source_size_kb,
        "full_image_size_kb": full_size_kb,
        "view_count": len(accepted),
        "view_labels": [view["label"] for view in accepted],
        "omitted_view_labels": omitted,
        "total_base64_image_mb": total_base64_mb,
        "render_dpi": source_dpi,
        "requested_full_dpi": RENDER_DPI,
        "detail_dpi": DETAIL_DPI,
    }
    cleanup_paths = [source_path] + [str(candidate["path"]) for candidate in candidates]
    return accepted, cleanup_paths, stats


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


def _call_chat(
    api_key: str,
    messages: List[Dict[str, Any]],
    context: str,
) -> Tuple[str, Dict[str, Any]]:
    url = f"{API_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body: Dict[str, Any] = {
        "model": MODEL_OCR,
        "max_tokens": MAX_TOKENS_OCR,
        "temperature": TEMPERATURE,
        "messages": messages,
    }
    if _supports_thinking_toggle(MODEL_OCR):
        body["enable_thinking"] = bool(ENABLE_THINKING_OCR)
    if QWEN_HIGH_RES_IMAGES:
        body["vl_high_resolution_images"] = True

    for attempt in range(1, MAX_RETRIES + 1):
        started = time.time()
        try:
            response = _get_http_session().post(
                url,
                headers=headers,
                json=body,
                timeout=(CONNECT_TIMEOUT_SECONDS, REQUEST_TIMEOUT_SECONDS),
            )
            if response.status_code == 200:
                try:
                    payload = response.json()
                except ValueError as exc:
                    if attempt < MAX_RETRIES:
                        delay = _backoff(attempt)
                        _log(f"⚠️ {context}: réponse 200 non JSON, reprise transport dans {delay:.1f}s")
                        time.sleep(delay)
                        continue
                    raise RuntimeError(f"{context}: réponse 200 non JSON") from exc

                choices = payload.get("choices", []) or []
                if not choices:
                    if attempt < MAX_RETRIES:
                        delay = _backoff(attempt)
                        _log(f"⚠️ {context}: aucune choice, reprise transport dans {delay:.1f}s")
                        time.sleep(delay)
                        continue
                    raise RuntimeError(f"{context}: réponse 200 sans choice")

                choice = choices[0] or {}
                message = choice.get("message", {}) or {}
                text = _extract_text(message.get("content")).strip("\n")
                reasoning = _extract_text(message.get("reasoning_content")).strip()
                finish_reason = choice.get("finish_reason")
                usage = payload.get("usage", {}) or {}
                partial_response = (
                    _response_header(response, "x-dashscope-partialresponse").lower()
                    == "true"
                )
                truncated = finish_reason == "length" or partial_response

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
                    "duration_ms": int((time.time() - started) * 1000),
                    "response_id": payload.get("id"),
                    "response_model": payload.get("model") or MODEL_OCR,
                    "request_id": _response_header(
                        response,
                        "x-dashscope-request-id",
                        "x-request-id",
                        "x-acs-request-id",
                    )
                    or None,
                    "reasoning_content_present": bool(reasoning),
                }
                if not stats["total_tokens"]:
                    stats["total_tokens"] = stats["input_tokens"] + stats["output_tokens"]

                if not text:
                    preview = json.dumps(message, ensure_ascii=False)[:EMPTY_RESPONSE_LOG_CHARS]
                    raise RuntimeError(f"{context}: réponse finale vide. message={preview}")

                if truncated:
                    _log(f"⚠️ {context}: réponse partielle/tronquée conservée pour salvage déterministe.")
                else:
                    _log(
                        f"✅ {context}: {stats['duration_ms'] / 1000:.2f}s, "
                        f"in={stats['input_tokens']} out={stats['output_tokens']}"
                    )
                return text, stats

            try:
                error_message = json.dumps(response.json(), ensure_ascii=False)[:800]
            except Exception:
                error_message = (response.text or "")[:800]
            retry, delay = _compute_retry_delay(response.status_code, error_message, attempt)
            _log(
                f"⚠️ {context}: HTTP {response.status_code}, retry={retry}, "
                f"délai={delay:.1f}s | {error_message[:200]}"
            )
            if not retry:
                raise RuntimeError(f"{context}: HTTP {response.status_code} {error_message}")
            time.sleep(delay)

        except requests.exceptions.Timeout as exc:
            retry, delay = _compute_retry_delay(None, str(exc), attempt)
            if not retry:
                raise
            _log(f"⚠️ {context}: timeout, reprise transport dans {delay:.1f}s")
            time.sleep(delay)
        except requests.exceptions.RequestException as exc:
            retry, delay = _compute_retry_delay(None, str(exc), attempt)
            if not retry:
                raise
            _log(f"⚠️ {context}: erreur réseau, reprise transport dans {delay:.1f}s")
            time.sleep(delay)

    raise RuntimeError(f"{context}: échec après {MAX_RETRIES} tentatives de transport")


def _build_ocr_messages(page_num: int, views: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    user_content: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"Page physique {page_num}. Toutes les images suivantes sont des vues "
                "de cette même page. Utilise la vue complète pour la géométrie et les "
                "vues détaillées pour vérifier les petits caractères."
            ),
        }
    ]
    for index, view in enumerate(views, start=1):
        user_content.append(
            {
                "type": "text",
                "text": f"Vue {index}/{len(views)} — {view['description']}.",
            }
        )
        user_content.append(
            {"type": "image_url", "image_url": {"url": view["data_url"]}}
        )
    user_content.append(
        {
            "type": "text",
            "text": (
                "Effectue les trois passages silencieux, puis retourne uniquement "
                "la transcription canonique demandée, terminée par [[END_PAGE]]."
            ),
        }
    )
    return [
        {"role": "system", "content": [_cacheable_text_block(OCR_PROMPT)]},
        {"role": "user", "content": user_content},
    ]


# =============================================================================
# Parsing canonique et qualité
# =============================================================================


def _strip_outer_fence(text: str) -> Tuple[str, bool]:
    normalized = (text or "").strip("\n")
    lines = normalized.splitlines()
    if len(lines) < 2:
        return normalized, False
    opening_match = FENCE_RE.match(lines[0])
    if not opening_match:
        return normalized, False
    token = opening_match.group(1)
    if not re.fullmatch(re.escape(token[0]) + "{" + str(len(token)) + ",}\\s*", lines[-1].strip()):
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
    normalized = normalized.replace("[TRONQUÉ]", "[TRONQUE]").replace("[TRONQUEE]", "[TRONQUE]")
    if normalized != cleaned:
        changes["token_aliases"] = 1
    return normalized, changes


def _parse_attributes(raw: str) -> Dict[str, str]:
    attributes: Dict[str, str] = {}
    for match in ATTRIBUTE_RE.finditer(raw or ""):
        value = next((group for group in match.groups()[1:] if group is not None), "")
        attributes[match.group(1).lower()] = value
    return attributes


def _normalized_role(kind: str, raw_role: str, warnings: List[str], element_id: str) -> str:
    role = ROLE_ALIASES.get((raw_role or "").lower(), (raw_role or "").lower())
    allowed = BLOCK_ROLES if kind == "BLOCK" else TABLE_ROLES
    fallback = "other" if kind == "BLOCK" else "other_table"
    if role not in allowed:
        warnings.append(f"{element_id}: role_invalide={raw_role or '<absent>'}, remplacé_par={fallback}")
        return fallback
    return role


def _derive_status(status: str, content: str) -> str:
    uncertain = "[ILLISIBLE]" in content
    truncated = "[TRONQUE]" in content
    if uncertain and truncated:
        return "uncertain_truncated"
    if uncertain:
        return "uncertain"
    if truncated:
        return "truncated"
    return status if status in ALLOWED_STATUSES else "readable"


def _normalize_table_rows(
    element_id: str,
    raw_lines: Sequence[str],
    declared_cols: Optional[int],
    warnings: List[str],
) -> Tuple[List[List[str]], int]:
    rows: List[List[str]] = []
    for row_number, raw_line in enumerate(raw_lines, start=1):
        if not raw_line.strip():
            warnings.append(f"{element_id}: ligne_table_vide_ignoree={row_number}")
            continue
        line = raw_line.replace("\t", "<TAB>")
        cells = line.split("<TAB>")
        normalized_cells = [cell if cell != "" else "<EMPTY>" for cell in cells]
        rows.append(normalized_cells)

    max_width = max((len(row) for row in rows), default=0)
    effective_cols = max(int(declared_cols or 0), max_width)
    if effective_cols <= 0:
        warnings.append(f"{element_id}: tableau_sans_cellule")
        return [], 0

    if declared_cols is None or declared_cols <= 0:
        warnings.append(f"{element_id}: cols_absent_derive={effective_cols}")
    elif declared_cols != max_width and max_width:
        warnings.append(
            f"{element_id}: cols_declare={declared_cols}, largeur_max={max_width}, largeur_effective={effective_cols}"
        )

    normalized_rows: List[List[str]] = []
    for row_number, row in enumerate(rows, start=1):
        if len(row) < effective_cols:
            warnings.append(
                f"{element_id}: ligne={row_number}, cellules={len(row)}, completee_a_droite={effective_cols}"
            )
            row = row + ["<EMPTY>"] * (effective_cols - len(row))
        normalized_rows.append(row)

    if not normalized_rows:
        return [], effective_cols

    if len(normalized_rows) == 1:
        warnings.append(f"{element_id}: en_tete_synthetique_ajoute")
        normalized_rows.insert(
            0,
            [f"[SANS_ENTETE_{index}]" for index in range(1, effective_cols + 1)],
        )

    header = normalized_rows[0]
    missing_header_index = 0
    for index, cell in enumerate(header):
        if cell.strip() in {"", "<EMPTY>"}:
            missing_header_index += 1
            header[index] = f"[SANS_ENTETE_{missing_header_index}]"
            warnings.append(
                f"{element_id}: en_tete_vide_colonne={index + 1}, token={header[index]}"
            )

    filtered_rows = [header]
    for row_number, row in enumerate(normalized_rows[1:], start=2):
        if all(cell.strip() in {"", "<EMPTY>"} for cell in row):
            warnings.append(f"{element_id}: ligne_entierement_vide_ignoree={row_number}")
            continue
        filtered_rows.append(row)
    return filtered_rows, effective_cols


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
    page_empty = False

    current: Optional[Dict[str, Any]] = None
    stray_lines: List[str] = []
    generated_block_counter = 0
    encountered_ids: Counter[str] = Counter()

    def flush_stray() -> None:
        nonlocal generated_block_counter, stray_lines
        meaningful = [line for line in stray_lines if line.strip()]
        stray_lines = []
        if not meaningful:
            return
        generated_block_counter += 1
        element_id = f"B_AUTO_{generated_block_counter:03d}"
        warnings.append(f"{element_id}: texte_hors_balise_preserve")
        elements.append(
            {
                "kind": "BLOCK",
                "id": element_id,
                "declared_order": None,
                "sequence": len(elements) + 1,
                "role": "other",
                "source": "mixed",
                "status": _derive_status("uncertain", "\n".join(meaningful)),
                "lines": meaningful,
            }
        )

    def finalize_current(reason: str = "normal") -> None:
        nonlocal current
        if current is None:
            return
        kind = str(current["kind"])
        attrs = dict(current["attrs"])
        raw_id = attrs.get("id", "").strip()
        if not raw_id:
            prefix = "B" if kind == "BLOCK" else "T"
            raw_id = f"{prefix}_AUTO_{len(elements) + 1:03d}"
            warnings.append(f"{raw_id}: id_absent_genere")
        encountered_ids[raw_id] += 1
        element_id = raw_id
        if encountered_ids[raw_id] > 1:
            element_id = f"{raw_id}_DUP{encountered_ids[raw_id]}"
            warnings.append(f"{raw_id}: id_duplique_renomme={element_id}")

        order_value: Optional[int]
        try:
            order_value = int(attrs.get("order", ""))
        except Exception:
            order_value = None
            warnings.append(f"{element_id}: order_absent_ou_invalide")

        raw_role = attrs.get("role") or attrs.get("role_hint") or ""
        role = _normalized_role(kind, raw_role, warnings, element_id)

        source = (attrs.get("source") or "").lower()
        if source not in ALLOWED_SOURCES:
            if role == "stamp":
                source = "stamp"
            elif role == "annotation":
                source = "handwritten"
            else:
                source = "printed"
            warnings.append(f"{element_id}: source_absente_ou_invalide, derivee={source}")

        status = (attrs.get("status") or "readable").lower()
        if status not in ALLOWED_STATUSES:
            warnings.append(f"{element_id}: status_invalide={status}, remplace=readable")
            status = "readable"

        raw_lines = list(current["lines"])
        while raw_lines and not raw_lines[0].strip():
            raw_lines.pop(0)
        while raw_lines and not raw_lines[-1].strip():
            raw_lines.pop()

        if kind == "BLOCK":
            if not raw_lines:
                warnings.append(f"{element_id}: block_vide_ignore")
                current = None
                return
            content = "\n".join(raw_lines)
            status = _derive_status(status, content)
            elements.append(
                {
                    "kind": kind,
                    "id": element_id,
                    "declared_order": order_value,
                    "sequence": len(elements) + 1,
                    "role": role,
                    "source": source,
                    "status": status,
                    "lines": raw_lines,
                }
            )
        else:
            try:
                declared_cols = int(attrs.get("cols", ""))
            except Exception:
                declared_cols = None
            rows, effective_cols = _normalize_table_rows(
                element_id, raw_lines, declared_cols, warnings
            )
            if not rows:
                warnings.append(f"{element_id}: table_vide_ignoree")
                current = None
                return
            content = "\n".join("<TAB>".join(row) for row in rows)
            status = _derive_status(status, content)
            elements.append(
                {
                    "kind": kind,
                    "id": element_id,
                    "declared_order": order_value,
                    "sequence": len(elements) + 1,
                    "role": role,
                    "source": source,
                    "status": status,
                    "cols": effective_cols,
                    "rows": rows,
                }
            )

        if reason != "normal":
            warnings.append(f"{element_id}: fermeture_salvage={reason}")
        current = None

    lines = (canonical_text or "").splitlines()
    for line_number, line in enumerate(lines, start=1):
        if END_PAGE_RE.match(line):
            flush_stray()
            finalize_current("end_page_avant_fermeture" if current else "normal")
            end_marker_present = True
            continue

        if line.strip() == "[PAGE VIDE]" and current is None:
            flush_stray()
            page_empty = True
            continue

        start_match = ELEMENT_START_RE.match(line)
        if start_match:
            flush_stray()
            if current is not None:
                finalize_current(f"nouvel_element_ligne_{line_number}")
            current = {
                "kind": start_match.group(1).upper(),
                "attrs": _parse_attributes(start_match.group(2)),
                "lines": [],
                "start_line": line_number,
            }
            continue

        if BLOCK_END_RE.match(line) or TABLE_END_RE.match(line):
            flush_stray()
            expected_kind = "BLOCK" if BLOCK_END_RE.match(line) else "TABLE"
            if current is None:
                warnings.append(f"ligne_{line_number}: fermeture_{expected_kind}_sans_ouverture")
                continue
            if current["kind"] != expected_kind:
                finalize_current(f"fermeture_inattendue_{expected_kind}_ligne_{line_number}")
            else:
                finalize_current()
            continue

        if current is not None:
            current["lines"].append(line)
        else:
            stray_lines.append(line)

    flush_stray()
    if current is not None:
        finalize_current("fin_de_reponse_sans_fermeture")

    declared_orders = [
        element["declared_order"]
        for element in elements
        if isinstance(element.get("declared_order"), int)
    ]
    if declared_orders and declared_orders != sorted(declared_orders):
        warnings.append("orders_non_croissants; ordre_de_sortie_conserve")
    if len(declared_orders) != len(set(declared_orders)):
        warnings.append("orders_dupliques; ordre_de_sortie_conserve")

    uncertain_ids = [
        element["id"]
        for element in elements
        if element.get("status") in {"uncertain", "uncertain_truncated"}
    ]
    truncated_ids = [
        element["id"]
        for element in elements
        if element.get("status") in {"truncated", "uncertain_truncated"}
    ]
    line_item_tables = [
        element for element in elements if element.get("role") == "line_items"
    ]
    totals_tables = [
        element for element in elements if element.get("role") == "totals_summary"
    ]

    if api_truncated:
        warnings.append("reponse_api_tronquee")
    if not end_marker_present:
        warnings.append("marqueur_END_PAGE_absent")
    if page_empty and elements:
        warnings.append("PAGE_VIDE_et_elements_presents")

    if page_empty and not elements and end_marker_present and not api_truncated:
        status = "validated"
    elif not elements:
        status = "unavailable"
        errors.append("aucun_element_canonique_exploitable")
    elif api_truncated or not end_marker_present or any(
        warning.startswith(("T", "B")) and "fermeture_salvage" in warning
        for warning in warnings
    ):
        status = "degraded"
    elif warnings or uncertain_ids or truncated_ids:
        status = "warning"
    else:
        status = "validated"

    quality = {
        "page_num": int(page_num),
        "status": status,
        "page_empty": page_empty,
        "end_marker_present": end_marker_present,
        "api_truncated": bool(api_truncated),
        "element_count": len(elements),
        "block_count": sum(1 for element in elements if element["kind"] == "BLOCK"),
        "table_count": sum(1 for element in elements if element["kind"] == "TABLE"),
        "line_item_table_count": len(line_item_tables),
        "totals_table_count": len(totals_tables),
        "has_line_items": bool(line_item_tables),
        "has_totals": bool(totals_tables),
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
        "quality": quality,
    }


# =============================================================================
# Rendu Markdown déterministe
# =============================================================================

SECTION_DEFINITIONS: List[Tuple[str, set[str], set[str]]] = [
    ("## Informations Émetteur", {"logo", "supplier"}, set()),
    ("## Informations Client", {"customer"}, set()),
    ("## Informations de Livraison", {"shipping"}, set()),
    ("## Détails du Document", {"document"}, {"document_meta"}),
    ("## Tableau des Lignes de Facturation", {"line_note"}, {"line_items"}),
    ("## Montants, Taxes et Totaux", set(), {"tax_summary", "totals_summary"}),
    ("## Informations de Paiement", {"payment", "bank"}, {"payment_table"}),
    (
        "## Annotations, Tampons et Signatures",
        {"annotation", "stamp", "signature_label"},
        set(),
    ),
    (
        "## Mentions Légales et Autres Contenus Visibles",
        {"legal", "marketing", "other"},
        {"other_table"},
    ),
]


def _display_tokens(text: str) -> str:
    return (text or "").replace("[TRONQUE]", "[TRONQUÉ]")


def _escape_markdown_cell(text: str) -> str:
    value = _display_tokens(text)
    if value == "<EMPTY>":
        return ""
    value = value.replace("<BR>", "<br>")
    value = value.replace("\n", "<br>")
    value = value.replace("\\", "\\\\").replace("|", "\\|")
    return value


def _render_table(element: Dict[str, Any]) -> str:
    rows = element.get("rows") or []
    if not rows:
        return ""
    header = rows[0]
    data_rows = rows[1:]
    output = [
        "| " + " | ".join(_escape_markdown_cell(cell) for cell in header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in data_rows:
        output.append("| " + " | ".join(_escape_markdown_cell(cell) for cell in row) + " |")
    return "\n".join(output)


def _render_block(element: Dict[str, Any]) -> str:
    lines = [_display_tokens(line) for line in element.get("lines", [])]
    content = "<br>\n".join(lines).strip()
    source = element.get("source")
    role = element.get("role")
    if source == "handwritten" or role == "annotation":
        return f"**Manuscrit :** {content}"
    if source == "stamp" or role == "stamp":
        return f"**Tampon :** {content}"
    if role == "signature_label":
        return f"**Zone de signature :** {content}"
    return content


def _quality_comment(quality: Dict[str, Any]) -> str:
    safe_status = re.sub(r"[^a-z_]+", "_", str(quality.get("status", "unknown")).lower())
    return (
        "<!-- OCR_QUALITY "
        f"status={safe_status} "
        f"warnings={int(quality.get('warning_count', 0) or 0)} "
        f"errors={int(quality.get('error_count', 0) or 0)} "
        f"uncertain={len(quality.get('uncertain_element_ids', []) or [])} "
        f"truncated={len(quality.get('truncated_element_ids', []) or [])} -->"
    )


def render_markdown_page(parsed: Dict[str, Any]) -> str:
    page_num = int(parsed["page_num"])
    quality = dict(parsed["quality"])
    lines: List[str] = [f"<!-- PAGE {page_num} -->", "", _quality_comment(quality)]

    if parsed.get("page_empty") and not parsed.get("elements"):
        lines.extend(["", "**[PAGE VIDE]**"])
        return "\n".join(lines).strip("\n")

    elements = list(parsed.get("elements") or [])
    if not elements:
        lines.extend(["", "## Extraction indisponible", "", "[PAGE NON EXTRAITE]"])
        return "\n".join(lines).strip("\n")

    used_ids: set[str] = set()
    for heading, block_roles, table_roles in SECTION_DEFINITIONS:
        section_elements = [
            element
            for element in elements
            if (
                (element["kind"] == "BLOCK" and element.get("role") in block_roles)
                or (element["kind"] == "TABLE" and element.get("role") in table_roles)
            )
        ]
        if not section_elements:
            continue
        lines.extend(["", heading, ""])
        for element in section_elements:
            rendered = _render_block(element) if element["kind"] == "BLOCK" else _render_table(element)
            if rendered:
                lines.append(rendered)
                lines.append("")
            used_ids.add(str(element["id"]))

    leftovers = [element for element in elements if str(element["id"]) not in used_ids]
    if leftovers:
        lines.extend(["", "## Autres Contenus Visibles", ""])
        for element in leftovers:
            rendered = _render_block(element) if element["kind"] == "BLOCK" else _render_table(element)
            if rendered:
                lines.append(rendered)
                lines.append("")

    return "\n".join(lines).strip("\n")


def render_summary_markdown_page(parsed: Dict[str, Any]) -> str:
    page_num = int(parsed["page_num"])
    quality = dict(parsed["quality"])
    selected_block_roles = {"supplier", "customer", "document"}
    selected_table_roles = {"document_meta", "tax_summary", "totals_summary"}
    selected = [
        element
        for element in parsed.get("elements", [])
        if (
            (element["kind"] == "BLOCK" and element.get("role") in selected_block_roles)
            or (element["kind"] == "TABLE" and element.get("role") in selected_table_roles)
        )
    ]
    lines = [f"<!-- PAGE {page_num} -->", "", _quality_comment(quality)]
    if not selected:
        lines.extend(["", "[AUCUNE DONNÉE SYNTHÉTIQUE EXTRAITE]"])
        return "\n".join(lines)

    groups = [
        ("## Identité et Parties", {"supplier", "customer"}),
        ("## Détails du Document", {"document", "document_meta"}),
        ("## Taxes et Totaux", {"tax_summary", "totals_summary"}),
    ]
    for heading, roles in groups:
        group = [element for element in selected if element.get("role") in roles]
        if not group:
            continue
        lines.extend(["", heading, ""])
        for element in group:
            rendered = _render_block(element) if element["kind"] == "BLOCK" else _render_table(element)
            if rendered:
                lines.extend([rendered, ""])
    return "\n".join(lines).strip("\n")


def render_canonical_page(parsed: Dict[str, Any]) -> str:
    """Rend une source canonique normalisée depuis la structure parsée."""
    if parsed.get("page_empty") and not parsed.get("elements"):
        return "[PAGE VIDE]\n[[END_PAGE]]"

    output: List[str] = []
    for element in parsed.get("elements", []) or []:
        kind = str(element.get("kind", "BLOCK")).upper()
        element_id = str(element.get("id", "B_AUTO"))
        order = int(element.get("sequence", 0) or 0)
        role = str(element.get("role", "other" if kind == "BLOCK" else "other_table"))
        source = str(element.get("source", "printed"))
        status = str(element.get("status", "readable"))
        if kind == "TABLE":
            cols = int(element.get("cols", 0) or 0)
            output.append(
                f"[[TABLE id={element_id} order={order:03d} role={role} "
                f"source={source} status={status} cols={cols}]]"
            )
            for row in element.get("rows", []) or []:
                output.append("<TAB>".join(str(cell) for cell in row))
            output.append("[[/TABLE]]")
        else:
            output.append(
                f"[[BLOCK id={element_id} order={order:03d} role={role} "
                f"source={source} status={status}]]"
            )
            output.extend(str(line) for line in element.get("lines", []) or [])
            output.append("[[/BLOCK]]")
    output.append("[[END_PAGE]]")
    return "\n".join(output).strip()


def wrap_canonical_source(canonical_text: str, page_num: int) -> str:
    return f"[[PAGE {int(page_num)}]]\n{canonical_text.strip()}".strip()


def build_unavailable_page(page_num: int, error: BaseException | str) -> Dict[str, Any]:
    message = str(error).replace("\n", " ")[:1000]
    quality = {
        "page_num": int(page_num),
        "status": "unavailable",
        "page_empty": False,
        "end_marker_present": False,
        "api_truncated": False,
        "element_count": 0,
        "block_count": 0,
        "table_count": 0,
        "line_item_table_count": 0,
        "totals_table_count": 0,
        "has_line_items": False,
        "has_totals": False,
        "uncertain_element_ids": [],
        "truncated_element_ids": [],
        "warnings": [],
        "errors": [message],
        "warning_count": 0,
        "error_count": 1,
    }
    parsed = {"page_num": int(page_num), "page_empty": False, "elements": [], "quality": quality}
    return {
        "page_num": int(page_num),
        "canonical": f"[[PAGE {int(page_num)}]]\n[EXTRACTION_INDISPONIBLE]\n[[END_PAGE]]",
        "markdown": render_markdown_page(parsed),
        "summary_markdown": render_summary_markdown_page(parsed),
        "quality": quality,
        "stats": {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
            "cache_creation_input_tokens": 0,
            "reasoning_tokens": 0,
            "image_tokens": 0,
            "duration_ms": 0,
            "quality_status": "unavailable",
            "page_error": message,
            "pipeline_version": PIPELINE_VERSION,
        },
    }


# =============================================================================
# Traitement d'une page — une génération Qwen
# =============================================================================


def process_page(
    pdf_path: str,
    page_num: int,
    api_key: str,
    image_dir: str,
) -> Dict[str, Any]:
    page_num = int(page_num)
    views: List[Dict[str, Any]] = []
    cleanup_paths: List[str] = []
    image_stats: Dict[str, Any] = {}
    try:
        views, cleanup_paths, image_stats = prepare_page_views(
            pdf_path=pdf_path,
            page_num=page_num,
            image_dir=image_dir,
        )
        _log(
            f"➡️ Page {page_num}: appel OCR canonique unique avec "
            f"{len(views)} vue(s), {image_stats['total_base64_image_mb']:.2f} Mo Base64"
        )
        messages = _build_ocr_messages(page_num, views)
        raw_text, api_stats = _call_chat(
            api_key=api_key,
            messages=messages,
            context=f"OCR canonique page {page_num}",
        )
        canonical_text, sanitizations = sanitize_canonical_response(raw_text)
        parsed = parse_canonical_page(
            canonical_text,
            page_num,
            api_truncated=bool(api_stats.get("truncated_output")),
        )
        quality = dict(parsed["quality"])
        markdown = render_markdown_page(parsed)
        summary_markdown = render_summary_markdown_page(parsed)
        normalized_canonical = render_canonical_page(parsed)
        canonical = wrap_canonical_source(normalized_canonical, page_num)

        stats = {
            **api_stats,
            **image_stats,
            "sanitizations": sanitizations,
            "raw_response_sha256": _sha256_text(raw_text),
            "canonical_sha256": _sha256_text(canonical),
            "markdown_sha256": _sha256_text(markdown),
            "summary_markdown_sha256": _sha256_text(summary_markdown),
            "canonical_generations": 1,
            "nominal_generations_per_page": NOMINAL_GENERATIONS_PER_PAGE,
            "semantic_retries": SEMANTIC_RETRIES,
            "quality_status": quality["status"],
            "quality_warning_count": quality["warning_count"],
            "quality_error_count": quality["error_count"],
            "uncertain_element_count": len(quality["uncertain_element_ids"]),
            "truncated_element_count": len(quality["truncated_element_ids"]),
            "has_line_items": bool(quality["has_line_items"]),
            "has_totals": bool(quality["has_totals"]),
            "canonical_ocr_only": CANONICAL_OCR_ONLY,
            "deterministic_markdown": DETERMINISTIC_MARKDOWN,
            "model": MODEL_OCR,
            "pipeline_version": PIPELINE_VERSION,
            "pipeline_fingerprint": get_pipeline_fingerprint(),
        }
        _log(
            f"✅ Page {page_num}: source canonique + Markdown déterministe, "
            f"qualité={quality['status']}, éléments={quality['element_count']}"
        )
        return {
            "page_num": page_num,
            "canonical": canonical,
            "markdown": markdown,
            "summary_markdown": summary_markdown,
            "quality": quality,
            "stats": stats,
        }
    finally:
        for view in views:
            view.pop("data_url", None)
        cleanup_page_images(cleanup_paths)


# Compatibilité minimale avec d'anciens tests.
def process_page_with_cache(
    pdf_path: str,
    page_num: int,
    api_key: str,
    is_first_page: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    del is_first_page
    with tempfile.TemporaryDirectory(prefix="qwen_canonical_page_") as image_dir:
        result = process_page(pdf_path, page_num, api_key, image_dir)
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
        "render_dpi": RENDER_DPI,
        "detail_dpi": DETAIL_DPI,
        "detail_views": ENABLE_DETAIL_VIEWS,
        "detail_upper_end": DETAIL_UPPER_END,
        "detail_lower_start": DETAIL_LOWER_START,
        "high_resolution": QWEN_HIGH_RES_IMAGES,
        "max_tokens_ocr": MAX_TOKENS_OCR,
        "temperature": TEMPERATURE,
        "thinking": ENABLE_THINKING_OCR,
        "prompt_sha256": _sha256_text(OCR_PROMPT),
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
        if not all(isinstance(record.get(name), str) for name in ("canonical", "markdown", "summary_markdown")):
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
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "pipeline_version": PIPELINE_VERSION,
        "pipeline_fingerprint": get_pipeline_fingerprint(),
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
# Validation finale et métriques
# =============================================================================


def _split_markdown_row(line: str) -> List[str]:
    r"""Découpe une ligne de tableau sans confondre ``\|`` avec un séparateur."""
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


def validate_markdown_quality(final_markdown: str, page_count: int) -> Dict[str, Any]:
    errors: List[str] = []
    try:
        validate_canonical_markdown_structure(final_markdown, page_count)
    except Exception as exc:
        errors.append(str(exc))
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": [],
        "summary": "Structure déterministe valide" if not errors else "KO: " + " | ".join(errors),
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
    "API_URL",
    "MODEL",
    "MODEL_OCR",
    "PIPELINE_VERSION",
    "CANONICAL_OCR_ONLY",
    "DETERMINISTIC_MARKDOWN",
    "NOMINAL_GENERATIONS_PER_PAGE",
    "SEMANTIC_RETRIES",
    "STOP_ON_CRITICAL",
    "PUBLISH_PARTIAL_DOCUMENT",
    "PUBLISH_DEGRADED_MARKDOWN",
    "RENDER_DPI",
    "DETAIL_DPI",
    "ENABLE_DETAIL_VIEWS",
    "QWEN_HIGH_RES_IMAGES",
    "ENABLE_THINKING_OCR",
    "ENABLE_EXPLICIT_CACHE",
    "validate_api_configuration",
    "configure_explicit_cache_for_batch",
    "get_pipeline_fingerprint",
    "get_progress_path",
    "get_pdf_info",
    "load_progress",
    "save_progress",
    "clear_progress",
    "process_page",
    "process_page_with_cache",
    "build_unavailable_page",
    "validate_canonical_markdown_structure",
    "validate_markdown_quality",
    "calculate_costs",
    "sanitize_canonical_response",
    "parse_canonical_page",
    "render_canonical_page",
    "render_markdown_page",
    "render_summary_markdown_page",
]

