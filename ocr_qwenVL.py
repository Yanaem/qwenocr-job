#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ocr_qwenVL.py — OCR Qwen validé + Markdown Python déterministe, compatible qwenocr_runner.py

Expose (d'après tes logs) :
- MODEL (str)
- INTER_REQUEST_DELAY (float)
- STOP_ON_CRITICAL (bool)  <-- attendu par le runner en cas d'erreur page
- get_pdf_info(pdf_path) -> dict {page_count,...}
- load_progress(pdf_path) -> Dict[str, Dict]
- save_progress(pdf_path, completed_pages) -> None
- clear_progress(pdf_path) -> None
- process_page_with_cache(pdf_path, page_num, api_key, is_first_page=False) -> (markdown_page, stats_payload)
- calculate_costs(stats_list) -> dict
- validate_markdown_quality(final_markdown, page_count) -> dict

Implémentation :
- 1 appel Qwen principal : OCR structuré image -> texte.
- Validation bloquante et retries ciblés si la structure OCR est incohérente.
- Markdown construit exclusivement par Python : aucun second appel Qwen, aucune réinterprétation générative.
- Annexe OCR brut ajoutée dans le markdown (côté code, 0 token).

Robustesse :
- DPI par défaut = 300 (configurable via env RENDER_DPI)
- Conversion PDF->PNG low-memory (pdf2image paths_only + fichiers temporaires)
- Retry court + logs en cas de 429/overloaded
- Retry spécifique si OCR invalide ou trop court (jusqu'à 3 reprises qualité par défaut)
- Payload stats contient des clés "flat" + sous-clé "stats" (compat runner)
"""

from __future__ import annotations

import base64
import hashlib
import html
import io
import json
import os
import re
import shlex
import tempfile
import threading
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import requests
from requests.adapters import HTTPAdapter
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

QWEN_WORKSPACE_ID = os.getenv("QWEN_WORKSPACE_ID", "").strip()
_QWEN_API_URL_OVERRIDE = os.getenv("QWEN_API_URL", "").strip().rstrip("/")

# Le Workspace ID est prioritaire. Ainsi, une ancienne variable QWEN_API_URL
# restée dans Cloud Run ne peut pas forcer silencieusement l'ancien domaine.
if QWEN_WORKSPACE_ID:
    API_URL = (
        f"https://{QWEN_WORKSPACE_ID}.ap-southeast-1.maas.aliyuncs.com/"
        "compatible-mode/v1"
    )
elif _QWEN_API_URL_OVERRIDE:
    API_URL = _QWEN_API_URL_OVERRIDE
else:
    API_URL = ""

MODEL_OCR = os.getenv("QWEN_MODEL_OCR", "qwen3.7-plus")
MODEL_MD = "python-deterministic-v4"

# Isole les checkpoints et les statistiques des anciennes versions du processeur.
PROCESSOR_VERSION = "python-markdown-v4-20260727"
_PROCESSOR_VERSION_SAFE = re.sub(r"[^A-Za-z0-9_.-]+", "-", PROCESSOR_VERSION).strip(".-")

# Attendu par le runner (affiché au démarrage)
MODEL = MODEL_OCR

# Conservé pour compatibilité avec le runner, sans pause fixe par défaut.
INTER_REQUEST_DELAY = _env_float("INTER_REQUEST_DELAY", 0.0)

# Attendu par le runner : stop ou non sur erreur page "critique"
# Recommandation : True pour ne pas générer de Markdown incomplet sans le savoir.
STOP_ON_CRITICAL = _env_bool("STOP_ON_CRITICAL", True)

# Qualité demandée : 300 DPI (configurable)
RENDER_DPI = _env_int("RENDER_DPI", 300)

MAX_TOKENS_OCR = _env_int("MAX_TOKENS_OCR", 16000)

TEMPERATURE = _env_float("TEMPERATURE", 0.0)

# Timeouts/retries réseau
REQUEST_TIMEOUT_SECONDS = _env_int("REQUEST_TIMEOUT_SECONDS", 600)
CONNECT_TIMEOUT_SECONDS = _env_int("CONNECT_TIMEOUT_SECONDS", 10)
HTTP_POOL_SIZE = max(1, _env_int("HTTP_POOL_SIZE", 8))

MAX_RETRIES = _env_int("MAX_RETRIES", 3)
BACKOFF_BASE = _env_float("BACKOFF_BASE", 2.0)
BACKOFF_MAX = _env_float("BACKOFF_MAX", 20.0)  # volontairement bas

# Logs
VERBOSE = _env_bool("VERBOSE", True)

# Rate-limit behavior
FAIL_FAST_ON_429 = _env_bool("FAIL_FAST_ON_429", False)  # False = retry court; True = fail vite

# Contrôles qualité OCR. Une page n'est acceptée que si la sortie structurée est valide.
OCR_MIN_CHARS = _env_int("OCR_MIN_CHARS", 40)
OCR_EMPTY_RETRIES = max(0, _env_int("OCR_EMPTY_RETRIES", 2))
OCR_QUALITY_RETRIES = max(0, _env_int("OCR_QUALITY_RETRIES", 3))
OCR_EMPTY_RETRY_SLEEP = _env_float("OCR_EMPTY_RETRY_SLEEP", 1.5)
OCR_EMPTY_PAGE_CONFIRMATIONS = max(1, _env_int("OCR_EMPTY_PAGE_CONFIRMATIONS", 2))
STRICT_OCR_STRUCTURE = _env_bool("STRICT_OCR_STRUCTURE", True)
STRICT_FUSED_CELL_HEURISTICS = _env_bool("STRICT_FUSED_CELL_HEURISTICS", True)
ALLOW_SAFE_STRUCTURE_REPAIR = _env_bool("ALLOW_SAFE_STRUCTURE_REPAIR", False)
BLANK_PAGE_DARK_PIXEL_RATIO = max(0.0, _env_float("BLANK_PAGE_DARK_PIXEL_RATIO", 0.001))

# Le mode haute résolution est explicitement demandé à Qwen-VL pour préserver les petits caractères.
QWEN_HIGH_RES_IMAGES = _env_bool("QWEN_HIGH_RES_IMAGES", True)
# Garde une marge sous la limite de transport des images encodées en base64.
MAX_BASE64_IMAGE_MB = max(1.0, _env_float("MAX_BASE64_IMAGE_MB", 9.5))

# Le Markdown est exclusivement produit par Python. Une éventuelle ancienne variable
# USE_QWEN_MARKDOWN présente dans Cloud Run est volontairement ignorée.
USE_QWEN_MARKDOWN = False

# Thinking mode : uniquement pour la lecture OCR.
ENABLE_THINKING_OCR = _env_bool("ENABLE_THINKING_OCR", True)
ENABLE_THINKING_MD = False
ALLOW_NO_THINK_FALLBACK = _env_bool("ALLOW_NO_THINK_FALLBACK", True)
EMPTY_RESPONSE_LOG_CHARS = _env_int("EMPTY_RESPONSE_LOG_CHARS", 1500)

# Le cache explicite porte uniquement sur le long prompt statique. Il ne change ni la hiérarchie
# des messages ni le contenu de l’image. Pour un lot d’une seule vague, il est désactivé automatiquement.
ENABLE_EXPLICIT_CACHE = _env_bool("ENABLE_EXPLICIT_CACHE", True)


def _log(msg: str) -> None:
    if VERBOSE:
        print(msg, flush=True)


if os.getenv("USE_QWEN_MARKDOWN", "").strip().lower() not in {"", "0", "false", "no", "off"}:
    _log("⚠️ USE_QWEN_MARKDOWN est ignoré : cette version impose le rendu Markdown Python.")


def validate_api_configuration() -> None:
    """Valide l'endpoint Qwen avant le premier appel réseau."""
    if not API_URL:
        raise RuntimeError(
            "Endpoint Qwen non configuré. Définis QWEN_WORKSPACE_ID avec l'identifiant "
            "du workspace Alibaba Cloud Model Studio (région Singapour), ou fournis "
            "QWEN_API_URL avec le nouvel endpoint workspace-specific."
        )

    if not API_URL.startswith("https://"):
        raise RuntimeError("QWEN_API_URL doit commencer par https://")

    if "dashscope-intl.aliyuncs.com" in API_URL.lower():
        raise RuntimeError(
            "L'ancien endpoint Qwen Singapour est encore configuré. Supprime "
            "QWEN_API_URL ou remplace-le, puis définis QWEN_WORKSPACE_ID afin "
            "d'utiliser le domaine dédié au workspace."
        )

    if QWEN_WORKSPACE_ID and (
        any(ch.isspace() for ch in QWEN_WORKSPACE_ID)
        or "/" in QWEN_WORKSPACE_ID
    ):
        raise RuntimeError("QWEN_WORKSPACE_ID est invalide : fournis uniquement l'identifiant du workspace.")


# =====================
# Helpers structure Markdown canonique
# =====================

OCR_PAGE_TOKEN_RE = re.compile(
    r"^\s*\[\[(?:PDF_)?PAGE\s+\d+\]\]\s*$",
    flags=re.IGNORECASE,
)

HTML_PAGE_MARKER_RE = re.compile(
    r"^\s*<!--\s*PAGE\s+(\d+)\s*:?\s*-->\s*$",
    flags=re.IGNORECASE,
)

FENCE_MARKER_RE = re.compile(r"^\s*(`{3,}|~{3,})")


def _fence_state_after_line(line: str, active_fence: Optional[str]) -> Tuple[Optional[str], bool]:
    """Retourne (nouvel état, la ligne est-elle une vraie ouverture/fermeture)."""
    match = FENCE_MARKER_RE.match(line or "")
    if not match:
        return active_fence, False
    token = match.group(1)
    if active_fence is None:
        return token, True
    remainder = (line or "")[match.end():].strip()
    if token[0] == active_fence[0] and len(token) >= len(active_fence) and not remainder:
        return None, True
    return active_fence, False


def _ocr_appendix_fence(ocr_text: str) -> str:
    """Choisit un fence plus long que toute suite de backticks contenue dans l'OCR."""
    runs = [len(match.group(0)) for match in re.finditer(r"`+", ocr_text or "")]
    return "`" * max(3, (max(runs) + 1) if runs else 3)


def _strip_model_page_tokens(text: str) -> str:
    """
    Supprime les tokens techniques [[PAGE n]] ou [[PDF_PAGE n]] produits par le modèle.

    La pagination physique est ajoutée exclusivement par le code Python.
    Les textes visibles de facture comme "Page 1/1" restent inchangés.
    """
    return "\n".join(
        line
        for line in (text or "").splitlines()
        if not OCR_PAGE_TOKEN_RE.match(line)
    ).strip()


def _strip_model_html_page_markers(markdown: str) -> str:
    """Supprime les marqueurs physiques générés hors fences, jamais ceux qui sont visibles dans un code block."""
    output: List[str] = []
    active_fence: Optional[str] = None
    for line in (markdown or "").splitlines():
        new_state, boundary = _fence_state_after_line(line, active_fence)
        if boundary:
            active_fence = new_state
            output.append(line)
            continue
        if active_fence is None and HTML_PAGE_MARKER_RE.match(line):
            continue
        output.append(line)
    return "\n".join(output).strip()


def _extract_html_page_markers_outside_fences(markdown: str) -> List[int]:
    """Retourne les numéros de balises <!-- PAGE n --> situées hors blocs de code."""
    markers: List[int] = []
    active_fence: Optional[str] = None
    for line in (markdown or "").splitlines():
        new_state, boundary = _fence_state_after_line(line, active_fence)
        if boundary:
            active_fence = new_state
            continue
        if active_fence is not None:
            continue
        match = HTML_PAGE_MARKER_RE.match(line)
        if match:
            markers.append(int(match.group(1)))
    return markers


def validate_canonical_markdown_structure(final_markdown: str, page_count: int) -> None:
    """
    Validation bloquante du contrat Markdown phase 1.

    Format attendu : exactement une balise HTML <!-- PAGE n --> par page physique,
    hors blocs de code, dans l'ordre [1..page_count].
    """
    expected_pages = list(range(1, int(page_count or 0) + 1))
    actual_pages = _extract_html_page_markers_outside_fences(final_markdown)

    if actual_pages != expected_pages:
        raise RuntimeError(
            "Structure Markdown physique invalide: "
            f"attendu={expected_pages}, obtenu={actual_pages}"
        )

    if re.search(r"(?:^|\n)\s*---\s*$", final_markdown.rstrip(), flags=re.MULTILINE):
        # Vérification stricte du séparateur final : le document ne doit pas finir par une ligne ---.
        lines = final_markdown.rstrip().splitlines()
        if lines and lines[-1].strip() == "---":
            raise RuntimeError("Le Markdown canonique ne doit pas finir par ---")


# =====================
# Prompts
# =====================

OCR_PROMPT = """Tu es un moteur OCR layout-aware spécialisé en documents comptables : factures, avoirs, notes de crédit, proformas.

OBJECTIF
Transcrire TOUT le texte visible d'une page en conservant le layout utile, pour générer ensuite un Markdown fidèle et exploitable.

SORTIE
- Texte OCR structuré uniquement.
- Interdiction : Markdown, JSON, explication, commentaire, bloc ```.
- Chaque appel traite une seule page.

PRIORITÉ DES RÈGLES

En cas de conflit, applique les règles dans cet ordre :

1. Fidélité absolue au texte visible : ne jamais inventer, corriger, normaliser, calculer ou compléter.
2. Ne jamais perdre un texte visible non vide.
3. Ne jamais fusionner deux zones visuellement distinctes.
4. Préserver l'intégrité des tableaux : N cellules par ligne, aucun padding.
5. Déterminer role_hint par le contenu et le layout, jamais par la position seule.
6. En cas de doute sur la structure d'un tableau, utiliser [[BLOCK]] plutôt que fabriquer un tableau.
7. En cas de doute sur un role_hint, utiliser role_hint=unknown.

FORMAT DES ÉLÉMENTS

[[BLOCK id=B001 order=001 pos=top-left role_hint=unknown]]
texte
[[/BLOCK]]

[[TABLE id=T001 order=001 pos=middle role_hint=unknown cols=N]]
cellule<TAB>cellule<TAB>cellule
[[/TABLE]]

bbox est optionnel.
- Ne l'ajoute que si les coordonnées sont utiles et fiables.
- Si bbox est ajouté, format strict : bbox=x1,y1,x2,y2, coordonnées normalisées 0-1000.
- Si bbox est incertain, ne mets aucun bbox.

Tokens autorisés dans le contenu :
<TAB>
<EMPTY>
<BR>
[ILLISIBLE]
[SANS_ENTETE_n]

Positions autorisées :
top-left, top, top-right,
middle-left, middle, middle-right,
bottom-left, bottom, bottom-right,
unknown.

role_hint autorisés :
supplier_identity
supplier_address
supplier_legal
supplier_contact
customer_identity
customer_address
customer_contact
customer_legal
billing_address
shipping_address
shipping_details
shipping_contact
delivery_confirmation
invoice_title
invoice_details
line_items
line_items_note
line_items_footer
tax_summary
totals_summary
payment_terms
bank_details
payment
legal_terms
marketing_badge
logo_text
stamp_signature
qr_barcode_text
notes
isolated_value
unknown

RÈGLES GÉNÉRALES

- N'ajoute aucun token technique de pagination comme [[PAGE n]].
- Ne génère jamais de marqueur technique [[PAGE n]] ou [[PDF_PAGE n]].
- Transcris uniquement les indications de pagination réellement visibles sur le document, par exemple "Page 1/1", "Page : 2" ou "2/3".
- Le code appelant gère lui-même la numérotation physique du PDF.
- Copie uniquement le texte visible.
- Conserve exactement lettres, chiffres, dates, montants, séparateurs, virgules, points, %, €, devises, majuscules, minuscules, abréviations, accents.
- Ne corrige pas.
- Ne reformule pas.
- Ne normalise pas.
- Ne calcule pas.
- Ne complète aucune information absente.
- N'ajoute aucun libellé, montant, symbole, devise, champ ou total absent de l'image.
- Transcris tout texte lisible : fournisseur, client, contacts, adresses, livraison, références, articles, prestations, taxes, totaux, échéances, banque, RIB, IBAN, BIC, conditions, mentions légales, pied de page, annotations, tampons, statut de paiement, texte lisible dans un logo.
- Ne transcris pas le contenu encodé d'un QR code ou d'un code-barres.
- Transcris seulement le texte imprimé lisible autour ou dans un logo, QR code ou code-barres si ce texte est réellement visible.
- Ignore uniquement les éléments purement graphiques sans texte lisible.
- Si une portion est illisible : écris [ILLISIBLE] à l'endroit concerné.
- Si la page est réellement vide : réponds exactement [PAGE VIDE].
- Un même texte visible ne doit apparaître qu'une seule fois, sauf s'il est répété visuellement.
- Ne déduis jamais une information à partir d'une autre page.

RÈGLES COMPTABLES — FIDÉLITÉ DES VALEURS

- Conserve exactement le signe des montants : "-", "+", "−", parenthèses comptables "(...)".
- Ne convertis jamais une parenthèse comptable en signe moins, ni un signe moins en parenthèses.
- Ne convertis jamais une virgule décimale en point décimal, ni l'inverse.
- Conserve les séparateurs de milliers visibles : espace, espace insécable, point, apostrophe.
- Ne modifie jamais les espaces à l'intérieur d'un montant.
- Conserve la devise exactement comme affichée : €, EUR, $, USD, CHF, £, HUF, etc.
- Conserve la position de la devise : avant ou après le montant.
- Conserve les taux de TVA exactement : 0%, 5,5%, 8,5%, 10%, 20%, 8.1%, etc.
- Ne fusionne jamais deux taux différents.
- Pour un avoir, une note de crédit, un remboursement ou un montant négatif : conserve le titre et les montants tels quels.
- Conserve les mentions de statut exactement : "Payé", "Acquittée", "Soldé", "Reste à payer", "Net à payer", "Échu", "À régler".
- Conserve les numéros de facture, avoir, commande, client, livraison, suivi et identifiants fiscaux exactement.

IDENTIFIANTS, ORDRE ET RÔLES

- Chaque [[BLOCK]] a un id unique B001, B002, B003...
- Chaque [[TABLE]] a un id unique T001, T002, T003...
- order est global à la page et croît selon l'ordre de lecture : 001, 002, 003...
- order s'applique aux blocs et aux tableaux ensemble.
- pos est seulement une position approximative.
- pos ne suffit jamais à déterminer le rôle d'un bloc.
- Deux blocs peuvent avoir le même pos sans devoir être fusionnés.
- role_hint doit être choisi selon le contenu visible et le layout, jamais selon la position seule.
- Si le rôle est incertain, utilise role_hint=unknown.
- Ne force jamais supplier_identity, supplier_address, customer_identity, customer_address ou shipping_address par position seule.

RÈGLES DE RÔLES

- Nom commercial, raison sociale ou logo textuel du vendeur/émetteur : role_hint=supplier_identity.
- Texte de logo non suffisant pour identifier l'émetteur : role_hint=logo_text.
- Adresse du vendeur/émetteur : role_hint=supplier_address.
- SIRET, SIREN, APE, NAF, TVA intracommunautaire, capital social, forme juridique, RCS : role_hint=supplier_legal.
- Téléphone, fax, email, site web du vendeur : role_hint=supplier_contact.
- Nom du client, acheteur, destinataire ou facturé à : role_hint=customer_identity.
- Adresse du client, facturé à, adresse de facturation : role_hint=customer_address ou billing_address.
- Contact client, email client, téléphone client, personne de contact client : role_hint=customer_contact.
- SIRET, TVA intra, identifiant fiscal ou information légale du client : role_hint=customer_legal.
- Adresse de livraison, livré à, expédié à, ship to, delivery address : role_hint=shipping_address.
- Mode de livraison, expédition, retrait, enlevé au comptoir, transporteur, incoterm, instruction de livraison, référence de livraison : role_hint=shipping_details.
- Contact de livraison, téléphone de livraison, email de livraison, personne à contacter pour la livraison : role_hint=shipping_contact.
- Confirmation de livraison, preuve de livraison, livré, reçu, nom ou signature liés à la livraison : role_hint=delivery_confirmation.
- Titre du document : FACTURE, AVOIR, NOTE DE CRÉDIT, PROFORMA, REÇU, etc. : role_hint=invoice_title.
- Numéro, date, référence, commande, vendeur, page imprimée, devise, objet, code client, statut de paiement isolé : role_hint=invoice_details.
- Statut de paiement dans la zone des totaux : role_hint=totals_summary.
- Tableau principal d'articles/prestations : role_hint=line_items.
- Note située au début ou au-dessus du tableau articles, liée aux articles mais ne décrivant pas une ligne article : role_hint=line_items_note.
- Pied de tableau articles, report, page, contact, signature ou total de report situé en bas du tableau articles : role_hint=line_items_footer.
- Tableau de TVA, taxes, bases, taux, montants de taxe : role_hint=tax_summary.
- Tableau de total HT, total TTC, acompte, remise globale, solde, net à payer : role_hint=totals_summary.
- Échéance, mode de règlement, conditions de paiement : role_hint=payment_terms.
- Banque, RIB, IBAN, BIC : role_hint=bank_details.
- Paiement générique si la distinction payment_terms / bank_details est impossible : role_hint=payment.
- Conditions légales, réserve de propriété, pénalités, indemnités, pied de page juridique : role_hint=legal_terms.
- Slogan, badge SAV, label qualité, argument marketing, texte promotionnel : role_hint=marketing_badge.
- Tampon ou signature non lié à une livraison : role_hint=stamp_signature.
- Texte lisible associé à QR code ou code-barres : role_hint=qr_barcode_text.
- Note libre : role_hint=notes.
- Valeur isolée sans libellé clair : role_hint=isolated_value.
- Rôle incertain : role_hint=unknown.

SÉPARATION DES BLOCS

- Ne fusionne jamais un slogan, badge SAV, label qualité, pictogramme, tampon, QR code ou texte marketing avec le fournisseur, le client ou la livraison.
- Ne fusionne jamais un bloc client avec un bloc marketing, même s'ils sont proches.
- Ne fusionne jamais un bloc fournisseur avec un bloc marketing, sauf si le texte est seulement le nom/logo de l'entreprise émettrice.
- Ne fusionne jamais une zone client avec une zone de livraison si elles sont visuellement séparées ou libellées différemment.
- Si une zone contient à la fois nom fournisseur et slogan marketing, sépare-les si visuellement possible.
- Si une zone contient à la fois marketing/logo et client, crée deux blocs séparés.
- Si une zone contient paiement et mentions légales, crée deux blocs séparés si une bordure, un espace ou un changement de style les sépare.
- Un bloc client doit contenir uniquement le destinataire, facturé à, acheteur, contact client, information légale client ou adresse de facturation.
- Les informations de livraison doivent aller dans shipping_address, shipping_details, shipping_contact ou delivery_confirmation, sauf si la facture ne distingue pas visuellement client et livraison.
- Tout texte proche du client mais sans lien explicite avec le destinataire doit rester dans un bloc séparé avec role_hint=marketing_badge, notes ou unknown.

LECTURE LAYOUT

- Lis par blocs visuels, pas par bande horizontale globale.
- Ordre des blocs : haut vers bas.
- À hauteur proche : gauche vers droite.
- Ne traverse jamais toute la page de gauche à droite si cela fusionne deux zones distinctes.
- Deux zones côte à côte restent deux blocs séparés si elles n'appartiennent pas à la même grille.
- Deux tableaux côte à côte restent deux [[TABLE]] séparés.
- Deux tableaux empilés mais séparés par bordure, espace, titre ou groupe d'en-têtes distinct restent deux [[TABLE]] séparés.
- Si une zone est ambiguë, utilise [[BLOCK]] ligne par ligne au lieu de fabriquer un tableau.
- Un titre situé au-dessus d'un tableau doit rester dans un [[BLOCK]] séparé, sauf s'il est clairement une cellule du tableau.

BLOCS

- Un [[BLOCK]] contient du texte non tabulaire.
- Chaque bloc commence par [[BLOCK ...]] et finit par [[/BLOCK]].
- N'utilise jamais <TAB> dans un [[BLOCK]].
- N'utilise jamais <BR> dans un [[BLOCK]].
- Si deux textes sont côte à côte mais ne forment pas une vraie grille, crée deux [[BLOCK]] séparés.
- Les adresses, contacts, mentions légales, notes, conditions, livraison et textes libres restent en [[BLOCK]].
- Les blocs de paiement sans vraie grille restent en [[BLOCK]], pas en [[TABLE]].
- Une ligne unique contenant des libellés et valeurs alignés reste en [[BLOCK]], jamais en [[TABLE]].
- Exemple : "Echéance Montant Conditions de Règlement 20/06/2025 09:24:16 Poids Brut:1,09Kg" doit être un [[BLOCK]], pas un [[TABLE]].
- Dans un [[BLOCK]], conserve les retours à la ligne utiles.
- Ne regroupe pas dans un même [[BLOCK]] des textes ayant des role_hint différents si une séparation visuelle existe.

TABLEAUX — DÉTECTION

- Chaque tableau visible commence par [[TABLE ... cols=N]] et finit par [[/TABLE]].
- N est obligatoire.
- N correspond au nombre réel de colonnes visuelles du tableau.
- Utilise <TAB> uniquement dans [[TABLE]].
- Un tableau = une grille continue OU un seul groupe logique d'en-têtes.
- Un [[TABLE]] doit contenir au minimum deux lignes OCR : une ligne d'en-tête et au moins une ligne de données.
- Si aucun en-tête n'est visible, crée d'abord une ligne d'en-têtes génériques [SANS_ENTETE_1], [SANS_ENTETE_2], etc., puis les lignes de données.
- Ne produis jamais un [[TABLE]] avec une seule ligne.
- Si une zone tabulaire ne contient qu'une seule ligne visible, transcris-la en [[BLOCK]] avec le role_hint approprié.
- Ne fusionne jamais deux groupes d'en-têtes indépendants dans une même [[TABLE]].
- Si deux zones ont des en-têtes, bordures, alignements ou espacements distincts, elles forment deux tableaux.
- Si un tableau de taxes et un tableau de totaux sont côte à côte, ils doivent rester deux [[TABLE]] séparés, sauf s'ils forment réellement une seule grille continue avec un seul groupe d'en-têtes.
- Si l'alignement ne permet pas de garantir les colonnes, ferme le tableau et transcris la zone en [[BLOCK]].

TABLEAUX — CELLULES

- Une ligne OCR = une ligne logique du tableau.
- Une cellule OCR = une cellule visuelle.
- Chaque ligne d'un tableau doit contenir exactement N cellules, donc exactement N-1 tokens <TAB>.
- Ne fusionne jamais deux cellules adjacentes.
- Ne divise jamais une cellule à cause d'espaces internes ordinaires.
- Détermine N avec toutes les colonnes réellement alignées : en-têtes visibles, lignes de données, totaux internes, codes, montants, taux, quantités.
- Ne détermine jamais N uniquement avec les libellés visibles de l'en-tête.
- Si les lignes de données ont plus de colonnes que les en-têtes visibles, ajoute [SANS_ENTETE_n] dans l'en-tête à la position exacte des colonnes sans libellé.
- Les marqueurs [SANS_ENTETE_n] sont numérotés séquentiellement dans chaque tableau, de gauche à droite, en recommençant à 1 pour chaque nouveau tableau.
- Une colonne sans en-tête n'est réelle que si au moins une ligne de données contient une valeur non vide dans cette colonne.
- Ne crée jamais [SANS_ENTETE_n] pour une colonne entièrement vide, un simple espace, une bordure, une marge ou une séparation graphique.
- Si une colonne n'a ni en-tête visible ni valeur visible dans aucune ligne, elle n'existe pas.
- N'ajoute jamais une colonne vide sans nom en fin d'en-tête.
- N'invente jamais un nom de colonne à partir du contenu des valeurs.
- Si une cellule réelle est vide dans une ligne réelle, utilise <EMPTY>.
- Si une cellule vide est en fin de ligne, écris quand même <EMPTY> pour conserver N cellules.
- Ne laisse jamais une cellule vide implicite.

TABLEAUX — EN-TÊTES

- Garde les en-têtes visibles exacts.
- Si un en-tête est écrit sur plusieurs lignes dans la même cellule, réunis les lignes avec <BR>.
- Si une ligne située juste sous les en-têtes contient uniquement des unités, devises ou marqueurs courts comme EUR, €, USD, HT, TTC, %, elle fait partie de l'en-tête.
- Fusionne ces unités dans les cellules d'en-tête correspondantes avec <BR>.
- Ne crée jamais une ligne de données composée uniquement d'unités, devises ou marqueurs courts.
- Les cellules vides d'une ligne d'unités restent vides et ne créent pas de nouvelles colonnes.
- Exemple : "Prix unit. HT" + ligne "EUR" devient "Prix unit. HT<BR>EUR".
- Exemple : "Total" + ligne "EUR" devient "Total<BR>EUR".

TABLEAUX — NOMBRES, TAUX, MONTANTS, CODES

- Si plusieurs valeurs courtes sont alignées en colonnes distinctes, elles doivent être séparées par <TAB>.
- Les nombres, montants, pourcentages, quantités, codes taxe, références et totaux alignés verticalement sont des cellules distinctes.
- Ne fusionne jamais un nombre et un pourcentage s'ils sont visuellement séparés ou répétés à la même position sur plusieurs lignes.
- Ne fusionne jamais un montant et un code taxe s'ils sont visuellement séparés ou répétés à la même position sur plusieurs lignes.
- Conserve le signe et les parenthèses des montants négatifs dans la cellule : -12,50 ou (12,50).
- Si une colonne sans en-tête contient des pourcentages répétés et qu'elle est visuellement située entre deux colonnes de nombres, prix ou montants, place [SANS_ENTETE_n] exactement à cette position.
- Ne place jamais [SANS_ENTETE_n] après la deuxième colonne numérique si les valeurs suivent l'ordre nombre/prix -> pourcentage -> nombre/prix.
- Exemple : si "7,430", "0%" et "7,430" sont trois valeurs alignées, l'en-tête doit être "Prix" <TAB> [SANS_ENTETE_1] <TAB> "Prix remisé" si seul le pourcentage n'a pas d'en-tête visible.
- Exemple : si "12,50", "0%" et "12,50" sont trois valeurs alignées en colonnes, transcris : 12,50<TAB>0%<TAB>12,50
- Exemple : si "100,00", "20%" et "120,00" sont trois valeurs alignées en colonnes, transcris : 100,00<TAB>20%<TAB>120,00
- Exemple : si "-15,00", "20%" et "-18,00" sont trois valeurs alignées, transcris : -15,00<TAB>20%<TAB>-18,00
- Une colonne contenant uniquement des pourcentages ou des codes sans en-tête visible doit avoir [SANS_ENTETE_n] dans l'en-tête.
- Ne remplace jamais [SANS_ENTETE_n] par "Remise", "TVA", "Code", "Taxe" ou autre libellé non visible.

TABLEAUX — ARTICLES

- Le tableau des articles contient seulement les vraies lignes d'articles ou prestations.
- Une ligne article réelle contient normalement une désignation et au moins une quantité, un prix, un montant, une taxe ou un code TVA.
- Une note, un contexte, une métadonnée documentaire, une information de livraison, une commande, un report ou un pied de tableau ne doit pas devenir une ligne article.
- Dans le tableau line_items, ne raisonne pas par mots exacts mais par fonction.
- Une ligne ou un segment appartient au tableau articles seulement s'il décrit un bien/prestation ou s'il porte une valeur commerciale de cette ligne : référence article, désignation, n° de série, quantité, prix, remise, montant, taxe, code TVA.
- Une ligne ou un segment est une métadonnée documentaire s'il a la forme libellé-valeur, instruction, contexte, report, pagination, contact, signature, livraison, référence de document ou information de suivi, et s'il ne porte pas les valeurs commerciales d'un article.
- Une métadonnée documentaire ne doit pas être fusionnée dans la désignation d'un article.
- Si cette métadonnée est située avant le premier article réel ou en tête du tableau, transcris-la en [[BLOCK ... role_hint=line_items_note]], invoice_details, shipping_details ou notes selon sa fonction.
- Si cette métadonnée est située après le dernier article réel ou en pied de tableau, transcris-la en [[BLOCK ... role_hint=line_items_footer]], shipping_details, delivery_confirmation ou notes selon sa fonction.
- Si une cellule contient à la fois une métadonnée documentaire et une vraie désignation produit, sépare les deux : la métadonnée sort du tableau, la désignation reste dans l'article.
- Ne retire jamais un mot simplement parce qu'il ressemble à un libellé documentaire : s'il fait partie d'une désignation produit normale et que la ligne contient quantité/prix/montant/taxe, il reste dans l'article.
- Exemples non exhaustifs de métadonnées documentaires : commande, référence commande, pièce site, report, à reporter, page, contact, signature, nom, expédition, livraison, transporteur, instruction de livraison.
- Une note située avant le premier article réel devient [[BLOCK ... role_hint=line_items_note]].
- Un pied situé après le dernier article réel devient [[BLOCK ... role_hint=line_items_footer]].
- Si une ligne située après un article réel contient seulement une référence secondaire, un EAN/GTIN, un code-barres imprimé, une garantie, une caractéristique produit ou une description longue, rattache-la à la ligne article précédente avec <BR>.
- Si la continuation est dans la colonne référence, rattache-la à la cellule référence précédente avec <BR>.
- Si la continuation est descriptive, rattache-la à la cellule désignation précédente avec <BR>.
- Si la continuation est dans la colonne N° de Série, rattache-la à la cellule N° de Série précédente avec <BR>.
- Rattacher une continuation avec <BR> ne modifie jamais le texte ; cela change seulement la cellule de rattachement.
- Ne conserve une ligne séparée dans line_items que si elle décrit clairement un nouvel article ou une nouvelle prestation.
- Une ligne de remise, d'avoir ou de correction avec montant négatif est une vraie ligne article si elle contient une quantité, un prix, un montant, une taxe ou un code TVA.
- Une ligne de report ou de pied ne doit jamais devenir une ligne article avec cellules vides.

TABLEAUX — ANTI-PADDING

- Ne crée jamais de ligne entièrement vide.
- Ne crée jamais de ligne composée uniquement de <EMPTY>.
- Ne crée jamais de lignes pour reproduire l'espace blanc d'un tableau haut.
- Une grande zone vide sous les articles ne doit produire aucune ligne OCR.
- Si une valeur isolée apparaît dans une zone vide du tableau sans former une ligne complète, ferme le tableau et transcris cette valeur dans un [[BLOCK ... role_hint=isolated_value]] séparé.
- Une valeur isolée ne doit pas devenir une ligne d'article.
- Une valeur isolée ne doit pas être supprimée.

RÈGLES FACTURES

- Les zones articles, prestations, notes d'articles, pieds de tableau, livraison, taxes, remises, acomptes, totaux, échéances, paiements et mentions peuvent être des tableaux ou des blocs séparés.
- Ne suppose jamais qu'un total, une taxe, un acompte ou un solde appartient au tableau voisin.
- Un montant reste dans le bloc ou tableau où il est visuellement placé.
- Un montant sous un en-tête de taxe reste dans le tableau de taxe.
- Un montant sous un en-tête de total reste dans le tableau de total.
- NET A PAYER, TOTAL A PAYER, SOLDE, AMOUNT DUE ou équivalent doit rester dans son bloc visuel d'origine.
- Ne mélange jamais un tableau de taxes avec un tableau de totaux s'ils ont des en-têtes, bordures, alignements ou espacements distincts.
- Ne place jamais un montant de taxe dans une colonne de total à payer.
- Ne place jamais un total à payer dans une colonne de taxe.
- Ne déplace jamais un montant d'un tableau vers un autre pour compléter une ligne.

RÈGLES IDENTIFIANTS ET CODES

- Pour SIRET, SIREN, TVA intracommunautaire, IBAN, BIC, RIB, numéros de facture, références, commandes, livraison, suivi et codes : conserve exactement les caractères visibles.
- Ne supprime pas d'espace visible.
- N'ajoute pas d'espace non visible.
- Si un code est imprimé sans espace, ne lui ajoute pas d'espace.
- Si un code est imprimé avec espaces, conserve les espaces visibles.
- Si un caractère est ambigu, utilise [ILLISIBLE] pour ce caractère ou segment.
- Ne transforme pas une virgule décimale en point décimal.
- Ne transforme pas un point décimal en virgule décimale.
- Ne modifie pas les espaces dans les montants.

CONTRÔLE FINAL SILENCIEUX AVANT SORTIE

- Tous les textes visibles utiles sont présents.
- Aucun texte visible n'est dupliqué sans duplication visuelle.
- Les signes, parenthèses comptables, devises et séparateurs de montants sont conservés à l'identique.
- Aucun <TAB> n'apparaît hors d'un [[TABLE]].
- Aucun <BR> n'apparaît hors d'un [[TABLE]].
- Chaque [[BLOCK]] est fermé par [[/BLOCK]].
- Chaque [[TABLE]] est fermé par [[/TABLE]].
- Chaque [[BLOCK]] possède id, order, pos et role_hint.
- Chaque [[TABLE]] possède id, order, pos, role_hint et cols=N.
- Chaque [[TABLE ... cols=N]] a exactement N cellules par ligne.
- Chaque ligne de tableau contient exactement N-1 tokens <TAB>.
- Aucun tableau ne contient une seule ligne.
- Aucun tableau ne contient de ligne vide de padding.
- Aucun tableau ne contient deux groupes d'en-têtes indépendants.
- Aucun tableau côte à côte n'a été fusionné.
- Aucune colonne [SANS_ENTETE_n] entièrement vide n'a été créée.
- Aucune colonne réelle sans en-tête n'a été supprimée.
- Aucune colonne réelle sans en-tête n'a reçu un nom inventé.
- Les colonnes de pourcentages sans en-tête sont placées à leur position visuelle exacte.
- Les notes, métadonnées documentaires et pieds de tableau articles ne sont pas dans le tableau line_items.
- Les lignes de continuation d'articles ont été rattachées à l'article précédent quand c'était visuellement justifié.
- Aucun bloc marketing, SAV, tampon, QR code textuel ou slogan n'a été fusionné avec supplier_identity, supplier_address, customer_identity, customer_address ou shipping_address.
"""

# Le Markdown est rendu par Python : aucun prompt Markdown Qwen n'est chargé.


# =====================
# Progress (attendu par le runner)
# =====================

PIPELINE_VERSION = "qwen-vl-ocr-python-markdown-v4"


def _pipeline_fingerprint() -> str:
    """Identifie tous les réglages susceptibles de modifier le contenu d'une page."""
    payload = {
        "pipeline_version": PIPELINE_VERSION,
        "processor_version": PROCESSOR_VERSION,
        "model_ocr": MODEL_OCR,
        "markdown_engine": MODEL_MD,
        "render_dpi": RENDER_DPI,
        "high_resolution_images": QWEN_HIGH_RES_IMAGES,
        "strict_ocr_structure": STRICT_OCR_STRUCTURE,
        "strict_fused_cell_heuristics": STRICT_FUSED_CELL_HEURISTICS,
        "allow_safe_structure_repair": ALLOW_SAFE_STRUCTURE_REPAIR,
        "ocr_prompt_sha256": hashlib.sha256(OCR_PROMPT.encode("utf-8")).hexdigest(),
        "renderer_contract": "exact-atom-roundtrip-v4",
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


PIPELINE_FINGERPRINT = _pipeline_fingerprint()


def _progress_path(pdf_path: str) -> str:
    # Un nouveau processeur ne doit jamais reprendre des pages produites par une
    # ancienne logique OCR/Markdown.
    return str(Path(pdf_path).with_suffix(f".{_PROCESSOR_VERSION_SAFE}.progress.json"))

def load_progress(
    pdf_path: str,
    expected_source_id: Optional[str] = None,
    expected_page_count: Optional[int] = None,
) -> Dict[str, Dict]:
    p = _progress_path(pdf_path)
    if not os.path.exists(p):
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict) or data.get("processor_version") != PROCESSOR_VERSION:
            _log("⚠️ Checkpoint ignoré : version du processeur différente ou absente.")
            return {}

        if "pages" in data and isinstance(data["pages"], dict):
            if data.get("pipeline_fingerprint") != PIPELINE_FINGERPRINT:
                _log(
                    "⚠️ Checkpoint ignoré : il provient d'une autre version du pipeline, "
                    "d'un autre modèle ou d'un autre réglage qualité."
                )
                return {}

            if expected_source_id is not None and data.get("source_id") != expected_source_id:
                _log("⚠️ Checkpoint ignoré : il ne correspond pas à cette version du PDF.")
                return {}

            if expected_page_count is not None:
                saved_page_count = data.get("page_count")
                if saved_page_count is None or int(saved_page_count) != int(expected_page_count):
                    _log("⚠️ Checkpoint ignoré : nombre de pages différent du PDF courant.")
                    return {}

            return data["pages"]

        # Un ancien checkpoint ne contient ni empreinte du pipeline ni garanties
        # suffisantes. Il est volontairement ignoré pour éviter de réutiliser un OCR
        # produit avant les contrôles qualité stricts.
        if isinstance(data, dict):
            _log("⚠️ Ancien checkpoint ignoré : qualité et version non vérifiables.")
        return {}
    except Exception:
        return {}

def save_progress(
    pdf_path: str,
    completed_pages: Dict[str, Dict],
    source_id: Optional[str] = None,
    page_count: Optional[int] = None,
) -> None:
    p = _progress_path(pdf_path)
    tmp = p + ".tmp"
    payload = {
        "version": 3,
        "processor_version": PROCESSOR_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "pipeline_fingerprint": PIPELINE_FINGERPRINT,
        "source_id": source_id,
        "page_count": int(page_count) if page_count is not None else None,
        "pages": completed_pages,
    }
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
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

    page_count: Optional[int] = None

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
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content["text"]
        if "content" in content:
            return _extract_text_from_response_content(content.get("content"))
        return ""

    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, str):
                if part.strip():
                    parts.append(part)
                continue

            if isinstance(part, dict):
                if isinstance(part.get("text"), str) and part["text"].strip():
                    parts.append(part["text"])
                    continue

                nested = part.get("content")
                if nested is not None:
                    nested_text = _extract_text_from_response_content(nested)
                    if nested_text:
                        parts.append(nested_text)

        return "\n\n".join(p for p in parts if p).strip()

    return ""


def _extract_message_texts(message: Dict[str, Any]) -> Tuple[str, str]:
    if not isinstance(message, dict):
        return "", ""

    content_text = _extract_text_from_response_content(message.get("content")).strip()
    reasoning_text = _extract_text_from_response_content(message.get("reasoning_content")).strip()
    return content_text, reasoning_text


def _supports_thinking_toggle(model: str) -> bool:
    m = (model or "").lower()
    return (
        m.startswith("qwen3")
        or m.startswith("qwen-plus")
        or m.startswith("qwen-flash")
        or m.startswith("qwen-turbo")
        or m.startswith("qwen-max")
    )

def _strip_triple_backticks(text: str) -> str:
    """Retire un éventuel fence englobant de longueur 3 ou plus, sans toucher au contenu interne."""
    t = (text or "").strip()
    opening = re.match(r"^(?P<fence>`{3,}|~{3,})(?:[A-Za-z0-9_-]+)?[ \t]*\n", t)
    if not opening:
        return t.strip("\n")
    fence = opening.group("fence")
    closing = re.search(r"\n" + re.escape(fence) + r"[ \t]*$", t)
    if not closing or closing.start() < opening.end():
        return t.strip("\n")
    return t[opening.end():closing.start()].strip("\n")

def _strip_existing_ocr_appendix(md: str) -> str:
    m = re.search(r"^##\s+Annexe\s*-\s*OCR\s+brut\s*$", md, flags=re.IGNORECASE | re.MULTILINE)
    if not m:
        return md.strip()
    return md[:m.start()].rstrip()

def _normalize_sans_entete_tokens(text: str) -> str:
    """
    Normalise les erreurs de forme du modèle : <SANS_ENTETE_1>
    devient [SANS_ENTETE_1]. Applicable OCR + Markdown.
    """
    if not text:
        return text
    return re.sub(r"<SANS_ENTETE_(\d+)>", r"[SANS_ENTETE_\1]", text)


def _is_md_table_row(line: str) -> bool:
    return bool(re.match(r"^\|.*\|\s*$", line or ""))


def _is_md_separator_row(line: str) -> bool:
    return bool(re.match(r"^\|[\s:\-|]+\|\s*$", line or ""))


def _split_md_cells(line: str) -> List[str]:
    """
    Découpe une ligne de tableau Markdown en respectant les pipes échappés.
    """
    raw = (line or "").strip()
    if raw.startswith("|"):
        raw = raw[1:]
    if raw.endswith("|"):
        raw = raw[:-1]

    cells: List[str] = []
    buf: List[str] = []
    escaped = False

    for ch in raw:
        if escaped:
            # On garde l'échappement pour ne pas modifier la valeur visible.
            buf.append("\\" + ch)
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == "|":
            cells.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)

    if escaped:
        buf.append("\\")

    cells.append("".join(buf).strip())
    return cells


def _build_md_row(cells: List[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _build_md_separator(n: int) -> str:
    return "| " + " | ".join(["---"] * max(n, 1)) + " |"



# =====================
# Validation OCR structurée et rendu Markdown Python déterministe
# =====================

OCR_OPEN_TAG_RE = re.compile(r"^\s*\[\[(BLOCK|TABLE)\s+(.+?)\]\]\s*$", re.IGNORECASE)
OCR_CLOSE_TAG_RE = re.compile(r"^\s*\[\[/(BLOCK|TABLE)\]\]\s*$", re.IGNORECASE)
OCR_ANY_TAG_RE = re.compile(r"\[\[(?:/?(?:BLOCK|TABLE)|(?:PDF_)?PAGE)\b", re.IGNORECASE)
_GENERIC_HEADER_RE = re.compile(r"\[SANS_ENTETE_(\d+)\]", re.IGNORECASE)
_ALLOWED_ATTRS = {"id", "order", "pos", "role_hint", "cols", "bbox"}

_ALLOWED_POSITIONS = {
    "top-left", "top", "top-right",
    "middle-left", "middle", "middle-right",
    "bottom-left", "bottom", "bottom-right", "unknown",
}

_ALLOWED_ROLES = {
    "supplier_identity", "supplier_address", "supplier_legal", "supplier_contact", "supplier",
    "customer_identity", "customer_address", "customer_contact", "customer_legal", "customer",
    "billing_address", "shipping_address", "shipping_details", "shipping_contact",
    "delivery_confirmation", "invoice_title", "invoice_details", "line_items",
    "line_items_note", "line_items_footer", "tax_summary", "totals_summary",
    "payment_terms", "bank_details", "payment", "legal_terms", "marketing_badge",
    "logo_text", "logo_marketing", "stamp_signature", "qr_barcode_text", "notes",
    "isolated_value", "unknown",
}

_SECTION_ORDER = [
    "## Informations Émetteur (Fournisseur)",
    "## Informations Client",
    "## Informations de Livraison",
    "## Détails de la Facture",
    "## Tableau des Lignes de Facturation",
    "## Montants Récapitulatifs",
    "## Informations de Paiement",
    "## Mentions Légales et Notes Complémentaires",
]

_SECTION_INDEX = {title: index for index, title in enumerate(_SECTION_ORDER)}

_UNIT_MARKERS = {
    "EUR", "€", "EUROS", "USD", "$", "CHF", "GBP", "£", "CAD", "AUD", "JPY", "¥",
    "HT", "TTC", "%", "U", "UN", "UNIT", "UNITE", "UNITÉ", "PCS", "PC", "PCE",
    "KG", "G", "MG", "L", "ML", "M", "M2", "M²", "M3", "M³", "MTR", "CM", "MM",
    "H", "HEURE", "HEURES", "J", "JOUR", "JOURS",
}

_LINE_ITEM_COMMERCIAL_TERMS = (
    "QTE", "QUANT", "NOMBRE", "NB", "PRIX", "MONTANT", "TOTAL", "TVA", "TAXE", "TAUX",
    "REMISE", "RABAIS", "P.U", "PU ", "UNITAIRE", "UNIT PRICE", "CODE TVA", "CODE TAXE",
    "HT", "TTC", "COUT", "COÛT",
)
_LINE_ITEM_DESCRIPTION_TERMS = ("DESIGN", "DÉSIGN", "DESCRIPTION", "LIBELLE", "LIBELLÉ", "PRESTATION", "ARTICLE", "PRODUIT", "SERVICE")
_LINE_ITEM_REFERENCE_TERMS = ("REFERENCE", "RÉFÉRENCE", "REF", "ARTICLE", "SKU", "CODE ARTICLE")
_LINE_ITEM_SERIAL_TERMS = ("SERIE", "SÉRIE", "SERIAL", "EAN", "GTIN", "LOT")


def _strip_accents(value: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFD", value or "")
        if unicodedata.category(ch) != "Mn"
    )


def _normalize_label(value: str) -> str:
    value = _strip_accents(value or "").upper()
    value = value.replace("<BR>", " ").replace("<EMPTY>", " ")
    return re.sub(r"\s+", " ", value).strip()


_UNIT_MARKERS_NORMALIZED = frozenset(_normalize_label(value) for value in _UNIT_MARKERS)


def _normalize_technical_tokens(text: str) -> str:
    """Normalise uniquement des variantes techniques, jamais le texte comptable."""
    if not text:
        return text
    text = re.sub(r"<SANS_ENTETE_(\d+)>", r"[SANS_ENTETE_\1]", text, flags=re.IGNORECASE)
    return text


def _parse_tag_attributes_strict(raw: str, line_no: int) -> Tuple[Dict[str, str], List[str]]:
    attrs: Dict[str, str] = {}
    errors: List[str] = []
    try:
        tokens = shlex.split(raw or "", posix=True)
    except ValueError as exc:
        return {}, [f"ligne {line_no}: attributs de balise illisibles: {exc}"]

    for token in tokens:
        if "=" not in token:
            errors.append(f"ligne {line_no}: attribut sans '=': {token!r}")
            continue
        key, value = token.split("=", 1)
        key = key.strip().lower()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", key):
            errors.append(f"ligne {line_no}: nom d'attribut invalide {key!r}")
            continue
        if key in attrs:
            errors.append(f"ligne {line_no}: attribut dupliqué {key}")
            continue
        if key not in _ALLOWED_ATTRS:
            errors.append(f"ligne {line_no}: attribut non autorisé {key}")
            continue
        attrs[key] = value
    return attrs, errors


def _parse_bbox(value: str, line_no: int) -> Tuple[Optional[Tuple[int, int, int, int]], List[str]]:
    if not value:
        return None, []
    errors: List[str] = []
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4 or any(not re.fullmatch(r"\d{1,4}", part) for part in parts):
        return None, [f"ligne {line_no}: bbox invalide {value!r}; format attendu x1,y1,x2,y2"]
    coords = tuple(int(part) for part in parts)
    x1, y1, x2, y2 = coords
    if any(coord < 0 or coord > 1000 for coord in coords):
        errors.append(f"ligne {line_no}: bbox hors intervalle 0-1000: {value!r}")
    if x1 >= x2 or y1 >= y2:
        errors.append(f"ligne {line_no}: bbox non croissante: {value!r}")
    return coords, errors


def _parse_structured_ocr(text: str) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    raw = (text or "").strip()
    if raw == "[PAGE VIDE]":
        return [], [], []

    items: List[Dict[str, Any]] = []
    errors: List[str] = []
    warnings: List[str] = []
    active: Optional[Dict[str, Any]] = None

    for line_no, raw_line in enumerate(raw.splitlines(), start=1):
        line = raw_line.rstrip("\r")
        open_match = OCR_OPEN_TAG_RE.match(line)
        close_match = OCR_CLOSE_TAG_RE.match(line)

        if open_match:
            if active is not None:
                errors.append(
                    f"ligne {line_no}: ouverture {open_match.group(1).upper()} avant fermeture de {active['kind']}"
                )
                continue
            kind = open_match.group(1).upper()
            attrs, attr_errors = _parse_tag_attributes_strict(open_match.group(2), line_no)
            errors.extend(attr_errors)
            active = {
                "kind": kind,
                "attrs": attrs,
                "lines": [],
                "start_line": line_no,
            }
            continue

        if close_match:
            close_kind = close_match.group(1).upper()
            if active is None:
                errors.append(f"ligne {line_no}: fermeture [[/{close_kind}]] sans ouverture")
                continue
            if close_kind != active["kind"]:
                errors.append(
                    f"ligne {line_no}: fermeture [[/{close_kind}]] alors que {active['kind']} est ouvert"
                )
                # Ne pas rattacher la suite au mauvais élément.
                active = None
                continue
            active["end_line"] = line_no
            items.append(active)
            active = None
            continue

        if active is None:
            if line.strip():
                errors.append(f"ligne {line_no}: texte hors [[BLOCK]]/[[TABLE]]: {line.strip()[:100]}")
            continue

        active["lines"].append(line.rstrip())

    if active is not None:
        errors.append(f"ligne {active['start_line']}: {active['kind']} non fermé en fin de réponse")

    return items, errors, warnings


def _looks_like_fused_vat_cell(value: str) -> bool:
    v = (value or "").strip()
    if not v or "%" in v or re.search(r"[A-Za-z€$£¥]", v):
        return False
    # Exemples suspects : 12,00 2 ; 55.80 0 ; (12,00) 2.
    return bool(re.fullmatch(r"[-+−]?\(?\d[\d\s.,']*\)?\s+\d{1,2}", v))


def _quantity_fusion_signature(value: str) -> Optional[Tuple[str, str]]:
    v = (value or "").strip()
    match = re.fullmatch(
        r"([-+−]?\d{1,5}(?:[.,]\d+)?)\s+([-+−]?\d+(?:[.,]\d+)?)(?:\s+[A-Za-zÀ-ÿ²³]{1,10})?",
        v,
    )
    if not match:
        return None
    return match.group(1), match.group(2)


def _row_header_score(row: List[str]) -> int:
    joined = " | ".join(_normalize_label(cell) for cell in row)
    terms = (
        "REFERENCE", "DESIGNATION", "DESCRIPTION", "QTE", "QUANTITE", "PRIX", "MONTANT",
        "TVA", "TAXE", "DATE", "FACTURE", "CLIENT", "ECHEANCE", "CODE", "LIBELLE",
        "IBAN", "BIC", "MODE DE REGLEMENT", "TOTAL", "TAUX", "BASE",
    )
    return sum(1 for term in terms if term in joined)


def _table_has_header(item: Dict[str, Any], rows: List[List[str]]) -> bool:
    if not rows:
        return False
    role = (item.get("attrs", {}).get("role_hint") or "unknown").lower()
    first = rows[0]
    if any(_GENERIC_HEADER_RE.fullmatch((cell or "").strip()) for cell in first):
        return True
    if role in {"line_items", "invoice_details", "tax_summary"}:
        return True
    score = _row_header_score(first)
    first_norm = " | ".join(_normalize_label(cell) for cell in first)
    looks_financial_data = any(
        token in first_norm
        for token in ("TOTAL HT", "TOTAL TTC", "NET A PAYER", "SOLDE A PAYER", "SOUS-TOTAL", "ACOMPTE")
    ) and any(re.search(r"\d", cell or "") for cell in first)
    if role == "totals_summary":
        return score >= 2 and not looks_financial_data
    return score >= 2 and not looks_financial_data


def _is_unit_only_row(row: List[str]) -> bool:
    nonempty = []
    for cell in row:
        value = (cell or "").strip()
        if value in {"", "<EMPTY>"}:
            continue
        segments = [segment.strip() for segment in value.split("<BR>") if segment.strip()]
        nonempty.extend(segments)
    if not nonempty:
        return False
    return all(_normalize_label(value) in _UNIT_MARKERS_NORMALIZED for value in nonempty)


def _header_indexes(header: List[str], terms: Tuple[str, ...], *, exclude: Tuple[str, ...] = ()) -> List[int]:
    indexes: List[int] = []
    for idx, cell in enumerate(header):
        normalized = _normalize_label(cell)
        if any(term in normalized for term in terms) and not any(term in normalized for term in exclude):
            indexes.append(idx)
    return indexes


def _line_item_indexes(header: List[str]) -> Dict[str, List[int]]:
    return {
        "commercial": _header_indexes(header, _LINE_ITEM_COMMERCIAL_TERMS),
        "description": _header_indexes(header, _LINE_ITEM_DESCRIPTION_TERMS),
        "reference": _header_indexes(header, _LINE_ITEM_REFERENCE_TERMS, exclude=("TVA", "TAXE")),
        "serial": _header_indexes(header, _LINE_ITEM_SERIAL_TERMS),
    }


def _is_long_identifier(value: str) -> bool:
    compact = re.sub(r"[\s-]", "", (value or "").strip())
    return bool(re.fullmatch(r"\d{8,18}", compact))


def _line_item_continuation_action(
    row: List[str],
    indexes: Dict[str, List[int]],
    has_previous: bool,
) -> Tuple[str, List[int]]:
    """Renvoie ('merge'|'data'|'error', indexes à fusionner)."""
    nonempty = [
        idx for idx, cell in enumerate(row)
        if (cell or "").strip() not in {"", "<EMPTY>"}
    ]
    if not nonempty:
        return "error", []

    commercial = set(indexes["commercial"])
    if any(idx in commercial for idx in nonempty):
        return "data", []

    if not has_previous:
        return "error", []

    description = set(indexes["description"])
    serial = set(indexes["serial"])
    reference = set(indexes["reference"])
    allowed = description | serial

    # Une continuation purement descriptive ou de numéro de série est sûre.
    if set(nonempty).issubset(allowed) and nonempty:
        return "merge", nonempty

    # Un identifiant long seul dans la colonne référence est un EAN/GTIN probable.
    if len(nonempty) == 1 and nonempty[0] in reference and _is_long_identifier(row[nonempty[0]]):
        return "merge", nonempty

    return "error", []


def _validate_line_item_continuations(item: Dict[str, Any], rows: List[List[str]]) -> List[str]:
    if len(rows) < 2:
        return []
    table_id = item.get("attrs", {}).get("id", "?")
    header = rows[0]
    indexes = _line_item_indexes(header)
    if not indexes["commercial"]:
        # Certaines factures présentent une liste descriptive et portent les montants
        # uniquement dans un récapitulatif séparé. On conserve alors chaque ligne telle quelle.
        return []
    if not (indexes["description"] or indexes["reference"]):
        return [f"table {table_id}: aucune colonne référence/désignation identifiable dans line_items"]

    errors: List[str] = []
    has_previous = False
    for row_index, row in enumerate(rows[1:], start=2):
        action, _ = _line_item_continuation_action(row, indexes, has_previous)
        if action == "data":
            has_previous = True
        elif action == "merge":
            continue
        else:
            preview = " | ".join(cell for cell in row if cell and cell != "<EMPTY>")[:160]
            errors.append(
                f"table {table_id} ligne logique {row_index}: ligne sans valeur commerciale ambiguë; "
                f"Qwen devait la placer dans un BLOCK ou la rattacher explicitement avec <BR>: {preview!r}"
            )
    return errors


def _validate_semantic_table_risks(item: Dict[str, Any], rows: List[List[str]]) -> List[str]:
    errors: List[str] = []
    role = (item.get("attrs", {}).get("role_hint") or "unknown").lower()
    table_id = item.get("attrs", {}).get("id", "?")
    flat = " | ".join(cell for row in rows for cell in row)
    flat_norm = _normalize_label(flat)

    if role == "tax_summary":
        tax_markers = any(token in flat_norm for token in ("CODES TVA", "CODE TVA", "EXONERE", "SOUMIS", "TAUX TVA", "BASE TVA"))
        total_markers = any(token in flat_norm for token in ("NET A PAYER", "TOTAL TTC", "SOLDE A PAYER", "AMOUNT DUE"))
        if tax_markers and total_markers:
            errors.append(f"table {table_id}: probable fusion du tableau TVA avec le tableau des totaux")
        elif tax_markers and "TOTAL HT" in flat_norm and rows and len(rows[0]) >= 7:
            errors.append(f"table {table_id}: TOTAL HT semble fusionné horizontalement au tableau TVA")

    if role == "totals_summary":
        if "CODES TVA" in flat_norm or ("EXONERE" in flat_norm and "SOUMIS" in flat_norm):
            errors.append(f"table {table_id}: probable fusion du tableau des totaux avec le tableau TVA")

    if role != "line_items" or len(rows) < 2:
        return errors

    header = [_normalize_label(cell) for cell in rows[0]]
    data_rows = rows[1:]
    quantity_indexes = [
        idx for idx, name in enumerate(header)
        if any(token in name for token in ("QTE", "QUANTITE", "QUANTITY", "NOMBRE"))
    ]
    vat_indexes = [
        idx for idx, name in enumerate(header)
        if name in {"TVA", "T", "TX", "TAX", "TAXE", "CODE TVA", "CODE TAXE"}
        or "CODE TVA" in name or "CODE TAXE" in name
    ]

    if STRICT_FUSED_CELL_HEURISTICS:
        for idx in vat_indexes:
            suspicious = [row[idx] for row in data_rows if idx < len(row) and _looks_like_fused_vat_cell(row[idx])]
            if suspicious:
                preview = ", ".join(repr(value) for value in suspicious[:4])
                errors.append(
                    f"table {table_id}: montant et code TVA probablement fusionnés dans la colonne {idx + 1}: {preview}"
                )

        for idx in quantity_indexes:
            signatures: List[Tuple[str, str, str]] = []
            for row in data_rows:
                if idx >= len(row):
                    continue
                signature = _quantity_fusion_signature(row[idx])
                if signature:
                    signatures.append((signature[0], signature[1], row[idx]))
            if len(signatures) >= 2:
                first_values = [value[0] for value in signatures]
                repeated_prefix = len(set(first_values)) < len(first_values)
                second_has_decimal_or_unit_pattern = any(
                    re.search(r"[,.]", value[1]) or len(re.sub(r"\D", "", value[1])) <= 3
                    for value in signatures
                )
                if repeated_prefix or second_has_decimal_or_unit_pattern:
                    preview = ", ".join(repr(value[2]) for value in signatures[:5])
                    errors.append(
                        f"table {table_id}: plusieurs quantités semblent contenir deux colonnes fusionnées: {preview}"
                    )

    # Un second groupe d'en-têtes au milieu d'une table est un signe de fusion de zones.
    for row_index, row in enumerate(data_rows, start=2):
        if _row_header_score(row) >= 3 and not any(re.search(r"\d", cell or "") for cell in row):
            errors.append(f"table {table_id} ligne logique {row_index}: second groupe d'en-têtes probable")
            break

    errors.extend(_validate_line_item_continuations(item, rows))
    return errors


def _validate_table_columns(item: Dict[str, Any], rows: List[List[str]]) -> List[str]:
    errors: List[str] = []
    role = (item.get("attrs", {}).get("role_hint") or "unknown").lower()
    table_id = item.get("attrs", {}).get("id", "?")
    if not rows:
        return errors
    has_header = _table_has_header(item, rows)
    if not has_header:
        return errors

    header = rows[0]
    data_rows = rows[1:]
    for idx, header_cell in enumerate(header):
        h = (header_cell or "").strip()
        unnamed = h in {"", "<EMPTY>"} or bool(_GENERIC_HEADER_RE.fullmatch(h))
        if not unnamed:
            continue
        values = [
            (row[idx] or "").strip() if idx < len(row) else ""
            for row in data_rows
        ]
        if values and all(value in {"", "<EMPTY>"} for value in values):
            errors.append(
                f"table {table_id}: colonne {idx + 1} sans en-tête et entièrement vide; "
                "elle ne devait pas être créée"
            )

    if role == "line_items":
        normalized = " | ".join(_normalize_label(cell) for cell in header)
        if not any(term in normalized for term in ("DESIGN", "DESCRIPTION", "LIBELLE", "REFERENCE", "ARTICLE")):
            errors.append(f"table {table_id}: en-tête line_items sans référence ni désignation")
    return errors


def validate_structured_ocr(text: str) -> Dict[str, Any]:
    """Valide la sortie OCR avant tout rendu. Aucune valeur visible n'est réparée ici."""
    raw = _normalize_technical_tokens((text or "").strip())
    errors: List[str] = []
    warnings: List[str] = []

    if not raw:
        return {"ok": False, "errors": ["réponse OCR vide"], "warnings": [], "items": [], "page_empty": False}

    if raw == "[PAGE VIDE]":
        return {"ok": True, "errors": [], "warnings": [], "items": [], "page_empty": True}

    if "[PAGE VIDE]" in raw:
        errors.append("[PAGE VIDE] ne peut pas être mélangé à d'autres contenus")
    if len(raw) < OCR_MIN_CHARS:
        errors.append(f"OCR anormalement court: {len(raw)} caractères")
    if "\x00" in raw:
        errors.append("caractère NUL interdit dans l'OCR")
    if "\t" in raw:
        errors.append("tabulation littérale interdite; Qwen doit utiliser <TAB>")
    if re.search(r"^\s*\[\[(?:PDF_)?PAGE\b", raw, flags=re.IGNORECASE | re.MULTILINE):
        errors.append("token de pagination technique interdit dans la sortie OCR")

    items, parse_errors, parse_warnings = _parse_structured_ocr(raw)
    errors.extend(parse_errors)
    warnings.extend(parse_warnings)

    seen_ids: set[Tuple[str, str]] = set()
    seen_orders: set[int] = set()
    appearance_orders: List[int] = []

    for item in items:
        kind = item["kind"]
        attrs = item.get("attrs", {})
        start_line = int(item.get("start_line", 0) or 0)
        item_id = attrs.get("id", "")
        order_raw = attrs.get("order", "")
        pos = attrs.get("pos", "")
        role = attrs.get("role_hint", "").lower()
        attrs["role_hint"] = role

        required = ["id", "order", "pos", "role_hint"] + (["cols"] if kind == "TABLE" else [])
        missing = [name for name in required if not attrs.get(name)]
        if missing:
            errors.append(f"ligne {start_line}: attribut(s) manquant(s) {', '.join(missing)}")

        if item_id:
            key = (kind, item_id.upper())
            if key in seen_ids:
                errors.append(f"ligne {start_line}: identifiant dupliqué {kind} {item_id}")
            seen_ids.add(key)
            expected_prefix = "B" if kind == "BLOCK" else "T"
            if not re.fullmatch(expected_prefix + r"\d+", item_id, flags=re.IGNORECASE):
                errors.append(f"ligne {start_line}: identifiant invalide {item_id!r} pour {kind}")
            elif not re.fullmatch(expected_prefix + r"\d{3}", item_id, flags=re.IGNORECASE):
                warnings.append(f"ligne {start_line}: identifiant non canonique {item_id!r}; format conseillé {expected_prefix}001")

        try:
            order = int(order_raw)
            if order <= 0:
                raise ValueError
            item["order_int"] = order
            appearance_orders.append(order)
            if order in seen_orders:
                errors.append(f"ligne {start_line}: order dupliqué {order_raw}")
            seen_orders.add(order)
            if not re.fullmatch(r"\d{3}", order_raw):
                warnings.append(f"ligne {start_line}: order non canonique {order_raw!r}; format conseillé 001")
        except (TypeError, ValueError):
            errors.append(f"ligne {start_line}: order invalide {order_raw!r}")
            item["order_int"] = 10**9

        if pos and pos not in _ALLOWED_POSITIONS:
            errors.append(f"ligne {start_line}: pos non autorisé {pos!r}")
        if role and role not in _ALLOWED_ROLES:
            errors.append(f"ligne {start_line}: role_hint non autorisé {role!r}")

        bbox, bbox_errors = _parse_bbox(attrs.get("bbox", ""), start_line)
        errors.extend(bbox_errors)
        item["bbox"] = bbox

        content_lines = item.get("lines", [])
        if not any(line.strip() for line in content_lines):
            errors.append(f"ligne {start_line}: {kind} vide")
            continue

        if kind == "BLOCK":
            for offset, line in enumerate(content_lines, start=1):
                absolute_line = start_line + offset
                if "<TAB>" in line:
                    errors.append(f"ligne {absolute_line}: <TAB> interdit dans un BLOCK")
                if "<BR>" in line:
                    errors.append(f"ligne {absolute_line}: <BR> interdit dans un BLOCK")
                if "<EMPTY>" in line:
                    errors.append(f"ligne {absolute_line}: <EMPTY> interdit dans un BLOCK")
                if OCR_ANY_TAG_RE.search(line):
                    errors.append(f"ligne {absolute_line}: balise technique imbriquée dans un BLOCK")
            continue

        try:
            declared_cols = int(attrs.get("cols", "0"))
        except (TypeError, ValueError):
            declared_cols = 0
        if declared_cols <= 0:
            errors.append(f"ligne {start_line}: cols invalide {attrs.get('cols')!r}")
            continue
        item["declared_cols"] = declared_cols

        rows: List[List[str]] = []
        for offset, line in enumerate(content_lines, start=1):
            absolute_line = start_line + offset
            if not line.strip():
                errors.append(f"ligne {absolute_line}: ligne vide interdite dans TABLE {item_id}")
                continue
            if "\t" in line:
                errors.append(f"ligne {absolute_line}: tabulation littérale interdite dans TABLE {item_id}")
            cells = line.split("<TAB>")
            rows.append(cells)
            if len(cells) != declared_cols:
                errors.append(
                    f"ligne {absolute_line}: TABLE {item_id} déclare {declared_cols} colonnes "
                    f"mais la ligne en contient {len(cells)}"
                )
            if all((cell or "").strip() in {"", "<EMPTY>"} for cell in cells):
                errors.append(f"ligne {absolute_line}: ligne de tableau entièrement vide")
            for cell_index, cell in enumerate(cells, start=1):
                stripped = (cell or "").strip()
                if OCR_ANY_TAG_RE.search(cell):
                    errors.append(f"ligne {absolute_line}: balise technique imbriquée dans TABLE {item_id}")
                if "<EMPTY>" in cell and stripped != "<EMPTY>":
                    errors.append(
                        f"ligne {absolute_line}, cellule {cell_index}: <EMPTY> doit occuper toute la cellule"
                    )
                if stripped.startswith("<BR>") or stripped.endswith("<BR>") or "<BR><BR>" in stripped:
                    errors.append(
                        f"ligne {absolute_line}, cellule {cell_index}: usage invalide de <BR>"
                    )

        if len(rows) < 2:
            errors.append(f"TABLE {item_id} contient moins de deux lignes; Qwen devait utiliser un BLOCK")
        if rows and all(len(row) == declared_cols for row in rows):
            errors.extend(_validate_table_columns(item, rows))
            errors.extend(_validate_semantic_table_risks(item, rows))
        item["rows"] = rows
        item["has_header"] = _table_has_header(item, rows) if rows else False

    if appearance_orders and appearance_orders != sorted(appearance_orders):
        errors.append("les valeurs order ne suivent pas l'ordre d'apparition")
    if appearance_orders:
        expected_sequence = list(range(min(appearance_orders), min(appearance_orders) + len(appearance_orders)))
        if appearance_orders != expected_sequence:
            warnings.append("les valeurs order contiennent des sauts; cela n'altère pas le rendu mais n'est pas canonique")

    if STRICT_OCR_STRUCTURE and not items:
        errors.append("aucun [[BLOCK]] ou [[TABLE]] détecté")

    errors = list(dict.fromkeys(errors))[:50]
    warnings = list(dict.fromkeys(warnings))[:50]
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "items": items,
        "page_empty": False,
    }


def _repair_unambiguous_ocr_structure(text: str) -> Tuple[str, List[str]]:
    """
    Répare uniquement des métadonnées techniques, jamais une valeur visible.
    Désactivé par défaut. Une table dont les lignes n'ont pas toutes la même largeur
    n'est jamais réparée.
    """
    lines = (text or "").splitlines()
    repairs: List[str] = []
    output = list(lines)

    active_kind: Optional[str] = None
    for index, line in enumerate(output):
        open_match = OCR_OPEN_TAG_RE.match(line)
        close_match = OCR_CLOSE_TAG_RE.match(line)
        if open_match and active_kind is None:
            active_kind = open_match.group(1).upper()
        elif close_match and active_kind is not None:
            close_kind = close_match.group(1).upper()
            if close_kind != active_kind:
                output[index] = f"[[/{active_kind}]]"
                repairs.append(
                    f"ligne {index + 1}: fermeture [[/{close_kind}]] remplacée par [[/{active_kind}]]"
                )
            active_kind = None

    index = 0
    while index < len(output):
        match = OCR_OPEN_TAG_RE.match(output[index])
        if not match or match.group(1).upper() != "TABLE":
            index += 1
            continue
        attrs, _ = _parse_tag_attributes_strict(match.group(2), index + 1)
        end = index + 1
        widths: List[int] = []
        while end < len(output):
            close = OCR_CLOSE_TAG_RE.match(output[end])
            if close and close.group(1).upper() == "TABLE":
                break
            if output[end].strip():
                widths.append(output[end].count("<TAB>") + 1)
            end += 1
        if end < len(output) and widths and len(set(widths)) == 1:
            actual = widths[0]
            try:
                declared = int(attrs.get("cols", "0"))
            except ValueError:
                declared = 0
            if declared > 0 and declared != actual:
                output[index] = re.sub(r"\bcols=\d+\b", f"cols={actual}", output[index], count=1, flags=re.IGNORECASE)
                repairs.append(
                    f"table ligne {index + 1}: cols={declared} remplacé par cols={actual}; toutes les lignes concordent"
                )
        index = max(index + 1, end + 1)
    return "\n".join(output).strip(), repairs


def _financial_text(value: str) -> bool:
    norm = _normalize_label(value)
    return any(
        token in norm
        for token in (
            "TOTAL", "NET A PAYER", "SOLDE", "MONTANT", "TVA", "TAXE", "REMISE",
            "ACOMPTE", "ECO-CONTRIBUTION", "SOUS-TOTAL", "AMOUNT DUE", "BALANCE DUE",
        )
    )


def _supplier_context(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    supplier_text = " ".join(
        " ".join(item.get("lines", []))
        for item in items
        if (item.get("attrs", {}).get("role_hint") or "").lower()
        in {"supplier_identity", "supplier_address", "supplier_legal", "supplier_contact", "supplier"}
    )
    tokens = {
        token for token in re.findall(r"[A-ZÀ-Ÿ0-9]{3,}", _normalize_label(supplier_text))
        if token not in {"SAS", "SARL", "EURL", "SA", "SIRET", "SIREN", "RCS", "TVA", "TEL", "FAX", "SITE"}
    }
    return {"tokens": tokens}


def _section_for_ocr_item(item: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> str:
    attrs = item.get("attrs", {})
    role = (attrs.get("role_hint") or "unknown").lower()
    text = "\n".join(item.get("lines", []))
    norm = _normalize_label(text)
    context = context or {}

    if role in {"supplier_identity", "supplier_address", "supplier_legal", "supplier_contact", "supplier"}:
        return _SECTION_ORDER[0]
    if role == "logo_text":
        logo_tokens = set(re.findall(r"[A-ZÀ-Ÿ0-9]{3,}", norm))
        shares_supplier_name = bool(logo_tokens & set(context.get("tokens", set())))
        return _SECTION_ORDER[0] if shares_supplier_name else _SECTION_ORDER[7]
    if role in {"customer_identity", "customer_address", "customer_contact", "customer_legal", "billing_address", "customer"}:
        return _SECTION_ORDER[1]
    if role in {"shipping_address", "shipping_details", "shipping_contact", "delivery_confirmation"}:
        return _SECTION_ORDER[2]
    if role in {"invoice_title", "invoice_details"}:
        return _SECTION_ORDER[3]
    if role in {"line_items_note", "line_items", "line_items_footer"}:
        return _SECTION_ORDER[4]
    if role in {"tax_summary", "totals_summary"}:
        return _SECTION_ORDER[5]
    if role in {"payment_terms", "bank_details", "payment"}:
        return _SECTION_ORDER[6]
    if role in {"legal_terms", "marketing_badge", "logo_marketing", "qr_barcode_text", "notes"}:
        return _SECTION_ORDER[7]
    if role == "stamp_signature":
        if any(token in norm for token in ("LIVRE", "LIVRÉ", "DELIVERED", "RECU", "REÇU", "NOM DATE ET SIGNATURE")):
            return _SECTION_ORDER[2]
        return _SECTION_ORDER[7]
    if role == "isolated_value":
        if _financial_text(text):
            return _SECTION_ORDER[5]
        if re.fullmatch(r"\s*\d{1,2}:\d{2}(?::\d{0,2})?\s*", text or ""):
            return _SECTION_ORDER[3]
        return _SECTION_ORDER[7]

    # unknown : classement conservateur, fondé uniquement sur des marqueurs explicites.
    if any(token in norm for token in ("ADRESSE DE LIVRAISON", "LIVRE A", "LIVRÉ À", "SHIP TO", "TRANSPORTEUR", "EXPEDITION", "EXPÉDITION", "INCOTERM")):
        return _SECTION_ORDER[2]
    if any(token in norm for token in ("IBAN", "BIC", "RIB", "MODE DE REGLEMENT", "MODE DE RÈGLEMENT", "ECHEANCE", "ÉCHÉANCE", "VIREMENT", "PRELEVEMENT", "PRÉLÈVEMENT")):
        return _SECTION_ORDER[6]
    if _financial_text(text):
        return _SECTION_ORDER[5]
    if any(token in norm for token in ("FACTURE", "AVOIR", "PROFORMA", "DATE", "N CLIENT", "REFERENCE CLIENT", "RÉFÉRENCE CLIENT", "COMMANDE")):
        return _SECTION_ORDER[3]
    return _SECTION_ORDER[7]


def _escape_table_segment(value: str) -> str:
    escaped = html.escape(value, quote=False)
    escaped = escaped.replace("\\", "\\\\")
    escaped = escaped.replace("|", r"\|")
    return escaped


def _render_ocr_cell(value: str) -> str:
    raw = (value or "").strip()
    if raw in {"", "<EMPTY>"}:
        return ""
    parts = raw.split("<BR>")
    return "<br>".join(_escape_table_segment(part) for part in parts)


def _escape_block_line(value: str) -> str:
    original = (value or "").rstrip()
    escaped = html.escape(original, quote=False)
    # Évite les structures Markdown accidentelles sans modifier le texte visible rendu.
    leading_spaces = len(original) - len(original.lstrip(" "))
    stripped = original.lstrip(" ")
    prefix = original[:leading_spaces]
    body = escaped[leading_spaces:]

    if leading_spaces >= 4 and prefix:
        prefix = "&#32;" + prefix[1:]
    if re.match(r"^(?:#{1,6}\s|[-+*]\s|\d+[.)]\s|```|~~~|\|)", stripped):
        first = stripped[0]
        entity = {
            "#": "&#35;", "-": "&#45;", "+": "&#43;", "*": "&#42;",
            "`": "&#96;", "~": "&#126;", "|": "&#124;",
        }.get(first)
        if entity:
            body = entity + body[1:]
        elif first.isdigit():
            # Encode le séparateur du numéro de liste, pas les chiffres.
            body = re.sub(r"^(\d+)([.)])", lambda m: m.group(1) + ("&#46;" if m.group(2) == "." else "&#41;"), body, count=1)
    if re.fullmatch(r"\s*(?:-{3,}|\*{3,}|_{3,}|={3,})\s*", original):
        char = stripped[0] if stripped else "-"
        entity = {"-": "&#45;", "*": "&#42;", "_": "&#95;", "=": "&#61;"}[char]
        body = entity + body[1:]
    return prefix + body


def _prepare_table(item: Dict[str, Any]) -> Tuple[List[str], List[List[str]], Dict[str, Any]]:
    rows = [list(row) for row in item.get("rows", [])]
    if not rows:
        rows = [line.split("<TAB>") for line in item.get("lines", []) if line.strip()]
    if not rows:
        return [], [], {"unit_row_merged": False, "continuations_merged": 0}

    has_header = bool(item.get("has_header", _table_has_header(item, rows)))
    if has_header:
        header = list(rows[0])
        data = [list(row) for row in rows[1:]]
    else:
        header = [f"[SANS_ENTETE_{index + 1}]" for index in range(len(rows[0]))]
        data = [list(row) for row in rows]

    unit_row_merged = False
    if data and _is_unit_only_row(data[0]):
        unit_row = data.pop(0)
        for index, unit in enumerate(unit_row):
            unit_value = (unit or "").strip()
            if unit_value in {"", "<EMPTY>"}:
                continue
            header_value = (header[index] or "").strip()
            header[index] = unit_value if header_value in {"", "<EMPTY>"} else header_value + "<BR>" + unit_value
        unit_row_merged = True

    continuations_merged = 0
    role = (item.get("attrs", {}).get("role_hint") or "unknown").lower()
    if role == "line_items":
        indexes = _line_item_indexes(header)
        if indexes["commercial"]:
            merged_rows: List[List[str]] = []
            for row in data:
                action, merge_indexes = _line_item_continuation_action(row, indexes, bool(merged_rows))
                if action == "merge":
                    previous = merged_rows[-1]
                    for index in merge_indexes:
                        value = (row[index] or "").strip()
                        prior = (previous[index] or "").strip()
                        previous[index] = value if prior in {"", "<EMPTY>"} else prior + "<BR>" + value
                    continuations_merged += 1
                else:
                    merged_rows.append(list(row))
            data = merged_rows

    unnamed_counter = 1
    for index, cell in enumerate(header):
        if (cell or "").strip() in {"", "<EMPTY>"}:
            header[index] = f"[SANS_ENTETE_{unnamed_counter}]"
            unnamed_counter += 1

    return header, data, {
        "unit_row_merged": unit_row_merged,
        "continuations_merged": continuations_merged,
    }


def _render_ocr_table(item: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    header_raw, data_raw, metadata = _prepare_table(item)
    if not header_raw:
        return "", metadata
    header = [_render_ocr_cell(cell) for cell in header_raw]
    output = [_build_md_row(header), _build_md_separator(len(header))]
    for row in data_raw:
        rendered = [_render_ocr_cell(cell) for cell in row]
        if all(not cell.strip() for cell in rendered):
            continue
        output.append(_build_md_row(rendered))
    metadata = {
        **metadata,
        "cols": len(header),
        "data_rows": len(output) - 2,
        "table_id": item.get("attrs", {}).get("id", "?"),
    }
    return "\n".join(output), metadata


def _render_markdown_from_items(items: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    context = _supplier_context(items)
    buckets: Dict[str, List[Tuple[int, int, str]]] = {section: [] for section in _SECTION_ORDER}
    table_signatures: List[Dict[str, Any]] = []

    for item in sorted(items, key=lambda value: (value.get("order_int", 10**9), value.get("start_line", 10**9))):
        section = _section_for_ocr_item(item, context)
        if item["kind"] == "TABLE":
            rendered, metadata = _render_ocr_table(item)
            if rendered:
                metadata["section"] = section
                metadata["order"] = item.get("order_int", 10**9)
                table_signatures.append(metadata)
        else:
            rendered = "\n".join(_escape_block_line(line) for line in item.get("lines", [])).strip()
        if rendered:
            buckets[section].append((item.get("order_int", 10**9), item.get("start_line", 10**9), rendered))

    parts: List[str] = []
    for section in _SECTION_ORDER:
        entries = sorted(buckets[section], key=lambda value: (value[0], value[1]))
        if not entries:
            continue
        parts.append(section + "\n\n" + "\n\n".join(value[2] for value in entries))
    return "\n\n".join(parts).strip(), {"tables": table_signatures}


def render_markdown_deterministic(ocr_text: str, page_num: int) -> str:
    validation = validate_structured_ocr(ocr_text)
    if not validation["ok"]:
        raise RuntimeError(
            f"Page {page_num}: OCR invalide avant rendu Markdown: "
            + "; ".join(validation["errors"][:10])
        )
    if validation.get("page_empty"):
        return "[PAGE VIDE]"
    markdown, manifest = _render_markdown_from_items(validation["items"])
    result = _validate_markdown_core(markdown, validation["items"], manifest=manifest)
    if not result["ok"]:
        raise RuntimeError(
            f"Page {page_num}: échec interne du rendu Markdown Python: "
            + "; ".join(result["errors"][:10])
        )
    return markdown


def _decode_table_cell(value: str) -> str:
    raw = value or ""
    output: List[str] = []
    index = 0
    while index < len(raw):
        if raw[index] == "\\" and index + 1 < len(raw) and raw[index + 1] in {"\\", "|"}:
            output.append(raw[index + 1])
            index += 2
            continue
        output.append(raw[index])
        index += 1
    return html.unescape("".join(output))


def _source_atoms(items: List[Dict[str, Any]]) -> Counter[Tuple[str, str]]:
    context = _supplier_context(items)
    atoms: Counter[Tuple[str, str]] = Counter()
    for item in items:
        section = _section_for_ocr_item(item, context)
        if item.get("kind") == "TABLE":
            rows = item.get("rows", [])
            for row in rows:
                for cell in row:
                    value = (cell or "").strip()
                    if value in {"", "<EMPTY>"} or _GENERIC_HEADER_RE.fullmatch(value):
                        continue
                    for part in value.split("<BR>"):
                        part = part.strip()
                        if part:
                            atoms[(section, part)] += 1
        else:
            for line in item.get("lines", []):
                candidate = (line or "").strip()
                if candidate:
                    atoms[(section, candidate)] += 1
    return atoms


def _parse_markdown_core(core: str) -> Tuple[Counter[Tuple[str, str]], List[Dict[str, Any]], List[str]]:
    atoms: Counter[Tuple[str, str]] = Counter()
    tables: List[Dict[str, Any]] = []
    errors: List[str] = []
    lines = (core or "").splitlines()
    current_section: Optional[str] = None
    index = 0

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if stripped.startswith("## "):
            if stripped not in _SECTION_INDEX:
                errors.append(f"ligne Markdown {index + 1}: section inattendue {stripped!r}")
                current_section = None
            else:
                current_section = stripped
            index += 1
            continue
        if not stripped:
            index += 1
            continue
        if current_section is None:
            errors.append(f"ligne Markdown {index + 1}: contenu hors section")
            index += 1
            continue

        if _is_md_table_row(line):
            if index + 1 >= len(lines) or not _is_md_separator_row(lines[index + 1]):
                errors.append(f"ligne Markdown {index + 1}: table sans séparateur")
                index += 1
                continue
            header = _split_md_cells(line)
            separator = _split_md_cells(lines[index + 1])
            if len(separator) != len(header):
                errors.append(
                    f"ligne Markdown {index + 2}: séparateur à {len(separator)} cellules; attendu {len(header)}"
                )
            rows: List[List[str]] = []
            cursor = index + 2
            while cursor < len(lines) and _is_md_table_row(lines[cursor]):
                row = _split_md_cells(lines[cursor])
                if len(row) != len(header):
                    errors.append(
                        f"ligne Markdown {cursor + 1}: {len(row)} cellules; attendu {len(header)}"
                    )
                rows.append(row)
                cursor += 1
            if not rows:
                errors.append(f"ligne Markdown {index + 1}: tableau sans ligne de données")
            tables.append({"section": current_section, "cols": len(header), "data_rows": len(rows)})
            for row in [header] + rows:
                for cell in row:
                    raw_cell = cell.strip()
                    if not raw_cell:
                        continue
                    # Le <br> généré par le code sépare des atomes OCR ; un <br> littéral
                    # provenant du document aurait été encodé en &lt;br&gt;.
                    for part in re.split(r"<br\s*/?>", raw_cell, flags=re.IGNORECASE):
                        decoded = _decode_table_cell(part).strip()
                        if not decoded or _GENERIC_HEADER_RE.fullmatch(decoded):
                            continue
                        atoms[(current_section, decoded)] += 1
            index = cursor
            continue

        decoded_line = html.unescape(line).strip()
        if decoded_line:
            atoms[(current_section, decoded_line)] += 1
        index += 1

    return atoms, tables, errors


def _validate_markdown_core(
    md: str,
    ocr_items: Optional[List[Dict[str, Any]]] = None,
    *,
    manifest: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    core = (md or "").strip()
    if not core:
        return {"ok": False, "errors": ["Markdown de page vide"], "warnings": []}

    if core == "[PAGE VIDE]":
        if ocr_items:
            errors.append("[PAGE VIDE] alors que l'OCR contient des éléments")
        return {"ok": not errors, "errors": errors, "warnings": warnings}

    for token in ("[[BLOCK", "[[TABLE", "[[/BLOCK]]", "[[/TABLE]]"):
        if token.lower() in core.lower():
            errors.append(f"token OCR résiduel dans le Markdown: {token}")
    for token in ("<TAB>", "<EMPTY>", "<BR>"):
        if token in core:
            errors.append(f"token OCR résiduel dans le Markdown: {token}")
    if HTML_PAGE_MARKER_RE.search(core):
        errors.append("balise physique <!-- PAGE n --> présente dans le cœur Markdown")
    if re.search(r"^##\s+Annexe\s*-\s*OCR\s+brut\s*$", core, flags=re.IGNORECASE | re.MULTILINE):
        errors.append("annexe OCR présente dans le cœur Markdown")
    if re.search(r"^\s*```", core, flags=re.MULTILINE):
        errors.append("bloc de code résiduel dans le cœur Markdown")

    headings = [line.strip() for line in core.splitlines() if line.strip().startswith("## ")]
    unexpected = [heading for heading in headings if heading not in _SECTION_INDEX]
    if unexpected:
        errors.append("section(s) inattendue(s): " + ", ".join(unexpected[:5]))
    positions = [_SECTION_INDEX[heading] for heading in headings if heading in _SECTION_INDEX]
    if positions != sorted(positions):
        errors.append("ordre des sections non canonique")
    if len(headings) != len(set(headings)):
        errors.append("section Markdown dupliquée")

    target_atoms, actual_tables, parse_errors = _parse_markdown_core(core)
    errors.extend(parse_errors)

    if ocr_items is not None:
        expected_atoms = _source_atoms(ocr_items)
        missing = expected_atoms - target_atoms
        added = target_atoms - expected_atoms
        if missing:
            preview = [f"{section}: {value!r}" for (section, value), count in missing.items() for _ in range(min(count, 2))]
            errors.append("contenu OCR absent ou mal classé: " + " | ".join(preview[:12]))
        if added:
            preview = [f"{section}: {value!r}" for (section, value), count in added.items() for _ in range(min(count, 2))]
            errors.append("contenu ajouté ou dupliqué par le Markdown: " + " | ".join(preview[:12]))

        if manifest is None:
            _, manifest = _render_markdown_from_items(ocr_items)
        expected_tables = [
            {
                "section": table["section"],
                "cols": int(table["cols"]),
                "data_rows": int(table["data_rows"]),
            }
            for table in manifest.get("tables", [])
        ]
        if actual_tables != expected_tables:
            errors.append(
                f"structure des tableaux différente du rendu attendu: attendu={expected_tables}, obtenu={actual_tables}"
            )

    errors = list(dict.fromkeys(errors))[:50]
    warnings = list(dict.fromkeys(warnings))[:50]
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "stats": {
            "tables": len(actual_tables),
            "visible_atoms": sum(target_atoms.values()),
        },
    }

# =====================
# Rendu PDF -> PNG base64 (low memory)
# =====================

def render_single_page_to_base64(pdf_path: str, page_num: int, dpi: int = RENDER_DPI) -> Tuple[str, float]:
    """
    Utilise paths_only=True pour réduire la RAM (important à 300 dpi).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
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
        except TypeError:
            # Fallback si pdf2image trop ancien (pas de paths_only)
            images = convert_from_path(
                pdf_path,
                dpi=dpi,
                first_page=page_num,
                last_page=page_num,
                fmt="png",
                output_folder=tmpdir,
            )
            if not images:
                raise ValueError(f"Aucune image générée pour la page {page_num}")
            png_path = os.path.join(tmpdir, f"page_{page_num}.png")
            images[0].save(png_path, format="PNG")
            with open(png_path, "rb") as f:
                b = f.read()

    b64 = base64.b64encode(b).decode("utf-8")
    return b64, (len(b) / 1024.0)


# =====================
# Appels API Qwen
# =====================

_HTTP_LOCAL = threading.local()
_CACHE_STATE_LOCK = threading.Lock()
_CACHE_PRIME_LOCKS: Dict[str, threading.Lock] = {}
_CACHE_READY_KEYS: set[str] = set()
_CACHE_RUNTIME_ENABLED = ENABLE_EXPLICIT_CACHE
_CACHE_SERIALIZE_PRIME = True


def configure_explicit_cache_for_batch(page_count: int, worker_count: int) -> bool:
    """
    Active le cache explicite uniquement lorsqu'au moins une seconde vague de
    pages pourra réellement en profiter.

    Avec une seule vague (par exemple 4 pages pour 4 workers), créer un cache
    explicite ne peut accélérer aucune page suivante et ajoute un coût de
    création. Dans ce cas, les marqueurs sont omis, sans modifier les prompts.

    Pour les lots plus longs, le premier appel de chaque prompt statique est
    sérialisé. Les appels suivants partent en parallèle une fois le cache créé.
    """
    global _CACHE_RUNTIME_ENABLED, _CACHE_SERIALIZE_PRIME

    remaining_pages = max(0, int(page_count))
    effective_workers = max(1, int(worker_count))
    runtime_enabled = bool(
        ENABLE_EXPLICIT_CACHE and remaining_pages > effective_workers
    )

    with _CACHE_STATE_LOCK:
        _CACHE_RUNTIME_ENABLED = runtime_enabled
        _CACHE_SERIALIZE_PRIME = runtime_enabled
        # Nouveau lot : ne pas considérer un cache vieux de plus de cinq minutes
        # comme encore amorcé dans la mémoire du processus.
        _CACHE_READY_KEYS.clear()
        _CACHE_PRIME_LOCKS.clear()

    return runtime_enabled


def _get_http_session() -> requests.Session:
    """Retourne une session HTTP persistante propre au thread courant."""
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


def _cacheable_text_block(text: str) -> Dict[str, Any]:
    """Construit un bloc texte statique, explicitement mis en cache si activé."""
    block: Dict[str, Any] = {"type": "text", "text": text}
    if _CACHE_RUNTIME_ENABLED:
        block["cache_control"] = {"type": "ephemeral"}
    return block


def _usage_to_stats(usage: Dict[str, Any]) -> Dict[str, int]:
    """Normalise les statistiques OpenAI-compatible, y compris le cache Qwen."""
    input_tokens = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
    output_tokens = int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
    total_tokens = int(usage.get("total_tokens") or (input_tokens + output_tokens))

    prompt_details = usage.get("prompt_tokens_details", {}) or {}
    cached_tokens = int(
        prompt_details.get("cached_tokens", usage.get("cached_tokens", 0)) or 0
    )
    cache_creation_input_tokens = int(
        prompt_details.get(
            "cache_creation_input_tokens",
            usage.get("cache_creation_input_tokens", 0),
        )
        or 0
    )

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cached_tokens": cached_tokens,
        "cache_creation_input_tokens": cache_creation_input_tokens,
        "uncached_input_tokens": max(
            0,
            input_tokens - cached_tokens - cache_creation_input_tokens,
        ),
    }


def _merge_token_stats(*items: Dict[str, Any]) -> Dict[str, int]:
    """Additionne les consommations de plusieurs tentatives réelles."""
    keys = (
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cached_tokens",
        "cache_creation_input_tokens",
        "uncached_input_tokens",
    )
    return {
        key: sum(int(item.get(key, 0) or 0) for item in items)
        for key in keys
    }


def _call_chat_with_cache_prime(
    *,
    cache_name: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int,
    context: str,
    enable_thinking: Optional[bool],
    extra_body: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, int]]:
    """
    Le premier appel réel d'un prompt statique crée le cache. Les autres threads
    attendent sa fin avant d'appeler Qwen, ce qui évite plusieurs créations de cache
    simultanées au démarrage d'un lot.
    """
    if not _CACHE_RUNTIME_ENABLED:
        return _call_chat(
            api_key=api_key,
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            context=context,
            enable_thinking=enable_thinking,
            extra_body=extra_body,
        )

    cache_key = f"{API_URL}|{model}|{cache_name}"

    if not _CACHE_SERIALIZE_PRIME:
        return _call_chat(
            api_key=api_key,
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            context=context,
            enable_thinking=enable_thinking,
            extra_body=extra_body,
        )

    with _CACHE_STATE_LOCK:
        if cache_key in _CACHE_READY_KEYS:
            prime_lock = None
        else:
            prime_lock = _CACHE_PRIME_LOCKS.setdefault(cache_key, threading.Lock())

    if prime_lock is None:
        return _call_chat(
            api_key=api_key,
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            context=context,
            enable_thinking=enable_thinking,
            extra_body=extra_body,
        )

    with prime_lock:
        with _CACHE_STATE_LOCK:
            already_ready = cache_key in _CACHE_READY_KEYS

        if not already_ready:
            _log(f"🧠 {context}: amorçage du cache explicite '{cache_name}'")
            text, stats = _call_chat(
                api_key=api_key,
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                context=context,
                enable_thinking=enable_thinking,
                extra_body=extra_body,
            )

            with _CACHE_STATE_LOCK:
                _CACHE_READY_KEYS.add(cache_key)

            created = int(stats.get("cache_creation_input_tokens", 0) or 0)
            hit = int(stats.get("cached_tokens", 0) or 0)
            _log(
                f"🧠 {context}: cache prêt "
                f"(création={created:,} tokens, hit={hit:,} tokens)"
            )
            return text, stats

    # Le cache est maintenant prêt. Le verrou est relâché avant l'appel réseau,
    # afin que tous les workers qui attendaient puissent repartir simultanément.
    return _call_chat(
        api_key=api_key,
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        context=context,
        enable_thinking=enable_thinking,
        extra_body=extra_body,
    )

def _backoff(attempt: int) -> float:
    delay = min((BACKOFF_BASE ** attempt), BACKOFF_MAX)
    return float(delay)

def _compute_retry_delay(http_status: Optional[int], err_msg: str, attempt: int) -> Tuple[bool, float]:
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
        return True, min(10.0 * attempt, 20.0)

    if "overloaded" in msg:
        return True, min(5.0 * attempt, 15.0)

    return True, _backoff(attempt)

def _call_chat(
    api_key: str,
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int,
    context: str,
    enable_thinking: Optional[bool] = None,
    allow_no_think_fallback: Optional[bool] = None,
    extra_body: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, int]]:
    validate_api_configuration()
    url = f"{API_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body: Dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": TEMPERATURE,
        "messages": messages,
    }

    if enable_thinking is not None and _supports_thinking_toggle(model):
        body["enable_thinking"] = bool(enable_thinking)

    if extra_body:
        protected = {"model", "messages", "max_tokens", "temperature"}
        for key, value in extra_body.items():
            if key not in protected:
                body[key] = value

    if allow_no_think_fallback is None:
        allow_no_think_fallback = ALLOW_NO_THINK_FALLBACK

    for attempt in range(1, MAX_RETRIES + 1):
        t0 = time.time()
        try:
            r = _get_http_session().post(
                url,
                headers=headers,
                json=body,
                timeout=(CONNECT_TIMEOUT_SECONDS, REQUEST_TIMEOUT_SECONDS),
            )

            if r.status_code == 200:
                js = r.json()
                usage = js.get("usage", {}) or {}
                stats = _usage_to_stats(usage)
                input_tokens = stats["input_tokens"]
                output_tokens = stats["output_tokens"]

                choices = js.get("choices", []) or []
                if not choices:
                    raise RuntimeError(f"{context}: réponse 200 mais aucune choice")

                choice0 = choices[0] or {}
                finish_reason = choice0.get("finish_reason")
                stats["finish_reason"] = finish_reason
                stats["truncated"] = int(str(finish_reason or "").lower() in {"length", "max_tokens"})
                message = choice0.get("message", {}) or {}
                text, reasoning_text = _extract_message_texts(message)

                if not text and output_tokens > 0:
                    finish_reason = choice0.get("finish_reason")
                    msg_preview = json.dumps(message, ensure_ascii=False)[:EMPTY_RESPONSE_LOG_CHARS]

                    if reasoning_text:
                        _log(
                            f"⚠️ {context}: 'content' vide mais 'reasoning_content' non vide "
                            f"({len(reasoning_text)} chars, finish_reason={finish_reason}). "
                            f"message={msg_preview}"
                        )
                    else:
                        _log(
                            f"⚠️ {context}: réponse vide malgré HTTP 200 / out={output_tokens} "
                            f"(finish_reason={finish_reason}). message={msg_preview}"
                        )

                    # Fallback ciblé : on garde le raisonnement par défaut.
                    # On ne coupe le thinking qu'en secours, UNE seule fois,
                    # uniquement si le modèle n'a pas fourni de réponse finale exploitable.
                    if reasoning_text and allow_no_think_fallback and enable_thinking is not False and _supports_thinking_toggle(model):
                        _log(f"↩️ {context}: retry unique avec enable_thinking=False pour récupérer la réponse finale")
                        fallback_text, fallback_stats = _call_chat(
                            api_key=api_key,
                            model=model,
                            messages=messages,
                            max_tokens=max_tokens,
                            context=context + " [final]",
                            enable_thinking=False,
                            allow_no_think_fallback=False,
                            extra_body=extra_body,
                        )
                        merged_stats: Dict[str, Any] = _merge_token_stats(stats, fallback_stats)
                        merged_stats["finish_reason"] = fallback_stats.get("finish_reason")
                        merged_stats["truncated"] = int(fallback_stats.get("truncated", 0) or 0)
                        return fallback_text, merged_stats

                _log(
                    f"✅ {context}: OK en {time.time()-t0:.2f}s "
                    f"(in={input_tokens} out={output_tokens} "
                    f"cache_hit={stats.get('cached_tokens', 0)} "
                    f"cache_create={stats.get('cache_creation_input_tokens', 0)})"
                )
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



def _is_probably_blank_image(image_b64: str) -> Optional[bool]:
    """Retourne True seulement si l'image rendue paraît réellement blanche."""
    try:
        from PIL import Image  # type: ignore

        raw = base64.b64decode(image_b64)
        with Image.open(io.BytesIO(raw)) as image:
            gray = image.convert("L")
            gray.thumbnail((256, 256))
            pixels = list(gray.getdata())
        if not pixels:
            return True
        dark_ratio = sum(1 for pixel in pixels if pixel < 245) / len(pixels)
        return dark_ratio <= BLANK_PAGE_DARK_PIXEL_RATIO
    except Exception as exc:
        _log(f"⚠️ Vérification visuelle de page vide indisponible: {exc}")
        return None


# =====================
# Pipeline qualité : OCR Qwen validé puis Markdown déterministe
# =====================

def _build_ocr_instruction(page_num: int, attempt: int, previous_errors: List[str]) -> str:
    if attempt == 0:
        return (
            f"OCR de la page {page_num}. Retourne uniquement le texte OCR brut structuré. "
            "Relis silencieusement chaque tableau avant de répondre et vérifie que chaque ligne "
            "contient exactement le nombre de cellules déclaré."
        )

    reasons = "; ".join(previous_errors[:8]) or "sortie vide ou trop courte"
    return (
        f"Nouvelle lecture indépendante de la page {page_num}. La sortie précédente a été rejetée "
        f"par le contrôle automatique pour : {reasons}. "
        "Repars entièrement de l'image, sans recopier ni corriger de mémoire la réponse précédente. "
        "Vérifie surtout : une fermeture du bon type pour chaque balise, le nombre exact de cellules "
        "de chaque ligne, la séparation quantité/conditionnement, la séparation montant/code TVA, "
        "et la séparation des tableaux TVA et totaux. N'invente rien. Retourne uniquement l'OCR structuré."
    )


def ocr_page_with_vl(api_key: str, pdf_path: str, page_num: int) -> Tuple[str, Dict[str, Any]]:
    _log(f"➡️ Page {page_num}: rendu image (dpi={RENDER_DPI})")
    image_b64, size_kb = render_single_page_to_base64(pdf_path, page_num, dpi=RENDER_DPI)
    base64_size_mb = len(image_b64) / (1024.0 * 1024.0)
    if base64_size_mb > MAX_BASE64_IMAGE_MB:
        del image_b64
        raise RuntimeError(
            f"Page {page_num}: image PNG encodée trop volumineuse "
            f"({base64_size_mb:.2f} Mo > {MAX_BASE64_IMAGE_MB:.2f} Mo). "
            "Ne réduis pas automatiquement la qualité : ajuste le PDF ou RENDER_DPI explicitement."
        )
    _log(
        f"➡️ Page {page_num}: image prête ({size_kb:.0f} KB; "
        f"base64={base64_size_mb:.2f} Mo), appel OCR"
    )
    data_url = f"data:image/png;base64,{image_b64}"
    blank_image_status = _is_probably_blank_image(image_b64)

    aggregate_stats: Dict[str, Any] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cached_tokens": 0,
        "cache_creation_input_tokens": 0,
        "uncached_input_tokens": 0,
    }
    attempt_details: List[Dict[str, Any]] = []
    previous_errors: List[str] = []
    empty_confirmations = 0
    accepted_text = ""

    max_extra_retries = max(
        OCR_EMPTY_RETRIES,
        OCR_QUALITY_RETRIES,
        OCR_EMPTY_PAGE_CONFIRMATIONS - 1,
    )
    max_attempts = 1 + max_extra_retries

    try:
        for attempt in range(max_attempts):
            page_instruction = _build_ocr_instruction(page_num, attempt, previous_errors)

            # Même hiérarchie que la version du 28/07 : prompt + image + instruction
            # dans un seul message user. Le cache éventuel ne modifie pas cette hiérarchie.
            messages = [
                {
                    "role": "user",
                    "content": [
                        _cacheable_text_block(OCR_PROMPT),
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": page_instruction},
                    ],
                }
            ]

            label = f"OCR page {page_num}" + (f" (qualité {attempt})" if attempt else "")
            text, stats = _call_chat_with_cache_prime(
                cache_name="ocr_static_prompt",
                api_key=api_key,
                model=MODEL_OCR,
                messages=messages,
                max_tokens=MAX_TOKENS_OCR,
                context=label,
                enable_thinking=ENABLE_THINKING_OCR,
                extra_body={"vl_high_resolution_images": True} if QWEN_HIGH_RES_IMAGES else None,
            )
            text = _strip_triple_backticks(text)
            text = _normalize_sans_entete_tokens(text)
            text = _strip_model_page_tokens(text)
            if ALLOW_SAFE_STRUCTURE_REPAIR:
                accepted_candidate, repairs = _repair_unambiguous_ocr_structure(text)
            else:
                accepted_candidate, repairs = text, []
            validation = validate_structured_ocr(accepted_candidate)

            if int(stats.get("truncated", 0) or 0):
                validation = {
                    **validation,
                    "ok": False,
                    "errors": [
                        f"réponse Qwen tronquée (finish_reason={stats.get('finish_reason')!r})",
                        *list(validation.get("errors", [])),
                    ],
                }

            if accepted_candidate.strip() == "[PAGE VIDE]":
                if blank_image_status is False:
                    validation = {
                        **validation,
                        "ok": False,
                        "errors": ["Qwen a répondu [PAGE VIDE] mais l'image contient des pixels non blancs"],
                    }
                elif blank_image_status is True:
                    empty_confirmations = OCR_EMPTY_PAGE_CONFIRMATIONS
                else:
                    empty_confirmations += 1
                    if empty_confirmations < OCR_EMPTY_PAGE_CONFIRMATIONS:
                        validation = {
                            **validation,
                            "ok": False,
                            "errors": [
                                f"[PAGE VIDE] doit être confirmé {OCR_EMPTY_PAGE_CONFIRMATIONS} fois "
                                "car la vérification visuelle est indisponible"
                            ],
                        }
            else:
                empty_confirmations = 0

            aggregate_stats = _merge_token_stats(aggregate_stats, stats)
            previous_errors = list(validation.get("errors", []))
            attempt_details.append(
                {
                    "attempt": attempt + 1,
                    "text_chars": len(accepted_candidate.strip()),
                    "valid": bool(validation.get("ok")),
                    "errors": previous_errors[:10],
                    "warnings": list(validation.get("warnings", []))[:10],
                    "repairs": repairs,
                    **stats,
                }
            )

            if repairs:
                _log(f"🛠️ Page {page_num}: réparations structurelles sûres: {'; '.join(repairs)}")

            if validation.get("ok"):
                accepted_text = accepted_candidate
                break

            preview = accepted_candidate.strip().replace("\n", " ")[:160]
            _log(
                f"⚠️ Page {page_num}: OCR rejeté par le contrôle qualité "
                f"({'; '.join(previous_errors[:5])}). Preview='{preview}'"
            )
            if attempt + 1 < max_attempts:
                time.sleep(OCR_EMPTY_RETRY_SLEEP)
    finally:
        del data_url
        del image_b64

    if not accepted_text:
        reason = "; ".join(previous_errors[:10]) or "aucune sortie exploitable"
        raise RuntimeError(
            f"Page {page_num}: OCR Qwen refusé après {max_attempts} tentative(s): {reason}"
        )

    aggregate_stats["attempt_count"] = len(attempt_details)
    aggregate_stats["attempts"] = attempt_details
    aggregate_stats["quality_validated"] = True
    aggregate_stats["high_resolution_images"] = QWEN_HIGH_RES_IMAGES
    aggregate_stats["image_png_kb"] = round(float(size_kb), 2)
    aggregate_stats["image_base64_mb"] = round(float(base64_size_mb), 4)
    aggregate_stats["processor_version"] = PROCESSOR_VERSION

    _log(
        f"✅ Page {page_num}: OCR validé après {len(attempt_details)} tentative(s) "
        f"({aggregate_stats.get('total_tokens', 0)} tokens cumulés)"
    )
    return accepted_text, aggregate_stats


def markdown_from_ocr(api_key: str, ocr_text: str, page_num: int) -> Tuple[str, Dict[str, Any]]:
    """Convertit l'OCR validé en Markdown sans aucun appel réseau ni modèle génératif."""
    del api_key  # Signature conservée pour compatibilité avec qwenocr_runner.py.
    validation = validate_structured_ocr(ocr_text)
    if not validation["ok"]:
        raise RuntimeError(
            f"Page {page_num}: OCR invalide transmis au moteur Markdown Python: "
            + "; ".join(validation["errors"][:10])
        )

    _log(f"➡️ Page {page_num}: rendu Markdown Python déterministe")
    if validation.get("page_empty"):
        markdown = "[PAGE VIDE]"
        manifest: Dict[str, Any] = {"tables": []}
    else:
        markdown, manifest = _render_markdown_from_items(validation["items"])

    md_validation = _validate_markdown_core(markdown, validation["items"], manifest=manifest)
    if not md_validation["ok"]:
        raise RuntimeError(
            f"Page {page_num}: erreur interne du moteur Markdown Python: "
            + "; ".join(md_validation["errors"][:10])
        )

    stats: Dict[str, Any] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cached_tokens": 0,
        "cache_creation_input_tokens": 0,
        "uncached_input_tokens": 0,
        "mode": "deterministic_python_v4",
        "quality_validated": True,
        "model": MODEL_MD,
        "tables_rendered": len(manifest.get("tables", [])),
        "visible_atoms": int(md_validation.get("stats", {}).get("visible_atoms", 0)),
    }
    _log(f"✅ Page {page_num}: Markdown Python validé")
    return markdown, stats


# =====================
# Fonction attendue par le runner
# =====================

def process_page_with_cache(pdf_path: str, page_num: int, api_key: str, is_first_page: bool = False) -> Tuple[str, Dict[str, Any]]:
    """
    Doit retourner (markdown_page, stats_payload).
    stats_payload contient :
      - champs flat (input_tokens/output_tokens/total_tokens)
      - ET une sous-clé "stats" qui répète ces champs
    -> compat avec runners qui font payload['stats'].
    """
    page_num = int(page_num)

    # 1) OCR brut
    ocr_text, ocr_stats = ocr_page_with_vl(api_key=api_key, pdf_path=pdf_path, page_num=page_num)

    # 2) Markdown structuré
    md_core, md_stats = markdown_from_ocr(api_key=api_key, ocr_text=ocr_text, page_num=page_num)

    # 3) Assemblage (inclut OCR brut en annexe)
    page_md = (
        f"<!-- PAGE {page_num} -->\n\n"
        f"{md_core.strip()}\n\n"
        "## Annexe - OCR brut\n"
        f"{_ocr_appendix_fence(ocr_text)}text\n"
        f"[[PAGE {page_num}]]\n\n"
        f"{ocr_text.rstrip()}\n"
        f"{_ocr_appendix_fence(ocr_text)}"
    ).strip()

    page_markers = _extract_html_page_markers_outside_fences(page_md)
    if page_markers != [page_num]:
        raise RuntimeError(
            f"Structure Markdown invalide pour la page physique {page_num}: "
            f"marqueurs détectés={page_markers}"
        )

    assembled_chunks = _extract_page_chunks(page_md)
    if len(assembled_chunks) != 1 or assembled_chunks[0][0] != page_num:
        raise RuntimeError(
            f"Page {page_num}: assemblage final incohérent: "
            f"chunks={[(number, len(chunk)) for number, chunk in assembled_chunks]}"
        )
    assembly_errors, assembly_warnings = _validate_page_chunk(page_num, assembled_chunks[0][1])
    if assembly_errors:
        raise RuntimeError(
            f"Page {page_num}: artefact final refusé: " + "; ".join(assembly_errors[:10])
        )
    if assembly_warnings:
        _log(f"⚠️ Page {page_num}: avertissements artefact: {'; '.join(assembly_warnings[:5])}")

    input_tokens = int(ocr_stats.get("input_tokens", 0)) + int(md_stats.get("input_tokens", 0))
    output_tokens = int(ocr_stats.get("output_tokens", 0)) + int(md_stats.get("output_tokens", 0))
    total_tokens = int(ocr_stats.get("total_tokens", 0)) + int(md_stats.get("total_tokens", 0))
    cached_tokens = int(ocr_stats.get("cached_tokens", 0)) + int(md_stats.get("cached_tokens", 0))
    cache_creation_input_tokens = int(ocr_stats.get("cache_creation_input_tokens", 0)) + int(
        md_stats.get("cache_creation_input_tokens", 0)
    )
    uncached_input_tokens = int(ocr_stats.get("uncached_input_tokens", 0)) + int(
        md_stats.get("uncached_input_tokens", 0)
    )

    ocr_validation_summary = validate_structured_ocr(ocr_text)
    integrity = {
        "ocr_sha256": hashlib.sha256(ocr_text.encode("utf-8")).hexdigest(),
        "markdown_core_sha256": hashlib.sha256(md_core.encode("utf-8")).hexdigest(),
        "page_artifact_sha256": hashlib.sha256(page_md.encode("utf-8")).hexdigest(),
        "ocr_elements": len(ocr_validation_summary.get("items", [])),
        "ocr_tables": sum(1 for item in ocr_validation_summary.get("items", []) if item.get("kind") == "TABLE"),
    }

    stats_core = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cached_tokens": cached_tokens,
        "cache_creation_input_tokens": cache_creation_input_tokens,
        "uncached_input_tokens": uncached_input_tokens,
        "details": {"ocr": ocr_stats, "md": md_stats},
        "models": {"ocr": MODEL_OCR, "md": MODEL_MD},
        "markdown_mode": md_stats.get("mode", "deterministic_python_v4"),
        "quality_validated": bool(ocr_stats.get("quality_validated") and md_stats.get("quality_validated")),
        "render_dpi": RENDER_DPI,
        "processor_version": PROCESSOR_VERSION,
        "integrity": integrity,
    }

    stats_payload: Dict[str, Any] = dict(stats_core)
    stats_payload["stats"] = dict(stats_core)

    # Libération immédiate de la référence locale, sans gc.collect() global.
    del ocr_text

    return page_md, stats_payload


# =====================
# Attendu: calculate_costs
# =====================

def calculate_costs(stats_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compat runner : renvoie des coûts à 0.0 mais garde les totaux tokens.
    Supporte 2 formats :
      - dict flat
      - dict wrapper avec sous-clé 'stats'
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

def _scan_fences(text: str) -> Tuple[bool, List[str]]:
    """Vérifie les fences Markdown, y compris les fences dynamiques de l'annexe OCR."""
    active: Optional[str] = None
    errors: List[str] = []
    for line in (text or "").splitlines():
        new_state, boundary = _fence_state_after_line(line, active)
        if boundary:
            active = new_state
    if active is not None:
        errors.append(f"bloc de code non fermé ({active})")
    return not errors, errors


def _extract_page_chunks(final_markdown: str) -> List[Tuple[int, str]]:
    markers: List[Tuple[int, int, int]] = []
    active_fence: Optional[str] = None
    offset = 0
    for line in (final_markdown or "").splitlines(keepends=True):
        new_state, boundary = _fence_state_after_line(line, active_fence)
        if boundary:
            active_fence = new_state
        elif active_fence is None:
            marker = HTML_PAGE_MARKER_RE.match(line)
            if marker:
                markers.append((int(marker.group(1)), offset, offset + len(line)))
        offset += len(line)

    chunks: List[Tuple[int, str]] = []
    for index, (page_num, _start, content_start) in enumerate(markers):
        content_end = markers[index + 1][1] if index + 1 < len(markers) else len(final_markdown)
        chunk = final_markdown[content_start:content_end]
        chunk = re.sub(r"\n?\s*---\s*\n?\s*$", "", chunk).strip()
        chunks.append((page_num, chunk))
    return chunks


_APPENDIX_PATTERN = re.compile(
    r"^##\s+Annexe\s*-\s*OCR\s+brut\s*$\s*"
    r"(?P<fence>`{3,}|~{3,})text\s*\n\[\[PAGE\s+(?P<page>\d+)\]\]\s*\n"
    r"(?P<ocr>.*?)\n(?P=fence)\s*$",
    flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
)



def _validate_page_chunk(page_num: int, chunk: str) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    _, fence_errors = _scan_fences(chunk)
    errors.extend(f"Page {page_num}: {error}" for error in fence_errors)

    appendix_headings = re.findall(
        r"^##\s+Annexe\s*-\s*OCR\s+brut\s*$",
        chunk or "",
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if len(appendix_headings) != 1:
        errors.append(
            f"Page {page_num}: une seule rubrique 'Annexe - OCR brut' attendue, trouvé={len(appendix_headings)}"
        )

    matches = list(_APPENDIX_PATTERN.finditer(chunk or ""))
    if len(matches) != 1:
        errors.append(f"Page {page_num}: une seule annexe OCR complète attendue, trouvé={len(matches)}")
        return errors, warnings

    match = matches[0]
    trailing = chunk[match.end():].strip()
    if trailing:
        errors.append(f"Page {page_num}: contenu interdit après la clôture de l'annexe OCR")

    appendix_page = int(match.group("page"))
    if appendix_page != page_num:
        errors.append(f"Page {page_num}: l'annexe annonce [[PAGE {appendix_page}]]")

    ocr_text = match.group("ocr").strip()
    ocr_validation = validate_structured_ocr(ocr_text)
    errors.extend(f"Page {page_num} OCR: {error}" for error in ocr_validation.get("errors", []))
    warnings.extend(f"Page {page_num} OCR: {warning}" for warning in ocr_validation.get("warnings", []))

    md_core = chunk[:match.start()].strip()
    if not md_core:
        errors.append(f"Page {page_num}: cœur Markdown absent avant l'annexe")
        return errors, warnings

    md_validation = _validate_markdown_core(md_core, ocr_validation.get("items", []))
    errors.extend(f"Page {page_num} Markdown: {error}" for error in md_validation.get("errors", []))
    warnings.extend(f"Page {page_num} Markdown: {warning}" for warning in md_validation.get("warnings", []))
    return errors, warnings


def validate_markdown_quality(final_markdown: str, page_count: int) -> Dict[str, Any]:
    """Contrôle strict et complet du document final, OCR annexé compris."""
    errors: List[str] = []
    warnings: List[str] = []
    expected_count = int(page_count or 0)

    if not isinstance(final_markdown, str) or not final_markdown.strip():
        errors.append("Markdown final vide.")
        chunks: List[Tuple[int, str]] = []
    else:
        _, fence_errors = _scan_fences(final_markdown)
        errors.extend(fence_errors)
        chunks = _extract_page_chunks(final_markdown)
        if final_markdown.rstrip().splitlines() and final_markdown.rstrip().splitlines()[-1].strip() == "---":
            errors.append("Le document final ne doit pas se terminer par un séparateur ---")

    actual_pages = [page_num for page_num, _ in chunks]
    expected_pages = list(range(1, expected_count + 1)) if expected_count else actual_pages
    if actual_pages != expected_pages:
        errors.append(f"Pagination physique invalide: attendu={expected_pages}, obtenu={actual_pages}")

    for page_num, chunk in chunks:
        page_errors, page_warnings = _validate_page_chunk(page_num, chunk)
        errors.extend(page_errors)
        warnings.extend(page_warnings)

    if expected_count and len(chunks) != expected_count:
        errors.append(f"Nombre de pages contrôlables: {len(chunks)}, attendu: {expected_count}")

    errors = list(dict.fromkeys(errors))[:200]
    warnings = list(dict.fromkeys(warnings))[:200]
    ok = not errors
    score = 1.0 if ok else max(0.0, 1.0 - min(len(errors), 25) / 25.0)
    stats = {
        "page_count": expected_count,
        "pages_checked": len(chunks),
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
        "stats": stats,
        "summary": ("OK" if ok else "KO") + ("" if not warnings else f" (warnings={len(warnings)})"),
    }

__all__ = [
    "API_URL",
    "PROCESSOR_VERSION",
    "MODEL",
    "MODEL_OCR",
    "MODEL_MD",
    "INTER_REQUEST_DELAY",
    "STOP_ON_CRITICAL",
    "RENDER_DPI",
    "ENABLE_EXPLICIT_CACHE",
    "QWEN_HIGH_RES_IMAGES",
    "MAX_BASE64_IMAGE_MB",
    "USE_QWEN_MARKDOWN",
    "ALLOW_SAFE_STRUCTURE_REPAIR",
    "PIPELINE_VERSION",
    "PIPELINE_FINGERPRINT",
    "validate_api_configuration",
    "configure_explicit_cache_for_batch",
    "get_pdf_info",
    "load_progress",
    "save_progress",
    "clear_progress",
    "process_page_with_cache",
    "calculate_costs",
    "validate_markdown_quality",
    "validate_structured_ocr",
    "render_markdown_deterministic",
    "validate_canonical_markdown_structure",
    "ocr_page_with_vl",
    "markdown_from_ocr",
]







