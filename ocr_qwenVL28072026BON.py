#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ocr_qwenVL.py — module compatible qwenocr_runner.py (sans modifier le runner)

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
- 2 étapes (fidélité) :
  1) OCR brut (image -> texte brut)
  2) Markdown (texte brut -> markdown structuré)
- Annexe OCR brut ajoutée dans le markdown (côté code, 0 token).

Robustesse :
- DPI par défaut = 300 (configurable via env RENDER_DPI)
- Conversion PDF->PNG low-memory (pdf2image paths_only + fichiers temporaires)
- Retry court + logs en cas de 429/overloaded
- Retry spécifique si OCR "trop court" (réessaye l'OCR 2 fois par défaut)
- Payload stats contient des clés "flat" + sous-clé "stats" (compat runner)
"""

from __future__ import annotations

import base64
import gc
import json
import os
import re
import tempfile
import threading
import time
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

API_URL = os.getenv("QWEN_API_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")

MODEL_OCR = os.getenv("QWEN_MODEL_OCR", "qwen3.7-plus")
MODEL_MD = os.getenv("QWEN_MODEL_MD", "qwen3.7-plus")

# Attendu par le runner (affiché au démarrage)
MODEL = MODEL_OCR

# Conservé pour compatibilité avec le runner, sans pause fixe par défaut.
INTER_REQUEST_DELAY = _env_float("INTER_REQUEST_DELAY", 0.0)

# Attendu par le runner : stop ou non sur erreur page "critique"
# Recommandation : True pour ne pas générer de Markdown incomplet sans le savoir.
STOP_ON_CRITICAL = _env_bool("STOP_ON_CRITICAL", True)

# Qualité demandée : 300 DPI (configurable)
RENDER_DPI = _env_int("RENDER_DPI", 300)

MAX_TOKENS_OCR = _env_int("MAX_TOKENS_OCR", 12000)
MAX_TOKENS_MD = _env_int("MAX_TOKENS_MD", 12000)

TEMPERATURE = _env_float("TEMPERATURE", 0.0)

# Timeouts/retries réseau
REQUEST_TIMEOUT_SECONDS = _env_int("REQUEST_TIMEOUT_SECONDS", 180)
CONNECT_TIMEOUT_SECONDS = _env_int("CONNECT_TIMEOUT_SECONDS", 10)
HTTP_POOL_SIZE = max(1, _env_int("HTTP_POOL_SIZE", 8))

MAX_RETRIES = _env_int("MAX_RETRIES", 3)
BACKOFF_BASE = _env_float("BACKOFF_BASE", 2.0)
BACKOFF_MAX = _env_float("BACKOFF_MAX", 20.0)  # volontairement bas

# Logs
VERBOSE = _env_bool("VERBOSE", True)

# Rate-limit behavior
FAIL_FAST_ON_429 = _env_bool("FAIL_FAST_ON_429", False)  # False = retry court; True = fail vite

# OCR "trop court" : retries spécifiques (utile si le modèle renvoie parfois 1-2 mots)
OCR_MIN_CHARS = _env_int("OCR_MIN_CHARS", 40)
OCR_EMPTY_RETRIES = _env_int("OCR_EMPTY_RETRIES", 2)          # nombre de retries supplémentaires
OCR_EMPTY_RETRY_SLEEP = _env_float("OCR_EMPTY_RETRY_SLEEP", 1.5)

# Thinking mode : on le garde activé par défaut pour conserver le raisonnement.
# Fallback ciblé : si le modèle renvoie uniquement reasoning_content et pas de content,
# on peut relancer UNE fois sans thinking pour récupérer la réponse finale.
ENABLE_THINKING_OCR = _env_bool("ENABLE_THINKING_OCR", True)
ENABLE_THINKING_MD = _env_bool("ENABLE_THINKING_MD", True)
ALLOW_NO_THINK_FALLBACK = _env_bool("ALLOW_NO_THINK_FALLBACK", True)
EMPTY_RESPONSE_LOG_CHARS = _env_int("EMPTY_RESPONSE_LOG_CHARS", 1500)


def _log(msg: str) -> None:
    if VERBOSE:
        print(msg, flush=True)


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

FENCE_MARKER_RE = re.compile(r"^\s*(```|~~~)")


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
    """
    Supprime les lignes autonomes <!-- PAGE n --> éventuellement générées par Qwen.

    Les marqueurs situés dans un bloc de code sont préservés.
    """
    output: List[str] = []
    active_fence: Optional[str] = None

    for line in (markdown or "").splitlines():
        fence_match = FENCE_MARKER_RE.match(line)

        if fence_match:
            fence_token = fence_match.group(1)
            if active_fence is None:
                active_fence = fence_token
            elif line.strip().startswith(active_fence):
                active_fence = None
            output.append(line)
            continue

        if active_fence is None and HTML_PAGE_MARKER_RE.match(line):
            continue

        output.append(line)

    return "\n".join(output).strip()


def _extract_html_page_markers_outside_fences(markdown: str) -> List[int]:
    """Retourne les numéros de balises <!-- PAGE n --> hors blocs de code."""
    markers: List[int] = []
    active_fence: Optional[str] = None

    for line in (markdown or "").splitlines():
        fence_match = FENCE_MARKER_RE.match(line)

        if fence_match:
            fence_token = fence_match.group(1)
            if active_fence is None:
                active_fence = fence_token
            elif line.strip().startswith(active_fence):
                active_fence = None
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

SYSTEM_PROMPT_MD = """Vous êtes un assistant spécialisé dans la conversion d'OCR layout-aware de documents comptables : factures, avoirs, notes de crédit, proformas, en Markdown fidèle.

Entrée : texte OCR structuré d'une ou plusieurs pages.

Sortie : Markdown uniquement.
Interdiction : JSON, explication, commentaire, annexe OCR, bloc de code autour de la réponse.

L'OCR peut contenir ces éléments :
[[BLOCK id=B001 order=001 pos=top-left role_hint=unknown]]
[[BLOCK id=B001 order=001 pos=top-left bbox=000,000,000,000 role_hint=unknown]]
[[/BLOCK]]
[[TABLE id=T001 order=001 pos=middle role_hint=line_items cols=N]]
[[TABLE id=T001 order=001 pos=middle bbox=000,000,000,000 role_hint=line_items cols=N]]
[[/TABLE]]

Tokens possibles dans le contenu :
<TAB>
<EMPTY>
<BR>
[ILLISIBLE]
[SANS_ENTETE_n]
<SANS_ENTETE_n>

PRIORITÉ DES RÈGLES

En cas de conflit, applique les règles dans cet ordre :

1. Fidélité : utiliser uniquement le contenu OCR, sans rien inventer, corriger, normaliser ou calculer.
2. Ne jamais perdre un contenu OCR non vide.
3. Respecter la séparation des [[TABLE]] OCR : ne jamais fusionner deux tables OCR distinctes.
4. Classer par role_hint, jamais par position seule.
5. Préserver les valeurs comptables : montants, signes, devises, taux, statuts.
6. En cas de doute de classement : Mentions Légales et Notes Complémentaires.
7. En cas de doute sur la structure d'un tableau : rendu en texte simple plutôt qu'un faux tableau.

RÈGLES ABSOLUES

- Utilise uniquement le contenu de l'OCR fourni.
- Ne corrige pas.
- Ne reformule pas.
- Ne normalise pas.
- Ne calcule pas.
- Ne complète aucune information absente.
- Ne déduis aucun montant, libellé, taxe, devise, adresse, identité ou référence.
- Ne crée jamais [CHAMP MANQUANT].
- Ne transforme jamais <EMPTY> en [CHAMP MANQUANT].
- Ne recopie jamais l'OCR brut complet.
- Ne crée jamais de section "Annexe - OCR brut".
- Supprime les tokens techniques [[...]], [[/BLOCK]], [[/TABLE]], id, order, pos, bbox, role_hint, cols=N du rendu final.
- Conserve [ILLISIBLE] exactement.
- Convertis <SANS_ENTETE_n> en [SANS_ENTETE_n].
- Conserve [SANS_ENTETE_n] exactement après normalisation.
- Convertis <EMPTY> en cellule vide dans un tableau Markdown.
- Convertis <BR> en <br> dans les cellules Markdown.
- Si <BR> apparaît hors tableau, remplace-le par un retour à la ligne simple.
- Si <TAB> apparaît hors tableau malgré l'OCR, ne crée pas de tableau : remplace <TAB> par quatre espaces.
- Tout contenu OCR non classé doit rester visible dans le Markdown, dans la section la plus sûre.

RÈGLES COMPTABLES — FIDÉLITÉ DES VALEURS

- Garde les montants exactement tels quels : chiffres, virgules, points, espaces, séparateurs de milliers.
- Conserve le signe et les parenthèses comptables des montants négatifs : -12,50 ou (12,50).
- Ne transforme jamais un signe moins en parenthèses, ni des parenthèses en signe moins.
- Conserve la devise exactement : €, EUR, $, USD, CHF, etc.
- Conserve la position de la devise : avant ou après le montant.
- Conserve les taux de TVA exactement.
- Ne fusionne jamais deux taux différents.
- Pour un avoir ou une note de crédit, conserve le titre et les montants tels quels, sans changer de signe ni recalculer.
- Conserve les mentions de statut : "Payé", "Acquittée", "Soldé", "Reste à payer", "Net à payer", "Échu", "À régler".
- Ne convertis jamais un montant d'une devise à une autre.

ORDRE

- Chaque appel traite exactement une seule page physique.
- Ne génère aucun commentaire HTML de pagination.
- Ne génère jamais de balise <!-- PAGE n -->.
- Le code appelant ajoute lui-même l'unique balise de page physique.
- Ne transforme jamais une pagination visible comme "Page 1/1", "Page : 2" ou "2/3" en commentaire HTML.
- À l'intérieur de la page, utilise order si présent.
- Si order est absent, respecte l'ordre d'apparition dans l'OCR.
- Les sections Markdown regroupent les contenus par rôle, mais l'ordre interne de chaque section doit suivre order.

SECTIONS MARKDOWN

Utilise ces sections, dans cet ordre.
Omettre une section seulement si aucun contenu ne s'y rattache.

## Informations Émetteur (Fournisseur)
## Informations Client
## Informations de Livraison
## Détails de la Facture
## Tableau des Lignes de Facturation
## Montants Récapitulatifs
## Informations de Paiement
## Mentions Légales et Notes Complémentaires

CLASSEMENT PAR role_hint

Utilise role_hint en priorité.
N'utilise jamais pos seul pour classer un bloc.

Vers "Informations Émetteur (Fournisseur)" :
- supplier_identity
- supplier_address
- supplier_legal
- supplier_contact
- supplier

Vers "Informations Client" :
- customer_identity
- customer_address
- customer_contact
- customer_legal
- billing_address
- customer

Vers "Informations de Livraison" :
- shipping_address
- shipping_details
- shipping_contact
- delivery_confirmation
- unknown si le contenu contient clairement une adresse de livraison, une expédition, un transporteur, un retrait, un enlevé au comptoir, une confirmation de livraison ou une instruction de livraison

Vers "Détails de la Facture" :
- invoice_title
- invoice_details
- unknown si le contenu est clairement un titre, une page imprimée, un numéro, une date, une référence, une commande, un vendeur, une devise, un code client, un statut de paiement ou un objet de facture

Vers "Tableau des Lignes de Facturation" :
- line_items_note
- line_items
- line_items_footer

Vers "Montants Récapitulatifs" :
- tax_summary
- totals_summary
- isolated_value si la valeur est proche ou située entre les articles et les récapitulatifs
- unknown si le contenu est clairement un sous-total, une taxe, un acompte, une remise globale, un solde, un total, un net à payer ou une valeur financière isolée

Vers "Informations de Paiement" :
- payment_terms
- bank_details
- payment
- unknown si le contenu contient clairement "Echéance", "Montant", "Conditions de Règlement", "Mode de règlement", "Règlement", "Acompte", "CB", "Virement" ou une date d'échéance

Vers "Mentions Légales et Notes Complémentaires" :
- legal_terms
- marketing_badge
- logo_text si le bloc ne permet pas d'identifier l'émetteur
- stamp_signature
- qr_barcode_text
- notes
- unknown non classable avec certitude

RÈGLES DE CLASSEMENT

- Ne place dans "Informations Client" que le destinataire, facturé à, acheteur, contact client, information légale client ou son adresse de facturation.
- Ne place pas shipping_address, shipping_details, shipping_contact ou delivery_confirmation dans "Informations Client" si la section "Informations de Livraison" existe.
- Une adresse de livraison n'est pas une adresse client par défaut : elle va dans "Informations de Livraison".
- Si la même adresse est visible à la fois dans une zone client et dans une zone livraison, conserve les deux occurrences dans leurs sections respectives.
- Les blocs "Expédition", "Adresse de livraison", "Livré à", "Transporteur", "Enlevé au comptoir", "Confirmation de Livraison", "Nom" ou "Signature" liés à la livraison vont dans "Informations de Livraison".
- Ne place jamais un slogan, badge SAV, label qualité, tampon, QR code, texte marketing ou logo secondaire dans "Informations Client".
- Ne place dans "Informations Émetteur" que l'identité, l'adresse, les coordonnées ou les mentions juridiques du fournisseur.
- Ne place jamais un badge SAV, slogan, label qualité, pictogramme, QR code ou texte promotionnel dans "Informations Émetteur".
- marketing_badge, stamp_signature, qr_barcode_text vont dans "Mentions Légales et Notes Complémentaires".
- logo_text va dans "Informations Émetteur" uniquement si le bloc contient le nom/logo évident de l'émetteur ou s'il correspond clairement à l'identité fournisseur.
- logo_text va dans "Mentions Légales et Notes Complémentaires" s'il ne permet pas d'identifier l'émetteur.
- Un bloc unknown proche du client ne devient pas client par proximité.
- Un bloc unknown proche du fournisseur ne devient pas fournisseur par proximité.
- Un bloc unknown contenant "Echéance", "Conditions de Règlement", "Mode de règlement", "Règlement", "Acompte", "CB" ou "Virement" va dans "Informations de Paiement".
- Un texte isolé d'en-tête sans libellé clair va dans "Détails de la Facture" ou dans "Mentions Légales et Notes Complémentaires", jamais dans le client par défaut.
- Ne déplace jamais une valeur d'un tableau vers un autre tableau.
- Ne fusionne jamais deux sections parce qu'elles sont proches visuellement.

COMPATIBILITÉ AVEC ANCIENS role_hint

Si l'OCR utilise d'anciens role_hint :
- supplier -> Informations Émetteur
- supplier_address -> Informations Émetteur
- customer -> Informations Client
- payment -> Informations de Paiement
- logo_marketing -> Mentions Légales et Notes Complémentaires, sauf si le bloc contient clairement le nom/logo de l'émetteur
- unknown -> classer par contenu explicite ; sinon Mentions Légales et Notes Complémentaires

BLOCS

- Un [[BLOCK]] devient du texte simple dans la section appropriée.
- Ne transforme pas un [[BLOCK]] en tableau Markdown.
- Ne fusionne pas deux [[BLOCK]] ayant des role_hint différents.
- Conserve les retours à la ligne utiles.
- Ne supprime aucun bloc non vide.
- Une valeur isolée doit être conservée.
- Pour une valeur isolée dans les montants, écrire simplement la valeur ou : Valeur isolée : X
- Ne crée pas de libellé plus précis que celui fourni par l'OCR.
- line_items_note doit être rendu en texte simple avant le tableau des articles.
- line_items_footer doit être rendu en texte simple après le tableau des articles.
- Ne transforme jamais line_items_note ou line_items_footer en ligne article.
- Ne transforme jamais line_items_note ou line_items_footer en tableau Markdown sauf si l'OCR les a explicitement marqués [[TABLE]].

TABLEAUX — RÈGLES GÉNÉRALES

- Chaque [[TABLE]] OCR devient un tableau Markdown séparé.
- Ne fusionne jamais deux [[TABLE]] OCR.
- Si un tableau de taxes et un tableau de totaux sont deux [[TABLE]] OCR distincts, ils doivent rester séparés.
- Si l'OCR contient une seule [[TABLE]] mêlant taxes et totaux dans une grille continue, rends cette seule table telle quelle ; ne crée pas une séparation artificielle.
- Ne fusionne jamais deux groupes d'en-têtes indépendants.
- Une table Markdown doit avoir un seul groupe logique d'en-têtes.
- Les cellules sont séparées par <TAB>.
- Utilise cols=N si présent pour vérifier le nombre de cellules.
- Chaque ligne Markdown doit avoir le même nombre de cellules que l'en-tête.
- Si une ligne a moins de cellules que l'en-tête, complète seulement par des cellules vides.
- Si une ligne a plus de cellules que l'en-tête, ajoute des colonnes [SANS_ENTETE_n] à l'en-tête plutôt que de fusionner ou supprimer des cellules.
- Si une cellule d'en-tête est vide ou <EMPTY> alors que la colonne contient des valeurs, remplace seulement cette cellule d'en-tête par [SANS_ENTETE_n].
- Numérote les [SANS_ENTETE_n] de gauche à droite, en recommençant à 1 pour chaque tableau.
- Ne renomme jamais [SANS_ENTETE_n].
- Ne remplace jamais [SANS_ENTETE_n] par "Remise", "TVA", "Code", "Taxe" ou autre libellé non visible.
- Ne crée aucune ligne de tableau entièrement vide.
- Ne crée aucune ligne composée uniquement de cellules vides.
- Si un tableau OCR est trop irrégulier pour être converti sûrement, rends ses lignes en texte simple dans la bonne section, cellules séparées par " | ", sans créer de faux tableau Markdown.

TABLEAUX — TABLES SANS DONNÉES

- Ne produis jamais un tableau Markdown avec seulement une ligne d'en-tête et aucune ligne de données.
- Si une [[TABLE]] OCR ne contient qu'une seule ligne, rends-la en texte simple dans la section appropriée.
- Ne crée pas de séparateur Markdown |---|---| si aucune ligne de données ne suit.
- Si un tableau Markdown serait vide après suppression de lignes vides, rends son contenu en texte simple.

TABLEAUX — TABLES SANS EN-TÊTE VISIBLE

- Si la première ligne d'une [[TABLE]] ressemble à une ligne de données et non à un en-tête, ajoute des en-têtes génériques [SANS_ENTETE_n].
- Indices de ligne de données : libellé financier + montant, par exemple "Sous-total 133.94", "Remise - 0.00", "Total TTC 168.94", "Montant HT (1) 133.94", "TVA (1) : 0.00% 0.00".
- Ne transforme jamais "Sous-total", "Remise", "Total TTC", "Montant HT", "TVA", "Frais encaissement", "Acompte", "Net à payer" en en-tête de tableau s'ils sont suivis d'un montant sur la même ligne.
- Pour une table sans en-tête visible, l'en-tête Markdown doit être [SANS_ENTETE_1], [SANS_ENTETE_2], etc.

TABLEAUX — UNITÉS D'EN-TÊTE

- Si la première ligne du corps du tableau contient uniquement des unités, devises ou marqueurs courts comme EUR, €, USD, HT, TTC, %, fusionne ces valeurs dans les en-têtes correspondants avec <br>.
- Supprime ensuite cette ligne du corps du tableau.
- Les cellules vides de cette ligne d'unités ne créent pas de colonnes.
- Exemple : "Prix unit. HT" + ligne "EUR" devient "Prix unit. HT<br>EUR".
- Exemple : "Total" + ligne "EUR" devient "Total<br>EUR".

TABLEAUX — COLONNES [SANS_ENTETE_n] VIDES

- Si une colonne [SANS_ENTETE_n] est entièrement vide dans toutes les lignes de données, supprime cette colonne du tableau Markdown.
- Ne supprime jamais une colonne [SANS_ENTETE_n] qui contient au moins une valeur visible.
- Après suppression d'une colonne [SANS_ENTETE_n] vide, renumérote les colonnes [SANS_ENTETE_n] restantes de gauche à droite.
- Ne supprime jamais une colonne nommée normalement, même si elle contient des cellules vides.

TABLEAUX — POSITION DES COLONNES DE POURCENTAGE

Applique ces règles uniquement aux tableaux où les valeurs montrent un motif nombre/prix -> pourcentage -> nombre/prix.

- Si les données suivent le motif nombre/prix -> pourcentage -> nombre/prix, mais que l'en-tête [SANS_ENTETE_n] est placé après la deuxième colonne numérique, corrige uniquement l'ordre des en-têtes sans déplacer les valeurs.
- Exemple mauvais : Px unitaire | Px unitaire remisé | [SANS_ENTETE_1] avec des valeurs 0,870 | 0% | 0,870.
- Exemple corrigé : Px unitaire | [SANS_ENTETE_1] | Px unitaire remisé.
- Ne modifie jamais les cellules de données pour cette correction ; seul l'ordre des en-têtes est corrigé.
- Ne crée pas un libellé "Remise" si ce mot n'est pas visible.

TABLEAUX — RÉPARATION DES LIGNES ARTICLES

Applique ces règles uniquement aux tables role_hint=line_items.

- Identifie les colonnes par leurs en-têtes visibles : référence, désignation, n° de série, quantité, prix, montant, taxe, TVA, code.
- Une vraie ligne article contient normalement une désignation et au moins une quantité, un prix, un montant, une taxe ou un code TVA.
- Une ligne qui ne contient ni quantité, ni prix, ni montant, ni taxe, ni code TVA est une note ou une continuation, pas un nouvel article, sauf preuve claire contraire.
- Dans une table line_items, distingue les vraies lignes articles des métadonnées documentaires.
- Une métadonnée documentaire a généralement la forme libellé-valeur, instruction, contexte, report, pagination, contact, signature, livraison ou référence de suivi.
- Si une métadonnée documentaire apparaît avant le premier article réel, rends-la comme texte simple avant le tableau articles.
- Si elle apparaît après le dernier article réel, rends-la comme texte simple après le tableau articles.
- Si elle apparaît dans une cellule d'article avec une vraie désignation produit, extrais seulement la métadonnée et conserve la désignation produit dans l'article.
- Ne laisse pas dans le tableau articles une ligne composée seulement d'une note, d'un contexte, d'une garantie, d'un code-barres secondaire, d'un report, d'un contact ou d'un pied de tableau.
- Si une ligne de continuation apparaît après un article réel, fusionne-la avec l'article précédent :
  - valeur de type EAN/GTIN/code-barres numérique long dans la colonne référence -> ajouter à la cellule Référence précédente avec <br> ;
  - phrase descriptive, garantie, caractéristique produit -> ajouter à la cellule Désignation précédente avec <br> ;
  - valeur située dans une colonne "N° de Série" -> ajouter à la cellule N° de Série précédente avec <br>.
- Si la ligne contient à la fois une référence secondaire et une description, rattache chaque valeur à la cellule correspondante de l'article précédent.
- Ne fusionne jamais une ligne qui contient une quantité, un prix, un montant ou une taxe non vide, sauf si elle est clairement un pied de tableau ou un report.
- Une ligne de remise, d'avoir ou de correction avec montant négatif est une vraie ligne du tableau : ne la fusionne pas, ne change pas son signe.
- Après fusion, supprime les lignes de continuation devenues inutiles.

FORMAT DES TABLEAUX MARKDOWN

- Échappe les caractères "|" présents dans les cellules en "\|".
- Utilise un séparateur Markdown simple :
|---|---|
- N'ajoute pas d'alignement avec ":".
- Garde les valeurs exactement telles qu'elles sont dans l'OCR.
- Ne modifie pas les virgules, points, espaces, devises, %, signes, parenthèses comptables ou symboles.

RÈGLES FACTURES

- Le tableau des articles/prestations doit rester dans "Tableau des Lignes de Facturation".
- Le tableau des articles ne doit contenir que les vraies lignes d'articles/prestations présentes ou réparables depuis l'OCR.
- line_items_note doit apparaître avant le tableau d'articles.
- line_items_footer doit apparaître après le tableau d'articles.
- Les consignes "fin du tableau articles" ne s'appliquent qu'au tableau des articles.
- Après le tableau des articles, continue toujours avec montants, taxes, totaux, échéances, paiement, livraison, mentions et pied de page.
- Les tableaux tax_summary et totals_summary doivent rester séparés s'ils sont deux [[TABLE]] distincts.
- Un montant de taxe reste dans le tableau de taxe.
- Un total à payer reste dans le tableau de total.
- Ne place jamais un montant de TVA dans NET A PAYER.
- Ne place jamais NET A PAYER dans un tableau de TVA.
- Le montant sous l'en-tête "NET A PAYER", "TOTAL A PAYER", "SOLDE" ou équivalent doit rester sous cet en-tête.
- Si une valeur semble ambiguë, conserve-la comme valeur isolée plutôt que de l'aligner dans un tableau voisin.

INFORMATIONS DE PAIEMENT

- Les blocs payment_terms, payment et bank_details vont dans "Informations de Paiement".
- Ne transforme pas un bloc paiement en tableau Markdown sauf si l'OCR l'a explicitement marqué [[TABLE]].
- Si une table payment_terms ne contient qu'une seule ligne, rends-la en texte simple.
- Conserve les libellés bancaires exactement : BRED, RIB, IBAN, BIC, Code BIC, échéance, mode de règlement.
- Ne modifie jamais les espaces des IBAN, BIC, RIB ou références bancaires.
- Ne rajoute pas d'espace dans un BIC.
- Ne supprime pas d'espace dans un IBAN.

CONTRÔLE FINAL SILENCIEUX

Avant de répondre, vérifie :
- La sortie contient uniquement le Markdown final.
- Il n'y a pas de section "Annexe - OCR brut".
- Aucun token [[...]], [[/BLOCK]], [[/TABLE]], <TAB>, <EMPTY>, bbox ne reste dans le Markdown.
- Aucun token <BR> ne reste ; il doit être converti en <br>.
- Aucun token <SANS_ENTETE_n> ne reste ; il doit être converti en [SANS_ENTETE_n].
- Aucun [CHAMP MANQUANT] n'a été créé.
- Les montants, signes, parenthèses comptables, devises et taux sont conservés à l'identique.
- Tous les [[TABLE]] OCR sont rendus comme des tableaux séparés, sauf les tables à une seule ligne qui sont rendues en texte simple.
- Aucun tableau Markdown ne contient deux groupes d'en-têtes indépendants.
- Aucun tableau Markdown ne fusionne taxes et totaux à partir de deux tables distinctes.
- Aucun tableau Markdown ne contient de ligne entièrement vide.
- Aucun tableau Markdown ne contient seulement un en-tête sans ligne de données.
- Les lignes d'unités comme EUR, €, USD, HT, TTC, % ne restent pas comme lignes d'articles.
- Les colonnes [SANS_ENTETE_n] entièrement vides sont supprimées.
- Les colonnes [SANS_ENTETE_n] de pourcentage sont placées au bon endroit par rapport aux données.
- Les blocs marketing, SAV, logo secondaire, QR code, tampon et slogans ne sont ni dans Informations Client ni dans Informations Émetteur.
- Les informations de livraison sont dans "Informations de Livraison", pas dans "Informations Client", quand elles sont distinguables.
- customer_contact et customer_legal sont dans Informations Client.
- line_items_note et line_items_footer sont autour du tableau articles, pas dedans comme lignes articles.
- Les métadonnées documentaires liées aux articles sont sorties du tableau articles.
- Les lignes de continuation d'articles ont été fusionnées quand elles ne contenaient ni quantité, ni prix, ni montant, ni taxe.
- Les informations après les articles sont présentes.
- Le Markdown respecte les sections demandées.
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


def _normalize_sans_entete_tokens(text: str) -> str:
    """
    Normalise les erreurs de forme du modèle : <SANS_ENTETE_1>
    devient [SANS_ENTETE_1]. Applicable OCR + Markdown.
    """
    if not text:
        return text
    return re.sub(r"<SANS_ENTETE_(\d+)>", r"[SANS_ENTETE_\1]", text)


def _normalize_markdown_tokens(md: str) -> str:
    """
    Nettoyage défensif du Markdown final produit par le modèle.
    Ne s'applique pas à l'annexe OCR ajoutée ensuite par le code.
    """
    if not md:
        return md
    md = _normalize_sans_entete_tokens(md)
    md = md.replace("<BR>", "<br>")
    md = md.replace("<EMPTY>", "")
    md = md.replace("<TAB>", "    ")
    return md


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


def _remove_header_only_md_tables(md: str) -> str:
    """
    Convertit les tableaux Markdown composés uniquement d'un en-tête
    et d'un séparateur en texte simple. Cela corrige notamment les blocs
    paiement/détails qu'un modèle transforme parfois en table vide.
    """
    lines = md.splitlines()
    out: List[str] = []
    i = 0

    while i < len(lines):
        if (
            i + 1 < len(lines)
            and _is_md_table_row(lines[i])
            and _is_md_separator_row(lines[i + 1])
        ):
            j = i + 2
            data_rows: List[str] = []

            while j < len(lines) and _is_md_table_row(lines[j]):
                data_rows.append(lines[j])
                j += 1

            if not data_rows:
                cells = [c for c in _split_md_cells(lines[i]) if c]
                if cells:
                    out.append(" | ".join(cells))
                i = j
                continue

        out.append(lines[i])
        i += 1

    return "\n".join(out)


def _drop_empty_unnamed_columns(md: str) -> str:
    """
    Supprime les colonnes [SANS_ENTETE_n] ou en-tête vide si elles sont
    entièrement vides dans les lignes de données.
    Si un en-tête vide contient des valeurs, il est renommé [SANS_ENTETE_n].
    """
    lines = md.splitlines()
    out: List[str] = []
    i = 0

    while i < len(lines):
        if (
            i + 1 < len(lines)
            and _is_md_table_row(lines[i])
            and _is_md_separator_row(lines[i + 1])
        ):
            header = _split_md_cells(lines[i])
            j = i + 2
            rows: List[List[str]] = []

            while j < len(lines) and _is_md_table_row(lines[j]):
                row = _split_md_cells(lines[j])
                if len(row) < len(header):
                    row += [""] * (len(header) - len(row))
                rows.append(row[:len(header)])
                j += 1

            if not rows:
                out.append(lines[i])
                out.append(lines[i + 1])
                i = j
                continue

            keep_indexes: List[int] = []
            for idx, h in enumerate(header):
                h_clean = h.strip()
                is_sans = bool(re.fullmatch(r"\[SANS_ENTETE_\d+\]", h_clean))
                is_blank_header = (h_clean == "")
                values = [
                    (row[idx].strip() if idx < len(row) else "")
                    for row in rows
                ]

                if (is_sans or is_blank_header) and all(v == "" for v in values):
                    continue

                keep_indexes.append(idx)

            new_header = [header[idx] for idx in keep_indexes]
            new_rows = [[row[idx] for idx in keep_indexes] for row in rows]

            # Renommer les en-têtes vides conservés, puis renuméroter les [SANS_ENTETE_n].
            counter = 1
            for k, h in enumerate(new_header):
                if h.strip() == "" or re.fullmatch(r"\[SANS_ENTETE_\d+\]", h.strip()):
                    new_header[k] = f"[SANS_ENTETE_{counter}]"
                    counter += 1

            if len(keep_indexes) != len(header) or new_header != header:
                out.append(_build_md_row(new_header))
                out.append(_build_md_separator(len(new_header)))
                for row in new_rows:
                    out.append(_build_md_row(row))
                i = j
                continue

        out.append(lines[i])
        i += 1

    return "\n".join(out)


def _looks_like_number_or_amount(value: str) -> bool:
    v = (value or "").strip()
    if not v:
        return False
    return bool(re.fullmatch(
        r"[-+−]?\(?\d[\d\s.,']*\)?\s*(?:€|EUR|USD|CHF|GBP|£|\$)?",
        v,
        flags=re.IGNORECASE,
    ))


def _looks_like_percent(value: str) -> bool:
    v = (value or "").strip()
    return bool(re.fullmatch(r"[-+−]?\d+(?:[,.]\d+)?\s*%", v))


def _fix_percent_header_position_in_tables(md: str) -> str:
    """
    Corrige uniquement les en-têtes dans le cas :
    Px unitaire | Px unitaire remisé | [SANS_ENTETE_1]
    alors que les données montrent : nombre | pourcentage | nombre.

    Les données ne sont jamais déplacées.
    """
    lines = md.splitlines()
    out: List[str] = []
    i = 0

    while i < len(lines):
        if (
            i + 1 < len(lines)
            and _is_md_table_row(lines[i])
            and _is_md_separator_row(lines[i + 1])
        ):
            header = _split_md_cells(lines[i])
            j = i + 2
            rows: List[List[str]] = []

            while j < len(lines) and _is_md_table_row(lines[j]):
                row = _split_md_cells(lines[j])
                if len(row) < len(header):
                    row += [""] * (len(header) - len(row))
                rows.append(row[:len(header)])
                j += 1

            changed = False
            for idx in range(0, max(len(header) - 2, 0)):
                h0 = header[idx].lower()
                h1 = header[idx + 1].lower()
                h2 = header[idx + 2].strip()

                is_bad_header_order = (
                    "unitaire" in h0
                    and "remis" in h1
                    and re.fullmatch(r"\[SANS_ENTETE_\d+\]", h2)
                )
                if not is_bad_header_order:
                    continue

                evidence = 0
                for row in rows:
                    if (
                        _looks_like_number_or_amount(row[idx])
                        and _looks_like_percent(row[idx + 1])
                        and _looks_like_number_or_amount(row[idx + 2])
                    ):
                        evidence += 1

                if evidence >= 1:
                    header[idx + 1], header[idx + 2] = header[idx + 2], header[idx + 1]
                    changed = True

            if changed:
                out.append(_build_md_row(header))
                out.append(_build_md_separator(len(header)))
                for row in rows:
                    out.append(_build_md_row(row))
                i = j
                continue

        out.append(lines[i])
        i += 1

    return "\n".join(out)


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
) -> Tuple[str, Dict[str, int]]:
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

                input_tokens = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
                output_tokens = int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)

                choices = js.get("choices", []) or []
                if not choices:
                    raise RuntimeError(f"{context}: réponse 200 mais aucune choice")

                choice0 = choices[0] or {}
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
                        return _call_chat(
                            api_key=api_key,
                            model=model,
                            messages=messages,
                            max_tokens=max_tokens,
                            context=context + " [final]",
                            enable_thinking=False,
                            allow_no_think_fallback=False,
                        )

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

    # On retente si le modèle renvoie un truc anormalement court
    last_text = ""
    last_stats: Dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    for k in range(0, OCR_EMPTY_RETRIES + 1):
        label = f"OCR page {page_num}" + (f" (retry {k})" if k > 0 else "")
        text, stats = _call_chat(api_key, MODEL_OCR, messages, MAX_TOKENS_OCR, label, enable_thinking=ENABLE_THINKING_OCR)
        text = _strip_triple_backticks(text)
        text = _normalize_sans_entete_tokens(text)
        text = _strip_model_page_tokens(text)

        last_text, last_stats = text, stats

        preview = text.strip().replace("\n", " ")[:120]
        if text.strip() == "[PAGE VIDE]":
            # Acceptable si page vraiment blanche; sinon la suite (Markdown) restera minimale.
            _log(f"⚠️ Page {page_num}: OCR indique [PAGE VIDE].")
            break

        if len(text.strip()) >= OCR_MIN_CHARS:
            break

        _log(
            f"⚠️ Page {page_num}: OCR trop court ({len(text.strip())} chars, out={stats.get('output_tokens', 0)} tokens). "
            f"Preview='{preview}'"
        )

        if k < OCR_EMPTY_RETRIES:
            time.sleep(OCR_EMPTY_RETRY_SLEEP)

    # Libère mémoire (base64 peut être gros)
    try:
        del image_b64
    except Exception:
        pass
    gc.collect()

    if len(last_text.strip()) < OCR_MIN_CHARS and last_text.strip() != "[PAGE VIDE]":
        raise RuntimeError("OCR trop court / vide (suspect)")

    _log(f"✅ Page {page_num}: OCR OK ({last_stats.get('total_tokens', 0)} tokens)")
    return last_text, last_stats


def markdown_from_ocr(api_key: str, ocr_text: str, page_num: int) -> Tuple[str, Dict[str, int]]:
    _log(f"➡️ Page {page_num}: appel Markdown (depuis OCR brut)")

    # Le modèle Markdown ne doit pas voir de token technique [[PAGE n]].
    # La pagination physique est ajoutée uniquement par le code Python.
    ocr_text_for_markdown = _strip_model_page_tokens(ocr_text)

    user_block = (
        f"Voici le texte OCR brut de la page physique {page_num} :\n\n"
        "```text\n"
        f"{ocr_text_for_markdown}\n"
        "```\n\n"
        "Génère uniquement le Markdown structuré pour cette page. "
        "N'inclus PAS la section '## Annexe - OCR brut'. "
        "Ne génère jamais de balise HTML <!-- PAGE n -->."
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

    md, stats = _call_chat(api_key, MODEL_MD, messages, MAX_TOKENS_MD, f"Markdown page {page_num}", enable_thinking=ENABLE_THINKING_MD)
    md = _strip_triple_backticks(md)
    md = _strip_existing_ocr_appendix(md)
    md = _normalize_markdown_tokens(md)
    md = _remove_empty_md_table_rows(md)
    md = _remove_header_only_md_tables(md)
    md = _fix_percent_header_position_in_tables(md)
    md = _drop_empty_unnamed_columns(md)
    md = _remove_empty_md_table_rows(md)
    md = _normalize_markdown_tokens(md)
    md = _strip_model_html_page_markers(md)

    _log(f"✅ Page {page_num}: Markdown OK ({stats.get('total_tokens', 0)} tokens)")
    return md.strip(), stats


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
        "```text\n"
        f"[[PAGE {page_num}]]\n\n"
        f"{ocr_text.rstrip()}\n"
        "```"
    ).strip()

    page_markers = _extract_html_page_markers_outside_fences(page_md)
    if page_markers != [page_num]:
        raise RuntimeError(
            f"Structure Markdown invalide pour la page physique {page_num}: "
            f"marqueurs détectés={page_markers}"
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

    stats_payload: Dict[str, Any] = dict(stats_core)
    stats_payload["stats"] = dict(stats_core)

    # libère mémoire
    try:
        del ocr_text
    except Exception:
        pass
    gc.collect()

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

def validate_markdown_quality(final_markdown: str, page_count: int) -> Dict[str, Any]:
    """
    Compat runner : validation légère, non bloquante.
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
    "MODEL",
    "MODEL_OCR",
    "MODEL_MD",
    "INTER_REQUEST_DELAY",
    "STOP_ON_CRITICAL",
    "RENDER_DPI",
    "get_pdf_info",
    "load_progress",
    "save_progress",
    "clear_progress",
    "process_page_with_cache",
    "calculate_costs",
    "validate_markdown_quality",
    "validate_canonical_markdown_structure",
    "ocr_page_with_vl",
    "markdown_from_ocr",
]





