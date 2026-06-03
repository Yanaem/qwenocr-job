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
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

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

MODEL_OCR = os.getenv("QWEN_MODEL_OCR", "qwen3.6-plus")
MODEL_MD = os.getenv("QWEN_MODEL_MD", "qwen3.6-plus")

# Attendu par le runner (affiché au démarrage)
MODEL = MODEL_OCR

# Attendu par le runner : pause entre pages
INTER_REQUEST_DELAY = _env_float("INTER_REQUEST_DELAY", 1.0)

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
# Prompts
# =====================

OCR_PROMPT = """Tu es un moteur OCR layout-aware spécialisé en factures.

Tâche : transcrire TOUT le texte visible d'une page de facture en conservant le layout utile.

Sortie : texte OCR structuré uniquement.
Interdiction : Markdown, JSON, explication, commentaire, bloc ```.

Chaque appel traite une seule page.

Tokens autorisés :
[[PAGE n]]
[[BLOCK position]]
[[/BLOCK]]
[[TABLE position cols=N]]
[[/TABLE]]
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

Règles générales :
- Commence toujours par [[PAGE n]].
- Si le numéro de page visible est connu, utilise ce numéro ; sinon utilise [[PAGE 1]].
- Le token [[PAGE n]] ne remplace pas le texte visible "Page : n" : si ce texte est visible, transcris-le aussi dans un bloc.
- Copie uniquement le texte visible.
- Conserve exactement lettres, chiffres, dates, montants, séparateurs, virgules, points, %, €, devises, majuscules, abréviations.
- Ne corrige pas.
- Ne reformule pas.
- Ne normalise pas.
- Ne calcule pas.
- Ne complète aucune information absente.
- N'ajoute aucun libellé, montant, symbole, devise, champ ou total absent de l'image.
- Transcris tout texte lisible : fournisseur, client, adresses, références, articles, prestations, taxes, totaux, échéances, banque, RIB, IBAN, BIC, conditions, mentions légales, pied de page, annotations, tampons, texte lisible dans logo.
- Ne transcris pas le contenu encodé d'un QR code ou d'un code-barres.
- Transcris seulement le texte imprimé lisible autour d'un QR code ou code-barres.
- Ignore uniquement les éléments purement graphiques sans texte lisible.
- Si une portion est illisible : écris [ILLISIBLE] à l'endroit concerné.
- Si la page est réellement vide : réponds exactement [PAGE VIDE].
- Un même texte visible ne doit apparaître qu'une seule fois, sauf s'il est répété visuellement.
- Ne fusionne jamais un logo, tampon, QR code, bloc marketing ou slogan avec le bloc client, même s'ils sont proches ou dans la même zone horizontale/verticale.
- Le bloc client doit contenir uniquement le destinataire/facturé/livré à et son adresse.
- Tout texte isolé près du client mais sans lien explicite avec le destinataire doit rester dans un [[BLOCK]] séparé.

Lecture layout :
- Lis par blocs visuels, pas par bande horizontale globale.
- Ordre des blocs : haut vers bas.
- À hauteur proche : gauche vers droite.
- Ne traverse jamais toute la page de gauche à droite si cela fusionne deux zones distinctes.
- Deux zones côte à côte restent deux blocs séparés si elles n'appartiennent pas à la même grille.
- Deux tableaux côte à côte restent deux [[TABLE]] séparés.
- Deux tableaux empilés mais séparés par bordure, espace, titre ou groupe d'en-têtes distinct restent deux [[TABLE]] séparés.
- Si une zone est ambiguë, utilise [[BLOCK]] ligne par ligne au lieu de fabriquer un tableau.
- Un titre situé au-dessus d'un tableau doit rester dans un [[BLOCK]] séparé, sauf s'il est clairement une cellule du tableau.

Blocs :
- Un [[BLOCK]] contient du texte non tabulaire.
- Chaque bloc commence par [[BLOCK position]] et finit par [[/BLOCK]].
- N'utilise jamais <TAB> dans un [[BLOCK]].
- Si deux textes sont côte à côte mais ne forment pas une vraie grille, crée deux [[BLOCK]] séparés.
- Les adresses, mentions légales, notes, conditions et textes libres restent en [[BLOCK]].
- Les blocs de paiement sans vraie grille doivent rester en [[BLOCK]], pas en [[TABLE]].
- Dans un [[BLOCK]], conserve les retours à la ligne utiles.
- N'utilise <BR> que dans les tableaux, jamais dans les blocs.

Tableaux — détection :
- Chaque tableau visible commence par [[TABLE position cols=N]] et finit par [[/TABLE]].
- N est obligatoire.
- N correspond au nombre réel de colonnes visuelles du tableau.
- Utilise <TAB> uniquement dans [[TABLE]].
- Un tableau = une grille continue OU un seul groupe logique d'en-têtes.
- Ne fusionne jamais deux groupes d'en-têtes indépendants dans une même [[TABLE]].
- Si deux zones ont des en-têtes, bordures, alignements ou espacements distincts, elles forment deux tableaux.
- Si l'alignement ne permet pas de garantir les colonnes, ferme le tableau et transcris la zone en [[BLOCK]].

Tableaux — cellules :
- Une ligne OCR = une ligne logique du tableau.
- Une cellule OCR = une cellule visuelle.
- Chaque ligne d'un tableau doit contenir exactement N cellules, donc exactement N-1 tokens <TAB>.
- Ne fusionne jamais deux cellules adjacentes.
- Ne divise jamais une cellule à cause d'espaces internes ordinaires.
- Détermine N avec toutes les colonnes réellement alignées : en-têtes visibles, lignes de données, totaux internes, codes, montants, taux, quantités.
- Ne détermine jamais N uniquement avec les libellés visibles de l'en-tête.
- Si les lignes de données ont plus de colonnes que les en-têtes visibles, ajoute [SANS_ENTETE_n] dans l'en-tête à la position exacte des colonnes sans libellé.
- n recommence à 1 dans chaque tableau.
- N'ajoute jamais une colonne vide sans nom en fin d'en-tête.
- N'invente jamais un nom de colonne à partir du contenu des valeurs.
- Si aucune ligne d'en-tête n'est visible, crée une ligne d'en-tête avec [SANS_ENTETE_1], [SANS_ENTETE_2], etc.
- Si une cellule réelle est vide dans une ligne réelle, utilise <EMPTY>.
- Si une cellule vide est en fin de ligne, écris quand même <EMPTY> pour conserver N cellules.
- Ne laisse jamais une cellule vide implicite.

Tableaux — en-têtes et retours ligne :
- Garde les en-têtes visibles exacts.
- Si un en-tête est écrit sur plusieurs lignes dans la même cellule, réunis les lignes avec <BR>.
- Si une désignation ou description continue sur plusieurs lignes dans la même cellule, réunis les lignes avec <BR>.
- Si une ligne contient seulement une continuation de texte dans une colonne et aucun autre champ significatif, rattache ce texte à la cellule correspondante de la ligne précédente avec <BR>.
- Si le rattachement est incertain, conserve une ligne réelle avec <EMPTY> dans les autres cellules, mais ne crée pas de ligne vide.

Tableaux — nombres, taux, montants, codes :
- Si plusieurs valeurs courtes sont alignées en colonnes distinctes, elles doivent être séparées par <TAB>.
- Les nombres, montants, pourcentages, quantités, codes taxe, références et totaux alignés verticalement sont des cellules distinctes.
- Ne fusionne jamais un nombre et un pourcentage s'ils sont visuellement séparés ou répétés à la même position sur plusieurs lignes.
- Ne fusionne jamais un montant et un code taxe s'ils sont visuellement séparés ou répétés à la même position sur plusieurs lignes.
- Exemple général : si "12,50", "0%" et "12,50" sont trois valeurs alignées en colonnes, transcris :
  12,50<TAB>0%<TAB>12,50
- Exemple général : si "100,00", "20%" et "120,00" sont trois valeurs alignées en colonnes, transcris :
  100,00<TAB>20%<TAB>120,00
- Une colonne contenant uniquement des pourcentages sans en-tête visible doit avoir [SANS_ENTETE_n] dans l'en-tête.
- Une colonne contenant uniquement des codes sans en-tête visible doit avoir [SANS_ENTETE_n] dans l'en-tête.
- Ne remplace jamais [SANS_ENTETE_n] par "Remise", "TVA", "Code", "Taxe" ou autre libellé non visible.

Tableaux — anti-padding :
- Ne crée jamais de ligne entièrement vide.
- Ne crée jamais de ligne composée uniquement de <EMPTY>.
- Ne crée jamais de lignes pour reproduire l'espace blanc d'un tableau haut.
- Le tableau des articles contient seulement les lignes réelles d'articles ou prestations.
- Une grande zone vide sous les articles ne doit produire aucune ligne OCR.
- Si une valeur isolée apparaît dans une zone vide du tableau sans former une ligne complète, ferme le tableau et transcris cette valeur dans un [[BLOCK position]] séparé.

Règles factures :
- Les zones articles, prestations, taxes, remises, acomptes, totaux, échéances, paiements et mentions peuvent être des tableaux ou des blocs séparés.
- Ne suppose jamais qu'un total, une taxe, un acompte ou un solde appartient au tableau voisin.
- Un montant reste dans le bloc ou tableau où il est visuellement placé.
- Un montant sous un en-tête de taxe reste dans le tableau de taxe.
- Un montant sous un en-tête de total reste dans le tableau de total.
- NET A PAYER, TOTAL A PAYER, SOLDE, AMOUNT DUE ou équivalent doit rester dans son bloc visuel d'origine.
- Ne mélange jamais un tableau de taxes avec un tableau de totaux s'ils ont des en-têtes, bordures, alignements ou espacements distincts.
- Ne place jamais un montant de taxe dans une colonne de total à payer.
- Ne place jamais un total à payer dans une colonne de taxe.
- Ne déplace jamais un montant d'un tableau vers un autre pour compléter une ligne.

Contrôle final silencieux avant sortie :
- Tous les textes visibles utiles sont présents.
- Aucun texte visible n'est dupliqué sans duplication visuelle.
- Aucun <TAB> n'apparaît hors d'un [[TABLE]].
- Aucun <BR> n'apparaît hors d'un [[TABLE]].
- Chaque [[BLOCK]] est fermé par [[/BLOCK]].
- Chaque [[TABLE]] est fermé par [[/TABLE]].
- Chaque [[TABLE position cols=N]] a exactement N cellules par ligne.
- Chaque ligne de tableau contient exactement N-1 tokens <TAB>.
- Aucun tableau ne contient de ligne vide de padding.
- Aucun tableau ne contient deux groupes d'en-têtes indépendants.
- Aucun tableau côte à côte n'a été fusionné.
- Aucune colonne réelle sans en-tête n'a été supprimée.
- Aucune colonne réelle sans en-tête n'a reçu un nom inventé : elle doit être [SANS_ENTETE_n].
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

⚠️ Règles supplémentaires Markdown :
- Ne transforme jamais <EMPTY> en [CHAMP MANQUANT]. Dans un tableau Markdown, <EMPTY> devient une cellule vide.
- Ne crée pas [CHAMP MANQUANT] si l'OCR ne le contient pas.
- Chaque [[TABLE]] OCR devient un tableau Markdown séparé.
- Ne fusionne jamais deux [[TABLE]] OCR.
- Si une table OCR contient [SANS_ENTETE_n], conserve exactement ce nom de colonne.
- Si un [[BLOCK]] contient plusieurs lignes avec une structure visuelle, rends-le en texte simple sauf si l'OCR l'a explicitement marqué [[TABLE]].
- Ne transforme pas un bloc paiement en tableau Markdown sauf s'il vient d'un [[TABLE]].
- Si une valeur isolée est dans un [[BLOCK]], conserve-la dans la section appropriée ou dans “Textes non classés”. Ne la supprime pas.

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

    md, stats = _call_chat(api_key, MODEL_MD, messages, MAX_TOKENS_MD, f"Markdown page {page_num}", enable_thinking=ENABLE_THINKING_MD)
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

    stats_payload: Dict[str, Any] = dict(stats_core)
    stats_payload["stats"] = dict(stats_core)

    # libère mémoire
    try:
        del ocr_text
    except Exception:
        pass
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
    "ocr_page_with_vl",
    "markdown_from_ocr",
]

