#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OCR FACTURES PDF -> OCR BRUT (page par page) -> MARKDOWN (page par page)

Changement de methode par rapport a ton script actuel :
1) Etape OCR : on demande au modele vision de renvoyer UNIQUEMENT le texte OCR brut de la page.
2) Etape Markdown : on donne ce texte OCR brut au modele (appel texte) pour produire le Markdown structure.

Avantages concrets :
- tu conserves une "source de verite" (OCR brut) par page
- tu peux reprendre un traitement sans refaire l'OCR
- tu peux ajouter l'annexe OCR brut en sortie sans la faire regenerer par le modele (moins de risques de troncature)

Notes :
- Ce script reste base sur Qwen (OpenAI-compatible) comme ton code d'origine.
- Il genere 2 fichiers :
  * <facture>.ocr.txt  : OCR brut, page par page
  * <facture>.md       : Markdown structure, page par page (+ annexe OCR brut ajoutee par le code)
"""

import base64
import json
import os
import re
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple

import requests
import tkinter as tk
from tkinter import filedialog, messagebox

from pypdf import PdfReader
from pdf2image import convert_from_path  # necessite pdf2image + poppler

# =====================
# Configuration
# =====================

# Endpoint OpenAI-compatible Qwen (region Singapore / International)
API_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

# Modeles
MODEL_OCR = "qwen-vl-max"   # vision (image -> OCR texte)
MODEL_MD = "qwen-vl-max"    # texte (OCR texte -> Markdown)

MODEL = MODEL_OCR

# Tokens
MAX_TOKENS_OCR = 20000
MAX_TOKENS_MD = 20000

TEMPERATURE = 0.0
REQUEST_TIMEOUT = 600
MAX_RETRIES = 5
BACKOFF_BASE = 2
BACKOFF_MAX = 120
INTER_REQUEST_DELAY = 2
STOP_ON_CRITICAL = False

# Qualite image
RENDER_DPI = 300

# =====================
# Prompts
# =====================

# 1) Prompt OCR (vision)
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

# 2) Prompt Markdown (texte) : basee sur ton prompt actuel, mais on retire l'annexe OCR brute
# (elle sera ajoutee automatiquement par le code a partir de l'etape 1).
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

➡️ Sortie finale : uniquement le document Markdown structuré, sans explication."""


# =====================
# Utilitaires
# =====================

def calculate_backoff_delay(attempt: int) -> int:
    return min(BACKOFF_BASE ** attempt, BACKOFF_MAX)


def handle_api_error(error: Exception, attempt: int, context: str) -> Tuple[bool, int]:
    error_str = str(error).lower()

    non_retryable = ["invalid api key", "authentication failed", "permission denied"]
    for non_retry in non_retryable:
        if non_retry in error_str:
            print(f"\n      ❌ Erreur non-récupérable : {error}")
            return False, 0

    if attempt >= MAX_RETRIES:
        print(f"\n      ❌ Échec après {MAX_RETRIES} tentatives")
        return False, 0

    wait_time = calculate_backoff_delay(attempt)

    if "timeout" in error_str:
        print(f"      ⏳ Timeout {context} (tentative {attempt}/{MAX_RETRIES})")
    elif "429" in error_str or "rate limit" in error_str:
        print(f"      🚦 Rate limit (tentative {attempt}/{MAX_RETRIES})")
        wait_time = max(wait_time, 60)
    elif "overloaded" in error_str:
        print(f"      🔥 API surchargée (tentative {attempt}/{MAX_RETRIES})")
        wait_time = max(wait_time, 30)
    else:
        print(f"      ⚠️  Erreur {context} (tentative {attempt}/{MAX_RETRIES}): {error}")

    print(f"      ⏱️  Attente {wait_time}s...")
    return True, wait_time


def choose_file() -> str:
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Sélectionner une facture PDF",
        filetypes=[("PDF", "*.pdf"), ("Tous", "*.*")]
    )
    root.destroy()

    if not path:
        sys.exit("❌ Aucun fichier sélectionné")

    return path


def get_pdf_info(pdf_path: str) -> Dict:
    with open(pdf_path, 'rb') as f:
        reader = PdfReader(f)
        page_count = len(reader.pages)

    file_size = os.path.getsize(pdf_path)

    return {
        "page_count": page_count,
        "file_size_bytes": file_size,
        "file_size_mb": file_size / (1024 * 1024)
    }


def render_single_page_to_base64(pdf_path: str, page_num: int, dpi: int = RENDER_DPI) -> Tuple[str, float]:
    """Rend une page en PNG et retourne base64 + taille KB."""
    images = convert_from_path(
        pdf_path,
        dpi=dpi,
        first_page=page_num,
        last_page=page_num
    )

    if not images:
        raise ValueError(f"Aucune image générée pour la page {page_num}")

    image = images[0]
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    image_bytes = buffer.read()

    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    size_kb = len(image_bytes) / 1024
    return image_base64, size_kb


def extract_text_from_response_content(content) -> str:
    """Supporte content string OU liste (OpenAI-compatible)."""
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
    """Enleve un enveloppement ```...``` si le modele en met un."""
    t = text.strip()
    if t.startswith("```"):
        # supprime la premiere ligne ```xxx
        t = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", t)
        # supprime le dernier ```
        t = re.sub(r"\n?```$", "", t)
    return t.strip("\n")


def remove_empty_md_table_rows(md: str) -> str:
    """Supprime les lignes de tableaux markdown dont toutes les cellules sont vides."""
    out: List[str] = []
    for line in md.splitlines():
        if re.match(r'^\|.*\|\s*$', line):
            cells = [c.strip() for c in line.strip().strip('|').split('|')]
            # garder la ligne separatrice d'en-tete: --- / :---:
            if cells and all(re.fullmatch(r':?-{3,}:?', c) for c in cells):
                out.append(line)
                continue
            if all(c == "" for c in cells):
                continue
        out.append(line)
    return "\n".join(out)


def strip_existing_ocr_appendix(md: str) -> str:
    """Si le modele a quand meme emis '## Annexe - OCR brut', on l'enleve."""
    # coupe a partir du titre "## Annexe - OCR brut" (si present)
    m = re.search(r"^##\s+Annexe\s*-\s*OCR\s+brut\s*$", md, flags=re.IGNORECASE | re.MULTILINE)
    if not m:
        return md.strip()
    return md[:m.start()].rstrip()


def call_chat_completions(
    api_key: str,
    model: str,
    messages: List[Dict],
    max_tokens: int,
    context: str,
    temperature: float = TEMPERATURE,
) -> Tuple[str, Dict]:
    """Appel generique a /chat/completions avec retries + stats tokens."""

    url = f"{API_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    body = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": messages,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(url, headers=headers, json=body, timeout=REQUEST_TIMEOUT)

            if response.status_code == 200:
                json_response = response.json()

                usage = json_response.get("usage", {})
                input_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0))
                output_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0))

                choices = json_response.get("choices", [])
                content = ""
                if choices:
                    message = choices[0].get("message", {})
                    content = message.get("content", "")

                text = extract_text_from_response_content(content).strip()

                stats = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                }
                return text, stats

            # Erreur HTTP
            error_msg = f"HTTP {response.status_code}"
            try:
                error_detail = response.json()
                if isinstance(error_detail, dict):
                    err = error_detail.get("error", {})
                    msg = err.get("message") or str(error_detail)
                    error_msg += f": {msg[:200]}"
                else:
                    error_msg += f": {str(error_detail)[:200]}"
            except Exception:
                error_msg += f": {response.text[:200]}"

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
# Etapes : OCR puis Markdown
# =====================

def ocr_page_with_vl(api_key: str, pdf_path: str, page_num: int) -> Tuple[str, Dict]:
    print(f"      🔎 OCR (vision) page {page_num}")

    image_base64, size_kb = render_single_page_to_base64(pdf_path, page_num)
    print(f"         📦 Image : {size_kb:.1f} KB")

    data_url = f"data:image/png;base64,{image_base64}"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": OCR_PROMPT},
                {"type": "image_url", "image_url": {"url": data_url}},
                {
                    "type": "text",
                    "text": (
                        f"OCR de la page {page_num}. "
                        "Retourne uniquement le texte OCR brut, avec les sauts de ligne."
                    ),
                },
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

    # Petit garde-fou : si le modele renvoie quasi rien, on considere que c'est suspect.
    if len(text.strip()) < 20:
        raise Exception("OCR trop court / vide (suspect)")

    return text, stats


def markdown_from_ocr(api_key: str, ocr_text: str, page_num: int) -> Tuple[str, Dict]:
    print(f"      🧩 Markdown (texte) page {page_num}")

    # On donne l'OCR dans un bloc text pour figer les retours a la ligne
    user_ocr_block = (
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
                {"type": "text", "text": user_ocr_block},
            ],
        }
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
# Progression / resume
# =====================

def progress_path_for(pdf_path: str) -> Path:
    return Path(pdf_path).with_suffix(".progress.json")


def save_progress(pdf_path: str, completed_pages: Dict):
    p = progress_path_for(pdf_path)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(completed_pages, f, indent=2, ensure_ascii=False)


def load_progress(pdf_path: str) -> Dict:
    p = progress_path_for(pdf_path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# =====================
# Cout / validation (optionnel)
# =====================

def calculate_costs(stats_list: List[Dict]) -> Dict:
    """Calcule un cout estime.

    ATTENTION : a ajuster selon la grille tarifaire reelle de ton endpoint / modele.
    Ici on laisse les memes variables que ton script d'origine, mais c'est indicatif.
    """

    total_input = sum(s.get("input_tokens", 0) for s in stats_list)
    total_output = sum(s.get("output_tokens", 0) for s in stats_list)
    total_tokens = total_input + total_output

    # A AJUSTER
    PRICE_INPUT = 0.20
    PRICE_OUTPUT = 1.60

    cost_input = (total_input * PRICE_INPUT) / 1_000_000
    cost_output = (total_output * PRICE_OUTPUT) / 1_000_000
    total_cost = cost_input + cost_output

    cost_per_call = total_cost / max(len(stats_list), 1)

    return {
        "total_input": total_input,
        "total_output": total_output,
        "total_tokens": total_tokens,
        "cost_input": cost_input,
        "cost_output": cost_output,
        "cost_total": total_cost,
        "cost_per_call": cost_per_call,
    }


def validate_markdown_quality(markdown: str, expected_pages: int) -> Dict:
    issues = {"critical": [], "warnings": [], "stats": {}}

    page_markers = re.findall(r'<!-- PAGE (\d+) -->', markdown)
    page_numbers = [int(p) for p in page_markers]

    if len(page_numbers) != expected_pages:
        issues["critical"].append(f"❌ Pages : {len(page_numbers)}/{expected_pages}")

    champ_manquant = len(re.findall(r'\[CHAMP MANQUANT\]', markdown, re.IGNORECASE))
    amounts = re.findall(r'\d{1,3}(?:[ \.]?\d{3})*,\d{2}\s*€', markdown)
    table_count = len(re.findall(r'\|.*\|.*\|', markdown))

    if champ_manquant > 0:
        issues["warnings"].append(f"⚠️  {champ_manquant} champ(s) manquant(s)")

    issues["stats"]["champs_manquants"] = champ_manquant
    issues["stats"]["montants_detectes"] = len(amounts)
    issues["stats"]["lignes_tableaux"] = table_count
    issues["stats"]["caracteres"] = len(markdown)

    return issues


# =====================
# Main
# =====================

def main():
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        messagebox.showerror("Erreur Configuration", "Variable DASHSCOPE_API_KEY non définie.")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("🔬 EXTRACTION FACTURES PDF -> OCR -> MARKDOWN (2 ETAPES)")
    print("=" * 70)
    print(f"💰 Modèles : OCR={MODEL_OCR} | MD={MODEL_MD}")
    print(f"🖼️  Rendu pages : {RENDER_DPI} dpi")
    print(f"🔁 Requêtes : 2 appels/page (OCR + Markdown)")
    print("=" * 70)

    try:
        pdf_path = choose_file()
        print(f"\n📄 Fichier : {Path(pdf_path).name}")

        pdf_info = get_pdf_info(pdf_path)
        page_count = pdf_info["page_count"]
        print(f"📊 Pages : {page_count}")
        print(f"💾 Taille : {pdf_info['file_size_mb']:.2f} MB")

        completed_pages = load_progress(pdf_path)
        if completed_pages:
            print(f"\n📂 Reprise : {len(completed_pages)} page(s) partiellement/totalement traitées")
            response = messagebox.askyesno(
                "Reprise détectée",
                f"{len(completed_pages)} page(s) déjà en cache (.progress.json).\n\nReprendre ?"
            )
            if not response:
                completed_pages = {}

        print("\n" + "=" * 70)
        print("🚀 DÉBUT DU TRAITEMENT")
        print("=" * 70 + "\n")

        start_time = time.time()

        all_page_markdown: List[str] = []
        all_page_ocr: List[str] = []
        all_stats_calls: List[Dict] = []  # une entree par appel API

        for page_num in range(1, page_count + 1):
            page_key = str(page_num)

            # Delai entre pages
            if page_num > 1 and INTER_REQUEST_DELAY > 0:
                time.sleep(INTER_REQUEST_DELAY)

            # Cache
            cached = completed_pages.get(page_key, {})
            ocr_text = cached.get("ocr_text")
            page_md = cached.get("page_markdown")

            try:
                print(f"      📄 Page {page_num}")

                # 1) OCR si absent
                if not ocr_text:
                    ocr_text, ocr_stats = ocr_page_with_vl(api_key, pdf_path, page_num)
                    all_stats_calls.append(ocr_stats)

                    completed_pages.setdefault(page_key, {})
                    completed_pages[page_key]["ocr_text"] = ocr_text
                    completed_pages[page_key]["ocr_stats"] = ocr_stats

                    # Sauvegarde immediate apres OCR (pour eviter de reperdre des tokens)
                    save_progress(pdf_path, completed_pages)
                else:
                    print(f"      🔎 OCR (vision) page {page_num} : ✓ cache")
                    ocr_stats = cached.get("ocr_stats", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})

                # Ajout OCR page par page (fichier .ocr.txt)
                ocr_block = f"<!-- PAGE {page_num} -->\n\n{ocr_text.strip()}\n\n---"

                # 2) Markdown si absent
                if not page_md:
                    md_core, md_stats = markdown_from_ocr(api_key, ocr_text, page_num)
                    all_stats_calls.append(md_stats)

                    # Ajoute l'annexe OCR BRUT par le code (0 token)
                    full_md = (
                        f"<!-- PAGE {page_num} -->\n\n"
                        f"{md_core.strip()}\n\n"
                        "## Annexe - OCR brut\n"
                        "```text\n"
                        f"{ocr_text.rstrip()}\n"
                        "```\n\n"
                        "---"
                    )

                    page_md = full_md

                    completed_pages.setdefault(page_key, {})
                    completed_pages[page_key]["page_markdown"] = page_md
                    completed_pages[page_key]["md_stats"] = md_stats

                    # Sauvegarde toutes les pages (robuste)
                    save_progress(pdf_path, completed_pages)
                else:
                    print(f"      🧩 Markdown (texte) page {page_num} : ✓ cache")
                    md_stats = cached.get("md_stats", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})

                # Logs tokens
                print(
                    "         📊 Tokens : "
                    f"OCR IN={ocr_stats.get('input_tokens', 0):,} OUT={ocr_stats.get('output_tokens', 0):,} | "
                    f"MD IN={md_stats.get('input_tokens', 0):,} OUT={md_stats.get('output_tokens', 0):,}"
                )

                all_page_ocr.append(ocr_block)
                all_page_markdown.append(page_md)

                print(f"         ✅ Page {page_num} terminée\n")

            except Exception as e:
                print(f"\n         ❌ Erreur page {page_num}: {e}")

                if STOP_ON_CRITICAL:
                    raise

                # Marqueur d'erreur dans les sorties
                all_page_ocr.append(f"<!-- PAGE {page_num} -->\n\n[ERREUR OCR] {e}\n\n---")
                all_page_markdown.append(f"<!-- PAGE {page_num} -->\n\n**[ERREUR EXTRACTION]** {e}\n\n---")

                print("         ⚠️  Marquée comme erreur, continuation...\n")

        duration = time.time() - start_time

        # Fusion
        final_ocr = "\n\n".join(all_page_ocr)
        final_markdown = "\n\n".join(all_page_markdown)

        # Sauvegarde
        md_path = Path(pdf_path).with_suffix(".md")
        ocr_path = Path(pdf_path).with_suffix(".ocr.txt")

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(final_markdown)

        with open(ocr_path, "w", encoding="utf-8") as f:
            f.write(final_ocr)

        # Validation
        validation = validate_markdown_quality(final_markdown, page_count)

        # Nettoyage progression
        p = progress_path_for(pdf_path)
        if p.exists():
            p.unlink()

        # Stats
        md_size_kb = len(final_markdown.encode("utf-8")) / 1024
        ocr_size_kb = len(final_ocr.encode("utf-8")) / 1024
        costs = calculate_costs(all_stats_calls)

        print("\n" + "=" * 70)
        print("✅ EXTRACTION TERMINÉE")
        print("=" * 70)
        print(f"📝 Markdown : {md_path.name} ({md_size_kb:.1f} KB)")
        print(f"🧾 OCR brut : {ocr_path.name} ({ocr_size_kb:.1f} KB)")
        print(f"📄 Pages    : {page_count}")
        print(f"⏱️  Durée    : {duration//60:.0f}min {duration%60:.0f}s")

        print("\n" + "-" * 70)
        print("💰 TOKENS (somme des appels)")
        print("-" * 70)
        print(f"📥 Input tokens  : {costs['total_input']:,}")
        print(f"📤 Output tokens : {costs['total_output']:,}")
        print(f"📊 Total tokens  : {costs['total_tokens']:,}")
        print(f"💵 Coût estimé   : ${costs['cost_total']:.4f} (indicatif)")

        print("\n" + "-" * 70)
        print("🔍 QUALITÉ")
        print("-" * 70)
        if not validation["critical"] and not validation["warnings"]:
            print("✅ Aucun problème détecté")
        elif not validation["critical"]:
            print(f"✅ Extraction avec {len(validation['warnings'])} avertissement(s)")
        else:
            print(f"⚠️  {len(validation['critical'])} problème(s) critique(s) détecté(s)")

        if validation["critical"]:
            for c in validation["critical"]:
                print("   ", c)
        if validation["warnings"]:
            for w in validation["warnings"]:
                print("   ", w)

        msg = (
            f"✅ Extraction terminée !\n\n"
            f"📝 Markdown : {md_path.name}\n"
            f"🧾 OCR brut : {ocr_path.name}\n"
            f"📄 Pages : {page_count}\n"
            f"⏱️  {duration//60:.0f}min {duration%60:.0f}s\n\n"
            f"Tokens total : {costs['total_tokens']:,}\n"
            f"Coût estimé (indicatif) : ${costs['cost_total']:.4f}"
        )
        messagebox.showinfo("✅ Extraction Terminée", msg)

    except KeyboardInterrupt:
        print("\n\n⚠️  Interruption (Ctrl+C)")
        print("💾 Progression sauvegardée (fichier .progress.json)\n")
        sys.exit(0)

    except Exception as e:
        print(f"\n❌ Erreur fatale : {e}")
        import traceback

        traceback.print_exc()
        messagebox.showerror("Erreur Fatale", str(e)[:300])
        sys.exit(1)


if __name__ == "__main__":
    main()
