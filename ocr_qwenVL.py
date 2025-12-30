#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OCR FACTURES PDF → MARKDOWN
Stratégie : Page par page + Qwen3-VL (API OpenAI-compatible)
"""

import os
import sys
import re
import requests
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
from pypdf import PdfReader
from io import BytesIO
import base64
import time
import json
from typing import Tuple, Dict, List

from pdf2image import convert_from_path  # nécessite pdf2image + poppler

# ====== Configuration ======
# Endpoint OpenAI-compatible Qwen (région Singapore).
# Si tu es en région Beijing, remplace par :
#   https://dashscope.aliyuncs.com/compatible-mode/v1
API_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

# Modèle de vision Qwen3-VL (tu peux mettre un autre Qwen3-VL si besoin)
MODEL = "qwen-vl-max"

# Nombre max de tokens de sortie par page
MAX_TOKENS = 20000

REQUEST_TIMEOUT = 600
MAX_RETRIES = 5
BACKOFF_BASE = 2
BACKOFF_MAX = 120
INTER_REQUEST_DELAY = 2
STOP_ON_CRITICAL = False

# ====== Prompt Système ======
SYSTEM_PROMPT = """Vous êtes un assistant spécialisé dans le traitement de documents comptables. Votre tâche est de convertir un texte brut issu d’un OCR d’une facture PDF (en français) en un document Markdown **strictement fidèle** au contenu original, sans aucune modification ni interprétation.

⚠️ Règles absolues :
- Ne jamais deviner ou supposer l’identité des parties.
- L’entreprise située en haut à gauche ou au début du texte est **le fournisseur** (émetteur de la facture).
- L’entreprise située en haut à droite ou en dessous du **fournisseur** est **le client** (recepteur de la facture). Si non présent, indiquez [CHAMP MANQUANT].
- Ne jamais remplacer un champ manquant par une hypothèse.
- Respectez **exactement** les libellés, dates, montants, unités, abréviations, majuscules, tirets, espaces, symboles (€, %, etc.).
- Ne reformulez **aucun mot** : copiez tel quel, même si le texte contient des fautes d’OCR ou des annotations manuscrites.
- Conservez les **structures visuelles** : tableaux, colonnes, lignes, séparateurs, barres verticales, valeurs alignées, etc.
- Ne fusionnez jamais des colonnes ni ne réorganisez les données.
- Utilisez `[CHAMP MANQUANT]` uniquement si une information attendue est illisible ou absente.

⚠️ Règles critiques sur les MONTANTS (priorité maximale) :
- Tout ce qui ressemble à un montant (chiffres avec virgule/point, espaces de milliers, signe -, parenthèses, symbole ou code devise comme €, EUR, etc.) doit être recopié **tel quel** (mêmes séparateurs, mêmes espaces, mêmes symboles). Ne jamais normaliser.
- Ne jamais supprimer, résumer, regrouper, dédupliquer ou “corriger” des montants, même si le même montant apparaît plusieurs fois : recopiez chaque occurrence là où elle apparaît.
- Si un tableau de récapitulatif (ex : TVA / taxes / codes / bases / HT / TVA / TTC) contient des lignes avec des cellules vides (ex : taux non renseigné), ces lignes doivent être reproduites **quand même** : ne pas les omettre.
- Si une cellule est réellement vide dans l’OCR, laissez-la vide. N’écrivez pas `[CHAMP MANQUANT]` à la place d’une cellule vide, sauf si l’OCR indique qu’une valeur est présente mais illisible.
- Ne jamais déduire un taux “0%” ou une taxe “0” si ce n’est pas explicitement écrit : recopiez uniquement ce qui est imprimé/OCRisé.
- Contrôle interne obligatoire (ne pas afficher) : avant de rendre la sortie, vérifiez que tous les montants du tableau des lignes + tous les montants de totaux (HT/TVA/TTC/Net à payer/Remises/Acomptes/Frais/Escompte, etc.) présents dans l’OCR apparaissent bien dans votre Markdown. Si un bloc de montants est difficile à classer, recopiez-le intégralement dans “## Montants Récapitulatifs” ou “## Mentions Légales et Notes Complémentaires” plutôt que de risquer de perdre un montant.

Structure de sortie (Markdown uniquement, sans commentaire) :

## Informations Émetteur (Fournisseur)
[Données exactes telles qu’elles apparaissent dans le texte]

## Informations Client
[Données du destinataire ou [CHAMP MANQUANT]]

## Détails de la Facture
- Numéro de facture : ...
- Date d'émission : ...
- Date de livraison / prestation : ...
- Référence client/commande : ...
- Autres éléments précisés (compte client, numéro de devis, etc.)

## Tableau des Lignes de Facturation
Reproduisez fidèlement le tableau original avec toutes ses colonnes, dans l'ordre exact où elles apparaissent dans le texte OCR.
Ne supprimez aucune ligne, y compris les lignes de sous-total/total, même si certaines cellules sont vides.
Recopiez **tous les montants** (prix unitaires, remises, montants HT, TVA, TTC, etc.) tels quels.

Utilisez la syntaxe Markdown standard :

| COLONNE_1 | COLONNE_2 | COLONNE_3 | ... |
|----------|----------|----------|-----|
| valeur1  | valeur2  | valeur3  | ... |

> 📌 Exemple typique :
> | RÉFÉRENCE | DÉSIGNATION | QUANTITÉ | PRIX UNITAIRE | TOTAL HT |
> |-----------|-------------|----------|----------------|----------|
> | 350110    | SAINT JUDE 1L5 | 6,000   | 0,31           | 1,86     |

Si certaines cellules sont mal lisibles ou barrées, conservez `[CHAMP MANQUANT]` ou indiquez `[CORRECTION MANUELLE]` **dans la cellule concernée**, sans modifier le montant lu.

## Montants Récapitulatifs
Reprenez ici **tous** les blocs de totaux et récapitulatifs présents après le tableau (ou ailleurs sur la page si c’est là que les totaux sont imprimés).
⚠️ Ne transformez pas un tableau en liste, et ne transformez pas une liste en tableau : gardez la forme d’origine.
Recopiez toutes les lignes/colonnes de récapitulatif (HT/TVA/TTC/Net à payer, bases par taux, codes, etc.), y compris celles avec des cellules vides.
Recopiez aussi tout montant isolé de paiement (ex : “Net à payer”, “Solde”, “Montant dû”, “Montant payé”, etc.) même s’il est hors du bloc principal.

## Informations de Paiement
- Modalités : ...
- Paiements effectués (espèces, carte, virement, etc.) : ...
- Conditions de paiement (ex: « payable comptant ») : ...
- Coordonnées bancaires (IBAN, BIC, etc.) si présentes
⚠️ Si des montants apparaissent dans cette zone (ex : montant payé, rendu monnaie, acompte, solde), recopiez-les tels quels.

## Mentions Légales et Notes Complémentaires
Copiez ici **toutes les informations supplémentaires** qui ne rentrent pas dans les sections précédentes :
- Capital social, RCS, SIRET, NAF, TVA intracommunautaire
- Agréments, clauses légales, conditions générales, pénalités de retard
- Mention de TVA exonérée, récupérable, etc.
- Chaque phrase sur une ligne distincte.
⚠️ Si des montants apparaissent dans les mentions (pénalités, indemnités, escompte, frais, seuils, etc.), recopiez-les tels quels.

➡️ Sortie finale : **Uniquement le document Markdown structuré**, sans explication, sans introduction, sans conclusion."""

def calculate_backoff_delay(attempt: int) -> int:
    """Backoff exponentiel"""
    return min(BACKOFF_BASE ** attempt, BACKOFF_MAX)


def handle_api_error(error: Exception, attempt: int, context: str) -> Tuple[bool, int]:
    """Gestion erreurs avec backoff"""
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
    """Sélection du fichier PDF"""
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
    """Récupère les infos du PDF"""
    with open(pdf_path, 'rb') as f:
        reader = PdfReader(f)
        page_count = len(reader.pages)

    file_size = os.path.getsize(pdf_path)

    return {
        "page_count": page_count,
        "file_size_bytes": file_size,
        "file_size_mb": file_size / (1024 * 1024)
    }


def extract_single_page_to_base64(pdf_path: str, page_num: int) -> Tuple[str, int]:
    """
    Extrait UNE page du PDF, la rend en PNG et la convertit en base64.

    Qwen limite la taille d'un fichier image local passé en data: URL
    à ~10 Mo après encodage base64. Si une page est énorme, réduire le DPI.

    Returns:
        (image_base64, size_kb)
    """
    # On rend uniquement la page demandée
    images = convert_from_path(
        pdf_path,
        dpi=300,               # 300 dpi suffit pour une facture
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


def process_page_with_cache(
    pdf_path: str,
    page_num: int,
    api_key: str,
    is_first_page: bool = False
) -> Tuple[str, Dict]:
    """
    Traite UNE page via Qwen3-VL (vision) en OpenAI-compatible.

    - Envoie la page rendue en image PNG base64
    - Injecte le prompt de structuration Markdown dans le message user
    - Retourne le Markdown + stats de tokens

    (Le nom de fonction conserve 'cache' mais Qwen n'expose pas ici de cache serveur.)
    """

    print(f"      📄 Page {page_num}")

    # Extraire la page en image base64
    print(f"         📦 Extraction image...", end=" ")
    image_base64, size_kb = extract_single_page_to_base64(pdf_path, page_num)
    print(f"{size_kb:.1f} KB")

    url = f"{API_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Data URL pour Qwen (OpenAI-compatible)
    data_url = f"data:image/png;base64,{image_base64}"

    # Les docs Qwen recommandent de mettre les consignes dans le message 'user'
    body = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                    {
                        "type": "text",
                        "text": (
                            f"Page {page_num} d'une facture PDF. "
                            "Applique STRICTEMENT les consignes ci-dessus et "
                            "renvoie UNIQUEMENT le Markdown structuré pour cette page."
                        ),
                    },
                ],
            }
        ],
    }

    print(f"         🔄 Traitement OCR...", end=" ")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                url,
                headers=headers,
                json=body,
                timeout=REQUEST_TIMEOUT
            )

            if response.status_code == 200:
                json_response = response.json()

                usage = json_response.get("usage", {})
                # OpenAI-compatible : prompt_tokens / completion_tokens
                input_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0))
                output_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0))

                print("✅")
                print(f"         📊 Tokens : IN={input_tokens:,} | OUT={output_tokens:,}")

                # Extraction du contenu renvoyé
                choices = json_response.get("choices", [])
                content = ""
                if choices:
                    message = choices[0].get("message", {})
                    content = message.get("content", "")

                if isinstance(content, str):
                    markdown_core = content.strip()
                elif isinstance(content, list):
                    parts = []
                    for part in content:
                        if isinstance(part, dict):
                            txt = part.get("text")
                            if txt:
                                parts.append(txt)
                    markdown_core = "\n\n".join(parts).strip()
                else:
                    markdown_core = str(content).strip()

                # Ajouter marqueur de page
                markdown = f"<!-- PAGE {page_num} -->\n\n{markdown_core}\n\n---"

                stats = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                }

                return markdown, stats

            # Gestion erreurs HTTP
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

            should_retry, wait_time = handle_api_error(
                Exception(error_msg),
                attempt,
                f"page {page_num}"
            )

            if not should_retry:
                raise Exception(error_msg)

            time.sleep(wait_time)

        except requests.exceptions.Timeout as e:
            should_retry, wait_time = handle_api_error(e, attempt, f"page {page_num} timeout")
            if not should_retry:
                raise
            time.sleep(wait_time)

        except requests.exceptions.RequestException as e:
            should_retry, wait_time = handle_api_error(e, attempt, f"page {page_num} réseau")
            if not should_retry:
                raise
            time.sleep(wait_time)

    raise Exception(f"Échec page {page_num} après {MAX_RETRIES} tentatives")


def save_progress(pdf_path: str, completed_pages: Dict):
    """Sauvegarde progression"""
    progress_file = Path(pdf_path).with_suffix(".progress.json")
    with open(progress_file, "w", encoding="utf-8") as f:
        json.dump(completed_pages, f, indent=2, ensure_ascii=False)


def load_progress(pdf_path: str) -> Dict:
    """Charge progression"""
    progress_file = Path(pdf_path).with_suffix(".progress.json")
    if progress_file.exists():
        with open(progress_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def calculate_costs(stats_list: List[Dict]) -> Dict:
    """
    Calcule les coûts estimés pour Qwen3-VL-Plus (région International / Singapour).

    Référence (0 < tokens ≤ 32K) :
      - Input  : 0.20 $ / million de tokens
      - Output : 1.60 $ / million de tokens
    Voir la doc Model Studio pour les autres régions. :contentReference[oaicite:5]{index=5}
    """

    total_input = sum(s.get("input_tokens", 0) for s in stats_list)
    total_output = sum(s.get("output_tokens", 0) for s in stats_list)
    total_tokens = total_input + total_output

    PRICE_INPUT = 0.20   # $ / 1M tokens (input)
    PRICE_OUTPUT = 1.60  # $ / 1M tokens (output)

    cost_input = (total_input * PRICE_INPUT) / 1_000_000
    cost_output = (total_output * PRICE_OUTPUT) / 1_000_000
    total_cost = cost_input + cost_output

    cost_per_page = total_cost / max(len(stats_list), 1)

    return {
        "total_input": total_input,
        "total_output": total_output,
        "total_tokens": total_tokens,
        "cost_input": cost_input,
        "cost_output": cost_output,
        "cost_total": total_cost,
        "cost_per_page": cost_per_page,
    }


def validate_markdown_quality(markdown: str, expected_pages: int) -> Dict:
    """Valide la qualité du markdown"""
    issues = {"critical": [], "warnings": [], "stats": {}}

    # Vérification des pages
    page_markers = re.findall(r'<!-- PAGE (\d+) -->', markdown)
    page_numbers = [int(p) for p in page_markers]

    if len(page_numbers) != expected_pages:
        issues["critical"].append(f"❌ Pages : {len(page_numbers)}/{expected_pages}")

    # Statistiques
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


def main():
    """Point d'entrée principal"""

    # Clé Qwen / DashScope (OpenAI-compatible)
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        messagebox.showerror(
            "Erreur Configuration",
            "Variable DASHSCOPE_API_KEY non définie."
        )
        sys.exit(1)

    print("\n" + "="*70)
    print("🔬 EXTRACTION FACTURES PDF → MARKDOWN (Qwen3-VL)")
    print("="*70)
    print(f"📄 Format : PDF → images PNG base64")
    print(f"🎯 Stratégie : Page par page")
    print(f"💾 Cache serveur : aucun (prompts envoyés à chaque requête)")
    print(f"📊 Affichage : Tokens + coût estimé")
    print(f"💰 Modèle : {MODEL}")
    print("="*70)

    try:
        # Sélection fichier
        pdf_path = choose_file()
        print(f"\n📄 Fichier : {Path(pdf_path).name}")

        # Analyse PDF
        pdf_info = get_pdf_info(pdf_path)
        page_count = pdf_info["page_count"]
        print(f"📊 Pages : {page_count}")
        print(f"💾 Taille : {pdf_info['file_size_mb']:.2f} MB")

        # Chargement progression
        completed_pages = load_progress(pdf_path)
        if completed_pages:
            print(f"\n📂 Reprise : {len(completed_pages)} page(s) déjà traitées")
            response = messagebox.askyesno(
                "Reprise détectée",
                f"{len(completed_pages)} page(s) déjà traitées.\n\nReprendre ?"
            )
            if not response:
                completed_pages = {}

        print("\n" + "="*70)
        print("🚀 DÉBUT DU TRAITEMENT")
        print("="*70 + "\n")

        start_time = time.time()
        all_markdown: List[str] = []
        all_stats: List[Dict] = []

        # Traitement page par page
        for page_num in range(1, page_count + 1):
            page_key = str(page_num)

            # Vérifier si déjà traitée
            if page_key in completed_pages:
                print(f"      ✓ Page {page_num} (déjà traitée)")
                saved_stats = completed_pages[page_key]["stats"]
                print(f"         📊 Tokens : IN={saved_stats.get('input_tokens', 0):,} | OUT={saved_stats.get('output_tokens', 0):,}")
                print()
                all_markdown.append(completed_pages[page_key]["markdown"])
                all_stats.append(saved_stats)
                continue

            # Délai entre requêtes
            if page_num > 1 and INTER_REQUEST_DELAY > 0:
                time.sleep(INTER_REQUEST_DELAY)

            try:
                # Première page du traitement ?
                is_first = (page_num == 1 and len(completed_pages) == 0)

                markdown, stats = process_page_with_cache(
                    pdf_path, page_num, api_key, is_first_page=is_first
                )

                all_markdown.append(markdown)
                all_stats.append(stats)

                completed_pages[page_key] = {
                    "markdown": markdown,
                    "stats": stats
                }

                # Sauvegarder toutes les 5 pages
                if page_num % 5 == 0:
                    save_progress(pdf_path, completed_pages)
                    print(f"         💾 Progression sauvegardée")

                print(f"         ✅ Page {page_num} terminée\n")

            except Exception as e:
                print(f"\n         ❌ Erreur page {page_num}: {e}")

                if STOP_ON_CRITICAL:
                    raise

                error_md = f"<!-- PAGE {page_num} -->\n\n**[ERREUR EXTRACTION]**\n\n---"
                all_markdown.append(error_md)
                all_stats.append({
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0
                })

                print(f"         ⚠️  Marquée comme erreur, continuation...\n")

        duration = time.time() - start_time

        # Fusion finale
        print("\n" + "="*70)
        print("🔧 FINALISATION")
        print("="*70)
        print("\n   🔗 Fusion des pages...")

        final_markdown = "\n\n".join(all_markdown)

        # Sauvegarde
        md_path = Path(pdf_path).with_suffix(".md")
        print(f"   💾 Sauvegarde : {md_path.name}")

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(final_markdown)

        # Statistiques
        md_size_kb = len(final_markdown.encode('utf-8')) / 1024
        costs = calculate_costs(all_stats)

        # Validation
        validation = validate_markdown_quality(final_markdown, page_count)

        # Nettoyage progression
        progress_file = Path(pdf_path).with_suffix(".progress.json")
        if progress_file.exists():
            progress_file.unlink()
            print("   🗑️  Fichier de progression supprimé")

        # Affichage résumé
        print("\n" + "="*70)
        print("✅ EXTRACTION TERMINÉE AVEC SUCCÈS")
        print("="*70)
        print(f"📝 Fichier Markdown : {md_path.name}")
        print(f"📄 Pages extraites  : {page_count}")
        print(f"💾 Taille Markdown  : {md_size_kb:.1f} KB")
        print(f"⏱️  Durée totale     : {duration//60:.0f}min {duration%60:.0f}s")
        print(f"⚡ Vitesse moyenne  : {duration/page_count:.1f}s/page")

        print("\n" + "-"*70)
        print("💰 CONSOMMATION DE TOKENS")
        print("-"*70)
        print(f"📥 Input (PDF)      : {costs['total_input']:,}")
        print(f"📤 Output tokens    : {costs['total_output']:,}")
        print(f"📊 TOTAL tokens     : {costs['total_tokens']:,}")

        print(f"\n💵 Coût input       : ${costs['cost_input']:.4f}")
        print(f"💵 Coût output      : ${costs['cost_output']:.4f}")
        print(f"💵 Coût TOTAL       : ${costs['cost_total']:.4f}")
        print(f"📄 Coût moyen/page  : ${costs['cost_per_page']:.4f}")

        print("\n" + "-"*70)
        print("🔍 QUALITÉ")
        print("-"*70)

        if not validation["critical"] and not validation["warnings"]:
            print("✅ Extraction parfaite")
        elif not validation["critical"]:
            print(f"✅ Extraction réussie avec {len(validation['warnings'])} avertissement(s)")
        else:
            print(f"⚠️  {len(validation['critical'])} problème(s) détectés")

        if validation["stats"]:
            stats = validation["stats"]
            print(f"📊 {stats.get('montants_detectes', 0)} montants, "
                  f"{stats.get('lignes_tableaux', 0)} lignes tableaux")
            if stats.get('champs_manquants', 0) > 0:
                print(f"⚠️  {stats['champs_manquants']} [CHAMP MANQUANT]")

        print("="*70 + "\n")

        # Message final
        msg = (
            f"✅ Extraction terminée !\n\n"
            f"📝 {md_path.name}\n"
            f"📄 {page_count} pages\n"
            f"💾 {md_size_kb:.1f} KB\n"
            f"⏱️  {duration//60:.0f}min {duration%60:.0f}s\n\n"
            f"💰 TOKENS :\n"
            f"   Input (PDF) : {costs['total_input']:,}\n"
            f"   Output      : {costs['total_output']:,}\n\n"
            f"💵 COÛT estimé : ${costs['cost_total']:.4f}"
        )

        messagebox.showinfo("✅ Extraction Terminée", msg)

    except KeyboardInterrupt:
        print("\n\n⚠️  Interruption (Ctrl+C)")
        print("💾 Progression sauvegardée\n")
        sys.exit(0)

    except Exception as e:
        print(f"\n❌ Erreur fatale : {e}")
        import traceback
        traceback.print_exc()
        messagebox.showerror("Erreur Fatale", str(e)[:300])
        sys.exit(1)


if __name__ == "__main__":
    main()

