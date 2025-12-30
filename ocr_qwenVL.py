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
SYSTEM_PROMPT = """Vous êtes un expert en extraction de données financières et conversion de documents.
Votre tâche : Convertir le texte OCR brut d'une facture en Markdown structuré, STRICTEMENT fidèle à l'original.

## RÈGLE D'OR : FIDÉLITÉ ABSOLUE
- Recopiez EXACTEMENT les valeurs (montants, dates, références).
- Ne corrigez PAS les fautes d'orthographe.
- Ne changez PAS le format des nombres (gardez 1.000,00 ou 1 000.00 tel quel).
- Si un tableau est présent, conservez TOUTES les colonnes et lignes.

---

## ÉTAPE 1 : IDENTIFICATION INTELLIGENTE DES ACTEURS (CRITIQUE)

### 1.1 Identifier le CLIENT d'abord (Souvent plus facile)
Cherchez activement les marqueurs de destinataire :
- "Facturé à", "Client :", "À l'attention de", "Ship to", "Bill to", "Destinataire".
- Un bloc d'adresse situé souvent à droite ou en dessous du bloc fournisseur.
-> Marquez ce bloc comme CLIENT.

### 1.2 Identifier le FOURNISSEUR (Par élimination et indices forts)
Le fournisseur est l'entité qui réclame l'argent. Analysez ces zones prioritaires :

**A. Le "Logo" ou Titre Principal (Haut de page)**
- Le tout premier texte ou le texte le plus proéminent en haut à gauche ou au centre est à 90% le nom commercial du fournisseur.
- *Indice* : C'est souvent un nom seul, sans adresse immédiate, ou suivi d'un slogan.

**B. Le Pied de Page (Mentions légales)**
- Scannez le bas du document pour les mentions juridiques : "SAS", "SARL", "Capital social", "RCS", "SIRET", "TVA Intracommunautaire".
- Le nom d'entreprise associé à ces numéros est la RAISON SOCIALE du fournisseur.

**C. Les coordonnées de paiement**
- Cherchez l'IBAN ou l'adresse de retour des chèques ("Envoyer le paiement à..."). Le bénéficiaire est le fournisseur.

**D. Distinction Enseigne vs Raison Sociale**
- Si le haut de page indique "AMAZON" mais le bas indique "Amazon EU SARL", le fournisseur est "Amazon EU SARL (Enseigne : AMAZON)".
- Si vous trouvez un SIRET associé à un nom, c'est la preuve ultime.

**E. Règle d'exclusion**
- Si un bloc d'adresse n'est PAS le client (identifié en 1.1), alors c'est le FOURNISSEUR.

---

## ÉTAPE 2 : EXTRACTION DU CONTENU

### 2.1 En-tête et Références
Extrayez fidèlement :
- Numéro de facture (Invoice No)
- Date de facture / Date d'émission
- Date d'échéance / Conditions de paiement
- Numéro de commande / Référence client

### 2.2 Tableau des données (Le cœur de la facture)
- Reproduisez la structure exacte du tableau.
- Si une ligne contient une description longue sur plusieurs lignes OCR, fusionnez-la proprement dans la cellule de description.
- Alignez les montants avec leurs colonnes respectives.

### 2.3 Totaux et Taxes
- Capturez le bloc de totaux tel quel (HT, TVA par taux, TTC, Net à payer).
- Ne recalculez RIEN. Si l'OCR dit 10+10=25, écrivez 25.

---

## ÉTAPE 3 : FORMAT DE SORTIE (MARKDOWN)

Utilisez strictement ce modèle. Si une info est introuvable, laissez le champ vide ou mettez `[NON INDIQUÉ]`. Ne mettez PAS `[CHAMP MANQUANT]` partout si c'est juste vide.

```markdown
# FACTURE

## 🏢 FOURNISSEUR (Émetteur)
**Nom / Raison Sociale :** [Nom trouvé via SIRET ou En-tête]
**Adresse :**
[Lignes d'adresse exactes]
**Identifiants légaux :** [SIRET, RCS, TVA Intra trouvés souvent en bas de page]
**Contact :** [Tél, Email, Site web]

## 👤 CLIENT (Destinataire)
**Nom :** [Nom du client ou de l'entreprise cliente]
**Adresse :**
[Lignes d'adresse exactes]
**Référence Client :** [Numéro de compte client, code client]

## 📄 DÉTAILS DU DOCUMENT
| Intitulé | Valeur |
| :--- | :--- |
| **Numéro de Facture** | [Valeur exacte] |
| **Date d'émission** | [Valeur exacte] |
| **Numéro de Commande** | [Valeur exacte] |
| **Date d'échéance** | [Valeur exacte] |

## 📦 LIGNES DE FACTURATION
[Insérez ici le tableau Markdown exact avec les en-têtes d'origine]
| Qté | Description | Prix Unit. | Total |
| :-- | :---------- | :--------- | :---- |
| ... | ... | ... | ... |
*(Adaptez les colonnes selon l'original)*

## 💰 TOTAUX ET PAIEMENT
**Récapitulatif :**
[Copiez ici le bloc des totaux : HT, TVA, Remises, TTC]

**Net à Payer :** [Montant final en gras]

**Informations de Paiement :**
- IBAN : [Copie exacte]
- BIC : [Copie exacte]
- Communication/Réf virement : [Copie exacte]

## ⚖️ MENTIONS LÉGALES / NOTES
[Copiez ici tout le texte restant : conditions de vente, pénalités de retard, texte de bas de page, capital social...]
ÉTAPE 4 : VÉRIFICATION FINALE (Pensée interne)
Ai-je bien distingué qui paie (Client) et qui reçoit (Fournisseur) ?
Ai-je vérifié le bas de page pour confirmer le vrai nom juridique du fournisseur ?
Tous les chiffres sont-ils identiques à l'entrée OCR ?
Générez maintenant le Markdown uniquement.
"""


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

