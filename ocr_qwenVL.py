#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""OCR FACTURES PDF → MARKDOWN

Stratégie : page par page + QVQ-Max (API OpenAI-compatible DashScope)

Différences clés vs Qwen-VL :
- QVQ (visual reasoning) est un modèle "thinking-only" et **ne supporte que le streaming**.
- Le flux renvoie d'abord le raisonnement dans `reasoning_content`, puis la réponse finale dans `content`.
  Ici on **ignore** `reasoning_content` et on ne garde que `content` pour écrire le Markdown.
"""

import os
import sys
import re
import json
import time
import base64
import requests
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
from pypdf import PdfReader
from io import BytesIO
from typing import Tuple, Dict, List, Optional

from pdf2image import convert_from_path  # nécessite pdf2image + poppler


# ====== Configuration ======
# Endpoint OpenAI-compatible Qwen (région Singapore).
# Si tu es en région Beijing, remplace par :
#   https://dashscope.aliyuncs.com/compatible-mode/v1
API_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

# Modèle visual reasoning (QVQ)
MODEL = "qvq-max"

# QVQ-Max : max response = 8192 tokens (chain-of-thought + réponse)
# On garde une petite marge.
MAX_TOKENS = 8000

# Streaming requis pour QVQ (voir doc "Visual reasoning")
STREAM = True

REQUEST_TIMEOUT = 600
MAX_RETRIES = 5
BACKOFF_BASE = 2
BACKOFF_MAX = 120
INTER_REQUEST_DELAY = 2
STOP_ON_CRITICAL = False

# Limite pratique pour l'upload en base64 via OpenAI-compatible.
# La doc indique que le fichier d'origine doit être < ~7MB pour passer en base64.
MAX_LOCAL_IMAGE_BYTES = int(6.5 * 1024 * 1024)


# ====== Prompt (injecté dans le message user) ======
SYSTEM_PROMPT = """Vous allez jouer le rôle d'un assistant qui reformate le texte brut d'une facture en un document structuré en **Markdown**, sans aucune perte d'information. Le texte d'entrée est le résultat d'un OCR d'une facture PDF en français (texte brut sans mise en page). **Votre objectif est de reproduire fidèlement toutes les informations extraites de la facture, en les organisant par sections et tableaux Markdown, sans rien inventer ni omettre.**

**Consignes importantes :** Ne générez **aucune information** qui n'apparaît pas explicitement dans le texte OCR fourni. Ne faites **aucune supposition** et ne tentez pas de deviner du contenu manquant. **N'ajoutez pas** de texte explicatif, **ne reformulez pas** le contenu original. Si le texte OCR comporte des erreurs ou des éléments incompréhensibles, laissez-les tels quels dans la limite du possible (ou signalez-les comme `[CHAMP MANQUANT]` s'ils sont illisibles). En particulier, si une donnée attendue n'est pas présente dans le texte (par exemple un numéro de facture manquant, une adresse illisible, etc.), indiquez clairement `[CHAMP MANQUANT]` à sa place plutôt que d'inventer quoi que ce soit.

**RÈGLE PRIORITAIRE pour identifier l'émetteur et le client :**
La position géographique des informations dans le document original est le critère principal d'identification :
- **À GAUCHE (ou apparaissant en premier dans le texte OCR)** = ÉMETTEUR/FOURNISSEUR (celui qui émet la facture)
- **À DROITE (ou apparaissant en second)** = CLIENT/DESTINATAIRE (celui qui reçoit la facture)

Cette règle de position est **prioritaire** sur tous les autres indices (mentions légales, SIRET, capital social, etc.).
Si le texte OCR présente deux blocs d'adresses distincts en début de document, considérez systématiquement :
- Le premier bloc = Informations émetteur
- Le second bloc = Informations client

**IMPORTANT :** Même si un bloc contient des mentions légales complètes (capital social, RCS, agrément, etc.) mais apparaît à droite ou en second, il s'agit quand même du CLIENT. Inversement, un bloc simple sans mentions légales mais à gauche/en premier est l'ÉMETTEUR.

Formatez la sortie en sections avec des titres **Markdown** clairs pour chaque catégorie d'informations de la facture. Utilisez par exemple la syntaxe de titre Markdown (`## Titre de la section`) pour chaque section principale. Respectez l'ordre et la hiérarchie suivants (si l'information est disponible dans le texte) :

- **Informations émetteur** : Identifiez le vendeur / l'émetteur de la facture en vous basant PRIORITAIREMENT sur la position (première adresse à gauche dans le document). Incluez le nom de la société ou du prestataire, l'adresse complète, et toute autre information le concernant présente dans le texte, comme son SIRET, son numéro de TVA intracommunautaire, coordonnées de contact, etc.

- **Informations client** : Identifiez le client / destinataire de la facture en vous basant PRIORITAIREMENT sur la position (deuxième adresse à droite dans le document). Incluez le nom ou raison sociale, adresse, et éventuelles autres infos comme un numéro de client, si mentionné.

- **Détails de la facture** : Regroupe les informations générales de la facture, par exemple le numéro de facture, la date d'émission, la date de la vente ou de la prestation, la date d'échéance de paiement, le numéro de commande ou de devis lié le cas échéant, etc. Listez chaque détail pertinent sur une ligne séparée ou sous-forme de sous-éléments si nécessaire (par exemple, « **Numéro de facture :** XXXXXX »).

- **Tableau des lignes** : Présentez sous forme de tableau Markdown toutes les lignes d'articles ou prestations figurant sur la facture. Chaque ligne du tableau doit correspondre à une ligne de facture. Conservez les colonnes telles qu'elles apparaissent dans le texte d'origine (par exemple : **Description**, **Quantité**, **Prix Unitaire**, **Total HT**, **TVA**, **Total TTC** ...). Utilisez la première ligne du tableau pour les en-têtes de colonnes si ces en-têtes sont présentes dans le texte OCR ; sinon, conservez la structure implicite. **Ne fusionnez pas** et ne réorganisez pas les colonnes : respectez l'ordre original. Si certaines valeurs dans le tableau sont manquantes ou illisibles, insérez `[CHAMP MANQUANT]` dans la cellule correspondante. Veillez à ce que le tableau Markdown soit correctement formaté avec des barres verticales `|` séparant chaque colonne et une ligne de séparation `---` sous la ligne d'en-têtes.

- **Montants** : Indiquez ici les totaux et récapitulatifs figurant après les lignes de détail. Cela comprend généralement le **Total HT** (hors taxes), le détail de la TVA (par taux, si disponible), le **Total TTC** (toutes taxes comprises), et éventuellement d'autres montants comme des frais annexes, remises ou acomptes déjà versés. Chaque ligne de ce récapitulatif doit reprendre exactement le libellé et le montant tels qu'ils apparaissent dans le texte OCR (par ex. « **Total HT :** 100,00 € », « **TVA 20% :** 20,00 € », « **Total TTC :** 120,00 € »). S'il manque un montant attendu, utilisez `[CHAMP MANQUANT]`.

- **Informations de paiement** : Si le texte comporte des indications sur le paiement, mentionnez-les dans cette section. Par exemple : modalités ou conditions de paiement (*paiement à 30 jours*, *à régler par virement bancaire*, etc.), coordonnées bancaires du bénéficiaire (IBAN, BIC) si présentes, ainsi que les mentions de pénalités de retard ou d'escompte en cas de paiement anticipé. Chaque information doit figurer sur une ligne distincte ou sous forme de liste à puces si cela s'y prête. Si aucune information de paiement n'est présente, vous pouvez omettre cette section ou la marquer `[CHAMP MANQUANT]` selon le contexte.

- **Mentions légales** : Recueillez ici toutes les autres mentions textuelles présentes sur la facture qui n'ont pas été incluses dans les sections ci-dessus. Cela peut inclure par exemple : la forme juridique et le capital de l'entreprise émettrice, son numéro SIRET/SIREN et RCS, son numéro de TVA intracommunautaire (s'il ne figurait pas déjà en section émetteur), des mentions du type *« TVA non applicable, article 293 B du CGI »*, l'adresse du site web, le contact du service client, ou toute note de bas de page (du style *« Merci de votre confiance »* ou conditions générales succinctes). **Aucune information visible dans le texte ne doit être ignorée.** Séparez les différentes mentions par des points ou mettez-les sur des lignes distinctes si besoin pour la lisibilité. Si aucune mention légale ou note complémentaire n'apparaît, indiquez `[CHAMP MANQUANT]` dans cette section également (sauf si toutes les infos étaient déjà classées ailleurs).

**Important :** Respectez **scrupuleusement le contenu et la formulation du texte original.** Ne reformulez pas les intitulés (par exemple si l'OCR a capturé « Montant total TTC » ne le transformez pas en « Total TTC » – laissez tel quel). Ne changez pas le format des dates, n'arrondissez pas les montants, n'interprétez pas les abréviations. Votre tâche n'est **que de structurer et organiser** le texte, pas de le traduire ni de le résumer. Enfin, la réponse que vous produirez **doit uniquement contenir le document Markdown formaté** (commençant par les sections ci-dessus), sans aucune explication supplémentaire en dehors des données de la facture.

Commencez maintenant la conversion en suivant ces consignes. Bonne organisation !"""


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
        filetypes=[("PDF", "*.pdf"), ("Tous", "*.*")],
    )
    root.destroy()

    if not path:
        sys.exit("❌ Aucun fichier sélectionné")

    return path


def get_pdf_info(pdf_path: str) -> Dict:
    """Récupère les infos du PDF"""
    with open(pdf_path, "rb") as f:
        reader = PdfReader(f)
        page_count = len(reader.pages)

    file_size = os.path.getsize(pdf_path)

    return {
        "page_count": page_count,
        "file_size_bytes": file_size,
        "file_size_mb": file_size / (1024 * 1024),
    }


def _render_pdf_page_to_image_bytes(
    pdf_path: str,
    page_num: int,
    dpi: int,
    fmt: str = "PNG",
    jpeg_quality: int = 90,
) -> bytes:
    """Rend une page PDF en image (bytes)."""
    images = convert_from_path(
        pdf_path,
        dpi=dpi,
        first_page=page_num,
        last_page=page_num,
    )
    if not images:
        raise ValueError(f"Aucune image générée pour la page {page_num}")

    image = images[0]
    buffer = BytesIO()

    if fmt.upper() == "JPEG":
        image.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
    else:
        image.save(buffer, format="PNG")

    buffer.seek(0)
    return buffer.read()


def extract_single_page_to_base64(pdf_path: str, page_num: int) -> Tuple[str, int, str]:
    """Extrait UNE page du PDF, la rend en image, puis encode en base64.

    Pour limiter les erreurs côté API (base64 trop gros), on essaie plusieurs DPI.

    Returns:
        (image_base64, size_kb, mime)
    """

    # Essais progressifs (PNG d'abord, puis JPEG si besoin)
    dpi_candidates = [300, 250, 200, 150]

    last_bytes: Optional[bytes] = None
    last_mime = "image/png"

    for dpi in dpi_candidates:
        img_bytes = _render_pdf_page_to_image_bytes(pdf_path, page_num, dpi=dpi, fmt="PNG")
        last_bytes = img_bytes
        last_mime = "image/png"
        if len(img_bytes) <= MAX_LOCAL_IMAGE_BYTES:
            break

    # Si toujours trop gros, basculer en JPEG compressé
    if last_bytes is not None and len(last_bytes) > MAX_LOCAL_IMAGE_BYTES:
        for dpi in [200, 150, 120]:
            img_bytes = _render_pdf_page_to_image_bytes(
                pdf_path,
                page_num,
                dpi=dpi,
                fmt="JPEG",
                jpeg_quality=85,
            )
            last_bytes = img_bytes
            last_mime = "image/jpeg"
            if len(img_bytes) <= MAX_LOCAL_IMAGE_BYTES:
                break

    if last_bytes is None:
        raise ValueError(f"Impossible de générer une image pour la page {page_num}")

    image_base64 = base64.b64encode(last_bytes).decode("utf-8")
    size_kb = int(len(last_bytes) / 1024)
    return image_base64, size_kb, last_mime


def _parse_sse_event_stream(resp: requests.Response) -> Tuple[str, Dict]:
    """Parse une réponse streaming (SSE) OpenAI-compatible.

    Retourne:
      - answer_content (str)
      - usage (dict) si présent via stream_options.include_usage
    """
    answer_parts: List[str] = []
    usage: Dict = {}

    for raw_line in resp.iter_lines(decode_unicode=True):
        if raw_line is None:
            continue
        line = raw_line.strip()
        if not line:
            continue
        if not line.startswith("data:"):
            continue

        data = line[len("data:") :].strip()
        if data == "[DONE]":
            break

        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue

        # Dernier chunk : choices=[] + usage={...}
        if not chunk.get("choices"):
            if isinstance(chunk.get("usage"), dict):
                usage = chunk["usage"]
            continue

        delta = chunk["choices"][0].get("delta") or {}

        # QVQ renvoie le raisonnement dans reasoning_content (on l'ignore)
        content_piece = delta.get("content")
        if content_piece is None:
            continue

        if isinstance(content_piece, str):
            answer_parts.append(content_piece)
        elif isinstance(content_piece, list):
            # Rare, mais on garde la compatibilité
            for part in content_piece:
                if isinstance(part, dict):
                    txt = part.get("text")
                    if txt:
                        answer_parts.append(txt)
        else:
            answer_parts.append(str(content_piece))

    return "".join(answer_parts), usage


def _build_error_message_from_response(resp: requests.Response) -> str:
    """Construit un message d'erreur lisible à partir d'une réponse HTTP."""
    error_msg = f"HTTP {resp.status_code}"
    try:
        error_detail = resp.json()
        if isinstance(error_detail, dict):
            err = error_detail.get("error", {})
            msg = err.get("message") or str(error_detail)
            error_msg += f": {msg[:400]}"
        else:
            error_msg += f": {str(error_detail)[:400]}"
    except Exception:
        # fallback texte brut
        try:
            error_msg += f": {resp.text[:400]}"
        except Exception:
            pass
    return error_msg


def process_page_with_cache(
    pdf_path: str,
    page_num: int,
    api_key: str,
    is_first_page: bool = False,
) -> Tuple[str, Dict]:
    """Traite UNE page via QVQ-Max (OpenAI-compatible, streaming requis)."""

    _ = is_first_page  # conservé pour compatibilité (non utilisé)

    print(f"      📄 Page {page_num}")

    # Extraire la page en image base64
    print(f"         📦 Extraction image...", end=" ")
    image_base64, size_kb, mime = extract_single_page_to_base64(pdf_path, page_num)
    print(f"{size_kb} KB ({mime})")

    url = f"{API_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    data_url = f"data:{mime};base64,{image_base64}"

    body = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": SYSTEM_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
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
        # QVQ : streaming obligatoire
        "stream": True,
        # Pour récupérer l'usage token dans le dernier chunk
        "stream_options": {"include_usage": True},
    }

    print("         🔄 Traitement OCR (stream)...", end=" ")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with requests.post(
                url,
                headers=headers,
                json=body,
                timeout=REQUEST_TIMEOUT,
                stream=True,
            ) as resp:
                if resp.status_code != 200:
                    raise Exception(_build_error_message_from_response(resp))

                ctype = (resp.headers.get("Content-Type") or "").lower()
                if "text/event-stream" not in ctype:
                    # Parfois l'API renvoie un JSON (ex: erreur) malgré status 200.
                    # On tente de le lire et d'échouer explicitement.
                    try:
                        j = resp.json()
                        raise Exception(f"Réponse inattendue (non-stream): {str(j)[:400]}")
                    except Exception as e:
                        raise Exception(f"Réponse inattendue (non-stream). Content-Type={ctype}. Détail={e}")

                answer_content, usage = _parse_sse_event_stream(resp)

            # Tokens (OpenAI-compatible)
            input_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0)) if isinstance(usage, dict) else 0
            output_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0)) if isinstance(usage, dict) else 0

            print("✅")
            print(f"         📊 Tokens : IN={input_tokens:,} | OUT={output_tokens:,}")

            markdown_core = (answer_content or "").strip()

            markdown = f"<!-- PAGE {page_num} -->\n\n{markdown_core}\n\n---"

            stats = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            }
            return markdown, stats

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

        except Exception as e:
            should_retry, wait_time = handle_api_error(e, attempt, f"page {page_num}")
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
    """Calcule les coûts estimés pour QVQ-Max (International/Singapour).

    Référence "Models" (QVQ → qvq-max) :
      - Input  : 1.2  $ / million tokens
      - Output : 4.8  $ / million tokens

    Note : sur QVQ, les tokens de sortie incluent le *raisonnement* (chain-of-thought)
    + la réponse finale.
    """

    total_input = sum(int(s.get("input_tokens", 0) or 0) for s in stats_list)
    total_output = sum(int(s.get("output_tokens", 0) or 0) for s in stats_list)
    total_tokens = total_input + total_output

    PRICE_INPUT = 1.2
    PRICE_OUTPUT = 4.8

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
    page_markers = re.findall(r"<!-- PAGE (\d+) -->", markdown)
    page_numbers = [int(p) for p in page_markers]

    if len(page_numbers) != expected_pages:
        issues["critical"].append(f"❌ Pages : {len(page_numbers)}/{expected_pages}")

    # Statistiques
    champ_manquant = len(re.findall(r"\[CHAMP MANQUANT\]", markdown, re.IGNORECASE))
    amounts = re.findall(r"\d{1,3}(?:[ \.]?\d{3})*,\d{2}\s*€", markdown)
    table_count = len(re.findall(r"\|.*\|.*\|", markdown))

    if champ_manquant > 0:
        issues["warnings"].append(f"⚠️  {champ_manquant} champ(s) manquant(s)")

    issues["stats"]["champs_manquants"] = champ_manquant
    issues["stats"]["montants_detectes"] = len(amounts)
    issues["stats"]["lignes_tableaux"] = table_count
    issues["stats"]["caracteres"] = len(markdown)

    return issues


def main():
    """Point d'entrée principal"""

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        messagebox.showerror("Erreur Configuration", "Variable DASHSCOPE_API_KEY non définie.")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("🔬 EXTRACTION FACTURES PDF → MARKDOWN (QVQ-Max)")
    print("=" * 70)
    print("📄 Format : PDF → images (base64)")
    print("🎯 Stratégie : Page par page")
    print(f"🧠 Reasoning : activé (QVQ = thinking-only, streaming)" )
    print("💾 Cache serveur : aucun (prompts envoyés à chaque requête)")
    print("📊 Affichage : Tokens + coût estimé")
    print(f"💰 Modèle : {MODEL}")
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
            print(f"\n📂 Reprise : {len(completed_pages)} page(s) déjà traitées")
            response = messagebox.askyesno(
                "Reprise détectée",
                f"{len(completed_pages)} page(s) déjà traitées.\n\nReprendre ?",
            )
            if not response:
                completed_pages = {}

        print("\n" + "=" * 70)
        print("🚀 DÉBUT DU TRAITEMENT")
        print("=" * 70 + "\n")

        start_time = time.time()
        all_markdown: List[str] = []
        all_stats: List[Dict] = []

        for page_num in range(1, page_count + 1):
            page_key = str(page_num)

            if page_key in completed_pages:
                print(f"      ✓ Page {page_num} (déjà traitée)")
                saved_stats = completed_pages[page_key]["stats"]
                print(
                    f"         📊 Tokens : IN={saved_stats.get('input_tokens', 0):,} | "
                    f"OUT={saved_stats.get('output_tokens', 0):,}"
                )
                print()
                all_markdown.append(completed_pages[page_key]["markdown"])
                all_stats.append(saved_stats)
                continue

            if page_num > 1 and INTER_REQUEST_DELAY > 0:
                time.sleep(INTER_REQUEST_DELAY)

            try:
                is_first = page_num == 1 and len(completed_pages) == 0

                markdown, stats = process_page_with_cache(
                    pdf_path, page_num, api_key, is_first_page=is_first
                )

                all_markdown.append(markdown)
                all_stats.append(stats)
                completed_pages[page_key] = {"markdown": markdown, "stats": stats}

                if page_num % 5 == 0:
                    save_progress(pdf_path, completed_pages)
                    print("         💾 Progression sauvegardée")

                print(f"         ✅ Page {page_num} terminée\n")

            except Exception as e:
                print(f"\n         ❌ Erreur page {page_num}: {e}")

                if STOP_ON_CRITICAL:
                    raise

                error_md = f"<!-- PAGE {page_num} -->\n\n**[ERREUR EXTRACTION]**\n\n---"
                all_markdown.append(error_md)
                all_stats.append({"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                print("         ⚠️  Marquée comme erreur, continuation...\n")

        duration = time.time() - start_time

        print("\n" + "=" * 70)
        print("🔧 FINALISATION")
        print("=" * 70)
        print("\n   🔗 Fusion des pages...")

        final_markdown = "\n\n".join(all_markdown)

        md_path = Path(pdf_path).with_suffix(".md")
        print(f"   💾 Sauvegarde : {md_path.name}")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(final_markdown)

        md_size_kb = len(final_markdown.encode("utf-8")) / 1024
        costs = calculate_costs(all_stats)
        validation = validate_markdown_quality(final_markdown, page_count)

        progress_file = Path(pdf_path).with_suffix(".progress.json")
        if progress_file.exists():
            progress_file.unlink()
            print("   🗑️  Fichier de progression supprimé")

        print("\n" + "=" * 70)
        print("✅ EXTRACTION TERMINÉE AVEC SUCCÈS")
        print("=" * 70)
        print(f"📝 Fichier Markdown : {md_path.name}")
        print(f"📄 Pages extraites  : {page_count}")
        print(f"💾 Taille Markdown  : {md_size_kb:.1f} KB")
        print(f"⏱️  Durée totale     : {duration // 60:.0f}min {duration % 60:.0f}s")
        print(f"⚡ Vitesse moyenne  : {duration / max(page_count, 1):.1f}s/page")

        print("\n" + "-" * 70)
        print("💰 CONSOMMATION DE TOKENS")
        print("-" * 70)
        print(f"📥 Input (PDF)      : {costs['total_input']:,}")
        print(f"📤 Output tokens    : {costs['total_output']:,}")
        print(f"📊 TOTAL tokens     : {costs['total_tokens']:,}")

        print(f"\n💵 Coût input       : ${costs['cost_input']:.4f}")
        print(f"💵 Coût output      : ${costs['cost_output']:.4f}")
        print(f"💵 Coût TOTAL       : ${costs['cost_total']:.4f}")
        print(f"📄 Coût moyen/page  : ${costs['cost_per_page']:.4f}")

        print("\n" + "-" * 70)
        print("🔍 QUALITÉ")
        print("-" * 70)

        if not validation["critical"] and not validation["warnings"]:
            print("✅ Extraction parfaite")
        elif not validation["critical"]:
            print(f"✅ Extraction réussie avec {len(validation['warnings'])} avertissement(s)")
        else:
            print(f"⚠️  {len(validation['critical'])} problème(s) détectés")

        if validation["stats"]:
            stats = validation["stats"]
            print(
                f"📊 {stats.get('montants_detectes', 0)} montants, "
                f"{stats.get('lignes_tableaux', 0)} lignes tableaux"
            )
            if stats.get("champs_manquants", 0) > 0:
                print(f"⚠️  {stats['champs_manquants']} [CHAMP MANQUANT]")

        print("=" * 70 + "\n")

        msg = (
            "✅ Extraction terminée !\n\n"
            f"📝 {md_path.name}\n"
            f"📄 {page_count} pages\n"
            f"💾 {md_size_kb:.1f} KB\n"
            f"⏱️  {duration // 60:.0f}min {duration % 60:.0f}s\n\n"
            "💰 TOKENS :\n"
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
