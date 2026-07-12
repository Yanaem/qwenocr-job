#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import ocr_qwenVL as ocr  # ton script OCR
from google.cloud import storage
import requests

# Bucket dédié Qwen (figé sur qwenvl par défaut)
QWEN_BUCKET = os.getenv("QWEN_BUCKET", "qwenvl")

try:
    PAGE_WORKERS = max(1, int(os.getenv("PAGE_WORKERS", "2")))
except ValueError:
    PAGE_WORKERS = 2


# ---------- GCS utils ----------

def parse_gs_uri(path: str) -> Tuple[str, str]:
    """
    Normalise un chemin GCS pour utiliser toujours le bucket QWEN_BUCKET.

    - Accepte "gs://bucket/chemin/fichier" ou "chemin/fichier"
    - Retourne (QWEN_BUCKET, "chemin/fichier")
    """
    if path.startswith("gs://"):
        rest = path[5:]
        parts = rest.split("/", 1)
        obj = parts[1] if len(parts) == 2 else ""
    else:
        obj = path.lstrip("/")

    if not obj:
        raise ValueError(f"Chemin objet GCS invalide: {path}")

    return QWEN_BUCKET, obj


def download_from_gcs(gs_uri: str, local_path: str) -> None:
    bucket_name, blob_name = parse_gs_uri(gs_uri)
    print("📥 Téléchargement GCS → local")
    print(f"   Bucket : {bucket_name}")
    print(f"   Objet  : {blob_name}")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(local_path)
    print(f"✅ Téléchargé dans : {local_path}")


def upload_to_gcs(local_path: str, gs_uri: str) -> None:
    bucket_name, blob_name = parse_gs_uri(gs_uri)
    print("📤 Upload local → GCS")
    print(f"   Fichier local : {local_path}")
    print(f"   Bucket        : {bucket_name}")
    print(f"   Objet         : {blob_name}")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)
    print("✅ Upload terminé")


# ---------- Runner logique ----------

def run_for_pdf(pdf_path: str, api_key: str, output_md_path: str | None = None):
    """
    Lance toute la chaîne OCR sur un PDF local.
    Retourne:
      - chemin du fichier .md généré
      - page_count
      - duration
      - md_size_kb
      - all_stats
      - costs
    """

    pdf_path = os.path.abspath(pdf_path)

    print("\n" + "=" * 70)
    print("🔬 EXTRACTION FACTURES PDF → MARKDOWN (Qwen3-VL)")
    print("=" * 70)
    print(f"📄 Fichier PDF      : {pdf_path}")
    print(f"💰 Modèle           : {ocr.MODEL}")
    print("=" * 70)

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF introuvable: {pdf_path}")

    pdf_info = ocr.get_pdf_info(pdf_path)
    page_count = pdf_info["page_count"]
    print(f"\n📊 Pages             : {page_count}")
    print(f"💾 Taille            : {pdf_info['file_size_mb']:.2f} MB")

    completed_pages: Dict[str, Dict] = ocr.load_progress(pdf_path)
    if completed_pages:
        print(f"\n📂 Reprise détectée : {len(completed_pages)} page(s) déjà traitées")
    else:
        print("\n📂 Aucune reprise, traitement complet du fichier")

    print("\n" + "=" * 70)
    print("🚀 DÉBUT DU TRAITEMENT (MODE BATCH)")
    print("=" * 70 + "\n")

    start_time = time.time()
    markdown_by_page: Dict[int, str] = {}
    stats_by_page: Dict[int, Dict] = {}
    pages_to_process: List[int] = []

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
            markdown_by_page[page_num] = completed_pages[page_key]["markdown"]
            stats_by_page[page_num] = saved_stats
        else:
            pages_to_process.append(page_num)

    if pages_to_process:
        worker_count = min(PAGE_WORKERS, len(pages_to_process))
        print(f"   ⚙️  Pages simultanées : {worker_count}\n")

        completed_since_save = 0
        critical_error = None

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_page = {
                executor.submit(
                    ocr.process_page_with_cache,
                    pdf_path,
                    page_num,
                    api_key,
                    page_num == 1 and len(completed_pages) == 0,
                ): page_num
                for page_num in pages_to_process
            }

            for future in as_completed(future_to_page):
                page_num = future_to_page[future]
                page_key = str(page_num)

                if future.cancelled():
                    continue

                try:
                    markdown, stats = future.result()

                    markdown_by_page[page_num] = markdown
                    stats_by_page[page_num] = stats
                    completed_pages[page_key] = {
                        "markdown": markdown,
                        "stats": stats,
                    }

                    completed_since_save += 1
                    if completed_since_save >= 5:
                        ocr.save_progress(pdf_path, completed_pages)
                        completed_since_save = 0
                        print("         💾 Progression sauvegardée")

                    print(f"         ✅ Page {page_num} terminée\n")

                except Exception as e:
                    print(f"\n         ❌ Erreur page {page_num}: {e}")

                    if ocr.STOP_ON_CRITICAL:
                        if critical_error is None:
                            critical_error = (page_num, e)
                            for pending_future in future_to_page:
                                if pending_future is not future:
                                    pending_future.cancel()
                        continue

                    error_md = f"<!-- PAGE {page_num} -->\n\n**[ERREUR EXTRACTION]**"
                    markdown_by_page[page_num] = error_md
                    stats_by_page[page_num] = {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                    }

                    print("         ⚠️  Marquée comme erreur, continuation...\n")

        if completed_since_save > 0:
            ocr.save_progress(pdf_path, completed_pages)
            print("         💾 Progression sauvegardée")

        if critical_error is not None:
            failed_page, failed_exception = critical_error
            raise RuntimeError(
                f"Échec critique lors du traitement de la page {failed_page}"
            ) from failed_exception

    all_markdown: List[str] = [
        markdown_by_page[page_num]
        for page_num in range(1, page_count + 1)
    ]
    all_stats: List[Dict] = [
        stats_by_page[page_num]
        for page_num in range(1, page_count + 1)
    ]

    duration = time.time() - start_time

    print("\n" + "=" * 70)
    print("🔧 FINALISATION")
    print("=" * 70)
    print("\n   🔗 Fusion des pages...")

    final_markdown = "\n\n---\n\n".join(
        page.strip() for page in all_markdown if str(page or "").strip()
    )

    ocr.validate_canonical_markdown_structure(final_markdown, page_count)

    if output_md_path:
        md_path = Path(output_md_path)
    else:
        md_path = Path(pdf_path).with_suffix(".md")

    md_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"   💾 Sauvegarde : {md_path}")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(final_markdown)

    md_size_kb = len(final_markdown.encode("utf-8")) / 1024
    costs = ocr.calculate_costs(all_stats)
    validation = ocr.validate_markdown_quality(final_markdown, page_count)

    progress_file = Path(pdf_path).with_suffix(".progress.json")
    if progress_file.exists():
        progress_file.unlink()
        print("   🗑️  Fichier de progression supprimé")

    print("\n" + "=" * 70)
    print("✅ EXTRACTION TERMINÉE AVEC SUCCÈS (MODE BATCH)")
    print("=" * 70)
    print(f"📝 Fichier Markdown : {md_path}")
    print(f"📄 Pages extraites  : {page_count}")
    print(f"💾 Taille Markdown  : {md_size_kb:.1f} KB")
    print(f"⏱️  Durée totale     : {duration // 60:.0f}min {duration % 60:.0f}s")
    print(f"⚡ Vitesse moyenne  : {duration / page_count:.1f}s/page")
    print(f"💵 Coût total       : ${costs['cost_total']:.4f}")
    if validation["stats"]:
        stats = validation["stats"]
        print(
            f"📊 {stats.get('montants_detectes', 0)} montants, "
            f"{stats.get('lignes_tableaux', 0)} lignes tableaux"
        )
    print("=" * 70 + "\n")

    return str(md_path), page_count, duration, md_size_kb, all_stats, costs


def main():
    try:
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY non définie.")

        gcs_input = os.getenv("GCS_INPUT_URI")
        gcs_output = os.getenv("GCS_OUTPUT_URI")
        local_input = os.getenv("INPUT_PDF_PATH")  # fallback éventuel

        if gcs_input:
            # Mode GCS
            local_pdf = "/tmp/input.pdf"
            download_from_gcs(gcs_input, local_pdf)

            # Si pas de GCS_OUTPUT_URI, on dérive la sortie :
            # Entrée : gs://qwenvl/in/xxx.pdf → Sortie : gs://qwenvl/out/xxx.md
            if not gcs_output:
                bucket, blob = parse_gs_uri(gcs_input)
                if blob.startswith("in/"):
                    rest = blob[len("in/"):]
                else:
                    rest = blob
                if "." in rest:
                    base = rest.rsplit(".", 1)[0]
                else:
                    base = rest
                out_blob = f"out/{base}.md"
                gcs_output = f"gs://{bucket}/{out_blob}"

            # Chemin local temporaire pour le .md
            local_md = "/tmp/output.md"
            (
                md_path,
                page_count,
                duration,
                md_size_kb,
                all_stats,
                costs,
            ) = run_for_pdf(local_pdf, api_key, output_md_path=local_md)

            upload_to_gcs(md_path, gcs_output)

            print("=" * 70)
            print(f"🔗 LOVABLE_MARKDOWN_GCS={gcs_output}")
            print("=" * 70)

            # Notifier Supabase / Lovable de la fin du job
            callback_url = os.getenv("CALLBACK_URL")
            ocr_job_id = os.getenv("OCR_JOB_ID")

            if callback_url and ocr_job_id:
                try:
                    total_in = sum(s.get("input_tokens", 0) for s in all_stats)
                    total_out = sum(s.get("output_tokens", 0) for s in all_stats)

                    payload = {
                        "ocrJobId": ocr_job_id,
                        "gcsOutputPath": gcs_output,
                        "status": "success",
                        "pageCount": page_count,
                        "durationSeconds": duration,
                        "markdownSizeKb": md_size_kb,
                        "stats": {
                            "inputTokens": total_in,
                            "outputTokens": total_out,
                            "cost": costs["cost_total"],
                        },
                    }

                    print(f"📡 Envoi du callback à {callback_url} ...")
                    resp = requests.post(callback_url, json=payload, timeout=30)
                    print(f"✅ Callback envoyé ({resp.status_code})")
                except Exception as e:
                    print(f"⚠️ Erreur callback: {e}")

        elif local_input:
            # Mode fichier local uniquement
            run_for_pdf(local_input, api_key)
        else:
            raise RuntimeError(
                "Ni GCS_INPUT_URI ni INPUT_PDF_PATH définis.\n"
                "Définis au moins GCS_INPUT_URI=gs://qwenvl/in/chemin/facture.pdf "
                "pour traiter un fichier depuis ton bucket dédié Qwen."
            )

    except Exception as e:
        print("\n❌ Erreur fatale dans qwenocr_runner.py :", e, file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
