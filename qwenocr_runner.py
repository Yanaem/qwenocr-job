#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List

import ocr_qwenVL as ocr  # ton script OCR


def run_for_pdf(pdf_path: str, api_key: str) -> None:
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

        if page_num > 1 and ocr.INTER_REQUEST_DELAY > 0:
            time.sleep(ocr.INTER_REQUEST_DELAY)

        try:
            is_first = (page_num == 1 and len(completed_pages) == 0)

            markdown, stats = ocr.process_page_with_cache(
                pdf_path, page_num, api_key, is_first_page=is_first
            )

            all_markdown.append(markdown)
            all_stats.append(stats)

            completed_pages[page_key] = {
                "markdown": markdown,
                "stats": stats,
            }

            if page_num % 5 == 0:
                ocr.save_progress(pdf_path, completed_pages)
                print(f"         💾 Progression sauvegardée")

            print(f"         ✅ Page {page_num} terminée\n")

        except Exception as e:
            print(f"\n         ❌ Erreur page {page_num}: {e}")

            if ocr.STOP_ON_CRITICAL:
                raise

            error_md = f"<!-- PAGE {page_num} -->\n\n**[ERREUR EXTRACTION]**\n\n---"
            all_markdown.append(error_md)
            all_stats.append(
                {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                }
            )

            print(f"         ⚠️  Marquée comme erreur, continuation...\n")

    duration = time.time() - start_time

    print("\n" + "=" * 70)
    print("🔧 FINALISATION")
    print("=" * 70)
    print("\n   🔗 Fusion des pages...")

    final_markdown = "\n\n".join(all_markdown)

    output_md = os.getenv("OUTPUT_MD_PATH")
    if output_md:
        md_path = Path(output_md)
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

    print("\n💰 TOKENS")
    print(f"📥 Input  : {costs['total_input']:,}")
    print(f"📤 Output : {costs['total_output']:,}")
    print(f"📊 Total  : {costs['total_tokens']:,}")
    print(f"💵 Coût total : ${costs['cost_total']:.4f}")

    if validation["stats"]:
        stats = validation["stats"]
        print(
            f"📊 {stats.get('montants_detectes', 0)} montants, "
            f"{stats.get('lignes_tableaux', 0)} lignes tableaux"
        )

    print("=" * 70 + "\n")


def main():
    try:
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY non définie.")

        pdf_path = os.getenv("INPUT_PDF_PATH")
        if not pdf_path:
            raise RuntimeError(
                "INPUT_PDF_PATH non définie. "
                "Donne le chemin du PDF à traiter via cette variable."
            )

        run_for_pdf(pdf_path, api_key)

    except Exception as e:
        print("\n❌ Erreur fatale dans qwenocr_runner.py :", e, file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
