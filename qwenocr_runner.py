#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import hashlib
import importlib
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

OCR_MODULE_NAME = "ocr_qwenVL"


def _load_ocr_module():
    """Charge exclusivement le module définitif /app/ocr_qwenVL.py."""
    try:
        module = importlib.import_module(OCR_MODULE_NAME)
    except ModuleNotFoundError as exc:
        if exc.name != OCR_MODULE_NAME:
            raise
        raise RuntimeError(
            "Module OCR introuvable : le runner exige un fichier nommé "
            "ocr_qwenVL.py placé dans le même dossier que qwenocr_runner.py."
        ) from exc
    return module


ocr = _load_ocr_module()


def _validate_ocr_contract() -> None:
    required_attributes = [
        "API_URL", "MODEL", "MODEL_MD", "STOP_ON_CRITICAL",
        "ENABLE_EXPLICIT_CACHE", "QWEN_HIGH_RES_IMAGES", "MARKDOWN_USES_IMAGE",
        "ENABLE_THINKING_OCR", "ENABLE_THINKING_MD",
        "ALLOW_NO_THINK_FALLBACK_MD", "MARKDOWN_FORMAT_RETRIES",
        "PIPELINE_VERSION",
    ]
    required_callables = [
        "validate_api_configuration", "configure_explicit_cache_for_batch",
        "get_pipeline_fingerprint", "get_progress_path", "get_pdf_info",
        "load_progress", "save_progress", "clear_progress",
        "process_page_with_cache", "calculate_costs",
        "validate_markdown_quality", "validate_canonical_markdown_structure",
    ]
    missing = [name for name in required_attributes if not hasattr(ocr, name)]
    missing += [name for name in required_callables if not callable(getattr(ocr, name, None))]
    if missing:
        loaded_path = getattr(ocr, "__file__", "chemin inconnu")
        raise RuntimeError(
            "Le fichier ocr_qwenVL.py chargé est ancien ou incompatible. "
            f"Chemin chargé : {loaded_path}. Éléments absents/non appelables : "
            + ", ".join(sorted(set(missing)))
            + ". Déploie ensemble les deux fichiers définitifs fournis."
        )
    if ocr.MARKDOWN_USES_IMAGE is not True:
        raise RuntimeError(
            "Contrat OCR incompatible : la génération Markdown doit recevoir l'image originale."
        )


def _loaded_ocr_path() -> str:
    return str(Path(getattr(ocr, "__file__", "chemin inconnu")).resolve())

try:
    from google.cloud import storage  # type: ignore
except Exception:
    storage = None  # type: ignore

import requests

# Bucket dédié Qwen (figé sur qwenvl par défaut)
QWEN_BUCKET = os.getenv("QWEN_BUCKET", "qwenvl")

DEFAULT_PAGE_WORKERS = 4
MIN_PAGE_WORKERS = 1
MAX_PAGE_WORKERS = 8


def _read_positive_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        parsed_value = int(raw_value)
    except ValueError as exc:
        raise RuntimeError(f"{name} doit être un entier strictement positif.") from exc
    if parsed_value < 1:
        raise RuntimeError(f"{name} doit être un entier strictement positif.")
    return parsed_value


PROGRESS_SAVE_EVERY = _read_positive_int_env("PROGRESS_SAVE_EVERY", 20)

_GCS_CLIENT: Optional[Any] = None


def read_page_workers() -> Tuple[str, int]:
    """Lit et valide le nombre de pages à traiter simultanément."""
    raw_value = os.getenv("PAGE_WORKERS", str(DEFAULT_PAGE_WORKERS)).strip()

    try:
        parsed_value = int(raw_value)
    except ValueError as exc:
        raise RuntimeError(
            f"PAGE_WORKERS doit être un entier entre {MIN_PAGE_WORKERS} et "
            f"{MAX_PAGE_WORKERS}. Valeur reçue : {raw_value!r}"
        ) from exc

    if not MIN_PAGE_WORKERS <= parsed_value <= MAX_PAGE_WORKERS:
        raise RuntimeError(
            f"PAGE_WORKERS doit être compris entre {MIN_PAGE_WORKERS} et "
            f"{MAX_PAGE_WORKERS}. Valeur reçue : {parsed_value}"
        )

    return raw_value, parsed_value


PAGE_WORKERS_RAW, PAGE_WORKERS = read_page_workers()


def _local_source_id(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"local-sha256:{digest.hexdigest()}"


# ---------- GCS utils ----------

def _get_gcs_client():
    global _GCS_CLIENT
    if storage is None:
        raise RuntimeError(
            "google-cloud-storage n'est pas installé. Il est requis uniquement pour le mode GCS."
        )
    if _GCS_CLIENT is None:
        _GCS_CLIENT = storage.Client()
    return _GCS_CLIENT

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


def download_from_gcs(gs_uri: str, local_path: str) -> str:
    bucket_name, blob_name = parse_gs_uri(gs_uri)
    print("📥 Téléchargement GCS → local")
    print(f"   Bucket : {bucket_name}")
    print(f"   Objet  : {blob_name}")
    client = _get_gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.reload(client=client)
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(local_path)
    print(f"✅ Téléchargé dans : {local_path}")
    return f"gs://{bucket_name}/{blob_name}#{blob.generation}"


def download_from_gcs_if_exists(gs_uri: str, local_path: str) -> bool:
    """Télécharge un objet seulement s'il existe, sans transformer l'absence en erreur."""
    bucket_name, blob_name = parse_gs_uri(gs_uri)
    client = _get_gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    if not blob.exists(client=client):
        return False

    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(local_path)
    return True


def upload_to_gcs(local_path: str, gs_uri: str, *, quiet: bool = False) -> None:
    bucket_name, blob_name = parse_gs_uri(gs_uri)
    if not quiet:
        print("📤 Upload local → GCS")
        print(f"   Fichier local : {local_path}")
        print(f"   Bucket        : {bucket_name}")
        print(f"   Objet         : {blob_name}")
    client = _get_gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)
    if not quiet:
        print("✅ Upload terminé")


def delete_from_gcs(gs_uri: str, *, quiet: bool = False) -> None:
    """Supprime un objet GCS s'il existe."""
    bucket_name, blob_name = parse_gs_uri(gs_uri)
    client = _get_gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    if blob.exists(client=client):
        blob.delete(client=client)
        if not quiet:
            print(f"🗑️  Objet GCS supprimé : gs://{bucket_name}/{blob_name}")


def derive_progress_gcs_uri(gcs_input: str) -> str:
    """Construit un chemin stable de checkpoint à partir de l'objet PDF d'entrée."""
    bucket_name, blob_name = parse_gs_uri(gcs_input)
    relative_name = blob_name[len("in/"):] if blob_name.startswith("in/") else blob_name
    base_name = relative_name.rsplit(".", 1)[0] if "." in relative_name else relative_name
    return f"gs://{bucket_name}/progress/{base_name}.progress.json"


# ---------- Runner logique ----------

def run_for_pdf(
    pdf_path: str,
    api_key: str,
    output_md_path: str | None = None,
    *,
    progress_gcs_uri: str | None = None,
    source_id: str | None = None,
):
    """
    Lance toute la chaîne OCR sur un PDF local.
    Retourne:
      - chemin du fichier .md généré
      - page_count
      - duration
      - md_size_kb
      - all_stats
      - costs
      - worker_count effectif
    """

    _validate_ocr_contract()
    pdf_path = os.path.abspath(pdf_path)

    print("\n" + "=" * 70)
    print("🔬 EXTRACTION FACTURES PDF → MARKDOWN (Qwen multimodal)")
    print("=" * 70)
    print(f"📄 Fichier PDF      : {pdf_path}")
    print(f"🧩 Module OCR       : {OCR_MODULE_NAME} ({ocr.PIPELINE_VERSION})")
    print(f"💰 Modèle OCR       : {ocr.MODEL}")
    print(f"📝 Modèle Markdown  : {getattr(ocr, 'MODEL_MD', ocr.MODEL)} (Qwen)")
    print(f"⚙️  PAGE_WORKERS     : {PAGE_WORKERS} (valeur reçue : {PAGE_WORKERS_RAW!r})")
    print(f"🧠 Cache explicite   : {'configuré' if ocr.ENABLE_EXPLICIT_CACHE else 'désactivé'}")
    print(f"🔎 Haute résolution : {'activée' if ocr.QWEN_HIGH_RES_IMAGES else 'désactivée'}")
    print("🖼️  Source Markdown  : image originale + comparaison OCR")
    print(
        "🧠 Thinking Markdown : "
        + ("activé" if ocr.ENABLE_THINKING_MD else "désactivé")
        + (
            " (aucun fallback sans thinking)"
            if ocr.ENABLE_THINKING_MD and not ocr.ALLOW_NO_THINK_FALLBACK_MD
            else ""
        )
    )
    print(f"🔁 Reprises format   : {ocr.MARKDOWN_FORMAT_RETRIES}")
    print(f"🔐 Empreinte        : {ocr.get_pipeline_fingerprint()[:16]}…")
    print(f"💾 Checkpoint        : toutes les {PROGRESS_SAVE_EVERY} page(s)")
    print("=" * 70)

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF introuvable: {pdf_path}")

    pdf_info = ocr.get_pdf_info(pdf_path)
    page_count = pdf_info["page_count"]
    print(f"\n📊 Pages             : {page_count}")
    print(f"💾 Taille            : {pdf_info['file_size_mb']:.2f} MB")

    if source_id is None:
        source_id = _local_source_id(pdf_path)

    completed_pages: Dict[str, Dict] = ocr.load_progress(
        pdf_path,
        expected_source_id=source_id,
        expected_page_count=page_count,
    )
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
    worker_count = 0

    def persist_progress_checkpoint() -> None:
        ocr.save_progress(
            pdf_path,
            completed_pages,
            source_id=source_id,
            page_count=page_count,
        )
        if progress_gcs_uri:
            local_progress_path = ocr.get_progress_path(pdf_path)
            upload_to_gcs(local_progress_path, progress_gcs_uri, quiet=True)

    for page_num in range(1, page_count + 1):
        page_key = str(page_num)
        record = completed_pages.get(page_key)
        if (
            isinstance(record, dict)
            and isinstance(record.get("markdown"), str)
            and record.get("markdown", "").strip()
            and isinstance(record.get("stats"), dict)
        ):
            print(f"      ✓ Page {page_num} (déjà traitée et validée)")
            saved_stats = record["stats"]
            print(
                f"         📊 Tokens : IN={saved_stats.get('input_tokens', 0):,} | "
                f"OUT={saved_stats.get('output_tokens', 0):,}"
            )
            print()
            markdown_by_page[page_num] = record["markdown"]
            stats_by_page[page_num] = saved_stats
        else:
            pages_to_process.append(page_num)

    if pages_to_process:
        worker_count = min(PAGE_WORKERS, len(pages_to_process))
        cache_active = ocr.configure_explicit_cache_for_batch(
            page_count=len(pages_to_process),
            worker_count=worker_count,
        )
        print(f"   ⚙️  Pages simultanées demandées : {PAGE_WORKERS}")
        print(f"   ⚙️  Pages simultanées effectives : {worker_count}")
        if ocr.ENABLE_EXPLICIT_CACHE:
            if cache_active:
                print(
                    "   🧠 Cache explicite : actif sur les prompts système statiques "
                    "pour les vagues suivantes"
                )
            else:
                print(
                    "   🧠 Cache explicite : omis pour cette reprise "
                    "(une seule vague, donc aucun hit utile)"
                )
        if worker_count < PAGE_WORKERS:
            print(
                "   ℹ️  Réduction automatique : le nombre de pages restantes "
                "est inférieur au nombre demandé."
            )
        print()

        completed_since_save = 0
        critical_error: Optional[Tuple[int, Exception]] = None
        page_errors: Dict[int, Exception] = {}

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
                except Exception as e:
                    page_errors[page_num] = e
                    print(f"\n         ❌ Erreur page {page_num}: {e}")

                    if ocr.STOP_ON_CRITICAL and critical_error is None:
                        critical_error = (page_num, e)
                        for pending_future in future_to_page:
                            if pending_future is not future:
                                pending_future.cancel()
                        print("         ⛔ Arrêt demandé ; annulation des pages non démarrées.\n")
                    else:
                        print(
                            "         ⚠️  La page n'est pas remplacée par un faux Markdown ; "
                            "les autres pages continuent pour alimenter le checkpoint.\n"
                        )
                    continue

                markdown_by_page[page_num] = markdown
                stats_by_page[page_num] = stats
                completed_pages[page_key] = {
                    "markdown": markdown,
                    "stats": stats,
                }

                completed_since_save += 1
                if completed_since_save >= PROGRESS_SAVE_EVERY:
                    try:
                        persist_progress_checkpoint()
                    except Exception as checkpoint_error:
                        raise RuntimeError(
                            f"Échec de sauvegarde du checkpoint après la page {page_num}."
                        ) from checkpoint_error
                    completed_since_save = 0
                    if progress_gcs_uri:
                        print("         💾 Checkpoint GCS sauvegardé")
                    else:
                        print("         💾 Progression locale sauvegardée")

                print(f"         ✅ Page {page_num} terminée\n")

        if completed_since_save > 0:
            persist_progress_checkpoint()
            if progress_gcs_uri:
                print("         💾 Checkpoint GCS sauvegardé")
            else:
                print("         💾 Progression locale sauvegardée")

        if critical_error is not None:
            failed_page, failed_exception = critical_error
            raise RuntimeError(
                f"Échec critique lors du traitement de la page {failed_page}. "
                "Le checkpoint contient uniquement les pages réellement validées."
            ) from failed_exception

        if page_errors:
            failed_pages = sorted(page_errors)
            first_page = failed_pages[0]
            raise RuntimeError(
                "Traitement incomplet : échec des pages "
                + ", ".join(str(page) for page in failed_pages)
                + ". Aucun Markdown final partiel n'a été produit ; le checkpoint conserve "
                "uniquement les pages validées."
            ) from page_errors[first_page]

    missing_pages = [
        page_num for page_num in range(1, page_count + 1)
        if page_num not in markdown_by_page or not str(markdown_by_page[page_num] or "").strip()
    ]
    missing_stats = [page_num for page_num in range(1, page_count + 1) if page_num not in stats_by_page]
    if missing_pages or missing_stats:
        raise RuntimeError(
            f"Finalisation impossible: pages Markdown manquantes={missing_pages}, "
            f"statistiques manquantes={missing_stats}."
        )

    all_markdown: List[str] = [markdown_by_page[page_num] for page_num in range(1, page_count + 1)]
    all_stats: List[Dict] = [stats_by_page[page_num] for page_num in range(1, page_count + 1)]

    duration = time.time() - start_time

    print("\n" + "=" * 70)
    print("🔧 FINALISATION")
    print("=" * 70)
    print("\n   🔗 Fusion des pages...")

    final_markdown = "\n\n---\n\n".join(page.strip() for page in all_markdown)
    ocr.validate_canonical_markdown_structure(final_markdown, page_count)
    validation = ocr.validate_markdown_quality(final_markdown, page_count)
    if not validation.get("ok"):
        raise RuntimeError("Validation finale refusée: " + " | ".join(validation.get("errors", [])))

    if output_md_path:
        md_path = Path(output_md_path)
    else:
        md_path = Path(pdf_path).with_suffix(".md")
    md_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"   💾 Sauvegarde atomique : {md_path}")
    temp_md_path = md_path.with_suffix(md_path.suffix + ".tmp")
    with open(temp_md_path, "w", encoding="utf-8") as handle:
        handle.write(final_markdown)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_md_path, md_path)

    md_size_kb = len(final_markdown.encode("utf-8")) / 1024
    costs = ocr.calculate_costs(all_stats)

    if progress_gcs_uri:
        print("   💾 Checkpoint GCS conservé jusqu'à l'upload du Markdown final")
    else:
        ocr.clear_progress(pdf_path)
        print("   🗑️  Fichier de progression supprimé")

    print("\n" + "=" * 70)
    print("✅ EXTRACTION TERMINÉE AVEC SUCCÈS (MODE BATCH)")
    print("=" * 70)
    print(f"📝 Fichier Markdown : {md_path}")
    print(f"📄 Pages extraites  : {page_count}")
    print(f"💾 Taille Markdown  : {md_size_kb:.1f} KB")
    print(f"⏱️  Durée totale     : {duration // 60:.0f}min {duration % 60:.0f}s")
    print(f"⚡ Vitesse moyenne  : {duration / page_count:.1f}s/page")
    if costs.get("cost_available"):
        print(f"💵 Coût total       : ${costs['cost_total']:.4f}")
    else:
        print("💵 Coût total       : non calculé (tarifs non configurés)")
    if validation["stats"]:
        stats = validation["stats"]
        print(
            f"📊 {stats.get('montants_detectes', 0)} montants, "
            f"{stats.get('lignes_tableaux', 0)} lignes tableaux"
        )
    print("=" * 70 + "\n")

    return (
        str(md_path),
        page_count,
        duration,
        md_size_kb,
        all_stats,
        costs,
        worker_count,
    )


def main():
    try:
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY non définie.")

        _validate_ocr_contract()
        ocr.validate_api_configuration()
        print(f"🧩 Module chargé    : {OCR_MODULE_NAME}")
        print(f"📦 Fichier chargé   : {_loaded_ocr_path()}")
        print(f"🌐 Endpoint Qwen    : {ocr.API_URL}")

        gcs_input = os.getenv("GCS_INPUT_URI")
        gcs_output = os.getenv("GCS_OUTPUT_URI")
        local_input = os.getenv("INPUT_PDF_PATH")  # fallback éventuel

        if gcs_input:
            # Mode GCS
            local_pdf = "/tmp/input.pdf"
            source_id = download_from_gcs(gcs_input, local_pdf)

            progress_gcs_uri = os.getenv("GCS_PROGRESS_URI") or derive_progress_gcs_uri(gcs_input)
            local_progress = ocr.get_progress_path(local_pdf)
            try:
                Path(local_progress).unlink(missing_ok=True)
            except Exception:
                pass

            if download_from_gcs_if_exists(progress_gcs_uri, local_progress):
                print(f"📂 Checkpoint GCS téléchargé : {progress_gcs_uri}")
            else:
                print(f"📂 Aucun checkpoint GCS : {progress_gcs_uri}")

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
                worker_count,
            ) = run_for_pdf(
                local_pdf,
                api_key,
                output_md_path=local_md,
                progress_gcs_uri=progress_gcs_uri,
                source_id=source_id,
            )

            upload_to_gcs(md_path, gcs_output)
            delete_from_gcs(progress_gcs_uri, quiet=True)
            ocr.clear_progress(local_pdf)
            print("🗑️  Checkpoint GCS supprimé après upload réussi du Markdown")

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
                    total_cached = sum(s.get("cached_tokens", 0) for s in all_stats)
                    total_cache_created = sum(
                        s.get("cache_creation_input_tokens", 0) for s in all_stats
                    )

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
                            "cachedTokens": total_cached,
                            "cacheCreationInputTokens": total_cache_created,
                            "cost": costs["cost_total"],
                            "pageWorkersRequested": PAGE_WORKERS,
                            "pageWorkersEffective": worker_count,
                        },
                    }

                    print(f"📡 Envoi du callback à {callback_url} ...")
                    resp = requests.post(callback_url, json=payload, timeout=30)
                    resp.raise_for_status()
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



