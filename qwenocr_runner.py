#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
qwenocr_runner.py — runner Cloud Run / local pour l'extraction canonique v5.

Chemin nominal par page :
    images -> un appel Qwen -> source canonique -> Markdown Python déterministe

Aucun second modèle ne reformule la première lecture. Une page dégradée reste
publiée avec son rapport qualité ; une erreur locale n'efface pas les autres
pages ni les totaux déjà extraits.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

OCR_MODULE_NAME = "ocr_qwenVL"


def _load_ocr_module():
    try:
        return importlib.import_module(OCR_MODULE_NAME)
    except ModuleNotFoundError as exc:
        if exc.name != OCR_MODULE_NAME:
            raise
        raise RuntimeError(
            "Module OCR introuvable : place ocr_qwenVL.py dans le même dossier "
            "que qwenocr_runner.py."
        ) from exc


ocr = _load_ocr_module()

try:
    from google.cloud import storage  # type: ignore
except Exception:
    storage = None  # type: ignore


# =============================================================================
# Contrat et configuration runner
# =============================================================================


def _validate_ocr_contract() -> None:
    required_attributes = [
        "API_URL",
        "MODEL",
        "MODEL_OCR",
        "PIPELINE_VERSION",
        "CANONICAL_OCR_ONLY",
        "DETERMINISTIC_MARKDOWN",
        "NOMINAL_GENERATIONS_PER_PAGE",
        "SEMANTIC_RETRIES",
        "STOP_ON_CRITICAL",
        "PUBLISH_PARTIAL_DOCUMENT",
        "ENABLE_EXPLICIT_CACHE",
        "QWEN_HIGH_RES_IMAGES",
        "RENDER_DPI",
        "DETAIL_DPI",
    ]
    required_callables = [
        "validate_api_configuration",
        "configure_explicit_cache_for_batch",
        "get_pipeline_fingerprint",
        "get_progress_path",
        "get_pdf_info",
        "load_progress",
        "save_progress",
        "clear_progress",
        "process_page",
        "build_unavailable_page",
        "validate_markdown_quality",
        "calculate_costs",
    ]
    missing = [name for name in required_attributes if not hasattr(ocr, name)]
    missing += [
        name for name in required_callables if not callable(getattr(ocr, name, None))
    ]
    if missing:
        raise RuntimeError(
            "ocr_qwenVL.py incompatible. Déploie ensemble les deux fichiers v5. "
            "Éléments absents : " + ", ".join(sorted(set(missing)))
        )
    if ocr.CANONICAL_OCR_ONLY is not True:
        raise RuntimeError("Contrat invalide : la lecture LLM doit être canonique uniquement.")
    if ocr.DETERMINISTIC_MARKDOWN is not True:
        raise RuntimeError("Contrat invalide : le Markdown doit être rendu par Python.")
    if int(ocr.NOMINAL_GENERATIONS_PER_PAGE) != 1:
        raise RuntimeError("Contrat invalide : une seule génération Qwen est autorisée par page.")
    if int(ocr.SEMANTIC_RETRIES) != 0:
        raise RuntimeError("Contrat invalide : aucune relance sémantique n'est autorisée.")


def _loaded_ocr_path() -> str:
    return str(Path(getattr(ocr, "__file__", "chemin inconnu")).resolve())


def _read_int_env(
    name: str,
    default: int,
    *,
    minimum: int = 0,
    maximum: Optional[int] = None,
) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        value = int(default)
    else:
        try:
            value = int(raw.strip())
        except ValueError as exc:
            raise RuntimeError(f"{name} doit être un entier.") from exc
    if value < minimum:
        raise RuntimeError(f"{name} doit être supérieur ou égal à {minimum}.")
    if maximum is not None and value > maximum:
        raise RuntimeError(f"{name} doit être inférieur ou égal à {maximum}.")
    return value


_workers_raw = os.getenv(
    "PAGE_WORKERS",
    os.getenv("PIPELINE_CONCURRENCY", "4"),
).strip()
try:
    PAGE_WORKERS = int(_workers_raw)
except ValueError as exc:
    raise RuntimeError("PAGE_WORKERS doit être un entier.") from exc
if not 1 <= PAGE_WORKERS <= 8:
    raise RuntimeError("PAGE_WORKERS doit être compris entre 1 et 8.")

PROGRESS_SAVE_EVERY = _read_int_env("PROGRESS_SAVE_EVERY", 10, minimum=1)
QWEN_BUCKET = os.getenv("QWEN_BUCKET", "qwenvl").strip() or "qwenvl"

_GCS_CLIENT: Optional[Any] = None


# =============================================================================
# Utilitaires généraux
# =============================================================================


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _local_source_id(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"local-sha256:{digest.hexdigest()}"


def _atomic_write_text(path: str | Path, content: str) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    return str(target)


def _atomic_write_json(path: str | Path, payload: Dict[str, Any]) -> str:
    return _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
    )


def _derive_local_sidecars(
    markdown_path: str,
    *,
    summary_path: Optional[str] = None,
    canonical_path: Optional[str] = None,
    quality_path: Optional[str] = None,
) -> Dict[str, str]:
    md = Path(markdown_path)
    base = md.with_suffix("")
    return {
        "markdown": str(md),
        "summary": summary_path or str(base) + ".summary.md",
        "canonical": canonical_path or str(base) + ".ocr.txt",
        "quality": quality_path or str(base) + ".quality.json",
    }


def _is_global_failure(error: BaseException) -> bool:
    message = str(error).lower()
    markers = (
        "invalid api key",
        "authentication failed",
        "permission denied",
        "http 401",
        "http 403",
        "http 404",
        "model not found",
        "endpoint qwen invalide",
        "workspace",
    )
    return any(marker in message for marker in markers)


# =============================================================================
# GCS
# =============================================================================


def _get_gcs_client():
    global _GCS_CLIENT
    if storage is None:
        raise RuntimeError(
            "google-cloud-storage n'est pas installé. Il est requis uniquement en mode GCS."
        )
    if _GCS_CLIENT is None:
        _GCS_CLIENT = storage.Client()
    return _GCS_CLIENT


def parse_gs_uri(path: str) -> Tuple[str, str]:
    """Compatibilité historique : tous les objets utilisent QWEN_BUCKET."""
    if path.startswith("gs://"):
        rest = path[5:]
        parts = rest.split("/", 1)
        obj = parts[1] if len(parts) == 2 else ""
    else:
        obj = path.lstrip("/")
    if not obj:
        raise ValueError(f"Chemin objet GCS invalide : {path}")
    return QWEN_BUCKET, obj


def _canonical_gs_uri(path: str) -> str:
    bucket, blob = parse_gs_uri(path)
    return f"gs://{bucket}/{blob}"


def download_from_gcs(gs_uri: str, local_path: str) -> str:
    bucket_name, blob_name = parse_gs_uri(gs_uri)
    print("📥 Téléchargement GCS → local")
    print(f"   Bucket : {bucket_name}")
    print(f"   Objet  : {blob_name}")
    client = _get_gcs_client()
    blob = client.bucket(bucket_name).blob(blob_name)
    blob.reload(client=client)
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(local_path)
    print(f"✅ Téléchargé dans : {local_path}")
    return f"gs://{bucket_name}/{blob_name}#{blob.generation}"


def download_from_gcs_if_exists(gs_uri: str, local_path: str) -> bool:
    bucket_name, blob_name = parse_gs_uri(gs_uri)
    client = _get_gcs_client()
    blob = client.bucket(bucket_name).blob(blob_name)
    if not blob.exists(client=client):
        return False
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(local_path)
    return True


def upload_to_gcs(local_path: str, gs_uri: str, *, quiet: bool = False) -> None:
    bucket_name, blob_name = parse_gs_uri(gs_uri)
    if not quiet:
        print(f"📤 {local_path} → gs://{bucket_name}/{blob_name}")
    client = _get_gcs_client()
    client.bucket(bucket_name).blob(blob_name).upload_from_filename(local_path)
    if not quiet:
        print("✅ Upload terminé")


def delete_from_gcs(gs_uri: str, *, quiet: bool = False) -> None:
    bucket_name, blob_name = parse_gs_uri(gs_uri)
    client = _get_gcs_client()
    blob = client.bucket(bucket_name).blob(blob_name)
    if blob.exists(client=client):
        blob.delete(client=client)
        if not quiet:
            print(f"🗑️  Objet supprimé : gs://{bucket_name}/{blob_name}")


def derive_progress_gcs_uri(gcs_input: str) -> str:
    bucket_name, blob_name = parse_gs_uri(gcs_input)
    relative = blob_name[len("in/") :] if blob_name.startswith("in/") else blob_name
    base = relative.rsplit(".", 1)[0] if "." in relative else relative
    return f"gs://{bucket_name}/progress/{base}.progress.json"


def _derive_gcs_sidecar(primary_uri: str, suffix: str) -> str:
    bucket, blob = parse_gs_uri(primary_uri)
    if blob.lower().endswith(".md"):
        blob = blob[:-3]
    return f"gs://{bucket}/{blob}{suffix}"


# =============================================================================
# Traitement du PDF
# =============================================================================


def _checkpoint_record(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": "done",
        "page_num": int(result["page_num"]),
        "canonical": str(result["canonical"]),
        "markdown": str(result["markdown"]),
        "summary_markdown": str(result["summary_markdown"]),
        "quality": dict(result["quality"]),
        "stats": dict(result["stats"]),
        "updated_at_utc": _utc_now(),
    }


def _record_to_result(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "page_num": int(record["page_num"]),
        "canonical": str(record["canonical"]),
        "markdown": str(record["markdown"]),
        "summary_markdown": str(record["summary_markdown"]),
        "quality": dict(record["quality"]),
        "stats": dict(record["stats"]),
    }


def _quality_status(page_qualities: Iterable[Dict[str, Any]]) -> str:
    statuses = [str(item.get("status", "unknown")) for item in page_qualities]
    if any(status == "unavailable" for status in statuses):
        return "partial"
    if any(status in {"degraded", "warning", "unknown"} for status in statuses):
        return "warning"
    return "ok"


def _build_quality_report(
    *,
    source_id: str,
    pdf_info: Dict[str, Any],
    page_results: List[Dict[str, Any]],
    validation: Dict[str, Any],
    duration_seconds: float,
    effective_workers: int,
) -> Dict[str, Any]:
    page_qualities = [dict(item["quality"]) for item in page_results]
    status_counts = Counter(str(item.get("status", "unknown")) for item in page_qualities)
    return {
        "schema": "qwen-canonical-ocr-quality-v1",
        "pipeline_version": ocr.PIPELINE_VERSION,
        "pipeline_fingerprint": ocr.get_pipeline_fingerprint(),
        "created_at_utc": _utc_now(),
        "source_id": source_id,
        "source_filename": pdf_info.get("filename"),
        "page_count": int(pdf_info["page_count"]),
        "quality_status": _quality_status(page_qualities),
        "status_counts": dict(sorted(status_counts.items())),
        "one_qwen_generation_per_page": True,
        "semantic_retries": 0,
        "deterministic_markdown": True,
        "markdown_structure_valid": bool(validation.get("ok")),
        "markdown_structure_errors": list(validation.get("errors", []) or []),
        "duration_seconds": round(float(duration_seconds), 3),
        "workers_effective": int(effective_workers),
        "pages": page_qualities,
    }


def run_for_pdf(
    pdf_path: str,
    api_key: str,
    *,
    output_md_path: Optional[str] = None,
    output_summary_path: Optional[str] = None,
    output_canonical_path: Optional[str] = None,
    output_quality_path: Optional[str] = None,
    progress_gcs_uri: Optional[str] = None,
    source_id: Optional[str] = None,
) -> Dict[str, Any]:
    started = time.time()
    pdf_path = str(Path(pdf_path).resolve())
    pdf_info = ocr.get_pdf_info(pdf_path)
    page_count = int(pdf_info["page_count"])
    source_id = source_id or _local_source_id(pdf_path)
    effective_workers = max(1, min(PAGE_WORKERS, page_count))
    ocr.configure_explicit_cache_for_batch(page_count, effective_workers)

    print("\n" + "=" * 78)
    print("🔬 EXTRACTION CANONIQUE QWEN → MARKDOWN PYTHON")
    print("=" * 78)
    print(f"📄 PDF                 : {pdf_path}")
    print(f"📄 Pages               : {page_count}")
    print(f"🧩 Module              : {_loaded_ocr_path()}")
    print(f"🤖 Modèle              : {ocr.MODEL_OCR}")
    print(f"📞 Générations/page    : {ocr.NOMINAL_GENERATIONS_PER_PAGE}")
    print(f"🧵 Workers             : {effective_workers}")
    print(
        f"🖼️  Rendu source        : "
        f"{max(ocr.RENDER_DPI, ocr.DETAIL_DPI if getattr(ocr, 'ENABLE_DETAIL_VIEWS', False) else ocr.RENDER_DPI)} DPI"
    )
    print(f"🔎 Vues détaillées     : {bool(getattr(ocr, 'ENABLE_DETAIL_VIEWS', False))}")
    print("📝 Markdown            : déterministe, aucun second LLM")
    print("=" * 78)

    checkpoint_pages = ocr.load_progress(
        pdf_path,
        expected_source_id=source_id,
        expected_page_count=page_count,
    )
    results_by_page: Dict[int, Dict[str, Any]] = {
        int(key): _record_to_result(record) for key, record in checkpoint_pages.items()
    }
    if results_by_page:
        print(f"📂 Reprise checkpoint  : {len(results_by_page)}/{page_count} page(s)")

    image_dir = tempfile.mkdtemp(prefix="qwen_canonical_images_")
    completed_since_gcs_sync = 0

    def persist_checkpoint(*, force_gcs: bool = False) -> None:
        nonlocal completed_since_gcs_sync
        records = {
            str(page_num): _checkpoint_record(result)
            for page_num, result in sorted(results_by_page.items())
        }
        ocr.save_progress(
            pdf_path,
            records,
            source_id=source_id,
            page_count=page_count,
        )
        completed_since_gcs_sync += 1
        if progress_gcs_uri and (
            force_gcs or completed_since_gcs_sync >= PROGRESS_SAVE_EVERY
        ):
            upload_to_gcs(ocr.get_progress_path(pdf_path), progress_gcs_uri, quiet=True)
            completed_since_gcs_sync = 0

    missing_pages = [page for page in range(1, page_count + 1) if page not in results_by_page]
    global_failure: Optional[Tuple[int, BaseException]] = None

    try:
        if missing_pages:
            with ThreadPoolExecutor(max_workers=effective_workers) as executor:
                futures: Dict[Future, int] = {
                    executor.submit(ocr.process_page, pdf_path, page_num, api_key, image_dir): page_num
                    for page_num in missing_pages
                }
                for future in as_completed(futures):
                    page_num = futures[future]
                    try:
                        result = future.result()
                    except BaseException as error:
                        if _is_global_failure(error) or (
                            bool(ocr.STOP_ON_CRITICAL) and not bool(ocr.PUBLISH_PARTIAL_DOCUMENT)
                        ):
                            global_failure = (page_num, error)
                            for pending in futures:
                                pending.cancel()
                            break
                        print(f"⚠️ Page {page_num}: extraction indisponible, page de secours publiée : {error}")
                        result = ocr.build_unavailable_page(page_num, error)
                    results_by_page[page_num] = result
                    persist_checkpoint()
                    print(
                        f"📌 Progression          : {len(results_by_page)}/{page_count} "
                        f"(page {page_num}, qualité={result['quality']['status']})"
                    )
        if global_failure is not None:
            page_num, error = global_failure
            raise RuntimeError(f"Échec global page {page_num}: {error}") from error
        persist_checkpoint(force_gcs=True)
    finally:
        shutil.rmtree(image_dir, ignore_errors=True)

    page_results = [results_by_page[page] for page in range(1, page_count + 1)]
    final_markdown = "\n\n".join(item["markdown"].rstrip("\n") for item in page_results) + "\n"
    final_summary = "\n\n".join(
        item["summary_markdown"].rstrip("\n") for item in page_results
    ) + "\n"
    final_canonical = "\n\n".join(
        item["canonical"].rstrip("\n") for item in page_results
    ) + "\n"

    validation = ocr.validate_markdown_quality(final_markdown, page_count)
    if not validation.get("ok"):
        # Le rendu Python ne doit normalement jamais échouer. En cas de défaut
        # technique, on conserve néanmoins tout le contenu et on le signale.
        print("⚠️ Validation Markdown déterministe : " + " | ".join(validation.get("errors", [])))

    duration = time.time() - started
    output_md_path = output_md_path or str(Path(pdf_path).with_suffix(".md"))
    paths = _derive_local_sidecars(
        output_md_path,
        summary_path=output_summary_path,
        canonical_path=output_canonical_path,
        quality_path=output_quality_path,
    )
    quality_report = _build_quality_report(
        source_id=source_id,
        pdf_info=pdf_info,
        page_results=page_results,
        validation=validation,
        duration_seconds=duration,
        effective_workers=effective_workers,
    )

    _atomic_write_text(paths["markdown"], final_markdown)
    _atomic_write_text(paths["summary"], final_summary)
    _atomic_write_text(paths["canonical"], final_canonical)
    _atomic_write_json(paths["quality"], quality_report)

    all_stats = [dict(item["stats"]) for item in page_results]
    costs = ocr.calculate_costs(all_stats)
    sizes_kb = {
        name: Path(path).stat().st_size / 1024.0 for name, path in paths.items()
    }

    print("\n" + "=" * 78)
    print("✅ EXTRACTION TERMINÉE")
    print("=" * 78)
    print(f"📝 Markdown complet    : {paths['markdown']}")
    print(f"🧾 Markdown synthèse   : {paths['summary']}")
    print(f"🔐 Source canonique    : {paths['canonical']}")
    print(f"🩺 Rapport qualité     : {paths['quality']}")
    print(f"📊 Qualité globale     : {quality_report['quality_status']}")
    print(f"⏱️  Durée               : {duration:.1f}s ({duration / page_count:.1f}s/page)")
    print("=" * 78 + "\n")

    return {
        "paths": paths,
        "page_count": page_count,
        "duration_seconds": duration,
        "sizes_kb": sizes_kb,
        "stats": all_stats,
        "costs": costs,
        "quality": quality_report,
        "worker_count": effective_workers,
        "source_id": source_id,
    }


# =============================================================================
# Main Cloud Run / local
# =============================================================================


def _send_callback(callback_url: str, payload: Dict[str, Any]) -> None:
    print(f"📡 Envoi callback : {callback_url}")
    response = requests.post(callback_url, json=payload, timeout=30)
    response.raise_for_status()
    print(f"✅ Callback envoyé ({response.status_code})")


def main() -> None:
    callback_url = os.getenv("CALLBACK_URL")
    ocr_job_id = os.getenv("OCR_JOB_ID")
    try:
        api_key = (os.getenv("DASHSCOPE_API_KEY") or "").strip()
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY non définie.")

        _validate_ocr_contract()
        ocr.validate_api_configuration()
        print(f"🧩 Module chargé       : {OCR_MODULE_NAME}")
        print(f"📦 Fichier chargé      : {_loaded_ocr_path()}")
        print(f"🌐 Endpoint Qwen       : {ocr.API_URL}")

        gcs_input = (os.getenv("GCS_INPUT_URI") or "").strip()
        gcs_output = (os.getenv("GCS_OUTPUT_URI") or "").strip()
        local_input = (os.getenv("INPUT_PDF_PATH") or "").strip()

        if gcs_input:
            local_pdf = "/tmp/input.pdf"
            source_id = download_from_gcs(gcs_input, local_pdf)
            progress_gcs_uri = (
                os.getenv("GCS_PROGRESS_URI") or derive_progress_gcs_uri(gcs_input)
            )
            local_progress = ocr.get_progress_path(local_pdf)
            Path(local_progress).unlink(missing_ok=True)
            if download_from_gcs_if_exists(progress_gcs_uri, local_progress):
                print(f"📂 Checkpoint repris   : {progress_gcs_uri}")
            else:
                print(f"📂 Aucun checkpoint    : {progress_gcs_uri}")

            if not gcs_output:
                bucket, blob = parse_gs_uri(gcs_input)
                relative = blob[len("in/") :] if blob.startswith("in/") else blob
                base = relative.rsplit(".", 1)[0] if "." in relative else relative
                gcs_output = f"gs://{bucket}/out/{base}.md"
            gcs_output = _canonical_gs_uri(gcs_output)

            gcs_summary = _canonical_gs_uri(
                os.getenv("GCS_SUMMARY_OUTPUT_URI")
                or _derive_gcs_sidecar(gcs_output, ".summary.md")
            )
            gcs_canonical = _canonical_gs_uri(
                os.getenv("GCS_OCR_OUTPUT_URI")
                or _derive_gcs_sidecar(gcs_output, ".ocr.txt")
            )
            gcs_quality = _canonical_gs_uri(
                os.getenv("GCS_QUALITY_OUTPUT_URI")
                or _derive_gcs_sidecar(gcs_output, ".quality.json")
            )

            result = run_for_pdf(
                local_pdf,
                api_key,
                output_md_path="/tmp/output.md",
                output_summary_path="/tmp/output.summary.md",
                output_canonical_path="/tmp/output.ocr.txt",
                output_quality_path="/tmp/output.quality.json",
                progress_gcs_uri=progress_gcs_uri,
                source_id=source_id,
            )
            upload_to_gcs(result["paths"]["markdown"], gcs_output)
            upload_to_gcs(result["paths"]["summary"], gcs_summary)
            upload_to_gcs(result["paths"]["canonical"], gcs_canonical)
            upload_to_gcs(result["paths"]["quality"], gcs_quality)
            delete_from_gcs(progress_gcs_uri, quiet=True)
            ocr.clear_progress(local_pdf)

            print("=" * 78)
            print(f"🔗 LOVABLE_MARKDOWN_GCS={gcs_output}")
            print(f"🔗 CANONICAL_OCR_GCS={gcs_canonical}")
            print(f"🔗 SUMMARY_MARKDOWN_GCS={gcs_summary}")
            print(f"🔗 QUALITY_REPORT_GCS={gcs_quality}")
            print("=" * 78)

            if callback_url and ocr_job_id:
                stats = result["stats"]
                total_in = sum(int(item.get("input_tokens", 0) or 0) for item in stats)
                total_out = sum(int(item.get("output_tokens", 0) or 0) for item in stats)
                total_reasoning = sum(int(item.get("reasoning_tokens", 0) or 0) for item in stats)
                total_image = sum(int(item.get("image_tokens", 0) or 0) for item in stats)
                callback_payload = {
                    "ocrJobId": ocr_job_id,
                    "gcsOutputPath": gcs_output,
                    "summaryGcsPath": gcs_summary,
                    "canonicalOcrGcsPath": gcs_canonical,
                    "qualityGcsPath": gcs_quality,
                    "status": "success",
                    "qualityStatus": result["quality"]["quality_status"],
                    "pageCount": result["page_count"],
                    "durationSeconds": result["duration_seconds"],
                    "markdownSizeKb": result["sizes_kb"]["markdown"],
                    "stats": {
                        "inputTokens": total_in,
                        "outputTokens": total_out,
                        "reasoningTokens": total_reasoning,
                        "imageTokens": total_image,
                        "workerCountEffective": result["worker_count"],
                        "generationsPerPage": 1,
                        "semanticRetries": 0,
                        "deterministicMarkdown": True,
                        "pipelineVersion": ocr.PIPELINE_VERSION,
                        "models": {"ocr": ocr.MODEL_OCR},
                    },
                }
                try:
                    _send_callback(callback_url, callback_payload)
                except Exception as callback_error:
                    print(f"⚠️ Erreur callback : {callback_error}")

        elif local_input:
            output_md = (os.getenv("OUTPUT_MD_PATH") or "").strip() or None
            result = run_for_pdf(local_input, api_key, output_md_path=output_md)
            ocr.clear_progress(local_input)
            print(json.dumps(result["paths"], ensure_ascii=False, indent=2))
        else:
            raise RuntimeError("Ni GCS_INPUT_URI ni INPUT_PDF_PATH définis.")

    except Exception as error:
        print(f"\n❌ Erreur fatale dans qwenocr_runner.py : {error}", file=sys.stderr)
        traceback.print_exc()
        if callback_url and ocr_job_id:
            try:
                _send_callback(
                    callback_url,
                    {
                        "ocrJobId": ocr_job_id,
                        "status": "error",
                        "error": str(error)[:2000],
                        "pipelineVersion": getattr(ocr, "PIPELINE_VERSION", None),
                    },
                )
            except Exception as callback_error:
                print(f"⚠️ Callback d'erreur impossible : {callback_error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

