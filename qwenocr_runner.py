#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
qwenocr_runner.py — runner Cloud Run/local v6.5.0, une seule sortie Markdown.

Par page :
    quatre vues JPEG ciblées de la même page -> un flux SSE Qwen nominal -> source canonique
    -> Markdown Python déterministe

La source canonique et la qualité sont conservées uniquement dans le checkpoint
technique temporaire afin de permettre une reprise. Après succès, le seul
artefact documentaire publié est le fichier Markdown final.
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
from typing import Any, Dict, Iterable, Optional, Tuple

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
# Contrat et configuration
# =============================================================================


def _validate_ocr_contract() -> None:
    required_attributes = [
        "API_URL",
        "MODEL",
        "MODEL_OCR",
        "PIPELINE_VERSION",
        "CANONICAL_OCR_ONLY",
        "DETERMINISTIC_MARKDOWN",
        "SINGLE_MARKDOWN_OUTPUT",
        "OCR_PROMPT_IN_USER_MESSAGE",
        "NOMINAL_GENERATIONS_PER_PAGE",
        "SEMANTIC_RETRIES",
        "STOP_ON_CRITICAL",
        "PUBLISH_PARTIAL_DOCUMENT",
        "PUBLISH_DEGRADED_MARKDOWN",
        "ENABLE_EXPLICIT_CACHE",
        "QWEN_HIGH_RES_IMAGES",
        "STREAMING_OCR",
        "STREAM_INCLUDE_USAGE",
        "THINKING_BUDGET_OCR",
        "MAX_COMPLETION_TOKENS_OCR",
        "OCR_SEED",
        "RENDER_DPI",
        "DETAIL_DPI",
        "CRITICAL_DPI",
        "BODY_VIEW_TOP",
        "BODY_VIEW_BOTTOM",
        "FOOTER_VIEW_TOP",
        "CRITICAL_VIEW_LEFT",
        "CRITICAL_VIEW_TOP",
        "CRITICAL_VIEW_BOTTOM",
        "VIEW_JPEG_QUALITY",
        "MAX_VIEW_PIXELS",
        "MAX_REQUEST_BODY_MB",
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
            "ocr_qwenVL.py incompatible. Déploie ensemble les deux fichiers v6.5.0. "
            "Éléments absents : " + ", ".join(sorted(set(missing)))
        )
    if ocr.CANONICAL_OCR_ONLY is not True:
        raise RuntimeError("Contrat invalide : Qwen doit produire uniquement la source canonique.")
    if ocr.DETERMINISTIC_MARKDOWN is not True:
        raise RuntimeError("Contrat invalide : le Markdown doit être rendu par Python.")
    if ocr.SINGLE_MARKDOWN_OUTPUT is not True:
        raise RuntimeError("Contrat invalide : une seule sortie Markdown est autorisée.")
    if ocr.OCR_PROMPT_IN_USER_MESSAGE is not True:
        raise RuntimeError("Contrat invalide : le prompt OCR doit être placé dans le message utilisateur multimodal.")
    if int(ocr.NOMINAL_GENERATIONS_PER_PAGE) != 1:
        raise RuntimeError("Contrat invalide : une seule génération Qwen est autorisée par page.")
    if int(ocr.SEMANTIC_RETRIES) != 0:
        raise RuntimeError("Contrat invalide : aucune relance sémantique n'est autorisée.")
    if ocr.STREAMING_OCR is not True or ocr.STREAM_INCLUDE_USAGE is not True:
        raise RuntimeError("Contrat invalide : le flux SSE avec usage final est obligatoire.")


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


_workers_raw = os.getenv("PAGE_WORKERS", os.getenv("PIPELINE_CONCURRENCY", "4")).strip()
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
# Utilitaires
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
        raise RuntimeError("google-cloud-storage n'est pas installé.")
    if _GCS_CLIENT is None:
        _GCS_CLIENT = storage.Client()
    return _GCS_CLIENT


def parse_gs_uri(path: str) -> Tuple[str, str]:
    """Compatibilité historique : les objets sont routés vers QWEN_BUCKET."""
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
            print(f"🗑️ Objet supprimé : gs://{bucket_name}/{blob_name}")


def derive_progress_gcs_uri(gcs_input: str) -> str:
    bucket_name, blob_name = parse_gs_uri(gcs_input)
    relative = blob_name[len("in/") :] if blob_name.startswith("in/") else blob_name
    base = relative.rsplit(".", 1)[0] if "." in relative else relative
    return f"gs://{bucket_name}/progress/{base}.progress.json"


# =============================================================================
# Traitement
# =============================================================================


def _checkpoint_record(result: Dict[str, Any]) -> Dict[str, Any]:
    """Le canonique reste interne au checkpoint et n'est jamais publié."""
    return {
        "status": "done",
        "page_num": int(result["page_num"]),
        "canonical": str(result["canonical"]),
        "markdown": str(result["markdown"]),
        "quality": dict(result["quality"]),
        "stats": dict(result["stats"]),
        "updated_at_utc": _utc_now(),
    }


def _record_to_result(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "page_num": int(record["page_num"]),
        "canonical": str(record["canonical"]),
        "markdown": str(record["markdown"]),
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


def _document_reference_conflicts(page_results: Iterable[Dict[str, Any]]) -> list[Dict[str, Any]]:
    groups: Dict[Tuple[str, str], list[Dict[str, Any]]] = {}
    for result in page_results:
        page_num = int(result.get("page_num", 0) or 0)
        quality = dict(result.get("quality") or {})
        for record in quality.get("reference_records", []) or []:
            suffix = str(record.get("suffix", ""))
            designation_key = str(record.get("designation_key", ""))
            reference = str(record.get("reference", ""))
            if not suffix or len(designation_key) < 6 or not reference:
                continue
            groups.setdefault((suffix, designation_key), []).append(
                {**dict(record), "page_num": page_num}
            )

    conflicts: list[Dict[str, Any]] = []
    for (suffix, designation_key), records in groups.items():
        references = sorted({str(record["reference"]) for record in records})
        if len(references) <= 1:
            continue
        conflicts.append(
            {
                "suffix": suffix,
                "designation_key": designation_key,
                "references": references,
                "pages": sorted({int(record["page_num"]) for record in records}),
            }
        )
    return conflicts


def run_for_pdf(
    pdf_path: str,
    api_key: str,
    *,
    output_md_path: Optional[str] = None,
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
    print("🔬 EXTRACTION CANONIQUE QWEN → MARKDOWN DÉTERMINISTE")
    print("=" * 78)
    print(f"📄 PDF                 : {pdf_path}")
    print(f"📄 Pages               : {page_count}")
    print(f"🧩 Module              : {_loaded_ocr_path()}")
    print(f"🤖 Modèle              : {ocr.MODEL_OCR}")
    print(f"📞 Générations/page    : {ocr.NOMINAL_GENERATIONS_PER_PAGE} nominale")
    print("🌊 Streaming OCR       : SSE activé, usage dans le dernier événement")
    print(f"🧠 Budget thinking     : {ocr.THINKING_BUDGET_OCR} tokens")
    print(f"🧾 Budget total sortie : {ocr.MAX_COMPLETION_TOKENS_OCR} tokens")
    print(f"🎯 Graine déterministe : {ocr.OCR_SEED}")
    print("🧭 Prompt OCR          : message utilisateur multimodal")
    print(f"🧵 Workers             : {effective_workers}")
    print(f"🖼️ Vue complète        : JPEG {ocr.RENDER_DPI} DPI")
    print(
        f"🔎 Corps / pied        : JPEG {ocr.DETAIL_DPI} DPI — "
        f"{int(round(ocr.BODY_VIEW_TOP * 100))}-{int(round(ocr.BODY_VIEW_BOTTOM * 100))}% / "
        f"{int(round(ocr.FOOTER_VIEW_TOP * 100))}-100%"
    )
    print(
        f"🎯 Vue numérique droite: JPEG {ocr.CRITICAL_DPI} DPI — "
        f"x={int(round(ocr.CRITICAL_VIEW_LEFT * 100))}-100%, "
        f"y={int(round(ocr.CRITICAL_VIEW_TOP * 100))}-{int(round(ocr.CRITICAL_VIEW_BOTTOM * 100))}%"
    )
    print(f"🧮 Pixels max/vue      : {ocr.MAX_VIEW_PIXELS:,}")
    print(f"📦 Corps HTTP maximal  : {ocr.MAX_REQUEST_BODY_MB:.1f} Mo (pré-contrôle exact)")
    print("🛟 Repli 413            : compression/résolution seulement, aucune réanalyse")
    print("📝 Sortie documentaire : un seul fichier Markdown")
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
                            bool(ocr.STOP_ON_CRITICAL)
                            and not bool(ocr.PUBLISH_PARTIAL_DOCUMENT)
                        ):
                            global_failure = (page_num, error)
                            for pending in futures:
                                pending.cancel()
                            break
                        print(f"⚠️ Page {page_num}: page de secours publiée : {error}")
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
    reference_conflicts = _document_reference_conflicts(page_results)
    document_comment = (
        "<!-- DOCUMENT_QUALITY "
        f"reference_conflicts={len(reference_conflicts)} "
        f"critical={'failed' if reference_conflicts else 'passed'} -->"
    )
    final_markdown = document_comment + "\n\n" + "\n\n".join(
        item["markdown"].rstrip("\n") for item in page_results
    ) + "\n"

    validation = ocr.validate_markdown_quality(final_markdown, page_count)
    if not validation.get("ok"):
        # Une anomalie de rendu technique est signalée sans supprimer le contenu.
        print("⚠️ Validation Markdown : " + " | ".join(validation.get("errors", [])))

    output_md_path = output_md_path or str(Path(pdf_path).with_suffix(".md"))
    _atomic_write_text(output_md_path, final_markdown)

    duration = time.time() - started
    page_qualities = [dict(item["quality"]) for item in page_results]
    status_counts = Counter(str(item.get("status", "unknown")) for item in page_qualities)
    quality_status = _quality_status(page_qualities)
    if reference_conflicts and quality_status == "ok":
        quality_status = "warning"
    all_stats = [dict(item["stats"]) for item in page_results]
    costs = ocr.calculate_costs(all_stats)

    print("\n" + "=" * 78)
    print("✅ EXTRACTION TERMINÉE")
    print("=" * 78)
    print(f"📝 Markdown            : {output_md_path}")
    print(f"📊 Qualité globale     : {quality_status}")
    print(f"📊 Statuts pages       : {dict(sorted(status_counts.items()))}")
    print(f"🔎 Conflits références : {len(reference_conflicts)}")
    print(f"⏱️ Durée               : {duration:.1f}s ({duration / page_count:.1f}s/page)")
    print("=" * 78 + "\n")

    return {
        "path": output_md_path,
        "page_count": page_count,
        "duration_seconds": duration,
        "size_kb": Path(output_md_path).stat().st_size / 1024.0,
        "stats": all_stats,
        "costs": costs,
        "quality_status": quality_status,
        "status_counts": dict(sorted(status_counts.items())),
        "worker_count": effective_workers,
        "source_id": source_id,
        "markdown_structure_valid": bool(validation.get("ok")),
        "markdown_structure_errors": list(validation.get("errors", []) or []),
        "reference_conflicts": reference_conflicts,
    }


# =============================================================================
# Main
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
            progress_gcs_uri = os.getenv("GCS_PROGRESS_URI") or derive_progress_gcs_uri(gcs_input)
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

            result = run_for_pdf(
                local_pdf,
                api_key,
                output_md_path="/tmp/output.md",
                progress_gcs_uri=progress_gcs_uri,
                source_id=source_id,
            )
            upload_to_gcs(result["path"], gcs_output)
            delete_from_gcs(progress_gcs_uri, quiet=True)
            ocr.clear_progress(local_pdf)

            print("=" * 78)
            print(f"🔗 LOVABLE_MARKDOWN_GCS={gcs_output}")
            print("=" * 78)

            if callback_url and ocr_job_id:
                stats = result["stats"]
                callback_payload = {
                    "ocrJobId": ocr_job_id,
                    "gcsOutputPath": gcs_output,
                    "status": "success",
                    "qualityStatus": result["quality_status"],
                    "pageCount": result["page_count"],
                    "durationSeconds": result["duration_seconds"],
                    "markdownSizeKb": result["size_kb"],
                    "stats": {
                        "inputTokens": sum(int(item.get("input_tokens", 0) or 0) for item in stats),
                        "outputTokens": sum(int(item.get("output_tokens", 0) or 0) for item in stats),
                        "reasoningTokens": sum(int(item.get("reasoning_tokens", 0) or 0) for item in stats),
                        "imageTokens": sum(int(item.get("image_tokens", 0) or 0) for item in stats),
                        "payloadFallbacks": sum(int(item.get("payload_fallback_count", 0) or 0) for item in stats),
                        "workerCountEffective": result["worker_count"],
                        "generationsPerPage": 1,
                        "semanticRetries": 0,
                        "referenceConflicts": len(result.get("reference_conflicts", []) or []),
                        "ocrSeed": ocr.OCR_SEED,
                        "thinkingBudget": ocr.THINKING_BUDGET_OCR,
                        "maxCompletionTokens": ocr.MAX_COMPLETION_TOKENS_OCR,
                        "deterministicMarkdown": True,
                        "singleMarkdownOutput": True,
                        "ocrPromptInUserMessage": True,
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
            print(result["path"])
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

