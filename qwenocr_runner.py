#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import hashlib
import importlib
import os
import shutil
import sys
import time
import traceback
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

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
        "MARKDOWN_INDEPENDENT_FROM_OCR", "MARKDOWN_STRUCTURAL_CLEANUP",
        "ENABLE_THINKING_OCR", "ENABLE_THINKING_MD",
        "ALLOW_NO_THINK_FALLBACK_OCR", "ALLOW_NO_THINK_FALLBACK_MD",
        "MARKDOWN_FORMAT_RETRIES", "TWO_QUEUE_PIPELINE",
        "NOMINAL_TWO_GENERATIONS", "TARGETED_RECOVERY_ENABLED",
        "OCR_RECOVERY_ATTEMPTS", "MARKDOWN_RECOVERY_ATTEMPTS",
        "PIPELINE_VERSION",
    ]
    required_callables = [
        "validate_api_configuration", "configure_explicit_cache_for_batch",
        "get_pipeline_fingerprint", "get_progress_path", "get_pdf_info",
        "load_progress", "save_progress", "clear_progress",
        "run_ocr_stage", "run_markdown_stage", "cleanup_page_image",
        "get_page_image_path", "calculate_costs",
        "validate_markdown_quality", "validate_canonical_markdown_structure",
    ]
    missing = [name for name in required_attributes if not hasattr(ocr, name)]
    missing += [
        name for name in required_callables
        if not callable(getattr(ocr, name, None))
    ]
    if missing:
        loaded_path = getattr(ocr, "__file__", "chemin inconnu")
        raise RuntimeError(
            "Le fichier ocr_qwenVL.py chargé est ancien ou incompatible. "
            f"Chemin chargé : {loaded_path}. Éléments absents/non appelables : "
            + ", ".join(sorted(set(missing)))
            + ". Déploie ensemble les deux fichiers fournis."
        )
    if ocr.MARKDOWN_USES_IMAGE is not True:
        raise RuntimeError(
            "Contrat incompatible : la phase Markdown doit recevoir l'image."
        )
    if ocr.MARKDOWN_INDEPENDENT_FROM_OCR is not True:
        raise RuntimeError(
            "Contrat incompatible : aucun texte OCR ne doit être transmis au modèle Markdown."
        )
    if ocr.MARKDOWN_STRUCTURAL_CLEANUP is not False:
        raise RuntimeError(
            "Contrat incompatible : Python ne doit pas restructurer le Markdown."
        )
    if ocr.TWO_QUEUE_PIPELINE is not True:
        raise RuntimeError(
            "Contrat incompatible : le pipeline OCR et Markdown doit utiliser deux files."
        )
    if ocr.NOMINAL_TWO_GENERATIONS is not True:
        raise RuntimeError(
            "Contrat incompatible : le chemin nominal doit rester à 1 OCR + 1 Markdown."
        )
    if ocr.TARGETED_RECOVERY_ENABLED is not True:
        raise RuntimeError(
            "Contrat incompatible : la récupération ciblée doit être activée."
        )
    if bool(ocr.ALLOW_NO_THINK_FALLBACK_OCR) or bool(ocr.ALLOW_NO_THINK_FALLBACK_MD):
        raise RuntimeError(
            "Contrat incompatible : aucun second appel sans thinking n'est autorisé."
        )
    if int(ocr.MARKDOWN_FORMAT_RETRIES) != 0:
        raise RuntimeError(
            "Contrat incompatible : les avertissements structurels ne doivent "
            "déclencher aucune régénération Markdown."
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

MIN_CONCURRENCY = 1
MAX_CONCURRENCY = 8


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


# PAGE_WORKERS reste accepté comme alias de migration.
_pipeline_default_raw = os.getenv(
    "PIPELINE_CONCURRENCY",
    os.getenv("PAGE_WORKERS", "4"),
).strip()
try:
    PIPELINE_CONCURRENCY = int(_pipeline_default_raw)
except ValueError as exc:
    raise RuntimeError("PIPELINE_CONCURRENCY doit être un entier.") from exc
if not MIN_CONCURRENCY <= PIPELINE_CONCURRENCY <= MAX_CONCURRENCY:
    raise RuntimeError(
        f"PIPELINE_CONCURRENCY doit être compris entre {MIN_CONCURRENCY} "
        f"et {MAX_CONCURRENCY}."
    )

OCR_MAX_CONCURRENCY = _read_int_env(
    "OCR_MAX_CONCURRENCY",
    PIPELINE_CONCURRENCY,
    minimum=1,
    maximum=PIPELINE_CONCURRENCY,
)
MARKDOWN_MAX_CONCURRENCY = _read_int_env(
    "MARKDOWN_MAX_CONCURRENCY",
    PIPELINE_CONCURRENCY,
    minimum=1,
    maximum=PIPELINE_CONCURRENCY,
)
BALANCED_OCR_SLOTS = _read_int_env(
    "BALANCED_OCR_SLOTS",
    max(1, PIPELINE_CONCURRENCY // 2),
    minimum=0,
    maximum=OCR_MAX_CONCURRENCY,
)
BALANCED_MARKDOWN_SLOTS = _read_int_env(
    "BALANCED_MARKDOWN_SLOTS",
    PIPELINE_CONCURRENCY - BALANCED_OCR_SLOTS,
    minimum=0,
    maximum=MARKDOWN_MAX_CONCURRENCY,
)
if BALANCED_OCR_SLOTS + BALANCED_MARKDOWN_SLOTS > PIPELINE_CONCURRENCY:
    raise RuntimeError(
        "BALANCED_OCR_SLOTS + BALANCED_MARKDOWN_SLOTS ne doit pas dépasser "
        "PIPELINE_CONCURRENCY."
    )
if BALANCED_OCR_SLOTS + BALANCED_MARKDOWN_SLOTS < 1:
    raise RuntimeError(
        "Au moins un slot équilibré OCR ou Markdown doit être disponible."
    )

MARKDOWN_READY_BUFFER = _read_int_env(
    "MARKDOWN_READY_BUFFER",
    max(4, PIPELINE_CONCURRENCY * 2),
    minimum=1,
)
RECOVERY_CONCURRENCY = _read_int_env(
    "RECOVERY_CONCURRENCY",
    1,
    minimum=1,
    maximum=PIPELINE_CONCURRENCY,
)

# Alias uniquement pour compatibilité des métriques existantes.
PAGE_WORKERS = PIPELINE_CONCURRENCY
PAGE_WORKERS_RAW = _pipeline_default_raw

_GCS_CLIENT: Optional[Any] = None


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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_page_state(page_num: int) -> Dict[str, Any]:
    return {
        "page_num": int(page_num),
        "status": "pending_ocr",
        "ocr_text": None,
        "ocr_stats": None,
        "markdown": None,
        "stats": None,
        "ocr_attempts": 0,
        "markdown_attempts": 0,
        "recovered": False,
        "last_error": None,
        "last_error_phase": None,
        "updated_at_utc": _utc_now(),
    }


def _normalize_page_state(page_num: int, record: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    state = _new_page_state(page_num)
    if isinstance(record, dict):
        state.update(record)
    state["page_num"] = int(page_num)
    state["ocr_attempts"] = max(0, int(state.get("ocr_attempts", 0) or 0))
    state["markdown_attempts"] = max(
        0, int(state.get("markdown_attempts", 0) or 0)
    )
    state["recovered"] = bool(state.get("recovered", False))
    return state


def _set_error(
    state: Dict[str, Any],
    *,
    phase: str,
    error: BaseException,
    final: bool,
) -> None:
    state["status"] = "failed_final" if final else f"{phase}_retry_pending"
    state["last_error_phase"] = phase
    state["last_error"] = f"{type(error).__name__}: {error}"
    state["updated_at_utc"] = _utc_now()


def _is_global_failure(error: BaseException) -> bool:
    message = str(error).lower()
    global_markers = (
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
    return any(marker in message for marker in global_markers)


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
    Pipeline à deux files et quatre slots globaux.

    - file OCR et file Markdown séparées ;
    - OCR durablement sauvegardé avant l'assemblage final ;
    - génération Markdown depuis l'image seule, sans texte OCR ;
    - erreurs locales isolées ;
    - une récupération ciblée maximum par phase ;
    - aucun faux Markdown partiel.
    """
    _validate_ocr_contract()
    pdf_path = os.path.abspath(pdf_path)

    print("\n" + "=" * 78)
    print("🔬 EXTRACTION PDF → OCR + MARKDOWN (Qwen, deux files)")
    print("=" * 78)
    print(f"📄 Fichier PDF          : {pdf_path}")
    print(f"🧩 Pipeline             : {ocr.PIPELINE_VERSION}")
    print(f"💰 Modèle OCR           : {ocr.MODEL}")
    print(f"📝 Modèle Markdown      : {getattr(ocr, 'MODEL_MD', ocr.MODEL)}")
    print(f"🔢 Slots Qwen globaux   : {PIPELINE_CONCURRENCY}")
    print(f"🔎 Maximum OCR          : {OCR_MAX_CONCURRENCY}")
    print(f"📝 Maximum Markdown     : {MARKDOWN_MAX_CONCURRENCY}")
    print(
        f"⚖️  Cible équilibrée     : "
        f"{BALANCED_OCR_SLOTS} OCR + {BALANCED_MARKDOWN_SLOTS} Markdown"
    )
    print(f"📥 Buffer Markdown      : {MARKDOWN_READY_BUFFER}")
    print(f"🩹 Récupération         : {RECOVERY_CONCURRENCY} slot(s), ciblée")
    print(f"🧠 Cache explicite      : {'configuré' if ocr.ENABLE_EXPLICIT_CACHE else 'désactivé'}")
    print(f"🔎 Haute résolution     : {'activée' if ocr.QWEN_HIGH_RES_IMAGES else 'désactivée'}")
    print("🔒 Markdown indépendant : image seule, aucun OCR transmis")
    print("🧽 Python               : enveloppe uniquement, aucune restructuration")
    print("=" * 78)

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF introuvable: {pdf_path}")

    pdf_info = ocr.get_pdf_info(pdf_path)
    page_count = int(pdf_info["page_count"])
    print(f"\n📊 Pages                : {page_count}")
    print(f"💾 Taille               : {pdf_info['file_size_mb']:.2f} MB")

    if source_id is None:
        source_id = _local_source_id(pdf_path)

    loaded_states = ocr.load_progress(
        pdf_path,
        expected_source_id=source_id,
        expected_page_count=page_count,
    )
    page_states: Dict[str, Dict[str, Any]] = {
        str(page_num): _normalize_page_state(
            page_num, loaded_states.get(str(page_num))
        )
        for page_num in range(1, page_count + 1)
    }

    image_dir = "/tmp/qwen_pages"
    Path(image_dir).mkdir(parents=True, exist_ok=True)

    pending_ocr: Deque[int] = deque()
    ready_markdown: Deque[int] = deque()
    ocr_recovery_pages: Set[int] = set()
    markdown_recovery_pages: Set[int] = set()
    completed_pages: Set[int] = set()
    failed_final_pages: Set[int] = set()

    for page_num in range(1, page_count + 1):
        state = page_states[str(page_num)]
        status = str(state.get("status", "pending_ocr"))
        if status == "markdown_done":
            completed_pages.add(page_num)
        elif status == "ocr_done":
            ready_markdown.append(page_num)
        elif status == "markdown_retry_pending":
            markdown_recovery_pages.add(page_num)
        elif status == "ocr_retry_pending":
            ocr_recovery_pages.add(page_num)
        elif status == "failed_final":
            failed_final_pages.add(page_num)
        else:
            pending_ocr.append(page_num)

    if loaded_states:
        print(
            f"📂 Reprise : {len(completed_pages)} page(s) finie(s), "
            f"{len(ready_markdown)} OCR sauvegardé(s), "
            f"{len(ocr_recovery_pages) + len(markdown_recovery_pages)} "
            "page(s) en récupération"
        )
    else:
        print("📂 Aucune reprise : traitement complet")

    remaining_for_cache = page_count - len(completed_pages)
    cache_active = ocr.configure_explicit_cache_for_batch(
        page_count=max(0, remaining_for_cache),
        worker_count=min(PIPELINE_CONCURRENCY, max(1, remaining_for_cache)),
    )
    if ocr.ENABLE_EXPLICIT_CACHE:
        print(
            "🧠 Cache explicite      : "
            + ("actif" if cache_active else "omis pour une seule vague")
        )

    def persist_checkpoint(reason: str) -> None:
        ocr.save_progress(
            pdf_path,
            page_states,
            source_id=source_id,
            page_count=page_count,
        )
        if progress_gcs_uri:
            upload_to_gcs(
                ocr.get_progress_path(pdf_path),
                progress_gcs_uri,
                quiet=True,
            )
        print(f"         💾 Checkpoint sauvegardé : {reason}")

    start_time = time.time()
    global_failure: Optional[Tuple[int, str, BaseException]] = None

    active: Dict[Future, Tuple[str, int]] = {}

    def active_counts() -> Tuple[int, int]:
        active_ocr = sum(1 for phase, _page in active.values() if phase == "ocr")
        active_md = sum(
            1 for phase, _page in active.values() if phase == "markdown"
        )
        return active_ocr, active_md

    def submit_ocr(
        executor: ThreadPoolExecutor,
        page_num: int,
        *,
        recovery: bool = False,
    ) -> None:
        state = page_states[str(page_num)]
        state["ocr_attempts"] = int(state.get("ocr_attempts", 0) or 0) + 1
        state["updated_at_utc"] = _utc_now()
        future = executor.submit(
            ocr.run_ocr_stage,
            pdf_path,
            page_num,
            api_key,
            image_dir,
            recovery=recovery,
        )
        active[future] = ("ocr", page_num)
        active_ocr, active_md = active_counts()
        print(
            f"         ▶ OCR page {page_num} "
            f"(actifs: OCR={active_ocr}, MD={active_md}, total={len(active)})"
        )

    def submit_markdown(
        executor: ThreadPoolExecutor,
        page_num: int,
        *,
        recovery: bool = False,
    ) -> None:
        state = page_states[str(page_num)]
        if not isinstance(state.get("ocr_text"), str):
            raise RuntimeError(
                f"Page {page_num}: OCR absent pour l'annexe finale. "
                "Le modèle Markdown resterait indépendant, mais l'artefact complet exige les deux sorties."
            )
        if not isinstance(state.get("ocr_stats"), dict):
            raise RuntimeError(
                f"Page {page_num}: statistiques OCR absentes pour le rapport final."
            )
        state["markdown_attempts"] = int(
            state.get("markdown_attempts", 0) or 0
        ) + 1
        state["updated_at_utc"] = _utc_now()
        future = executor.submit(
            ocr.run_markdown_stage,
            pdf_path,
            page_num,
            api_key,
            image_dir,
            state["ocr_text"],
            state["ocr_stats"],
            recovery=recovery,
        )
        active[future] = ("markdown", page_num)
        active_ocr, active_md = active_counts()
        print(
            f"         ▶ Markdown page {page_num} "
            f"(actifs: OCR={active_ocr}, MD={active_md}, total={len(active)})"
        )

    def dispatch_available(executor: ThreadPoolExecutor) -> None:
        while len(active) < PIPELINE_CONCURRENCY:
            active_ocr, active_md = active_counts()

            can_md = bool(ready_markdown) and (
                active_md < MARKDOWN_MAX_CONCURRENCY
            )
            can_ocr = (
                bool(pending_ocr)
                and active_ocr < OCR_MAX_CONCURRENCY
                and len(ready_markdown) < MARKDOWN_READY_BUFFER
            )

            if not can_md and not can_ocr:
                break

            both_queues = bool(ready_markdown) and bool(pending_ocr)
            phase: Optional[str] = None

            if both_queues:
                if can_md and active_md < BALANCED_MARKDOWN_SLOTS:
                    phase = "markdown"
                elif can_ocr and active_ocr < BALANCED_OCR_SLOTS:
                    phase = "ocr"
                elif can_md:
                    phase = "markdown"
                elif can_ocr:
                    phase = "ocr"
            elif can_md:
                phase = "markdown"
            elif can_ocr:
                phase = "ocr"

            if phase == "markdown":
                submit_markdown(executor, ready_markdown.popleft())
            elif phase == "ocr":
                submit_ocr(executor, pending_ocr.popleft())
            else:
                break

    print("\n" + "=" * 78)
    print("🚀 PASSAGE PRINCIPAL")
    print("=" * 78)

    executor = ThreadPoolExecutor(max_workers=PIPELINE_CONCURRENCY)
    try:
        while pending_ocr or ready_markdown or active:
            dispatch_available(executor)

            if not active:
                if pending_ocr or ready_markdown:
                    raise RuntimeError(
                        "Planificateur bloqué alors que des pages restent en attente."
                    )
                break

            done, _pending = wait(
                list(active.keys()),
                return_when=FIRST_COMPLETED,
            )
            ocr_pages_ready_after_checkpoint: List[int] = []
            images_to_cleanup_after_checkpoint: List[str] = []
            checkpoint_reasons: List[str] = []

            for future in done:
                phase, page_num = active.pop(future)
                state = page_states[str(page_num)]

                try:
                    result = future.result()
                except Exception as error:
                    print(f"         ❌ {phase.upper()} page {page_num}: {error}")
                    _set_error(
                        state,
                        phase=phase,
                        error=error,
                        final=False,
                    )
                    checkpoint_reasons.append(
                        f"{phase} page {page_num} en erreur"
                    )
                    if _is_global_failure(error):
                        global_failure = (page_num, phase, error)
                        break

                    if phase == "ocr":
                        ocr_recovery_pages.add(page_num)
                    else:
                        markdown_recovery_pages.add(page_num)
                    print(
                        f"         ↪ Page {page_num} isolée ; le lot principal continue."
                    )
                    continue

                if phase == "ocr":
                    state["status"] = "ocr_done"
                    state["ocr_text"] = result["ocr_text"]
                    state["ocr_stats"] = result["ocr_stats"]
                    state["last_error"] = None
                    state["last_error_phase"] = None
                    state["updated_at_utc"] = _utc_now()
                    ocr_pages_ready_after_checkpoint.append(page_num)
                    checkpoint_reasons.append(f"OCR page {page_num} terminé")
                else:
                    state["status"] = "markdown_done"
                    state["markdown"] = result["markdown"]
                    state["stats"] = result["stats"]
                    state["recovered"] = False
                    state["last_error"] = None
                    state["last_error_phase"] = None
                    state["updated_at_utc"] = _utc_now()
                    completed_pages.add(page_num)
                    images_to_cleanup_after_checkpoint.append(
                        result["image_path"]
                    )
                    checkpoint_reasons.append(
                        f"Markdown page {page_num} terminé"
                    )

            if checkpoint_reasons:
                persist_checkpoint("; ".join(checkpoint_reasons))

            # L'OCR est persisté avant l'assemblage final. Il n'est jamais
            # transmis au modèle Markdown ; la mise en file reste une règle de reprise.
            for page_num in ocr_pages_ready_after_checkpoint:
                ready_markdown.append(page_num)
                print(
                    f"         ✅ OCR page {page_num} sauvegardé avant Markdown."
                )
            for image_path in images_to_cleanup_after_checkpoint:
                ocr.cleanup_page_image(image_path)

            if global_failure is not None:
                for pending_future in active:
                    pending_future.cancel()
                break
    finally:
        executor.shutdown(
            wait=global_failure is None,
            cancel_futures=global_failure is not None,
        )

    if global_failure is not None:
        page_num, phase, error = global_failure
        raise RuntimeError(
            f"Échec global pendant {phase} page {page_num}. "
            "Le checkpoint contient tous les états persistés."
        ) from error

    print("\n" + "=" * 78)
    print("🩹 RÉCUPÉRATION CIBLÉE")
    print("=" * 78)

    # Récupération OCR : une seule nouvelle génération OCR par page concernée.
    for page_num in sorted(ocr_recovery_pages):
        state = page_states[str(page_num)]
        max_total_attempts = 1 + int(ocr.OCR_RECOVERY_ATTEMPTS)
        if int(state.get("ocr_attempts", 0) or 0) >= max_total_attempts:
            _set_error(
                state,
                phase="ocr",
                error=RuntimeError("budget de récupération OCR épuisé"),
                final=True,
            )
            failed_final_pages.add(page_num)
            persist_checkpoint(f"OCR page {page_num} échec final")
            continue

        try:
            state["ocr_attempts"] = int(state.get("ocr_attempts", 0) or 0) + 1
            result = ocr.run_ocr_stage(
                pdf_path,
                page_num,
                api_key,
                image_dir,
                recovery=True,
            )
            state["status"] = "ocr_done"
            state["ocr_text"] = result["ocr_text"]
            state["ocr_stats"] = result["ocr_stats"]
            state["recovered"] = True
            state["last_error"] = None
            state["last_error_phase"] = None
            state["updated_at_utc"] = _utc_now()
            persist_checkpoint(f"OCR page {page_num} récupéré")
            markdown_recovery_pages.add(page_num)
            print(f"         ✅ OCR page {page_num} récupéré.")
        except Exception as error:
            if _is_global_failure(error):
                raise RuntimeError(
                    f"Échec global pendant la récupération OCR page {page_num}."
                ) from error
            _set_error(state, phase="ocr", error=error, final=True)
            failed_final_pages.add(page_num)
            persist_checkpoint(f"OCR page {page_num} échec final")
            print(f"         ❌ OCR page {page_num} échec final: {error}")

    # Récupération Markdown : image seule. L'OCR sauvegardé sert uniquement
    # à l'annexe et au rapport, jamais au contexte du modèle Markdown.
    def recover_markdown(page_num: int) -> Tuple[int, Optional[Dict[str, Any]], Optional[BaseException]]:
        state = page_states[str(page_num)]
        try:
            result = ocr.run_markdown_stage(
                pdf_path,
                page_num,
                api_key,
                image_dir,
                state["ocr_text"],
                state["ocr_stats"],
                recovery=True,
            )
            return page_num, result, None
        except BaseException as error:
            return page_num, None, error

    markdown_recovery_list = sorted(markdown_recovery_pages)
    if markdown_recovery_list:
        with ThreadPoolExecutor(max_workers=RECOVERY_CONCURRENCY) as recovery_executor:
            recovery_futures: Dict[Future, int] = {}
            for page_num in markdown_recovery_list:
                state = page_states[str(page_num)]
                max_total_attempts = 1 + int(
                    ocr.MARKDOWN_RECOVERY_ATTEMPTS
                )
                if int(state.get("markdown_attempts", 0) or 0) >= max_total_attempts:
                    _set_error(
                        state,
                        phase="markdown",
                        error=RuntimeError(
                            "budget de récupération Markdown épuisé"
                        ),
                        final=True,
                    )
                    failed_final_pages.add(page_num)
                    persist_checkpoint(
                        f"Markdown page {page_num} échec final"
                    )
                    continue

                state["markdown_attempts"] = int(
                    state.get("markdown_attempts", 0) or 0
                ) + 1
                recovery_futures[
                    recovery_executor.submit(recover_markdown, page_num)
                ] = page_num

            for future in recovery_futures:
                page_num, result, error = future.result()
                state = page_states[str(page_num)]
                if error is not None:
                    if _is_global_failure(error):
                        raise RuntimeError(
                            f"Échec global pendant la récupération Markdown "
                            f"page {page_num}."
                        ) from error
                    _set_error(
                        state,
                        phase="markdown",
                        error=error,
                        final=True,
                    )
                    failed_final_pages.add(page_num)
                    persist_checkpoint(
                        f"Markdown page {page_num} échec final"
                    )
                    print(
                        f"         ❌ Markdown page {page_num} échec final: {error}"
                    )
                    continue

                assert result is not None
                state["status"] = "markdown_done"
                state["markdown"] = result["markdown"]
                state["stats"] = result["stats"]
                state["recovered"] = True
                state["last_error"] = None
                state["last_error_phase"] = None
                state["updated_at_utc"] = _utc_now()
                completed_pages.add(page_num)
                persist_checkpoint(
                    f"Markdown page {page_num} récupéré"
                )
                ocr.cleanup_page_image(result["image_path"])
                print(f"         ✅ Markdown page {page_num} récupéré.")

    failed_final_pages = {
        page_num
        for page_num in range(1, page_count + 1)
        if page_states[str(page_num)].get("status") != "markdown_done"
    }
    if failed_final_pages:
        raise RuntimeError(
            "Traitement incomplet après récupération ciblée : pages "
            + ", ".join(str(page) for page in sorted(failed_final_pages))
            + ". Aucun Markdown final partiel n'a été publié."
        )

    all_markdown: List[str] = [
        str(page_states[str(page_num)]["markdown"])
        for page_num in range(1, page_count + 1)
    ]
    all_stats: List[Dict[str, Any]] = [
        dict(page_states[str(page_num)]["stats"])
        for page_num in range(1, page_count + 1)
    ]

    duration = time.time() - start_time

    print("\n" + "=" * 78)
    print("🔧 FINALISATION")
    print("=" * 78)
    final_markdown = "\n\n".join(
        page.rstrip("\n") for page in all_markdown
    )
    ocr.validate_canonical_markdown_structure(final_markdown, page_count)
    validation = ocr.validate_markdown_quality(final_markdown, page_count)
    if not validation.get("ok"):
        raise RuntimeError(
            "Validation finale refusée: "
            + " | ".join(validation.get("errors", []))
        )

    md_path = (
        Path(output_md_path)
        if output_md_path
        else Path(pdf_path).with_suffix(".md")
    )
    md_path.parent.mkdir(parents=True, exist_ok=True)
    temp_md_path = md_path.with_suffix(md_path.suffix + ".tmp")
    with open(temp_md_path, "w", encoding="utf-8") as handle:
        handle.write(final_markdown)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_md_path, md_path)

    md_size_kb = len(final_markdown.encode("utf-8")) / 1024
    costs = ocr.calculate_costs(all_stats)
    recovered_pages = sorted(
        page_num
        for page_num in range(1, page_count + 1)
        if bool(page_states[str(page_num)].get("recovered"))
        or int(page_states[str(page_num)].get("ocr_attempts", 0) or 0) > 1
        or int(page_states[str(page_num)].get("markdown_attempts", 0) or 0) > 1
    )

    if progress_gcs_uri:
        print("💾 Checkpoint GCS conservé jusqu'à l'upload final")
    else:
        ocr.clear_progress(pdf_path)
        print("🗑️  Checkpoint local supprimé après succès")

    shutil.rmtree(image_dir, ignore_errors=True)

    warned_pages = sum(
        1 for stats in all_stats
        if int(stats.get("markdown_warning_count", 0) or 0) > 0
    )
    markdown_warnings = sum(
        int(stats.get("markdown_warning_count", 0) or 0)
        for stats in all_stats
    )
    degraded_ocr_outputs = sum(
        1 for stats in all_stats
        if str(stats.get("ocr_output_status", stats.get("ocr_audit_status", "unknown"))) != "ok"
    )
    quality_status = (
        "warning"
        if markdown_warnings or degraded_ocr_outputs
        else "ok"
    )

    print("\n" + "=" * 78)
    print("✅ EXTRACTION TERMINÉE")
    print("=" * 78)
    print(f"📝 Fichier Markdown    : {md_path}")
    print(f"📄 Pages extraites     : {page_count}")
    print(f"💾 Taille Markdown     : {md_size_kb:.1f} KB")
    print(f"⏱️  Durée totale        : {duration // 60:.0f}min {duration % 60:.0f}s")
    print(f"⚡ Vitesse moyenne     : {duration / page_count:.1f}s/page")
    print(f"🩹 Pages récupérées    : {recovered_pages or 'aucune'}")
    print(
        f"⚠️  Markdown           : {markdown_warnings} avertissement(s) "
        f"sur {warned_pages} page(s)"
    )
    print(
        "✅ Qualité finale      : OK"
        if quality_status == "ok"
        else "⚠️  Qualité finale     : WARNING — contrôle recommandé"
    )
    print("=" * 78 + "\n")

    return (
        str(md_path),
        page_count,
        duration,
        md_size_kb,
        all_stats,
        costs,
        min(PIPELINE_CONCURRENCY, page_count),
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
        local_input = os.getenv("INPUT_PDF_PATH")

        if gcs_input:
            local_pdf = "/tmp/input.pdf"
            source_id = download_from_gcs(gcs_input, local_pdf)

            progress_gcs_uri = (
                os.getenv("GCS_PROGRESS_URI")
                or derive_progress_gcs_uri(gcs_input)
            )
            local_progress = ocr.get_progress_path(local_pdf)
            Path(local_progress).unlink(missing_ok=True)

            if download_from_gcs_if_exists(progress_gcs_uri, local_progress):
                print(f"📂 Checkpoint GCS téléchargé : {progress_gcs_uri}")
            else:
                print(f"📂 Aucun checkpoint GCS : {progress_gcs_uri}")

            if not gcs_output:
                bucket, blob = parse_gs_uri(gcs_input)
                rest = blob[len("in/"):] if blob.startswith("in/") else blob
                base = rest.rsplit(".", 1)[0] if "." in rest else rest
                gcs_output = f"gs://{bucket}/out/{base}.md"

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
            print("🗑️  Checkpoint GCS supprimé après upload réussi")

            print("=" * 78)
            print(f"🔗 LOVABLE_MARKDOWN_GCS={gcs_output}")
            print("=" * 78)

            callback_url = os.getenv("CALLBACK_URL")
            ocr_job_id = os.getenv("OCR_JOB_ID")

            if callback_url and ocr_job_id:
                try:
                    total_in = sum(
                        int(item.get("input_tokens", 0) or 0)
                        for item in all_stats
                    )
                    total_out = sum(
                        int(item.get("output_tokens", 0) or 0)
                        for item in all_stats
                    )
                    total_cached = sum(
                        int(item.get("cached_tokens", 0) or 0)
                        for item in all_stats
                    )
                    total_cache_created = sum(
                        int(item.get("cache_creation_input_tokens", 0) or 0)
                        for item in all_stats
                    )
                    total_reasoning = sum(
                        int(item.get("reasoning_tokens", 0) or 0)
                        for item in all_stats
                    )
                    total_image_tokens = sum(
                        int(item.get("image_tokens", 0) or 0)
                        for item in all_stats
                    )
                    total_partial_responses = sum(
                        int(item.get("partial_response_count", 0) or 0)
                        for item in all_stats
                    )

                    warning_pages = sorted(
                        index + 1
                        for index, item in enumerate(all_stats)
                        if int(item.get("markdown_warning_count", 0) or 0) > 0
                    )
                    warning_type_counts: Counter[str] = Counter()
                    for item in all_stats:
                        for warning in item.get("markdown_warnings", []) or []:
                            warning_type = str(warning).split(":", 1)[0].strip()
                            if warning_type:
                                warning_type_counts[warning_type] += 1

                    ocr_status_counts: Counter[str] = Counter(
                        str(item.get("ocr_output_status", item.get("ocr_audit_status", "unknown")) or "unknown")
                        for item in all_stats
                    )
                    degraded_ocr_pages = sum(
                        count
                        for status, count in ocr_status_counts.items()
                        if status != "ok"
                    )
                    total_markdown_warnings = sum(
                        int(item.get("markdown_warning_count", 0) or 0)
                        for item in all_stats
                    )
                    recovered_pages = sorted(
                        index + 1
                        for index, item in enumerate(all_stats)
                        if bool(item.get("recovered", False))
                    )
                    callback_quality_status = (
                        "warning"
                        if total_markdown_warnings or degraded_ocr_pages
                        else "ok"
                    )

                    payload = {
                        "ocrJobId": ocr_job_id,
                        "gcsOutputPath": gcs_output,
                        "status": "success",
                        "qualityStatus": callback_quality_status,
                        "envelopeValid": True,
                        "warningPages": warning_pages,
                        "warningTypes": dict(sorted(warning_type_counts.items())),
                        "ocrOutputStatus": dict(sorted(ocr_status_counts.items())),
                        "ocrAuditStatus": dict(sorted(ocr_status_counts.items())),  # compatibilité
                        "recoveredPages": recovered_pages,
                        "pageCount": page_count,
                        "durationSeconds": duration,
                        "markdownSizeKb": md_size_kb,
                        "stats": {
                            "inputTokens": total_in,
                            "outputTokens": total_out,
                            "reasoningTokens": total_reasoning,
                            "imageTokens": total_image_tokens,
                            "cachedTokens": total_cached,
                            "cacheCreationInputTokens": total_cache_created,
                            "partialResponses": total_partial_responses,
                            "cost": (
                                costs["cost_total"]
                                if costs.get("cost_available")
                                else None
                            ),
                            "costAvailable": bool(costs.get("cost_available")),
                            "pipelineConcurrency": PIPELINE_CONCURRENCY,
                            "ocrMaxConcurrency": OCR_MAX_CONCURRENCY,
                            "markdownMaxConcurrency": MARKDOWN_MAX_CONCURRENCY,
                            "balancedOcrSlots": BALANCED_OCR_SLOTS,
                            "balancedMarkdownSlots": BALANCED_MARKDOWN_SLOTS,
                            "recoveryConcurrency": RECOVERY_CONCURRENCY,
                            "workerCountEffective": worker_count,
                            "markdownWarnings": total_markdown_warnings,
                            "ocrOutputDegradedPages": degraded_ocr_pages,
                            "ocrAuditDegradedPages": degraded_ocr_pages,  # compatibilité
                            "twoQueuePipeline": True,
                            "markdownIndependentFromOcr": True,
                            "targetedRecovery": True,
                            "pipelineVersion": ocr.PIPELINE_VERSION,
                            "models": {
                                "ocr": getattr(ocr, "MODEL_OCR", ocr.MODEL),
                                "markdown": getattr(ocr, "MODEL_MD", ocr.MODEL),
                            },
                        },
                    }

                    print(f"📡 Envoi du callback à {callback_url} ...")
                    response = requests.post(
                        callback_url,
                        json=payload,
                        timeout=30,
                    )
                    response.raise_for_status()
                    print(f"✅ Callback envoyé ({response.status_code})")
                except Exception as error:
                    print(f"⚠️ Erreur callback: {error}")

        elif local_input:
            run_for_pdf(local_input, api_key)
        else:
            raise RuntimeError(
                "Ni GCS_INPUT_URI ni INPUT_PDF_PATH définis."
            )

    except Exception as error:
        print(
            "\n❌ Erreur fatale dans qwenocr_runner.py :",
            error,
            file=sys.stderr,
        )
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


