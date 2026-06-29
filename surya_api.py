#!/home/al/local/AI-tools/surya-api/.venv/bin/python
# -*- coding: utf-8 -*-

from typing import Optional
from filelock import FileLock, Timeout
import logging.config
import copy
import sys

from fastapi import FastAPI, File, UploadFile, Query, Path, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import io
import subprocess
# import pathlib
import argparse
from time import perf_counter
from PIL import Image, ImageChops

# import pypdfium2
import asyncio

from pydantic import BaseModel

from crown.table_rec import TableExtPredictor
from crown.utils import crop_by_percent, crop_by_side_percent, get_page_image
from crown.utils import poligon_expand
from crown.utils import trim_empty_background
from surya.layout.schema import LayoutBox, LayoutResult
from surya.recognition.schema import PageOCRResult
from surya.settings import settings
from crown.inference import ApiInferenceManager
from surya.recognition import RecognitionPredictor

# from surya.detection import DetectionPredictor
from surya.layout import LayoutPredictor
from surya.logging import get_logger
import httpx
import surya.inference.backends.spawn as spawn_module
from surya.inference.backends.spawn import (
    _cache_dir,
    _lock_path,
    _read_sentinel,
    _write_sentinel,
    _delete_sentinel,
    _stop_docker_container,
    _stop_process,
)

# Localhost must bypass system proxy (Windows SOCKS/HTTP proxy breaks Docker /health).
os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
os.environ.setdefault("no_proxy", "127.0.0.1,localhost")


def _patch_openai_no_system_proxy() -> None:
    """OpenAI uses httpx with trust_env=True; SOCKS proxies break localhost vLLM."""
    from openai import OpenAI

    if getattr(OpenAI, "_surya_api_trust_env_false", False):
        return
    _orig_init = OpenAI.__init__

    def _init(self, *args, **kwargs):
        kwargs.setdefault("http_client", httpx.Client(trust_env=False))
        _orig_init(self, *args, **kwargs)

    OpenAI.__init__ = _init  # type: ignore[method-assign]
    OpenAI._surya_api_trust_env_false = True


_patch_openai_no_system_proxy()

from contextlib import asynccontextmanager

from anyio.lowlevel import RunVar
from anyio import CapacityLimiter

from surya.table_rec import TableRecPredictor
from surya.table_rec.schema import TableCell, TableCol, TableResult, TableRow
import portalocker


logger = get_logger()

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,  # Crucial: Don't kill third-party loggers
    "formatters": {
        "default": {
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": "%(levelprefix)s %(asctime)s [%(name)s] %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        },
    },
    "root": {"handlers": ["console"], "level": "INFO"},
    "loggers": {
        # Explicitly configure your chosen third-party library logger here
        "surya": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": True,
        },
    },
}

logging.config.dictConfig(LOGGING_CONFIG)

backend_type = settings.SURYA_INFERENCE_BACKEND or "vllm"
LOCK_FILE = _cache_dir() / "surya-api_server.lock"
LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)


def update_request_count(delta: int = 1) -> None:
    lock = FileLock(str(_lock_path(backend_type)))
    try:
        with lock.acquire(timeout=1):
            data = _read_sentinel(backend_type)
            if not data:
                return
            count = data.get("request_count", 0) + delta
            if count < 0:
                count = 0
            data["request_count"] = count
            data["last_updated"] = perf_counter()
            _write_sentinel(backend_type, data)
    except Timeout:
        pass


def get_request_count() -> tuple[int, float | None, int | None]:
    lock = FileLock(str(_lock_path(backend_type)))
    try:
        with lock.acquire(timeout=1):
            data = _read_sentinel(backend_type) or {}
            if data and "request_count" not in data:
                data["request_count"] = 0
                data["last_updated"] = perf_counter()
                _write_sentinel(backend_type, data)
            return (
                data.get("request_count", 0),
                data.get("last_updated"),
                data.get("port"),
            )
    except Timeout:
        return 0, None, None


def probe_inference_health(base_url: str, timeout: float = 5.0) -> bool:
    """Probe vLLM /health; trust_env=False avoids proxy hijacking localhost."""
    url = f"{base_url.rstrip('/')}/health"
    try:
        with httpx.Client(timeout=timeout, trust_env=False) as client:
            r = client.get(url)
            if r.status_code == 200:
                return True
            logger.warning("Health check %s returned HTTP %s", url, r.status_code)
    except Exception as exc:
        logger.warning("Health check %s failed: %s", url, exc)
    return False


# Surya's spawn loop uses the same probe; patch so start() benefits too.
spawn_module.probe_health = lambda base_url, timeout=1.0: probe_inference_health(
    base_url, timeout=max(timeout, 5.0)
)


def _backend_handle():
    """ServerHandle from the active backend — only set after backend.start() succeeds."""
    return getattr(inference_manager.backend, "handle", None)


def _backend_client():
    return getattr(inference_manager.backend, "_client", None)


def _repair_openai_client_if_needed() -> None:
    """VllmBackend.start() returns early when handle exists, even if _client is None."""
    backend = inference_manager.backend
    handle = getattr(backend, "handle", None)
    if handle is None or getattr(backend, "_client", None) is not None:
        return
    from openai import OpenAI

    logger.info("Repairing missing OpenAI client for inference backend")
    backend._client = OpenAI(
        api_key=settings.VLLM_API_KEY,
        base_url=handle.base_url,
        http_client=httpx.Client(trust_env=False),
    )


def _handle_health_url(handle) -> str:
    base = handle.base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return base


async def ensure_inference_server() -> None:
    """Ensure the inference backend is connected.

    backend.handle is None until start() finishes — that is normal before the
    first OCR request. The backend's start() owns the full container lifecycle
    (create / start / recreate-on-unhealthy with a retry cap), so here we only
    decide whether to (re)connect.
    """
    handle = _backend_handle()
    client = _backend_client()
    if handle is not None and client is not None:
        if probe_inference_health(_handle_health_url(handle)):
            return
        logger.info("Stale inference handle; reconnecting")
        inference_manager.stop()
    elif handle is not None and client is None:
        logger.info("Inference handle without OpenAI client; reconnecting")
        inference_manager.stop()
    await asyncio.to_thread(inference_manager.start)
    _repair_openai_client_if_needed()
    if _backend_handle() is None:
        raise HTTPException(
            status_code=503,
            detail="Inference server failed to start (no backend handle).",
        )
    if _backend_client() is None:
        raise HTTPException(
            status_code=503,
            detail="Inference server failed to start (no OpenAI client).",
        )


def cleanup():
    logger.info("Cleaning up resources...")
    lock = FileLock(str(_lock_path(backend_type)))
    try:
        with lock.acquire(timeout=1):
            sentinel = _read_sentinel(backend_type)
            if not sentinel or not sentinel.get("cleanup_kind"):
                logger.info("No active server detected; skipping cleanup.")
                return
            cleanup_id = sentinel.get("cleanup_id")
            pid = sentinel.get("pid")
            if sentinel.get("cleanup_kind") == "docker" and cleanup_id:
                _stop_docker_container(cleanup_id)
            elif sentinel.get("cleanup_kind") == "process" and pid:
                _stop_process(pid, backend_type)
    except Timeout:
        pass
    finally:
        _delete_sentinel(backend_type)
    logger.info("Cleanup complete.")


def release_inference_on_idle() -> None:
    """Stop the inference container on idle while keeping it for fast restart."""
    inference_manager.stop()
    cleanup()


async def resource_management_loop():
    try:
        timer = 60.0
        while True:
            await asyncio.sleep(timer)
            cnt, last_updated, _ = get_request_count()
            if (
                cnt == 0
                and last_updated is not None
                and (perf_counter() - last_updated) > timer
            ):
                logger.warning(
                    f"No requests in the last {int(timer)} seconds; stopping inference container (persistent, will restart on next request)."
                )
                release_inference_on_idle()
    except asyncio.CancelledError:
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Reduce FastAPI threads because we use separate worker processes.
    RunVar("_default_thread_limiter").set(CapacityLimiter(10))

    # Open or create the lock file
    lock_file = open(LOCK_FILE, "w")
    background_task = None

    try:
        # Attempt non-blocking exclusive lock
        portalocker.lock(lock_file, portalocker.LOCK_EX | portalocker.LOCK_NB)
        print("Server starting...")
        background_task = asyncio.create_task(resource_management_loop())
    except portalocker.LockException:
        # Another worker process already has the lock
        pass

    yield

    # Gracefully close tasks and release the lock on shutdown
    if background_task:
        background_task.cancel()
    try:
        portalocker.unlock(lock_file)
        lock_file.close()
    except Exception:
        pass

    # atexit._run_exitfuncs()
    print("Done")


def load_and_preprocess_image(
    file: UploadFile,
    dpi: int | None,
    trim: float,
    crop: float,
    crop_left: float = 0.0,
    crop_right: float = 0.0,
    crop_top: float = 0.0,
    crop_bottom: float = 0.0,
) -> Image.Image:
    """Load PDF/image upload and apply shared trim/crop preprocessing."""
    if file.content_type == "application/pdf":
        image = get_page_image(file, page_num=1, dpi=dpi or 300)
    else:
        image = Image.open(file.file).convert("RGB")
    if trim > 0.0:
        image = trim_empty_background(image, threshold=trim)
    if crop_left or crop_right or crop_top or crop_bottom:
        image = crop_by_side_percent(
            image,
            left=crop_left or crop,
            right=crop_right or crop,
            top=crop_top or crop,
            bottom=crop_bottom or crop,
        )
    elif crop > 0.0:
        image = crop_by_percent(image, crop)
    return image


# Initialize FastAPI
app = FastAPI(lifespan=lifespan)

# Allow CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load models once when the application starts
inference_manager = ApiInferenceManager()


@app.post("/ocr/full/")
async def ocr_full_page(file: UploadFile = File(...),
    dpi: int | None = Query(
        default=300,
        ge=72,
        le=600,
        description="Optional: DPI for rendering PDF pages to images. Higher DPI can improve OCR accuracy but increases processing time and memory usage. 300 (the default) is a common choice for good quality OCR."
        ),
    trim: float = Query(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Entropy threshold to trim empty/solid borders before OCR. 0 disables trimming; 0.2–0.5 recommended."
        ),
    crop: float = Query(
        default=0.0,
        ge=0.0,
        le=50.0,
        description="Optional percent to crop from each side of the image before OCR. Can help with noisy borders that can provoke hallucinations. 0 means no cropping, 50 means crop half of the image from each side, 0.2 .. 0.5 recommended value."
        ),
    crop_left: float = Query(
        default=0.0,
        ge=0.0,
        le=50.0,
        description="Percent to crop from the LEFT side; overrides crop for that side when non-zero."
        ),
    crop_right: float = Query(
        default=0.0,
        ge=0.0,
        le=50.0,
        description="Percent to crop from the RIGHT side; overrides crop for that side when non-zero."
        ),
    crop_top: float = Query(
        default=0.0,
        ge=0.0,
        le=50.0,
        description="Percent to crop from the TOP side; overrides crop for that side when non-zero."
        ),
    crop_bottom: float = Query(
        default=0.0,
        ge=0.0,
        le=50.0,
        description="Percent to crop from the BOTTOM side; overrides crop for that side when non-zero."
        ),
    ):
    """Full-page OCR that extracts text and returns structured HTML.

    Can be inaccurate, slow and even unstable on large documents or complex tables.
    Returns HTML with block-level structure including bounding boxes and labels.
    For small, easy pages only.
    """
    try:
        logger.info(
            f"Received file: {file.filename}, content_type: {file.content_type}"
        )
        recognizer = RecognitionPredictor(inference_manager)
        await ensure_inference_server()
        update_request_count()
        start_time = perf_counter()
        image = load_and_preprocess_image(
            file,
            dpi=dpi,
            trim=trim,
            crop=crop,
            crop_left=crop_left,
            crop_right=crop_right,
            crop_top=crop_top,
            crop_bottom=crop_bottom,
        )
        # Use full_page=True for direct HTML extraction with HIGH_ACCURACY_BBOX_PROMPT
        predictions = recognizer([image], full_page=True)

        if not predictions:
            return {"html": "", "blocks": []}

        # Build structured block list
        blocks_data = []
        for prediction in predictions:
            for block in prediction.blocks:
                if block.html:  # Skip empty/skipped blocks
                    blocks_data.append(
                        {
                            "label": block.label,
                            "html": block.html,
                            "polygon": block.polygon,
                            "confidence": block.confidence,
                            "reading_order": block.reading_order,
                        }
                    )

        # Assemble full-page HTML by combining all blocks in reading order
        html_parts = []
        for block in blocks_data:
            # Each block already has HTML with proper structure from the model
            html_parts.append(
                f'<div class="block" data-label="{block["label"]}">{block["html"]}</div>'
            )

        full_html = '<div id="ocr-page">' + "\n".join(html_parts) + "</div>"
        end_time = perf_counter()
        logger.info(
            f"OCR completed for {file.filename}, extracted {len(blocks_data)} blocks in {end_time - start_time:.2f} seconds."
        )
        return {
            "html": full_html,
            "blocks": blocks_data,
            "page_bbox": predictions[0].image_bbox if predictions else [],
        }
    except Exception as e:
        msg = str(e)
        logger.error(f"OCR failed for {file.filename}: {msg}")
        raise HTTPException(
            status_code=500, detail=msg or "An error occurred during OCR processing."
        )
    finally:
        update_request_count(delta=-1)

def text_recognition(
    img: Image.Image,
    layouts: list[LayoutResult],
) -> tuple[list[PageOCRResult], list[tuple[int, ...]]]:
    from surya.inference.prompts import LAYOUT_LABEL_SET
    # desired_labels = set(LAYOUT_LABEL_SET) - {"Image", "Figure", "Diagram", "Table", "Table-Of-Contents", "Complex-Block"}
    desired_labels = set(LAYOUT_LABEL_SET) - {"Image", "Figure", "Diagram", "Table", "Table-Of-Contents"}
    layout: LayoutResult = layouts[0]
    texts = [
        b
        for b in layout.bboxes
        if b.label in desired_labels or b.raw_label in desired_labels or b.label.startswith("Cell")
    ]
    if not texts:
        return [], []
    filterred_layout = copy.copy(layout)
    filterred_layout.bboxes = texts
    recognizer = RecognitionPredictor(inference_manager)
    predictions = recognizer([img], [filterred_layout])  # , full_page=False
    filtered_bboxes = [tuple(int(c) for c in b.bbox) for b in texts]
    return predictions, filtered_bboxes


def table_recognition(
    img: Image.Image,
    layout: LayoutResult,
    mode: str
) -> tuple[list[TableResult], list[tuple[int, ...]]]:
    tables = [b for b in layout.bboxes if b.label in ("Table", "Table-Of-Contents")]
    if not tables:
        return [], []
    table_bboxes = [tuple(int(c) for c in b.bbox) for b in tables]
    table_imgs = [img.crop(b) for b in table_bboxes]
    table_counts = [b.count for b in tables]
    table_rec_predictor = TableExtPredictor(inference_manager)
    table_preds = table_rec_predictor.predict_flexible(table_imgs, counts=table_counts, mode="td")
    if mode in ["td"]:
        table_preds2 = table_rec_predictor.predict_simple(table_imgs)
        if table_preds2:
            for pred, pred2 in zip(table_preds, table_preds2):
                if pred2.error:
                    continue
                pred.cells = pred2.cells
                pred.cols = pred2.cols
                pred.rows = pred2.rows
    return table_preds, table_bboxes


@app.post("/ocr/block/")
async def ocr_blocks(file: UploadFile = File(...),
    dpi: int | None = Query(
        default=300,
        ge=72,
        le=600,
        description="Optional: DPI for rendering PDF pages to images. Higher DPI can improve OCR accuracy but increases processing time and memory usage. 300 (the default) is a common choice for good quality OCR."
        ),
    trim: float = Query(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Entropy threshold to trim empty/solid borders before OCR. 0 disables trimming; 0.2–0.5 recommended."
        ),
    crop: float = Query(
        default=0.0,
        ge=0.0,
        le=50.0,
        description="Optional percent to crop from each side of the image before OCR. Can help with noisy borders that can provoke hallucinations. 0 means no cropping, 50 means crop half of the image from each side, 0.2 .. 0.5 recommended value."
        ),
    crop_left: float = Query(
        default=0.0,
        ge=0.0,
        le=50.0,
        description="Percent to crop from the LEFT side; overrides crop for that side when non-zero."
        ),
    crop_right: float = Query(
        default=0.0,
        ge=0.0,
        le=50.0,
        description="Percent to crop from the RIGHT side; overrides crop for that side when non-zero."
        ),
    crop_top: float = Query(
        default=0.0,
        ge=0.0,
        le=50.0,
        description="Percent to crop from the TOP side; overrides crop for that side when non-zero."
        ),
    crop_bottom: float = Query(
        default=0.0,
        ge=0.0,
        le=50.0,
        description="Percent to crop from the BOTTOM side; overrides crop for that side when non-zero."
        ),
    # tblmode: str = Query(
    #     default="td",
    #     enum=["td", "div"],
    #     description="Table recognition mode: 'td' for cell-level HTML using <td> and <tr> tags, 'div' for cell-level HTML with <div> and data-bbox attributes"
    #     )
    ):
    """Block-based page OCR that extracts text and tables and returns structured HTML.  
    Faster than full-page OCR and able to process very large pages with a lot of content 
    that should be filtered out (like images), also is more accurate for complex table layouts.
    """
    try:
        logger.warning(
            f"Received file: {file.filename}, content_type: {file.content_type}"
        )
        await ensure_inference_server()
        update_request_count()
        start_time = perf_counter()
        image = load_and_preprocess_image(
            file,
            dpi=dpi,
            trim=trim,
            crop=crop,
            crop_left=crop_left,
            crop_right=crop_right,
            crop_top=crop_top,
            crop_bottom=crop_bottom,
        )
        layout_predictor = LayoutPredictor(inference_manager)
        layouts = layout_predictor([image])
        if not layouts or not layouts[0].bboxes:
            return {"html": "", "blocks": []}
        width, height = image.size
        margin = int(max(width, height) / 200)  # Dynamic margin based on image size (e.g., 2px for 1000px image)
        for block in layouts[0].bboxes:
            poligon_expand(block.polygon, margin=margin)
        texts, text_bboxes  = text_recognition(image, layouts)
        tables, table_bboxes = table_recognition(image, layouts[0], mode="td")
        blocks_data = []
        for pred in texts:
            for block in pred.blocks:
                if block.html:  # Skip empty/skipped blocks
                    blocks_data.append(
                        {
                            "label": block.label,
                            "html": block.html,
                            "polygon": block.polygon,
                            "confidence": block.confidence,
                            "reading_order": block.reading_order,
                        }
                    )
        for table, bbox in zip(tables, table_bboxes):
            if table.html:  # Skip empty/skipped tables
                blocks_data.append(
                    {
                        "label": "Table",
                        "html": table.html,
                        "rows": table.rows,
                        "cols": table.cols,
                        "cells": table.cells,
                        "bbox": bbox,
                    }
                )

        html_parts = []
        for block in blocks_data:
            # Each block already has HTML with proper structure from the model
            html_parts.append(
                f'<div class="block" data-label="{block["label"]}">{block["html"]}</div>'
            )

        full_html = '<div id="ocr-page">' + "\n".join(html_parts) + "</div>"
        end_time = perf_counter()
        logger.warning(
            f"OCR completed for {file.filename}, extracted {len(blocks_data)} blocks in {end_time - start_time:.2f} seconds."
        )
        return {
            "blocks": blocks_data,
            "html": full_html,
        }
    except Exception as e:
        msg = str(e)
        logger.error(f"OCR failed for {file.filename}: {msg}")
        raise HTTPException(
            status_code=500, detail=msg or "An error occurred during OCR processing."
        )
    finally:
        update_request_count(delta=-1)


# Run the application
if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Surya API Server")
    parser.add_argument(
        "--port", type=int, default=8522, help="Port number to run the server on"
    )
    args = parser.parse_args()

    n_workers = settings.SURYA_INFERENCE_PARALLEL or 1
    if sys.platform == "win32":
        # uvicorn multiprocess workers pass sockets to children via fork on
        # Unix only; on Windows this fails with WinError 10022 at sock.listen().
        if n_workers > 1:
            logger.warning(
                "Multiple uvicorn workers are not supported on Windows; using workers=1"
            )
        n_workers = 1
    uvicorn.run(
        "surya_api:app",
        host="0.0.0.0",
        port=args.port,
        workers=n_workers,
        log_config=None,
    )  #  , reload=True
