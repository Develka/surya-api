# Surya API

HTTP API for [Surya OCR](https://github.com/datalab-to/surya) — document OCR, layout analysis, and table recognition. This project wraps the Surya 2 foundation model with a FastAPI server so AI agents and other clients can extract text from images and PDFs over REST.

Built on top of upstream Surya with custom extensions in the `crown` package (persistent vLLM container management, improved table recognition, and image preprocessing).

## Why this project

- **Local, private processing.** Everything runs on your own hardware — the document never leaves your machine. There is no third-party OCR service in the loop, so sensitive technical documentation stays confidential.
- **Large-format technical documents and drawings.** Handles big engineering pages and schematics (e.g. **A1**-sized drawings) by rendering PDFs at a configurable DPI and using layout-aware block OCR that filters out images/figures and focuses on text and tables.
- **Higher accuracy through preprocessing.** Automatic empty-background trimming and optional per-side cropping remove noisy borders that otherwise trigger hallucinations, measurably improving recognition quality before the page ever reaches the model.
- **Spatial metadata for downstream correction.** The `blocks` section returns per-region polygons/bounding boxes, labels, and confidence alongside the HTML. This lets you feed individual regions (especially complex tables) into a larger, more capable model for a second-pass correction, grounding, or citation.

## Features

- **Full-page OCR** (`POST /ocr/full/`) — single-pass OCR for simple pages
- **Block-based OCR** (`POST /ocr/block/`) — layout-aware OCR with table extraction; recommended for complex and large-format documents
- **PDF support** — first page rendered at configurable DPI
- **Image preprocessing** — automatic background trimming and optional per-side border cropping
- **Auto-managed inference** — keeps a single persistent vLLM container; stops it on idle and restarts it on the next request without reloading from scratch

## Requirements

### Surya version

This project targets **Surya 2** (`surya-ocr >= 0.20.0`). Surya 2 is a ground-up rework: a single ~650M-parameter VLM handles OCR, layout, and table recognition, served by `vllm` (NVIDIA GPU) or `llama.cpp` (CPU / Apple Silicon). Earlier Surya 1.x releases use a different API and schema and are **not** compatible.

### Inference backend

Surya requires an inference backend. See the [upstream installation guide](https://github.com/datalab-to/surya#installation) for details.

| Platform | Backend | Requirement |
|----------|---------|-------------|
| NVIDIA GPU | [vLLM](https://github.com/vllm-project/vllm) | Docker + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) |
| CPU / Apple Silicon | [llama.cpp](https://github.com/ggml-org/llama.cpp) | `llama-server` binary (`brew install llama.cpp` on macOS, or a release binary) |

This API server is cross-platform (**Linux, Windows, macOS**) — single-instance locking uses `portalocker`/`filelock` rather than `fcntl`. Note that on Windows, uvicorn is forced to a single worker (multi-worker socket sharing is Unix-only).

**Python:** 3.10+

### GPU memory

The vLLM backend is the memory-hungry part. The model weights are small (~650M params, ~1.5 GB), but most VRAM goes to the KV cache, which scales with `--max-model-len` and concurrency.

| GPU VRAM | Notes |
|----------|-------|
| **24 GB (e.g. RTX 4090) — recommended** | Matches the shipped defaults (`VLLM_GPU_TYPE=4090`, `VLLM_MAX_MODEL_LEN=18000`, `VLLM_GPU_MEMORY_UTILIZATION=0.85`). Comfortable for large pages and moderate concurrency. Upstream benchmarks were run on a 32 GB RTX 5090. |
| **~12–16 GB — workable** | Reduce `VLLM_MAX_MODEL_LEN` (e.g. 8000–12000), lower `VLLM_GPU_MEMORY_UTILIZATION`, and cut concurrency (`SURYA_INFERENCE_PARALLEL`). Large A1 drawings at high DPI may not fit. |
| **No NVIDIA GPU** | Use the `llama.cpp` backend on CPU / Apple Silicon. Much slower (see below). |

### Approximate recognition time

Per-page time depends heavily on hardware, page content density, DPI, and concurrency.

- **Warm GPU (vLLM), single page:** typically a few seconds up to ~20 s for dense, large-format pages. Upstream reports ~5 pages/s aggregate throughput on an RTX 5090 at high concurrency (median ~19 s/page latency under heavy batching).
- **First request after idle (cold start):** add the container start time. With caches warm (see [persistent caching](#performance--caching)) this is usually ~30–90 s; a true cold start with no cached weights or `torch.compile` artifacts can take several minutes.
- **CPU / Apple Silicon (llama.cpp):** on the order of ~0.1 pages/s — viable for offline single-document use, not for batch workloads.

## Download

Clone the repository:

```bash
git clone git@github.com:Develka/surya-api.git surya-api
cd surya-api
```

Or download a release archive and extract it.

## Installation

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

pip install surya-ocr fastapi "uvicorn[standard]" filelock portalocker httpx openai pypdfium2 numpy pillow
# For the integration test:
pip install pytest requests
```

## Configuration

Copy `.env.example` to `.env` and adjust as needed; every value has a sensible default. Key variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `SURYA_INFERENCE_BACKEND` | auto | `vllm`, `llamacpp`, or auto-detect |
| `SURYA_INFERENCE_URL` | unset | Attach to an existing OpenAI-compatible server instead of managing a container |
| `SURYA_INFERENCE_PARALLEL` | `8` | Client concurrency; also sets the uvicorn worker count (forced to 1 on Windows) |
| `SURYA_INFERENCE_STARTUP_TIMEOUT` | `600` | Seconds to wait for the container to become healthy before recreating |
| `SURYA_INFERENCE_PORT` | unset | Fixed host port for the inference server (auto-picks a free one if unset) |
| `SURYA_MODEL_CHECKPOINT` | `datalab-to/surya-ocr-2` | Model served by vLLM |
| `SURYA_DOCKER_CACHE_MODE` | `volume` | `volume` (Docker named volumes, fast + cross-platform) or `bind` (host directory mounts) |
| `SURYA_HF_CACHE_VOLUME` | `surya-hf-cache` | Named volume for HuggingFace weights (volume mode) |
| `SURYA_VLLM_CACHE_VOLUME` | `surya-vllm-cache` | Named volume for the vLLM `torch.compile` cache (volume mode) |
| `DOCKER_HF_CACHE_PATH` | `~/.cache/huggingface` | Host path for weights (bind mode) |
| `DOCKER_VLLM_CACHE_PATH` | `~/.cache/surya-vllm` | Host path for the `torch.compile` cache (bind mode) |
| `VLLM_DOCKER_IMAGE` | `vllm/vllm-openai:v0.20.1` | vLLM container image |
| `VLLM_GPUS` / `VLLM_GPU_TYPE` | `0` / `4090` | GPU device id and tuning profile |
| `VLLM_DTYPE` | `bfloat16` | Inference dtype |
| `VLLM_MAX_MODEL_LEN` | `18000` | Max sequence length (largest VRAM lever) |
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.85` | Fraction of VRAM vLLM may use |
| `VLLM_ENABLE_MTP` / `VLLM_MTP_TOKENS` | `true` / `2` | Speculative (multi-token prediction) decoding |
| `VLLM_EXTRA_ARGS` | unset | Extra space-separated args appended to the vLLM command |

The full annotated list lives in [`.env.example`](.env.example); Surya/vLLM settings are read by `surya.settings`.

## Performance & caching

Two caches dominate startup time, and both are persisted across container restarts so that a stop/start cycle does **not** reload the model from scratch:

- **HuggingFace weights** → `/root/.cache/huggingface`
- **vLLM `torch.compile` artifacts** → `/root/.cache/vllm`

By default these are stored in **Docker named volumes** (`SURYA_DOCKER_CACHE_MODE=volume`). Named volumes live inside the Docker storage backend (ext4 in the WSL2/Docker-Desktop VM on Windows/macOS), which avoids the slow 9P bind-mount path that makes weight loading crawl on Windows. Use `bind` mode only if you want the cache files directly visible on the host.

The container itself is **persistent and canonical**: exactly one container named `surya-vllm` is kept. On idle (no requests for ~60 s) it is *stopped*, not deleted, and restarted on the next request. If it ever fails to become healthy it is recreated, up to 3 attempts, after which the API returns `503`.

## Usage

### Start the server

```bash
python surya_api.py
```

Default URL: `http://0.0.0.0:8522`

Custom port:

```bash
python surya_api.py --port 9000
```

Interactive API docs are available at `http://localhost:8522/docs` once the server is running.

The server starts the Surya inference backend on the first request. When idle for 60 seconds with no active requests, it stops the backend container to free GPU resources, then restarts it (fast, from cache) on the next request.

### Optional: attach to an existing inference server

If you already run vLLM or llama.cpp, point Surya at it instead of letting this project manage the container:

```bash
export SURYA_INFERENCE_BACKEND=vllm          # or llamacpp
export SURYA_INFERENCE_URL=http://localhost:8000/v1
```

### Full-page OCR

Best for small, simple pages. Can be slow or less accurate on large documents or complex tables.

```bash
curl -X POST "http://localhost:8522/ocr/full/" \
  -F "file=@document.png"
```

With PDF and options:

```bash
curl -X POST "http://localhost:8522/ocr/full/?dpi=300&crop=0.5" \
  -F "file=@scan.pdf"
```

### Block-based OCR (recommended)

Runs layout detection first, then OCRs text blocks and tables separately. Faster and more accurate on large-format or complex pages, and able to filter out non-text regions (images/figures).

```bash
curl -X POST "http://localhost:8522/ocr/block/" \
  -F "file=@document.png"
```

```bash
curl -X POST "http://localhost:8522/ocr/block/?dpi=300&crop=0.3" \
  -F "file=@report.pdf"
```

### Query parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `dpi` | `300` | 72–600 | PDF render resolution (images ignore this) |
| `trim` | `0.5` | 0–1 | Entropy threshold to trim empty/solid borders before OCR; 0 disables |
| `crop` | `0` | 0–50 | Percent to crop from each edge before OCR; try `0.2`–`0.5` for noisy borders |
| `crop_left` / `crop_right` / `crop_top` / `crop_bottom` | `0` | 0–50 | Per-side crop percent; overrides `crop` for that side when non-zero |

### Response format

Both endpoints return JSON:

```json
{
  "html": "<div id=\"ocr-page\">...</div>",
  "blocks": [
    {
      "label": "Text",
      "html": "<p>...</p>",
      "polygon": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
      "confidence": 0.95,
      "reading_order": 0
    }
  ],
  "page_bbox": [0, 0, width, height]
}
```

The `/ocr/block/` endpoint may also include table blocks with `rows`, `cols`, `cells`, and `bbox` fields. The per-block `polygon`/`bbox` spatial metadata is what enables a downstream model to re-process specific regions (e.g. a complex table) for correction.

### Example: Python client

```python
import requests

with open("page.png", "rb") as f:
    response = requests.post(
        "http://localhost:8522/ocr/block/",
        files={"file": ("page.png", f, "image/png")},
        params={"dpi": 300, "crop": 0},
    )
response.raise_for_status()
data = response.json()
print(data["html"])
```

### Example: AI agent / tool definition

Point your agent at `POST /ocr/block/` with a multipart file upload. The `html` field contains structured page content; `blocks` provides per-region metadata (labels, bounding boxes, confidence) for grounding or citation.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Unknown scheme for proxy URL 'socks4://...'` or health checks hang on localhost | A system SOCKS/HTTP proxy hijacks localhost connections (common on Windows) | Handled automatically (`NO_PROXY` for localhost + OpenAI client patched with `trust_env=False`). For `pip`, install with `--proxy=""` and `NO_PROXY=*`. |
| `'NoneType' object has no attribute 'chat'` | Backend handle existed but the OpenAI client wasn't initialized | Fixed: the server now verifies and repairs the client before serving requests. Restart the server if you hit it on an old build. |
| Very long startup / model reloads every time | `torch.compile` and/or weight cache lost between runs | Keep `SURYA_DOCKER_CACHE_MODE=volume` (default) so both caches persist in Docker named volumes. |
| Slow weight loading on Windows | Bind-mounting a Windows path into WSL2 uses the slow 9P filesystem | Use `volume` cache mode (default) instead of `bind`. |
| Container disappears after idle | (Legacy) container ran with `--rm` | Fixed: the canonical `surya-vllm` container is *stopped*, not removed, on idle and restarted on demand. |
| `503 Inference server failed to start` | Container couldn't become healthy within `SURYA_INFERENCE_STARTUP_TIMEOUT` after 3 recreate attempts | Check `docker logs surya-vllm`; usually GPU OOM or a bad config. Lower `VLLM_MAX_MODEL_LEN` / `VLLM_GPU_MEMORY_UTILIZATION`. |
| CUDA out of memory | `--max-model-len`/concurrency too high for the card | Reduce `VLLM_MAX_MODEL_LEN`, `VLLM_GPU_MEMORY_UTILIZATION`, and/or `SURYA_INFERENCE_PARALLEL`. |
| Garbled text / hallucinations near page borders | Noisy scan margins | Increase `trim` or use `crop`/per-side crop params. |
| `/ocr/full/` slow or unstable on large A1 drawings | Full-page mode struggles with very large, content-dense pages | Use `/ocr/block/`, which filters non-text regions and handles large layouts better. |
| `destroy_process_group() was not called` warning in container logs | Benign NCCL shutdown warning from vLLM on container stop | Safe to ignore. |

## Project layout

```
surya-api/
├── surya_api.py          # FastAPI application and endpoints
├── .env.example          # documented environment variables
└── crown/
    ├── utils.py          # PDF rendering, crop, background trimming
    ├── inference/        # persistent vLLM container management
    │   ├── __init__.py   # ApiInferenceManager
    │   └── vllm.py       # VllmPersistentBackend (single canonical container)
    └── table_rec/        # Extended table recognition (TableExtPredictor)
```

## License

This API layer is provided alongside [Surya](https://github.com/datalab-to/surya), which is licensed under Apache 2.0. Surya model weights use a [modified AI Pubs Open Rail-M license](https://github.com/datalab-to/surya/blob/master/MODEL_LICENSE) — review upstream terms for commercial use.

## Credits

- [Surya](https://github.com/datalab-to/surya) by [Datalab](https://www.datalab.to)
