# Surya API

HTTP API for [Surya OCR](https://github.com/datalab-to/surya) — document OCR, layout analysis, and table recognition. This project wraps the Surya model with a FastAPI server so AI agents and other clients can extract text from images and PDFs over REST.

Built on top of upstream Surya with custom extensions in the `crown` package (improved table recognition and image preprocessing).

## Features

- **Full-page OCR** (`POST /ocr/full/`) — single-pass OCR for simple pages
- **Block-based OCR** (`POST /ocr/block/`) — layout-aware OCR with table extraction; recommended for complex documents
- **PDF support** — first page rendered at configurable DPI
- **Image preprocessing** — automatic background trimming and optional border cropping
- **Auto-managed inference** — starts and stops the Surya VLM backend (vLLM or llama.cpp) as needed

## Prerequisites

Surya requires an inference backend. See the [upstream installation guide](https://github.com/datalab-to/surya#installation) for details.

| Platform | Backend | Requirement |
|----------|---------|-------------|
| NVIDIA GPU | [vLLM](https://github.com/vllm-project/vllm) | Docker + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) |
| CPU / Apple Silicon | [llama.cpp](https://github.com/ggml-org/llama.cpp) | `llama-server` binary (`brew install llama.cpp` on macOS, or a release binary) |

This API server is intended to run on **Linux** (it uses `fcntl` for single-instance locking across workers).

**Python:** 3.10+

## Download

Clone the repository:

```bash
git clone <your-repo-url> surya-api
cd surya-api
```

Or download a release archive and extract it.

## Installation

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows (API server itself expects Linux for production)

pip install surya-ocr fastapi "uvicorn[standard]" filelock pypdfium2 numpy pillow
```

On first use, Surya downloads model weights automatically.

### Optional: attach to an existing inference server

If you already run vLLM or llama.cpp, point Surya at it instead of auto-spawning:

```bash
export SURYA_INFERENCE_BACKEND=vllm          # or llamacpp
export SURYA_INFERENCE_URL=http://localhost:8000/v1
```

Other useful environment variables (full list in [upstream `surya/settings.py`](https://github.com/datalab-to/surya/blob/master/surya/settings.py)):

| Variable | Default | Description |
|----------|---------|-------------|
| `SURYA_INFERENCE_BACKEND` | auto | `vllm`, `llamacpp`, or auto-detect |
| `SURYA_INFERENCE_PARALLEL` | `8` | Client concurrency; also sets uvicorn worker count |
| `SURYA_INFERENCE_KEEP_ALIVE` | `false` | Keep the inference server running after requests |

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

The server spawns the Surya inference backend on the first request. When idle for 60 seconds with no active requests, it shuts the backend down to save GPU/CPU resources.

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

Runs layout detection first, then OCRs text blocks and tables separately. Faster and more accurate on large or complex pages.

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
| `crop` | `0` | 0–50 | Percent to crop from each edge before OCR; try `0.2`–`0.5` for noisy borders |

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

The `/ocr/block/` endpoint may also include table blocks with `rows`, `cols`, `cells`, and `bbox` fields.

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

## Project layout

```
surya-api/
├── surya_api.py          # FastAPI application and endpoints
└── crown/
    ├── utils.py          # PDF rendering, crop, background trimming
    └── table_rec/        # Extended table recognition (TableExtPredictor)
```

## License

This API layer is provided alongside [Surya](https://github.com/datalab-to/surya), which is licensed under Apache 2.0. Surya model weights use a [modified AI Pubs Open Rail-M license](https://github.com/datalab-to/surya/blob/master/MODEL_LICENSE) — review upstream terms for commercial use.

## Credits

- [Surya](https://github.com/datalab-to/surya) by [Datalab](https://www.datalab.to)
