from typing import List, Optional

from PIL import Image

from surya.table_rec import TableRecPredictor, _polygon_from_bbox, _intersect_bbox, logger
from surya.table_rec.schema import TableCell, TableCol, TableResult, TableRow
from surya.inference.prompts import PROMPT_TYPE_TABLE_REC, TABLE_REC_JSON_SCHEMA, TABLE_REC_LABEL_SET
from surya.inference.schema import PROMPT_TYPE_BLOCK, BatchInputItem
from surya.inference.util import image_token_budget
from surya.inference import SuryaInferenceManager, get_default_manager
from surya.logging import get_logger
from surya.settings import settings
from surya.inference.parsers import clean_block_html, denorm_bbox, parse_table_rec

BLOCK_PROMPT_TBL = ("OCR this image to HTML Each block is a div with data-label and data-bbox "
    "(x0 y0 x1 y1, normalized 0-1000)."
)

# TABLE_REC_PROMPT = (
#     "OCR this image to JSON. Each entry is a dict with "
#     '"label" ("Row" or "Col"), "text" (raw OCR text), and "bbox" (x0 y0 x1 y1, normalized 0-1000).'
# )

TABLE_REC_PROMPT = (
    "OCR this image to JSON. Each entry is a dict with "
    '"text" (raw OCR text), and "bbox" (x0 y0 x1 y1, normalized 0-1000).'
)

TABLE_REC_JSON_SCHEMA_EXT = {
    "type": "array",
    "maxItems": 200,
    "items": {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": TABLE_REC_LABEL_SET},
            "text": {"type": "string", "description": "OCR text for this block."},
            "bbox": {
                "type": "string",
                "pattern": r"^\d{1,4} \d{1,4} \d{1,4} \d{1,4}$",
            },
        },
        "required": ["bbox", "text"],
        "additionalProperties": False,
    },
}


class TableExtPredictor(TableRecPredictor):
    def __init__(self, manager: Optional[SuryaInferenceManager] = None):
        super().__init__(manager)

    def predict_flexible(
        self, images: List[Image.Image],
        counts: Optional[List[int]] = None,
        mode: str = "div"
    ) -> List[TableResult]:
        """Full-HTML path: BLOCK_PROMPT on table crops. Use when complex
        structure (spanning cells, headers) matters and ground-truth-style
        HTML is preferred. `counts` (one per image) shapes max_tokens."""
        if not images:
            return []
        manager = self.manager or get_default_manager()
        if counts is None:
            counts = [0] * len(images)
        batch = []
        if mode == "td":
            prompt = None
            prompt_type = PROMPT_TYPE_BLOCK
        elif mode == "div":
            prompt = BLOCK_PROMPT_TBL
            prompt_type = "" # Emty string by intention!
        for img, count in zip(images, counts):
            batch.append(
                BatchInputItem(
                    image=img,
                    prompt=prompt,
                    prompt_type=prompt_type,
                    max_tokens=image_token_budget(
                        count,
                        ceiling=settings.SURYA_MAX_TOKENS_BLOCK_CEILING,
                        floor=1024,
                    ),
                )
            )
        outputs = manager.generate(batch)
        results: List[TableResult] = []
        for img, out in zip(images, outputs):
            w, h = img.size
            page_bbox = [0, 0, float(w), float(h)]
            if out.error:
                results.append(
                    TableResult(
                        rows=[],
                        cols=[],
                        cells=[],
                        image_bbox=page_bbox,
                        raw=out.raw,
                        mode="full",
                        error=True,
                    )
                )
                continue
            html = clean_block_html(out.raw)
            results.append(
                TableResult(
                    rows=[],
                    cols=[],
                    cells=[],
                    image_bbox=page_bbox,
                    raw=out.raw,
                    html=html,
                    mode="full",
                    error=False,
                )
            )
        return results


    def predict_simple(self, images: List[Image.Image]) -> List[TableResult]:
        if not images:
            return []
        manager = self.manager or get_default_manager()
        guided = TABLE_REC_JSON_SCHEMA_EXT if settings.SURYA_GUIDED_TABLE_REC else None
        batch = [
            BatchInputItem(
                image=img,
                # prompt=TABLE_REC_PROMPT,
                prompt_type=PROMPT_TYPE_TABLE_REC,
                max_tokens=settings.SURYA_MAX_TOKENS_TABLE_REC,
                guided_json=guided,
            )
            for img in images
        ]
        outputs = manager.generate(batch)

        results: List[TableResult] = []
        for img, out in zip(images, outputs):
            w, h = img.size
            page_bbox = [0, 0, float(w), float(h)]
            if out.error or not out.raw:
                results.append(
                    TableResult(
                        rows=[],
                        cols=[],
                        cells=[],
                        image_bbox=page_bbox,
                        raw=out.raw,
                        mode="simple",
                        error=True,
                    )
                )
                continue
            try:
                elements = parse_table_rec(out.raw)
            except Exception as e:
                logger.warning(
                    f"Table rec parse failed: {e}; raw[:200]={out.raw[:200]!r}"
                )
                results.append(
                    TableResult(
                        rows=[],
                        cols=[],
                        cells=[],
                        image_bbox=page_bbox,
                        raw=out.raw,
                        mode="simple",
                        error=True,
                    )
                )
                continue

            rows: List[TableRow] = []
            cols: List[TableCol] = []
            for el in elements:
                pixel_bbox = denorm_bbox(el.bbox, w, h, scale=settings.BBOX_SCALE)
                poly = _polygon_from_bbox(pixel_bbox)
                if el.label == "Row":
                    rows.append(TableRow(polygon=poly, row_id=len(rows)))
                else:
                    cols.append(TableCol(polygon=poly, col_id=len(cols)))

            # Derive cells geometrically (row × column intersections)
            cells: List[TableCell] = []
            cell_id = 0
            for row in rows:
                for col in cols:
                    inter = _intersect_bbox(row.bbox, col.bbox)
                    if inter is None:
                        continue
                    cells.append(
                        TableCell(
                            polygon=_polygon_from_bbox(inter),
                            row_id=row.row_id,
                            col_id=col.col_id,
                            cell_id=cell_id,
                        )
                    )
                    cell_id += 1
            results.append(
                TableResult(
                    rows=rows,
                    cols=cols,
                    cells=cells,
                    image_bbox=page_bbox,
                    raw=out.raw,
                    mode="simple",
                    error=False,
                )
            )
        return results
