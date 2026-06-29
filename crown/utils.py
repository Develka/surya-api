import io
import os

import numpy as np
import pypdfium2
from PIL import Image
from fastapi import UploadFile
from concurrent.futures import ProcessPoolExecutor

from surya.settings import settings


def get_page_image(
    pdf_file: UploadFile, page_num: int, dpi: int | None = None
) -> Image.Image:
    if dpi is None:
        dpi = settings.IMAGE_DPI_HIGHRES
    doc = pypdfium2.PdfDocument(pdf_file.file.read())
    renderred = doc.render(
        pypdfium2.PdfBitmap.to_pil,
        page_indices=[page_num - 1],
        scale=dpi / 72,
    )
    png = list(renderred)[0]
    png_image = png.convert("RGB")
    doc.close()
    return png_image


def poligon_expand(polygon: list[list[float]], margin: float):
    """
    Expands a polygon by a certain margin (inplace).
                polygon = [
                    [x_min, y_min],
                    [x_max, y_min],
                    [x_max, y_max],
                    [x_min, y_max],
                ]
    """
    if len(polygon) != 4:
        return polygon
    polygon[0][0] -= margin  # x_min
    polygon[0][1] -= margin  # y_min
    polygon[1][0] += margin  # x_max
    polygon[1][1] -= margin  # y_min
    polygon[2][0] += margin  # x_max
    polygon[2][1] += margin  # y_max
    polygon[3][0] -= margin  # x_min
    polygon[3][1] += margin  # y_max

def image_white_cnt_points(image: Image.Image, white_threshold: int = 200) -> int:
    """Counts the number of white points in b/w image."""
    import numpy as np

    array_img = np.array(image)
    count = int(np.sum(array_img > white_threshold))
    return count
    
def crop_by_percent(image: Image.Image, crop_percent: float) -> Image.Image:
    """Crops a percentage from each side of the image."""
    if crop_percent > 0.0:
        w, h = image.size
        crop_margin_h = int(h * crop_percent / 100)
        crop_margin_w = int(w * crop_percent / 100)
        cropped_image = image.crop((
            crop_margin_w,
            crop_margin_h,
            w - crop_margin_w,
            h - crop_margin_h,
        ))
        return cropped_image
    return image


def crop_by_side_percent(
    image: Image.Image,
    left: float = 0.0,
    right: float = 0.0,
    top: float = 0.0,
    bottom: float = 0.0,
) -> Image.Image:
    """Crops a percentage from each side of the image independently."""
    left = max(0.0, float(left or 0.0))
    right = max(0.0, float(right or 0.0))
    top = max(0.0, float(top or 0.0))
    bottom = max(0.0, float(bottom or 0.0))
    if left == 0.0 and right == 0.0 and top == 0.0 and bottom == 0.0:
        return image
    w, h = image.size
    crop_left = int(w * left / 100)
    crop_right = w - int(w * right / 100)
    crop_top = int(h * top / 100)
    crop_bottom = h - int(h * bottom / 100)
    if crop_left >= crop_right or crop_top >= crop_bottom:
        return image
    return image.crop((crop_left, crop_top, crop_right, crop_bottom))


def entropy(values):
    values = np.asarray(values)
    _, counts = np.unique(values, return_counts=True)
    probs = counts / len(values)
    return -np.sum(probs * np.log2(probs))


def _entropy_from_chunks(
    image_bytes: bytes, chunk_boxes: list[tuple[int, int, int, int]]
) -> list[float]:
    img = Image.open(io.BytesIO(image_bytes))
    return [entropy(img.crop(box).getdata()) for box in chunk_boxes]


def _split_round_robin(items, n_slices):
    n_slices = max(1, n_slices)
    slices: list[list] = [[] for _ in range(n_slices)]
    for idx, item in enumerate(items):
        slices[idx % n_slices].append(item)
    return slices


def stripe_entropy(
    image: Image.Image,
    aggressive: bool = False,
    executor: ProcessPoolExecutor | None = None,
) -> float:
    if aggressive:
        return entropy(image.getdata())
    w, h = image.size
    chunk_boxes: list[tuple[int, int, int, int]] = []
    if h > w:
        h_step = w
        for y in range(0, h, h_step):
            chunk_boxes.append((0, y, w, min(y + h_step, h)))
    else:
        w_step = h
        for x in range(0, w, w_step):
            chunk_boxes.append((x, 0, min(x + w_step, w), h))
    if not chunk_boxes:
        return 1.0
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    image_bytes = buf.getvalue()
    n_slices = min(len(chunk_boxes), (os.cpu_count() or 1))
    splits = _split_round_robin(chunk_boxes, n_slices)
    owns_pool = executor is None
    pool = executor if executor is not None else ProcessPoolExecutor()
    try:
        per_worker_results = list(
            pool.map(
                _entropy_from_chunks,
                [image_bytes] * len(splits),
                splits,
            )
        )
    finally:
        if owns_pool:
            pool.shutdown()
    entropies: list[float] = []
    for result in per_worker_results:
        entropies.extend(result)
    return max(entropies)


def trim_empty_background(
    image: Image.Image,
    threshold: float = 0.5,
    executor: ProcessPoolExecutor | None = None,
) -> Image.Image:
    w, h = image.size
    h_step = max(1, h // 40)
    w_step = max(1, w // 40)
    crop_top = 0
    crop_left = 0
    crop_bottom = h
    crop_right = w
    pal_image = image.convert("P", palette=Image.Palette.ADAPTIVE, colors=256)
    owns_pool = executor is None
    pool = executor if executor is not None else ProcessPoolExecutor()
    try:
        for y in range(0, h, h_step):
            stripe = pal_image.crop((0, y, w, min(y + h_step, h)))
            stripe_ent = stripe_entropy(stripe, executor=pool)
            if stripe_ent < threshold:
                crop_top = y + h_step
            else:
                break
        for y in range(h, 0, -h_step):
            stripe = pal_image.crop((0, max(y - h_step, 0), w, y))
            stripe_ent = stripe_entropy(stripe, executor=pool)
            if stripe_ent < threshold:
                crop_bottom = y - h_step
            else:
                break
        for x in range(0, w, w_step):
            stripe = pal_image.crop((x, 0, min(x + w_step, w), h))
            stripe_ent = stripe_entropy(stripe, executor=pool)
            if stripe_ent < threshold:
                crop_left = x + w_step
            else:
                break
        for x in range(w, 0, -w_step):
            stripe = pal_image.crop((max(x - w_step, 0), 0, x, h))
            stripe_ent = stripe_entropy(stripe, executor=pool)
            if stripe_ent < threshold:
                crop_right = x - w_step
            else:
                break
    finally:
        if owns_pool:
            pool.shutdown()

    if crop_left >= crop_right or crop_top >= crop_bottom:
        return image
    if crop_left > 0 or crop_top > 0 or crop_right < w or crop_bottom < h:
        image = image.crop((crop_left, crop_top, crop_right, crop_bottom))
    return image
