import base64
import io
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (
    MODEL_PATH,
    MODEL_ARCH,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_THRESHOLD,
    DEFAULT_ALPHA,
    IMAGE_SIZES,
    DEFAULT_THRESHOLD_MODE,
    DEFAULT_PERCENTILE,
    DEFAULT_MIN_AREA,
    DEFAULT_ALIGN,
    DEFAULT_MULTISCALE,
    MULTISCALE_SIZES,
    DEFAULT_FAST_MODE,
)
from inference import (
    load_model,
    build_transforms,
    run_inference,
    compute_mask,
    compute_overlay,
    run_multiscale_inference,
    calibrate_threshold,
    remove_small_components,
    align_images,
    hash_bytes,
    make_cache_key,
    model_cache_stamp,
)
from model import build_model


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="Urban Change Detection API", version="1.0.0")
API_PREFIX = "/api"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model(lambda: build_model(MODEL_ARCH), MODEL_PATH, DEVICE)
cache = {}


def np_to_png_bytes(np_img):
    buffer = io.BytesIO()
    Image.fromarray(np_img).save(buffer, format="PNG")
    return buffer.getvalue()


def heatmap_to_png_bytes(prob_map):
    fig = plt.figure(figsize=(3, 3), dpi=160)
    ax = fig.add_subplot(111)
    ax.imshow(prob_map, cmap="magma")
    ax.axis("off")
    buffer = io.BytesIO()
    fig.savefig(buffer, format="PNG", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return buffer.getvalue()




def png_bytes_to_base64(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


def resize_image_bytes(img: Image.Image, size: int = 256) -> bytes:
    buffer = io.BytesIO()
    img = img.resize((size, size), resample=Image.BILINEAR)
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def build_csv_report(rows):
    header = "name,change_percentage,pixels_changed\n"
    lines = [
        f"{r['name']},{r['change_percentage']:.4f},{r['pixels_changed']}"
        for r in rows
    ]
    return header + "\n".join(lines)


def build_html_report(rows):
    parts = [
        "<html><head><meta charset='utf-8'><title>Batch Report</title>"
        "<style>body{font-family:Arial,sans-serif} table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #ddd;padding:8px} th{background:#f4f4f4}</style>"
        "</head><body>",
        "<h2>Urban Change Detection Batch Report</h2>",
        "<table><tr><th>Image</th><th>Change %</th><th>Pixels Changed</th><th>Overlay</th></tr>",
    ]
    for r in rows:
        img_tag = f"<img src='data:image/png;base64,{r['overlay_thumb']}' width='120'/>" if r.get("overlay_thumb") else ""
        parts.append(
            f"<tr><td>{r['name']}</td><td>{r['change_percentage']:.2f}</td>"
            f"<td>{r['pixels_changed']}</td><td>{img_tag}</td></tr>"
        )
    parts.append("</table></body></html>")
    return "".join(parts)


@app.get(f"{API_PREFIX}/health")
def health():
    return {"status": "ok", "device": DEVICE, "arch": MODEL_ARCH}


@app.get(f"{API_PREFIX}/config")
def get_config():
    return {
        "model_arch": MODEL_ARCH,
        "default_threshold": DEFAULT_THRESHOLD,
        "default_image_size": DEFAULT_IMAGE_SIZE,
        "image_sizes": IMAGE_SIZES,
        "default_threshold_mode": DEFAULT_THRESHOLD_MODE,
        "default_percentile": DEFAULT_PERCENTILE,
        "default_min_area": DEFAULT_MIN_AREA,
        "default_align": DEFAULT_ALIGN,
        "default_multiscale": DEFAULT_MULTISCALE,
        "multiscale_sizes": MULTISCALE_SIZES,
        "default_fast_mode": DEFAULT_FAST_MODE,
    }


@app.post(f"{API_PREFIX}/infer")
async def infer(
    t1: UploadFile = File(...),
    t2: UploadFile = File(...),
    threshold: float = Form(DEFAULT_THRESHOLD),
    image_size: int = Form(DEFAULT_IMAGE_SIZE),
    use_tiling: bool = Form(True),
    threshold_mode: str = Form(DEFAULT_THRESHOLD_MODE),
    percentile: float = Form(DEFAULT_PERCENTILE),
    min_area: int = Form(DEFAULT_MIN_AREA),
    align: bool = Form(DEFAULT_ALIGN),
    multiscale: bool = Form(DEFAULT_MULTISCALE),
    fast_mode: bool = Form(DEFAULT_FAST_MODE),
):
    if fast_mode:
        image_size = 256
        use_tiling = False
        multiscale = False
        align = False
        threshold_mode = "fixed"

    if image_size not in IMAGE_SIZES:
        image_size = DEFAULT_IMAGE_SIZE

    t1_bytes = await t1.read()
    t2_bytes = await t2.read()

    t1_img = Image.open(io.BytesIO(t1_bytes)).convert("RGB")
    t2_img = Image.open(io.BytesIO(t2_bytes)).convert("RGB")

    t2_img, align_info = align_images(t1_img, t2_img, align)
    align_score = None
    if align_info.get("debug"):
        dbg = align_info["debug"]
        matches = dbg.get("num_matches") or 0
        inliers = dbg.get("inliers") or 0
        if matches > 0:
            align_score = float(inliers) / float(matches)

    model_stamp = model_cache_stamp(MODEL_PATH)
    cache_key = make_cache_key(
        hash_bytes(t1_bytes),
        hash_bytes(t2_bytes),
        image_size,
        threshold,
        use_tiling,
        model_stamp,
        threshold_mode=threshold_mode,
        percentile=percentile,
        min_area=min_area,
        align=align,
        multiscale=multiscale,
    )
    cached = cache.get(cache_key)
    if cached:
        return cached

    if multiscale:
        prob = run_multiscale_inference(
            model,
            DEVICE,
            t1_img,
            t2_img,
            MULTISCALE_SIZES,
            use_tiling,
            progress=None,
        )
        output_size = prob.shape[0]
    else:
        transform_resize, transform_tile = build_transforms(image_size)
        prob = run_inference(
            model,
            DEVICE,
            t1_img,
            t2_img,
            image_size,
            use_tiling,
            transform_resize,
            transform_tile,
            progress=None,
        )
        output_size = image_size
    calibrated, mode_used = calibrate_threshold(prob, threshold_mode, percentile, threshold)
    mask = compute_mask(prob, calibrated)
    mask, post_info = remove_small_components(mask, min_area)
    base = np.array(t2_img.resize((prob.shape[1], prob.shape[0]))).astype(np.uint8)
    overlay = compute_overlay(base, mask, alpha=DEFAULT_ALPHA)

    change_percentage = float((mask.sum() / mask.size) * 100)

    mask_png = np_to_png_bytes((mask * 255).astype(np.uint8))
    overlay_png = np_to_png_bytes(overlay)
    confidence_png = heatmap_to_png_bytes(prob)

    payload = {
        "change_percentage": change_percentage,
        "threshold": float(calibrated),
        "threshold_mode": mode_used,
        "image_size": int(output_size),
        "multiscale": bool(multiscale),
        "pixels_changed": int(mask.sum()),
        "mask_png_base64": png_bytes_to_base64(mask_png),
        "overlay_png_base64": png_bytes_to_base64(overlay_png),
        "confidence_png_base64": png_bytes_to_base64(confidence_png),
        "output_width": int(prob.shape[1]),
        "output_height": int(prob.shape[0]),
        "alignment": align_info,
        "alignment_score": align_score,
        "postprocess": post_info,
    }
    cache[cache_key] = payload
    return payload


@app.post(f"{API_PREFIX}/batch")
async def batch_infer(
    archive: UploadFile = File(...),
    threshold: float = Form(DEFAULT_THRESHOLD),
    image_size: int = Form(DEFAULT_IMAGE_SIZE),
    use_tiling: bool = Form(True),
    threshold_mode: str = Form(DEFAULT_THRESHOLD_MODE),
    percentile: float = Form(DEFAULT_PERCENTILE),
    min_area: int = Form(DEFAULT_MIN_AREA),
    align: bool = Form(DEFAULT_ALIGN),
    multiscale: bool = Form(DEFAULT_MULTISCALE),
):
    if image_size not in IMAGE_SIZES:
        image_size = DEFAULT_IMAGE_SIZE

    data = await archive.read()
    zf = zipfile.ZipFile(io.BytesIO(data))
    t1_files = {}
    t2_files = {}
    for name in zf.namelist():
        norm = name.replace("\\", "/")
        if "/t1/" in norm:
            key = os.path.basename(norm)
            t1_files[key] = norm
        elif "/t2/" in norm:
            key = os.path.basename(norm)
            t2_files[key] = norm

    common = sorted(set(t1_files.keys()) & set(t2_files.keys()))
    rows = []

    for key in common:
        t1_bytes = zf.read(t1_files[key])
        t2_bytes = zf.read(t2_files[key])
        t1_img = Image.open(io.BytesIO(t1_bytes)).convert("RGB")
        t2_img = Image.open(io.BytesIO(t2_bytes)).convert("RGB")

        t2_img, _ = align_images(t1_img, t2_img, align)

        if multiscale:
            prob = run_multiscale_inference(
                model,
                DEVICE,
                t1_img,
                t2_img,
                MULTISCALE_SIZES,
                use_tiling,
                progress=None,
            )
        else:
            transform_resize, transform_tile = build_transforms(image_size)
            prob = run_inference(
                model,
                DEVICE,
                t1_img,
                t2_img,
                image_size,
                use_tiling,
                transform_resize,
                transform_tile,
                progress=None,
            )

        calibrated, _ = calibrate_threshold(prob, threshold_mode, percentile, threshold)
        mask = compute_mask(prob, calibrated)
        mask, _ = remove_small_components(mask, min_area)
        base = np.array(t2_img.resize((prob.shape[1], prob.shape[0]))).astype(np.uint8)
        overlay = compute_overlay(base, mask, alpha=DEFAULT_ALPHA)

        change_percentage = float((mask.sum() / mask.size) * 100)
        overlay_thumb = png_bytes_to_base64(resize_image_bytes(Image.fromarray(overlay), size=200))

        rows.append(
            {
                "name": key,
                "change_percentage": change_percentage,
                "pixels_changed": int(mask.sum()),
                "overlay_thumb": overlay_thumb,
            }
        )

    csv_text = build_csv_report(rows)
    html_text = build_html_report(rows)

    return {
        "count": len(rows),
        "csv_base64": png_bytes_to_base64(csv_text.encode("utf-8")),
        "html_base64": png_bytes_to_base64(html_text.encode("utf-8")),
        "items": rows,
    }
