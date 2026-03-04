import hashlib
import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms


def load_model(model_factory, model_path, device):
    model = model_factory().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def build_transforms(image_size):
    transform_resize = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    transform_tile = transforms.Compose([
        transforms.ToTensor()
    ])
    return transform_resize, transform_tile


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def model_cache_stamp(model_path: str) -> str:
    try:
        mtime = os.path.getmtime(model_path)
    except OSError:
        mtime = 0
    return f"{model_path}:{mtime}"


def make_cache_key(
    t1_hash,
    t2_hash,
    image_size,
    threshold,
    use_tiling,
    model_stamp,
    threshold_mode="fixed",
    percentile=90,
    min_area=0,
    align=False,
    multiscale=False,
):
    return (
        f"{t1_hash}:{t2_hash}:{image_size}:{threshold:.4f}:{use_tiling}:{model_stamp}:"
        f"{threshold_mode}:{percentile}:{min_area}:{align}:{multiscale}"
    )


def compute_mask(prob_map: np.ndarray, threshold: float) -> np.ndarray:
    return (prob_map > threshold).astype(np.uint8)


def compute_overlay(base_rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    base = base_rgb.astype(np.float32)
    red = np.zeros_like(base)
    red[..., 0] = 255
    mask3 = np.repeat(mask[..., None], 3, axis=2)
    overlay = np.where(mask3 == 1, (1 - alpha) * base + alpha * red, base)
    return overlay.astype(np.uint8)


def otsu_threshold(prob_map: np.ndarray) -> float:
    flat = (prob_map.clip(0, 1) * 255).astype(np.uint8).ravel()
    hist = np.bincount(flat, minlength=256).astype(np.float64)
    total = flat.size
    if total == 0:
        return 0.5
    sum_total = np.dot(np.arange(256), hist)
    sum_b = 0.0
    w_b = 0.0
    w_f = 0.0
    var_max = -1.0
    threshold = 127
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > var_max:
            var_max = var_between
            threshold = t
    return float(threshold / 255.0)


def calibrate_threshold(prob_map: np.ndarray, mode: str, percentile: float, fallback: float):
    mode = (mode or "fixed").lower()
    if mode == "otsu":
        return otsu_threshold(prob_map), "otsu"
    if mode == "percentile":
        pct = float(np.clip(percentile, 50, 99.9))
        return float(np.percentile(prob_map, pct)), f"percentile:{pct:.1f}"
    return float(fallback), "fixed"


def remove_small_components(mask: np.ndarray, min_area: int):
    if min_area <= 0:
        return mask, {"removed": 0}
    try:
        import cv2
    except Exception:
        return mask, {"removed": 0, "reason": "cv2_missing"}
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros_like(mask, dtype=np.uint8)
    removed = 0
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            keep[labels == label] = 1
        else:
            removed += 1
    return keep, {"removed": removed}


def align_images(t1_img: Image.Image, t2_img: Image.Image, enabled: bool):
    if not enabled:
        return t2_img, {"aligned": False, "reason": "disabled"}
    try:
        from align import align_t2_to_t1
    except Exception as exc:
        return t2_img, {"aligned": False, "reason": f"align_import_failed:{exc.__class__.__name__}"}
    try:
        t1_arr = np.array(t1_img)
        t2_arr = np.array(t2_img)
        aligned, _, debug = align_t2_to_t1(t1_arr, t2_arr)
        return Image.fromarray(aligned), {"aligned": True, "debug": debug}
    except Exception as exc:
        return t2_img, {"aligned": False, "reason": f"align_failed:{exc.__class__.__name__}"}


def run_inference(
    model,
    device,
    t1_img: Image.Image,
    t2_img: Image.Image,
    tile_size: int,
    use_tiling: bool,
    transform_resize,
    transform_tile,
    progress=None,
):
    if not use_tiling:
        t1_tensor = transform_resize(t1_img).unsqueeze(0).to(device)
        t2_tensor = transform_resize(t2_img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(t1_tensor, t2_tensor)
        prob = output.squeeze().cpu().numpy()
        if progress is not None:
            progress.progress(1.0)
        return prob

    t1_arr = np.array(t1_img)
    t2_arr = np.array(t2_img)
    h, w = t1_arr.shape[:2]
    pad_h = (tile_size - (h % tile_size)) % tile_size
    pad_w = (tile_size - (w % tile_size)) % tile_size

    pad_cfg = ((0, pad_h), (0, pad_w), (0, 0))
    t1_pad = np.pad(t1_arr, pad_cfg, mode="reflect")
    t2_pad = np.pad(t2_arr, pad_cfg, mode="reflect")

    ph, pw = t1_pad.shape[:2]
    prob_map = np.zeros((ph, pw), dtype=np.float32)
    stride = tile_size

    ys = list(range(0, max(ph - tile_size + 1, 1), stride))
    xs = list(range(0, max(pw - tile_size + 1, 1), stride))
    if ys[-1] != ph - tile_size:
        ys.append(ph - tile_size)
    if xs[-1] != pw - tile_size:
        xs.append(pw - tile_size)

    total_tiles = len(ys) * len(xs)
    done_tiles = 0
    for y in ys:
        for x in xs:
            t1_tile = Image.fromarray(t1_pad[y:y + tile_size, x:x + tile_size])
            t2_tile = Image.fromarray(t2_pad[y:y + tile_size, x:x + tile_size])
            t1_tensor = transform_tile(t1_tile).unsqueeze(0).to(device)
            t2_tensor = transform_tile(t2_tile).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(t1_tensor, t2_tensor)
            prob_tile = output.squeeze().cpu().numpy()
            prob_map[y:y + tile_size, x:x + tile_size] = prob_tile
            done_tiles += 1
            if progress is not None:
                progress.progress(done_tiles / total_tiles)

    return prob_map[:h, :w]


def resize_prob(prob_map: np.ndarray, size: int) -> np.ndarray:
    if prob_map.shape[0] == size and prob_map.shape[1] == size:
        return prob_map
    img = Image.fromarray((prob_map * 255).astype(np.uint8))
    img = img.resize((size, size), resample=Image.BILINEAR)
    return np.array(img).astype(np.float32) / 255.0


def run_multiscale_inference(
    model,
    device,
    t1_img: Image.Image,
    t2_img: Image.Image,
    sizes,
    use_tiling: bool,
    progress=None,
):
    sizes = sorted(list(set(int(s) for s in sizes)))
    if len(sizes) == 1:
        transform_resize, transform_tile = build_transforms(sizes[0])
        return run_inference(
            model,
            device,
            t1_img,
            t2_img,
            sizes[0],
            use_tiling,
            transform_resize,
            transform_tile,
            progress=progress,
        )

    class _ProgressProxy:
        def __init__(self, outer, base, total):
            self.outer = outer
            self.base = base
            self.total = total

        def progress(self, value):
            if self.outer is None:
                return
            self.outer.progress((self.base + value) / self.total)

    target = max(sizes)
    prob_sum = None
    total = len(sizes)
    for idx, size in enumerate(sizes):
        transform_resize, transform_tile = build_transforms(size)
        step_progress = _ProgressProxy(progress, idx, total) if progress is not None else None
        prob = run_inference(
            model,
            device,
            t1_img,
            t2_img,
            size,
            use_tiling,
            transform_resize,
            transform_tile,
            progress=step_progress,
        )
        prob = resize_prob(prob, target)
        if prob_sum is None:
            prob_sum = prob
        else:
            prob_sum += prob

    prob_map = prob_sum / float(total)
    if progress is not None:
        progress.progress(1.0)
    return prob_map
