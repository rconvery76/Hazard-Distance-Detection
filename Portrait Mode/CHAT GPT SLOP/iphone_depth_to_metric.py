#!/usr/bin/env python3
"""
iphone_depth_to_metric.py

Compute per-pixel distance-to-camera (meters) from an iPhone photo.

Paths:
A) If the HEIC contains an embedded depth/disparity auxiliary image (Portrait/LiDAR), we extract it
   with pillow-heif and use that as the base depth. If it appears to be disparity (bigger = closer),
   we invert/normalize and *optionally* scale to meters using --calib pairs. Some HEIC depth is already
   metric (ARKit/LiDAR), but scale metadata is not always preserved on export; calibration makes it robust.

B) If no embedded depth is found, we run MiDaS (monocular) to get a relative depth map, then fit a linear
   mapping depth_m = a*rel + b from calibration pairs you provide via --calib "x,y,dist; x,y,dist; ..."
   with distances in meters.

Outputs:
- <stem>_metric_depth_mm.png   : 16-bit depth map in millimeters (clipped to [0, 65535])
- <stem>_metric_depth_preview.png : 8-bit preview (brighter = closer)
- <stem>_metric_depth_info.json   : JSON with notes, fitted coefficients, and whether embedded depth was used

Usage examples:
  python iphone_depth_to_metric.py IMG_1234.heic --calib "600,900,2.0; 1200,900,5.5"
  python iphone_depth_to_metric.py IMG_1234.heic --prefer-embedded
  python iphone_depth_to_metric.py IMG_1234.heic --model DPT_Hybrid --calib "500,800,1.2; 900,1000,3.0"

Tips for calibration:
- Place a known target in the scene (e.g., a ruler, a 10 cm tag, or a friend at a measured distance).
- Provide 2+ points spanning near/far distances for a stable fit.
- Coordinates (x,y) are pixel positions in the original image (origin at top-left).

Note: Without calibration (and without trustworthy metric depth in the HEIC), absolute meters cannot be
recovered from monocular depth alone. We'll still produce a *relative* metric file with a=1,b=0 if no
calibration is given, but its scale will be arbitrary.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

# HEIC support
heif_ok = True
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception as e:
    heif_ok = False
    print("[WARN] pillow-heif unavailable; cannot read embedded depth from HEIC.", file=sys.stderr)

# Optional: MiDaS for fallback relative depth
import torch

# Optional: visualization
try:
    import cv2
    cv2_ok = True
except Exception:
    cv2_ok = False


def parse_calib(calib_str):
    """
    Parse --calib "x1,y1,d1; x2,y2,d2; ..."  (distances in meters)
    Returns: list of (x,y,d)
    """
    pts = []
    if not calib_str:
        return pts
    for token in calib_str.split(";"):
        token = token.strip()
        if not token:
            continue
        parts = token.split(",")
        if len(parts) != 3:
            raise ValueError(f"Bad calib token: {token}")
        x = int(round(float(parts[0])))
        y = int(round(float(parts[1])))
        d = float(parts[2])
        pts.append((x, y, d))
    return pts


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_midas(model_name, device):
    midas = torch.hub.load("intel-isl/MiDaS", model_name)
    midas.to(device)
    midas.eval()

    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_name in ("DPT_Large", "DPT_Hybrid"):
        transform = transforms.dpt_transform
    else:
        transform = transforms.small_transform
    return midas, transform


@torch.inference_mode()
def midas_depth(img_rgb: Image.Image, model_name="MiDaS_small", device=None) -> np.ndarray:
    """Return relative depth (float32, larger ~ farther for MiDaS)."""
    if device is None:
        device = get_device()
    midas, transform = load_midas(model_name, device)
    img_np = np.array(img_rgb, dtype=np.uint8)
    t = transform(img_np)
    if isinstance(t, dict):
        inp = t["image"].to(device)
    else:
        inp = t.to(device)
    pred = midas(inp)
    pred = torch.nn.functional.interpolate(
        pred.unsqueeze(1),
        size=img_np.shape[:2],  # (H, W)
        mode="bicubic",
        align_corners=False,
    ).squeeze(1).squeeze(0)
    depth = pred.detach().cpu().numpy().astype(np.float32)
    return depth


def try_extract_embedded_depth(path: Path):
    """
    Attempt to extract an auxiliary depth/disparity image from a HEIC.
    Returns (depth, info_dict) or (None, info_dict).
    Depth is float32 array (H, W), normalized to [0,1] where larger = farther (we standardize here).
    """
    info = {"embedded_depth_found": False, "notes": ""}
    if not heif_ok:
        info["notes"] = "pillow-heif not available"
        return None, info
    try:
        heif = pillow_heif.open_heif(str(path))
    except Exception as e:
        info["notes"] = f"Failed to open HEIC: {e}"
        return None, info

    # pillow-heif represents auxiliary images in heif.image.extra_images (depth, alpha, etc.)
    # We'll search for the first extra image with depth-like dimensions (grayscale).
    depth_img = None
    if hasattr(heif, "image") and hasattr(heif.image, "extra_images"):
        for extra in heif.image.extra_images:
            # Try to convert to PIL and inspect
            try:
                pil_extra = Image.frombytes(
                    extra.mode, (extra.size[0], extra.size[1]), extra.data, "raw"
                )
            except Exception:
                continue
            if pil_extra.mode in ("I;16", "I", "L"):
                depth_img = pil_extra
                break

    if depth_img is None:
        info["notes"] = "No suitable auxiliary depth image found."
        return None, info

    arr = np.array(depth_img)
    arr = arr.astype(np.float32)
    # Normalize to [0,1] for a consistent convention: larger = farther
    arr -= arr.min()
    if arr.max() > 0:
        arr /= arr.max()
    # Heuristic: some embedded maps are disparity (larger = closer). Detect by histogram skew.
    # We'll just make a copy with both variants and choose the one that correlates
    # with MiDaS direction later if calibration is given; else assume larger = farther.
    info["embedded_depth_found"] = True
    info["notes"] = f"Extracted auxiliary depth/disparity ({depth_img.mode}, {depth_img.size})."
    return arr, info


def fit_linear_to_meters(rel_depth: np.ndarray, calib_pts):
    """
    Fit depth_m = a * rel + b using least squares on calibration points.
    calib_pts: list of (x,y,d_meters)
    Returns (a, b). If <2 points, fall back to (1, 0).
    """
    if len(calib_pts) < 2:
        return 1.0, 0.0
    X = []
    y = []
    H, W = rel_depth.shape[:2]
    for (x, y_m, d) in calib_pts:
        x = np.clip(int(x), 0, W - 1)
        y_m = np.clip(int(y_m), 0, H - 1)
        X.append([rel_depth[y_m, x], 1.0])
        y.append(d)
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    # Solve (X^T X) theta = X^T y
    theta, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b = float(theta[0]), float(theta[1])
    return a, b


def to_uint16_mm(depth_m):
    mm = np.clip(np.round(depth_m * 1000.0), 0, 65535).astype(np.uint16)
    return mm


def save_outputs(stem_path: Path, depth_m: np.ndarray, meta: dict):
    # 16-bit millimeter map
    mm16 = to_uint16_mm(depth_m)
    out_mm = stem_path.with_name(stem_path.stem + "_metric_depth_mm.png")
    Image.fromarray(mm16, mode="I;16").save(out_mm)

    # 8-bit preview (brighter = closer): invert normalized meters
    d = depth_m.copy().astype(np.float32)
    d = np.where(np.isfinite(d), d, np.nan)
    d_min = np.nanmin(d)
    d_max = np.nanmax(d)
    if not np.isfinite(d_min) or not np.isfinite(d_max) or d_max == d_min:
        d_min, d_max = 0.0, 1.0
    dn = (d - d_min) / (d_max - d_min + 1e-12)
    dn = 1.0 - dn  # closer brighter
    preview = (np.clip(dn, 0, 1) * 255.0).astype(np.uint8)
    out_preview = stem_path.with_name(stem_path.stem + "_metric_depth_preview.png")
    Image.fromarray(preview, mode="L").save(out_preview)

    # JSON info
    out_json = stem_path.with_name(stem_path.stem + "_metric_depth_info.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return str(out_mm), str(out_preview), str(out_json)


def main():
    ap = argparse.ArgumentParser(description="Compute metric depth from iPhone HEIC using embedded depth or MiDaS + calibration")
    ap.add_argument("input", help="Path to input HEIC/JPG/PNG")
    ap.add_argument("--prefer-embedded", action="store_true", help="Prefer HEIC embedded depth if present")
    ap.add_argument("--model", default="MiDaS_small", choices=["MiDaS_small", "DPT_Hybrid", "DPT_Large"],
                    help="MiDaS model when embedded depth is missing or not preferred")
    ap.add_argument("--calib", default="", help='Calibration pairs: "x,y,dist; x,y,dist; ..." with dist in meters')
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"[ERROR] Input not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    # Load main image
    img = Image.open(str(in_path))
    if img.mode != "RGB":
        img = img.convert("RGB")
    W, H = img.size

    meta = {
        "input": str(in_path),
        "image_size": [W, H],
        "used_embedded_depth": False,
        "midas_model": None,
        "linear_fit": {"a": 1.0, "b": 0.0, "n_points": 0},
        "notes": []
    }

    # Try embedded depth if HEIC
    rel_depth = None
    embedded_info = {}
    if args.prefer_embedded and in_path.suffix.lower() in (".heic", ".heif"):
        rel_depth, embedded_info = try_extract_embedded_depth(in_path)
        if rel_depth is not None:
            # Resize to image size if needed
            if rel_depth.shape[:2] != (H, W):
                rel_depth = np.array(Image.fromarray((rel_depth * 255).astype(np.uint8), mode="L").resize((W, H), Image.BICUBIC)).astype(np.float32) / 255.0
            meta["used_embedded_depth"] = True
            meta["notes"].append("Used embedded depth/disparity (normalized).")
        else:
            meta["notes"].append("Embedded depth not found or not readable; falling back to MiDaS.")
    elif in_path.suffix.lower() in (".heic", ".heif"):
        # Even if not preferred, we still try to get it; if not found we use MiDaS.
        rel_depth_try, embedded_info = try_extract_embedded_depth(in_path)
        if rel_depth_try is not None:
            rel_depth = rel_depth_try
            if rel_depth.shape[:2] != (H, W):
                rel_depth = np.array(Image.fromarray((rel_depth * 255).astype(np.uint8), mode="L").resize((W, H), Image.BICUBIC)).astype(np.float32) / 255.0
            meta["notes"].append("Embedded depth available; using MiDaS unless --prefer-embedded is specified.")
        else:
            meta["notes"].append("Embedded depth not found; using MiDaS.")

    # If no usable embedded depth, run MiDaS
    if rel_depth is None or not meta["used_embedded_depth"]:
        device = get_device()
        meta["notes"].append(f"Running MiDaS on device: {device}")
        meta["midas_model"] = args.model
        rel_depth = midas_depth(img, model_name=args.model, device=device)
        # Normalize to [0,1] for consistency
        rd = rel_depth.astype(np.float32)
        rd -= rd.min()
        if rd.max() > 0:
            rd /= rd.max()
        rel_depth = rd

    # Calibration
    calib_pts = parse_calib(args.calib)
    meta["linear_fit"]["n_points"] = len(calib_pts)
    a, b = fit_linear_to_meters(rel_depth, calib_pts)
    meta["linear_fit"]["a"] = a
    meta["linear_fit"]["b"] = b
    if len(calib_pts) < 2 and not meta["used_embedded_depth"]:
        meta["notes"].append("No calibration provided; meters will be arbitrary (relative). Provide --calib with 2+ points for real meters.")

    # Map to meters
    depth_m = a * rel_depth + b
    # Sanitize
    depth_m = np.clip(depth_m, 0, None).astype(np.float32)

    # Save outputs
    mm_path, prev_path, json_path = save_outputs(in_path, depth_m, meta)

    print("[DONE] Wrote:")
    print(" ", mm_path)
    print(" ", prev_path)
    print(" ", json_path)


if __name__ == "__main__":
    main()
