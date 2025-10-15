#!/usr/bin/env python3
"""
iphone_depth_to_metric_gui.py

Interactive version:
1) Loads an iPhone image (HEIC/JPG/PNG).
2) Tries to extract embedded depth (Portrait/LiDAR); otherwise uses MiDaS to estimate relative depth.
3) Lets you CLICK calibration points on the image (OpenCV window). Press:
     - Left click: add point
     - 'u': undo last point
     - 'q': finish point selection
4) After you finish clicking, the script asks you (in the terminal) to enter the real-world distance (in meters)
   for each clicked point. It fits a linear mapping depth_m = a*rel + b using least squares.
5) Saves:
     - *_metric_depth_mm.png (16-bit millimeters)
     - *_metric_depth_preview.png (8-bit preview, brighter = closer)
     - *_metric_depth_info.json (metadata, model used, coefficients)

Notes:
- Without calibration points, monocular depth cannot yield true meters; embedded depth MAY be metric,
  but this varies with export pipeline. Click 2+ calibration points for best results.
- Coordinates are taken directly from mouse clicks; no need to type (x,y) manually.

Dependencies:
pip install pillow pillow-heif numpy torch torchvision timm opencv-python
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

import torch

import cv2


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
    """Return relative depth (float32, larger ~ farther)."""
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
    # Normalize to [0,1] for consistency
    depth -= depth.min()
    if depth.max() > 0:
        depth /= depth.max()
    return depth


def try_extract_embedded_depth(path: Path):
    """
    Extract auxiliary depth/disparity from HEIC if present.
    Returns (depth_norm, info_dict) or (None, info_dict).
    depth_norm: float32 [0,1], larger = farther (normalized).
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

    depth_img = None
    if hasattr(heif, "image") and hasattr(heif.image, "extra_images"):
        for extra in heif.image.extra_images:
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

    arr = np.array(depth_img).astype(np.float32)
    arr -= arr.min()
    if arr.max() > 0:
        arr /= arr.max()
    info["embedded_depth_found"] = True
    info["notes"] = f"Extracted auxiliary depth/disparity ({depth_img.mode}, {depth_img.size})."
    return arr, info


def fit_linear_to_meters(rel_depth: np.ndarray, pts_xy, dists_m):
    """
    Fit depth_m = a*rel + b via least squares.
    pts_xy: list of (x,y)
    dists_m: list of distances (meters), same length as pts_xy
    """
    if len(pts_xy) < 2:
        return 1.0, 0.0
    H, W = rel_depth.shape[:2]
    X = []
    y = []
    for (x, ypix), d in zip(pts_xy, dists_m):
        x = int(np.clip(round(x), 0, W - 1))
        ypix = int(np.clip(round(ypix), 0, H - 1))
        X.append([rel_depth[ypix, x], 1.0])
        y.append(float(d))
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    theta, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b = float(theta[0]), float(theta[1])
    return a, b


def to_uint16_mm(depth_m):
    mm = np.clip(np.round(depth_m * 1000.0), 0, 65535).astype(np.uint16)
    return mm


def save_outputs(stem_path: Path, depth_m: np.ndarray, meta: dict):
    mm16 = to_uint16_mm(depth_m)
    out_mm = stem_path.with_name(stem_path.stem + "_metric_depth_mm.png")
    Image.fromarray(mm16, mode="I;16").save(out_mm)

    d = depth_m.copy().astype(np.float32)
    d_min = float(np.nanmin(d)) if np.isfinite(np.nanmin(d)) else 0.0
    d_max = float(np.nanmax(d)) if np.isfinite(np.nanmax(d)) else 1.0
    if d_max == d_min:
        d_max = d_min + 1.0
    dn = (d - d_min) / (d_max - d_min)
    dn = 1.0 - dn  # closer brighter
    preview = (np.clip(dn, 0, 1) * 255.0).astype(np.uint8)
    out_preview = stem_path.with_name(stem_path.stem + "_metric_depth_preview.png")
    Image.fromarray(preview, mode="L").save(out_preview)

    out_json = stem_path.with_name(stem_path.stem + "_metric_depth_info.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return str(out_mm), str(out_preview), str(out_json)


def click_points(image_bgr):
    """
    OpenCV UI to collect points.
    - Left click to add point
    - 'u' to undo last
    - 'q' to finish
    Returns list of (x,y) in original image coords.
    """
    pts = []
    disp = image_bgr.copy()

    def redraw():
        disp[:] = image_bgr
        for i, (x, y) in enumerate(pts):
            cv2.circle(disp, (int(x), int(y)), 6, (0, 255, 255), -1)
            cv2.putText(disp, f"{i+1}", (int(x)+8, int(y)-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    def on_mouse(event, x, y, flags, param):
        nonlocal pts
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
            redraw()

    redraw()
    cv2.namedWindow("Click calibration points (L-click add, 'u' undo, 'q' finish)", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Click calibration points (L-click add, 'u' undo, 'q' finish)", on_mouse)

    while True:
        cv2.imshow("Click calibration points (L-click add, 'u' undo, 'q' finish)", disp)
        key = cv2.waitKey(20) & 0xFF
        if key == ord('q'):
            break
        if key == ord('u') and pts:
            pts.pop()
            redraw()

    cv2.destroyAllWindows()
    return pts


def main():
    ap = argparse.ArgumentParser(description="Interactive metric depth from iPhone image using embedded depth or MiDaS + clicked calibration")
    ap.add_argument("input", help="Path to input HEIC/JPG/PNG")
    ap.add_argument("--prefer-embedded", action="store_true", help="Prefer HEIC embedded depth if present")
    ap.add_argument("--model", default="MiDaS_small", choices=["MiDaS_small", "DPT_Hybrid", "DPT_Large"],
                    help="MiDaS model when embedded depth is missing or not preferred")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"[ERROR] Input not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    # Load image for display and processing (RGB for depth, BGR for OpenCV display)
    img = Image.open(str(in_path))
    if img.mode != "RGB":
        img = img.convert("RGB")
    W, H = img.size
    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    meta = {
        "input": str(in_path),
        "image_size": [W, H],
        "used_embedded_depth": False,
        "midas_model": None,
        "linear_fit": {"a": 1.0, "b": 0.0, "n_points": 0},
        "notes": []
    }

    # Depth source
    rel_depth = None
    embedded_info = {}
    if in_path.suffix.lower() in (".heic", ".heif"):
        rel_depth_try, embedded_info = try_extract_embedded_depth(in_path)
        if rel_depth_try is not None:
            if args.prefer_embedded:
                rel_depth = rel_depth_try
                meta["used_embedded_depth"] = True
                meta["notes"].append("Using embedded depth/disparity (normalized).")
            else:
                rel_depth = rel_depth_try
                meta["notes"].append("Embedded depth available; you can use --prefer-embedded to force its use.")
        else:
            meta["notes"].append("Embedded depth not found or not readable.")

    if rel_depth is None or not meta["used_embedded_depth"]:
        device = get_device()
        meta["notes"].append(f"Running MiDaS on device: {device}")
        meta["midas_model"] = args.model
        rel_depth = midas_depth(img, model_name=args.model, device=device)

    # Interactive calibration
    print("\n=== Calibration ===")
    print("An image window will open. Left-click to add calibration points; 'u' to undo; 'q' to finish.\n")
    pts_xy = click_points(img_bgr)
    print(f"Selected {len(pts_xy)} point(s).")

    dists_m = []
    if len(pts_xy) >= 2:
        print("Enter the real-world distance in METERS for each point (order shown on image).")
        for i, (x, y) in enumerate(pts_xy, start=1):
            while True:
                try:
                    val = float(input(f"Distance for point #{i} at (x={x}, y={y}) [m]: ").strip())
                    dists_m.append(val)
                    break
                except Exception:
                    print("Please enter a numeric value (e.g., 2.0).")
    else:
        print("Fewer than 2 points selected. Output will be relative unless embedded depth is already metric.")

    a, b = fit_linear_to_meters(rel_depth, pts_xy, dists_m) if len(pts_xy) >= 2 else (1.0, 0.0)
    meta["linear_fit"]["a"] = a
    meta["linear_fit"]["b"] = b
    meta["linear_fit"]["n_points"] = len(pts_xy)

    if len(pts_xy) < 2 and not meta["used_embedded_depth"]:
        meta["notes"].append("No/insufficient calibration points; meters will be arbitrary (relative). Provide 2+ points for real meters.")

    # Map to meters and save
    depth_m = a * rel_depth + b
    depth_m = np.clip(depth_m, 0, None).astype(np.float32)

    mm_path, prev_path, json_path = save_outputs(in_path, depth_m, meta)
    print("\n[DONE] Wrote:")
    print(" ", mm_path)
    print(" ", prev_path)
    print(" ", json_path)


if __name__ == "__main__":
    main()
