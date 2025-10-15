#!/usr/bin/env python3
"""
detect_then_depth.py

End-to-end pipeline:
- INPUT: RGB image (HEIC/JPG/PNG). HEIC supported via pillow-heif.
- STEP 1: Object detection on the RGB image (torchvision Faster R-CNN, COCO classes).
- STEP 2: Build depth map
    A) If HEIC with embedded auxiliary depth/disparity: optionally use it.
       - If --trust-embedded-meters is set, depth is assumed to be in METERS (no scaling).
       - Else we normalize to [0,1] (relative), and optional --calib "x,y,dist;..." fits meters.
    B) Otherwise run MiDaS (MiDaS_small/DPT_Hybrid/DPT_Large) to estimate RELATIVE depth, then
       fit meters via --calib (needs ≥2 points). Without calibration, output distances are RELATIVE only.
- STEP 3: For each detection, compute the median distance inside the bbox.
- OUTPUTS:
    - <stem>_overlay.png: colorized depth with detection boxes + labels
    - <stem>_detections.csv: id, class, score, bbox, median_distance_m (or None if unavailable)
    - <stem>_metric_depth_mm.png: 16-bit depth (millimeters) if meters are known (embedded trusted or calib given)
    - <stem>_metric_depth_preview.png + <stem>_metric_depth_info.json: optional metadata for depth

Usage:
  python detect_then_depth.py input.heic --out overlay.png --csv dets.csv \
    --prefer-embedded --trust-embedded-meters

  python detect_then_depth.py input.jpg --model DPT_Hybrid \
    --calib "600,900,2.0; 1200,900,5.5"

Args:
  --prefer-embedded           : If HEIC and embedded depth exists, prefer it over MiDaS
  --trust-embedded-meters     : Treat embedded depth as already in meters (skips calibration)
  --calib "x,y,dist;..."      : Pixel coords and real distances (meters) to fit linear scale
  --model {MiDaS_small,DPT_Hybrid,DPT_Large} : MiDaS model (fallback or forced)
  --score, --max              : Detector score threshold and max detections
  --near, --far               : Optional near/far clip in meters for depth color scaling
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
    print("[WARN] pillow-heif unavailable; HEIC decoding and embedded depth extraction may fail.", file=sys.stderr)

import torch
import torchvision
from torchvision.transforms import functional as F
import cv2
import pandas as pd


COCO_CLASSES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


def parse_calib(calib_str):
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
        x = float(parts[0]); y = float(parts[1]); d = float(parts[2])
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
    midas.to(device).eval()
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_name in ("DPT_Large", "DPT_Hybrid"):
        transform = transforms.dpt_transform
    else:
        transform = transforms.small_transform
    return midas, transform


@torch.inference_mode()
def midas_relative_depth(img_rgb: Image.Image, model_name="MiDaS_small", device=None) -> np.ndarray:
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
        size=img_np.shape[:2], mode="bicubic", align_corners=False
    ).squeeze(1).squeeze(0)
    d = pred.detach().cpu().numpy().astype(np.float32)
    # Normalize to [0,1]
    d -= d.min()
    if d.max() > 0:
        d /= d.max()
    return d


def try_extract_embedded_depth(path: Path):
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
    info["embedded_depth_found"] = True
    info["notes"] = f"Extracted auxiliary depth/disparity ({depth_img.mode}, {depth_img.size})."
    return arr, info


def fit_linear_to_meters(rel_depth: np.ndarray, calib_pts):
    if len(calib_pts) < 2:
        return 1.0, 0.0
    H, W = rel_depth.shape[:2]
    X = []; y = []
    for (x, ypix, dist) in calib_pts:
        xi = int(np.clip(round(x), 0, W - 1))
        yi = int(np.clip(round(ypix), 0, H - 1))
        X.append([rel_depth[yi, xi], 1.0])
        y.append(float(dist))
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    (a, b), *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(a), float(b)


def save_metric_depth_outputs(stem: Path, depth_m: np.ndarray, meta: dict):
    # 16-bit millimeter PNG
    mm = np.clip(np.round(depth_m * 1000.0), 0, 65535).astype(np.uint16)
    mm_path = stem.with_name(stem.stem + "_metric_depth_mm.png")
    Image.fromarray(mm, mode="I;16").save(mm_path)

    # 8-bit preview (closer = brighter)
    d = depth_m.astype(np.float32)
    dmin = float(np.nanmin(d)); dmax = float(np.nanmax(d))
    if not np.isfinite(dmin) or not np.isfinite(dmax) or dmax == dmin:
        dmin, dmax = 0.0, 1.0
    dn = (d - dmin) / (dmax - dmin + 1e-12)
    dn = 1.0 - dn
    preview = (np.clip(dn, 0, 1) * 255).astype(np.uint8)
    prev_path = stem.with_name(stem.stem + "_metric_depth_preview.png")
    Image.fromarray(preview, mode="L").save(prev_path)

    meta_path = stem.with_name(stem.stem + "_metric_depth_info.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return str(mm_path), str(prev_path), str(meta_path)


@torch.inference_mode()
def detect_objects(image_pil, score_thresh=0.5, max_det=50, device=None):
    if device is None:
        device = get_device()
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    model.to(device).eval()
    timg = F.to_tensor(image_pil).to(device)
    out = model([timg])[0]
    boxes = out["boxes"].detach().cpu().numpy()
    labels = out["labels"].detach().cpu().numpy()
    scores = out["scores"].detach().cpu().numpy()

    keep = scores >= score_thresh
    boxes = boxes[keep]; labels = labels[keep]; scores = scores[keep]
    if len(scores) > max_det:
        idx = np.argsort(-scores)[:max_det]
        boxes = boxes[idx]; labels = labels[idx]; scores = scores[idx]
    return boxes, labels, scores


def colorize_depth(depth_m, near=None, far=None):
    d = depth_m.copy()
    d[~np.isfinite(d)] = np.nan
    if near is None: near = np.nanmin(d)
    if far  is None: far  = np.nanmax(d)
    if not np.isfinite(near) or not np.isfinite(far) or near == far:
        near, far = 0.0, 1.0
    dn = (d - near) / (far - near + 1e-12)
    dn = 1.0 - np.clip(dn, 0, 1)
    dn_u8 = (dn * 255).astype(np.uint8)
    return cv2.applyColorMap(dn_u8, cv2.COLORMAP_PLASMA)  # BGR


def median_depth_in_box(depth_m, box):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1 = max(x1, 0); y1 = max(y1, 0)
    x2 = max(x2, 0); y2 = max(y2, 0)
    if x2 <= x1 or y2 <= y1: return np.nan
    crop = depth_m[y1:y2, x1:x2]
    crop = crop[np.isfinite(crop) & (crop > 0)]
    return float(np.median(crop)) if crop.size else np.nan


def draw_boxes(col_bgr, boxes, labels, scores, depth_m, units="m", calibrated=True):
    out = col_bgr.copy()
    H, W = out.shape[:2]
    rows = []
    for i, (box, lab, sc) in enumerate(zip(boxes, labels, scores), start=1):
        x1, y1, x2, y2 = box
        x1 = int(np.clip(round(x1), 0, W-1)); y1 = int(np.clip(round(y1), 0, H-1))
        x2 = int(np.clip(round(x2), 0, W-1)); y2 = int(np.clip(round(y2), 0, H-1))
        med = median_depth_in_box(depth_m, (x1, y1, x2, y2))
        color = tuple(int(c) for c in np.random.default_rng(int(lab)).integers(0,255,3))
        cv2.rectangle(out, (x1,y1), (x2,y2), color, 2)
        cls = COCO_CLASSES[lab] if 0 <= lab < len(COCO_CLASSES) else f"id{lab}"
        if np.isfinite(med):
            prefix = "" if calibrated else "~"
            text = f"{cls} {sc:.2f} | {prefix}{med:.2f} {units}"
        else:
            text = f"{cls} {sc:.2f} | n/a"
        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ty = max(y1 - th - bl - 4, 0)
        cv2.rectangle(out, (x1, ty), (x1 + tw + 6, ty + th + bl + 6), (0,0,0), -1)
        cv2.putText(out, text, (x1+3, ty + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        rows.append({
            "id": i, "class_id": int(lab), "class_name": cls, "score": float(sc),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "median_distance_m": float(med) if np.isfinite(med) else None
        })
    return out, rows


def main():
    ap = argparse.ArgumentParser(description="Detect objects, build depth map, and report per-object distances")
    ap.add_argument("input", help="Path to HEIC/JPG/PNG image")
    ap.add_argument("--prefer-embedded", action="store_true", help="Prefer embedded HEIC depth if present")
    ap.add_argument("--trust-embedded-meters", action="store_true", help="Assume embedded depth units are meters")
    ap.add_argument("--calib", default="", help='Calibration pairs: "x,y,dist; x,y,dist; ..." (meters)')
    ap.add_argument("--model", default="MiDaS_small", choices=["MiDaS_small","DPT_Hybrid","DPT_Large"], help="MiDaS model if needed")
    ap.add_argument("--score", type=float, default=0.5, help="Detection score threshold")
    ap.add_argument("--max", type=int, default=50, help="Max detections")
    ap.add_argument("--near", type=float, default=None, help="Near clip (meters) for color scaling")
    ap.add_argument("--far", type=float, default=None, help="Far clip (meters) for color scaling")
    ap.add_argument("--out", default=None, help="Output overlay PNG (default: <stem>_overlay.png)")
    ap.add_argument("--csv", default=None, help="Output CSV (default: <stem>_detections.csv)")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"[ERROR] Input not found: {in_path}", file=sys.stderr); sys.exit(1)

    # Load visual for detection
    img_pil = Image.open(str(in_path))
    if img_pil.mode != "RGB":
        img_pil = img_pil.convert("RGB")
    W, H = img_pil.size
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # Depth build
    meta = {
        "input": str(in_path),
        "image_size": [W, H],
        "used_embedded_depth": False,
        "trusted_embedded_meters": bool(args.trust_embedded_meters),
        "midas_model": None,
        "linear_fit": {"a": 1.0, "b": 0.0, "n_points": 0},
        "notes": []
    }

    rel_depth = None
    depth_m = None
    calibrated = False

    # Try embedded for HEIC
    if in_path.suffix.lower() in (".heic", ".heif"):
        emb, emb_info = try_extract_embedded_depth(in_path)
        if emb is not None:
            meta["notes"].append("Embedded depth found.")
            if args.trust_embedded_meters:
                # Assume embedded already in meters
                if emb.ndim == 2:
                    depth_m = emb.astype(np.float32)
                else:
                    depth_m = emb[...,0].astype(np.float32)
                # If range looks huge (e.g., 0..65535), try interpret as mm or 16-bit normalized
                if depth_m.max() > 1000:  # likely in millimeters or arbitrary units
                    depth_m = depth_m / 1000.0  # interpret as mm -> m heuristic
                meta["used_embedded_depth"] = True
                calibrated = True
            else:
                # Treat as relative, normalize to [0,1]
                rel = emb.astype(np.float32)
                rel -= rel.min()
                if rel.max() > 0: rel /= rel.max()
                rel_depth = rel
                meta["used_embedded_depth"] = True
                meta["notes"].append("Embedded depth treated as relative; use --trust-embedded-meters or --calib for meters.")
        else:
            meta["notes"].append(emb_info.get("notes","No embedded depth."))

    # If still no metric depth, run MiDaS to get relative
    if depth_m is None and rel_depth is None:
        device = get_device()
        meta["notes"].append(f"Running MiDaS on device: {device}")
        meta["midas_model"] = args.model
        rel_depth = midas_relative_depth(img_pil, model_name=args.model, device=device)

    # If we have only relative depth, fit meters if calibration provided
    calib_pts = parse_calib(args.calib)
    meta["linear_fit"]["n_points"] = len(calib_pts)
    if depth_m is None and rel_depth is not None:
        if len(calib_pts) >= 2:
            a, b = fit_linear_to_meters(rel_depth, calib_pts)
            depth_m = np.clip(a * rel_depth + b, 0, None).astype(np.float32)
            meta["linear_fit"]["a"] = a; meta["linear_fit"]["b"] = b
            calibrated = True
        else:
            # Relative only
            depth_m = rel_depth.copy().astype(np.float32)
            calibrated = False
            meta["notes"].append("No/insufficient calibration: distances are relative (~meters in labels).")

    # Resize depth to image size
    if depth_m.shape[:2] != (H, W):
        depth_m = cv2.resize(depth_m, (W, H), interpolation=cv2.INTER_CUBIC)

    # Save metric depth if calibrated
    stem = in_path
    if calibrated:
        save_metric_depth_outputs(stem, depth_m, meta)

    # Detection
    print("[INFO] Running object detection...")
    boxes, labels, scores = detect_objects(img_pil, score_thresh=args.score, max_det=args.max)

    # Overlay
    colored = colorize_depth(depth_m, near=args.near, far=args.far)
    overlay, rows = draw_boxes(colored, boxes, labels, scores, depth_m, units="m", calibrated=calibrated)

    out_path = args.out or str(stem.with_name(stem.stem + "_overlay.png"))
    cv2.imwrite(out_path, overlay)

    csv_path = args.csv or str(stem.with_name(stem.stem + "_detections.csv"))
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    print("[DONE] Wrote:")
    print(" ", out_path)
    print(" ", csv_path)
    if calibrated:
        print(" Saved calibrated metric depth files next to input (mm PNG + preview + JSON).")
    else:
        print(" NOTE: Distances are RELATIVE. Provide --calib (2+ points) or --trust-embedded-meters for true meters.")


if __name__ == "__main__":
    main()
