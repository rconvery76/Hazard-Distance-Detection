#!/usr/bin/env python3
"""
open_vocab_detect_v2.py

Open‑vocabulary detection with OWL‑ViT + prompt augmentation.

Key features
- HEIC/JPG/PNG input (HEIC via pillow-heif).
- Reads labels from --labels (one per line); auto‑generates prompt variants like:
    "{label}", "a {label}", "the {label}", "a photo of a {label}", "{label}s"
  This often improves recall for zero‑shot models.
- Maps results back to the *base* label so your CSV stays clean.
- Adjustable score/NMS thresholds.

Usage
  python open_vocab_detect_v2.py --image input.heic --labels labels.txt \
    --out overlay.png --csv dets.csv --score 0.18 --nms 0.5 \
    --model google/owlvit-large-patch14

Install
  pip install torch transformers pillow pillow-heif opencv-python pandas numpy
"""

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import cv2
from PIL import Image

# HEIC support
heif_ok = True
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    heif_ok = False

import torch
from transformers import AutoProcessor, OwlViTForObjectDetection


def load_image(path: str) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def read_labels(path_txt: str | None):
    if not path_txt:
        raise SystemExit("[ERROR] --labels is required (one label per line).")
    p = Path(path_txt)
    if not p.exists():
        raise SystemExit(f"[ERROR] Labels file not found: {p}")
    labels = []
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            labels.append(s)
    if not labels:
        raise SystemExit("[ERROR] Labels file is empty.")
    return labels


def make_variants_for_label(label: str):
    # Simple prompt expansion (no external resources needed).
    base = label.strip()
    plural = base if base.endswith("s") else base + "s"
    variants = list(dict.fromkeys([  # de-duplicate while preserving order
        base,
        f"a {base}",
        f"the {base}",
        f"a photo of a {base}",
        plural,
        f"a photo of {plural}",
    ]))
    return variants


def build_query_lists(base_labels):
    """
    Returns (flat_variants, variant_to_base_idx)
    flat_variants: list[str] with all prompt variants
    variant_to_base_idx: list[int] mapping each variant index -> base label index
    """
    flat = []
    mapping = []
    for i, lbl in enumerate(base_labels):
        vs = make_variants_for_label(lbl)
        flat.extend(vs)
        mapping.extend([i] * len(vs))
    return flat, mapping


def nms(boxes, scores, iou_thresh=0.5):
    if len(boxes) == 0:
        return []
    boxes = boxes.astype(np.float32)
    scores = scores.astype(np.float32)
    x1 = boxes[:, 0]; y1 = boxes[:, 1]; x2 = boxes[:, 2]; y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-12)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return keep


@torch.inference_mode()
def run_detection(image: Image.Image, base_labels, model_name="google/owlvit-base-patch16", score_thresh=0.25, iou_thresh=0.5, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build queries with variants
    flat_variants, variant_to_base = build_query_lists(base_labels)

    processor = AutoProcessor.from_pretrained(model_name)
    model = OwlViTForObjectDetection.from_pretrained(model_name).to(device).eval()

    inputs = processor(text=[flat_variants], images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]]).to(device)  # (H, W)

    results = processor.post_process_object_detection(
        outputs=outputs, target_sizes=target_sizes, threshold=score_thresh
    )[0]

    boxes = results["boxes"].detach().cpu().numpy()  # (N,4)
    scores = results["scores"].detach().cpu().numpy()
    variant_ids = results["labels"].detach().cpu().numpy().astype(int)

    # Class-agnostic NMS to reduce duplicates
    keep = nms(boxes, scores, iou_thresh=iou_thresh)
    boxes = boxes[keep]; scores = scores[keep]; variant_ids = variant_ids[keep]

    # Map variant -> base label
    base_ids = [variant_to_base[vi] if 0 <= vi < len(variant_to_base) else -1 for vi in variant_ids]
    dets = []
    for b, s, bi in zip(boxes, scores, base_ids):
        name = base_labels[bi] if 0 <= bi < len(base_labels) else "unknown"
        dets.append({"box": b.tolist(), "score": float(s), "label": name, "base_id": int(bi)})
    return dets


def draw( image: Image.Image, dets, alpha=0.35):
    bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    canvas = bgr.copy()
    rng = np.random.default_rng(1337)
    for d in dets:
        x1, y1, x2, y2 = [int(round(v)) for v in d["box"]]
        color = tuple(int(c) for c in rng.integers(0, 255, size=3))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 20)
        text = f'{d["label"]} {d["score"]:.2f}'
        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 20)
        ty = max(y1 - th - bl - 4, 0)
        cv2.rectangle(canvas, (x1, ty), (x1 + tw + 6, ty + th + bl + 6), (0,0,0), -1)
        cv2.putText(canvas, text, (x1 + 3, ty + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    out = (alpha * canvas + (1 - alpha) * bgr).astype(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser(description="Open‑vocabulary detection with prompt variants (OWL‑ViT)")
    ap.add_argument("--image", required=True, help="Path to image (HEIC/JPG/PNG)")
    ap.add_argument("--labels", required=True, help="Path to text file with base labels (one per line)")
    ap.add_argument("--out", default="overlay.png", help="Output overlay image")
    ap.add_argument("--csv", default="detections.csv", help="Detections CSV")
    ap.add_argument("--model", default="google/owlvit-base-patch16", help="HF model id (e.g., google/owlvit-large-patch14)")
    ap.add_argument("--score", type=float, default=0.20, help="Score threshold (lower increases recall)")
    ap.add_argument("--nms", type=float, default=0.5, help="IOU for NMS")
    args = ap.parse_args()

    img = load_image(args.image)
    base_labels = read_labels(args.labels)

    print(f"[INFO] Using {len(base_labels)} base labels with prompt augmentation ...")
    dets = run_detection(img, base_labels, model_name=args.model, score_thresh=args.score, iou_thresh=args.nms)

    overlay = draw(img, dets)
    cv2.imwrite(args.out, overlay)
    print(f"[DONE] Wrote overlay: {args.out}")

    rows = []
    for i, d in enumerate(dets, start=1):
        x1, y1, x2, y2 = d["box"]
        rows.append({
            "id": i, "label": d["label"], "score": d["score"],
            "x1": int(round(x1)), "y1": int(round(y1)), "x2": int(round(x2)), "y2": int(round(y2)),
            "width": int(round(x2 - x1)), "height": int(round(y2 - y1))
        })
    pd.DataFrame(rows).to_csv(args.csv, index=False)
    print(f"[DONE] Wrote CSV: {args.csv}")


if __name__ == "__main__":
    main()
