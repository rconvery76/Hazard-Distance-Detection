#!/usr/bin/env python3
"""
detect_on_depth.py

Run object detection on an RGB image, then overlay bounding boxes on a colorized depth map
(derived from a 16-bit metric depth PNG where pixel values are millimeters). Labels will include
the class, score, and the median distance (in meters) inside each box.

Usage:
    python detect_on_depth.py --image input.jpg --depth_mm input_metric_depth_mm.png \
        --out overlay.png --csv detections.csv --score 0.5 --max 50

Requirements:
    pip install torch torchvision pillow numpy opencv-python pandas

Notes:
    - Depth image must be a 16-bit PNG where value = distance (mm). You can generate this using
      the previously provided iphone_depth_to_metric.py/GUI scripts.
    - Detection uses torchvision's Faster R-CNN (COCO weights). It's reasonably fast and avoids extra deps.
"""

import argparse
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import torch
import torchvision
from torchvision.transforms import functional as F
import pandas as pd

from PIL import Image
import pillow_heif
pillow_heif.register_heif_opener()



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


def load_image(path):
    img = Image.open(path).convert("RGB")
    return img


def load_depth_mm(depth_path):
    """
    Load a 16-bit PNG where value = millimeters. Returns float32 meters.
    """
    depth_mm = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth_mm is None:
        raise FileNotFoundError(f"Could not read depth image: {depth_path}")
    if depth_mm.dtype != np.uint16:
        # Try to interpret anyway
        depth_mm = depth_mm.astype(np.uint16)
    depth_m = depth_mm.astype(np.float32) / 1000.0
    return depth_m


@torch.inference_mode()
def detect(image_pil, score_thresh=0.5, max_det=50, device=None):
    """
    Run Faster R-CNN on a PIL RGB image. Returns bboxes (N,4), labels (N), scores (N).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    model.to(device).eval()

    img_t = F.to_tensor(image_pil).to(device)  # [0,1] tensor
    outputs = model([img_t])[0]

    boxes = outputs["boxes"].detach().cpu().numpy()
    labels = outputs["labels"].detach().cpu().numpy()
    scores = outputs["scores"].detach().cpu().numpy()

    # Filter by score
    keep = scores >= score_thresh
    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]

    # Keep top max_det
    if len(scores) > max_det:
        idx = np.argsort(-scores)[:max_det]
        boxes = boxes[idx]
        labels = labels[idx]
        scores = scores[idx]

    return boxes, labels, scores


def colorize_depth(depth_m, near_clip=None, far_clip=None):
    """
    Create a colorized depth image for visualization.
    By default, auto-scales between min and max finite depths.
    Returns a BGR uint8 image for OpenCV drawing.
    """
    d = depth_m.copy()
    d[~np.isfinite(d)] = np.nan

    if near_clip is None:
        near_clip = np.nanmin(d)
    if far_clip is None:
        far_clip = np.nanmax(d)
    if not np.isfinite(near_clip) or not np.isfinite(far_clip) or near_clip == far_clip:
        near_clip, far_clip = 0.0, 1.0

    # Normalize so nearer = larger value for intuitive colormap
    dn = (d - near_clip) / (far_clip - near_clip + 1e-12)
    dn = 1.0 - np.clip(dn, 0, 1)  # nearer -> higher

    dn_u8 = (dn * 255).astype(np.uint8)
    colored = cv2.applyColorMap(dn_u8, cv2.COLORMAP_PLASMA)  # BGR
    return colored


def median_depth_in_box(depth_m, box):
    x1, y1, x2, y2 = box
    x1 = max(int(round(x1)), 0)
    y1 = max(int(round(y1)), 0)
    x2 = max(int(round(x2)), 0)
    y2 = max(int(round(y2)), 0)
    if x2 <= x1 or y2 <= y1:
        return np.nan
    crop = depth_m[y1:y2, x1:x2]
    # ignore zeros (often invalid) and NaNs
    crop_valid = crop[np.isfinite(crop) & (crop > 0)]
    if crop_valid.size == 0:
        return np.nan
    return float(np.median(crop_valid))


def draw_boxes_on_depth(colored_depth_bgr, boxes, labels, scores, depth_m, thickness=2):
    out = colored_depth_bgr.copy()
    H, W = out.shape[:2]
    rows = []
    for i, (box, lab, score) in enumerate(zip(boxes, labels, scores), start=1):
        x1, y1, x2, y2 = box
        x1 = int(np.clip(round(x1), 0, W-1))
        y1 = int(np.clip(round(y1), 0, H-1))
        x2 = int(np.clip(round(x2), 0, W-1))
        y2 = int(np.clip(round(y2), 0, H-1))

        med_m = median_depth_in_box(depth_m, (x1, y1, x2, y2))

        # Choose a consistent color based on class id
        color = tuple(int(c) for c in np.random.default_rng(lab).integers(0, 255, size=3))

        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        cls_name = COCO_CLASSES[lab] if 0 <= lab < len(COCO_CLASSES) else f"id{lab}"
        label = f"{cls_name} {score:.2f}"
        if np.isfinite(med_m):
            label += f" | {med_m:.2f} m"

        # Text background
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ty1 = max(y1 - th - baseline - 4, 0)
        cv2.rectangle(out, (x1, ty1), (x1 + tw + 6, ty1 + th + baseline + 6), (0, 0, 0), -1)
        cv2.putText(out, label, (x1 + 3, ty1 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        rows.append({
            "id": i,
            "class_id": int(lab),
            "class_name": cls_name,
            "score": float(score),
            "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
            "median_distance_m": med_m if np.isfinite(med_m) else None,
        })
    return out, rows


def main():
    ap = argparse.ArgumentParser(description="Object detection overlaid on a colorized metric depth map")
    ap.add_argument("--image", required=True, help="Path to RGB image (used for detection only)")
    ap.add_argument("--depth_mm", required=True, help="Path to 16-bit PNG depth map (millimeters)")
    ap.add_argument("--out", default="overlay.png", help="Output PNG with boxes drawn on colorized depth")
    ap.add_argument("--csv", default="detections.csv", help="CSV file to save detection info")
    ap.add_argument("--score", type=float, default=0.5, help="Score threshold (0-1)")
    ap.add_argument("--max", type=int, default=50, help="Max detections to keep")
    ap.add_argument("--near", type=float, default=None, help="Optional near clip (meters) for color scaling")
    ap.add_argument("--far", type=float, default=None, help="Optional far clip (meters) for color scaling")
    args = ap.parse_args()

    image_pil = load_image(args.image)
    depth_m = load_depth_mm(args.depth_mm)

    # Resize depth to match image if needed
    if depth_m.shape[:2] != (image_pil.height, image_pil.width):
        depth_m = cv2.resize(depth_m, (image_pil.width, image_pil.height), interpolation=cv2.INTER_CUBIC)

    print("[INFO] Running detection...")
    boxes, labels, scores = detect(image_pil, score_thresh=args.score, max_det=args.max)

    print("[INFO] Creating overlay...")
    colored = colorize_depth(depth_m, near_clip=args.near, far_clip=args.far)
    overlay, rows = draw_boxes_on_depth(colored, boxes, labels, scores, depth_m)

    # Save overlay
    cv2.imwrite(args.out, overlay)
    print(f"[DONE] Wrote overlay: {args.out}")

    # Save CSV
    df = pd.DataFrame(rows, columns=[
        "id","class_id","class_name","score","x1","y1","x2","y2","median_distance_m"
    ])
    df.to_csv(args.csv, index=False)
    print(f"[DONE] Wrote CSV: {args.csv}")


if __name__ == "__main__":
    main()
