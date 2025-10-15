#!/usr/bin/env python3
"""
open_vocab_detect.py

Open‑vocabulary object detection & classification with Google's OWL‑ViT (Transformers).
- Supports HEIC/JPG/PNG input (HEIC via pillow-heif).
- Uses a large, customizable label list (default covers many categories beyond COCO).
- Outputs an overlay image with bounding boxes and a CSV of detections.

Usage:
  python open_vocab_detect.py --image input.heic --out overlay.png --csv dets.csv \
    --labels labels.txt --score 0.25 --nms 0.5 --model google/owlvit-base-patch16

Requirements:
  pip install torch transformers pillow pillow-heif opencv-python pandas numpy

Notes:
  - For best results, provide a label list (labels.txt) with 200–1000+ classes you care about.
  - OWL‑ViT is zero‑shot; quality depends on label phrasing. Include synonyms (e.g., "cell phone, mobile phone, smartphone").
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
except Exception as e:
    heif_ok = False
    print("[WARN] pillow-heif unavailable; HEIC may not open.", file=sys.stderr)

import torch
from transformers import AutoProcessor, OwlViTForObjectDetection


DEFAULT_LABELS = [
    # People / clothing / accessories
    "person","man","woman","child","baby","face","hand","head","foot","leg","arm",
    "hat","cap","helmet","glasses","sunglasses","mask","backpack","handbag","purse",
    "umbrella","jacket","coat","hoodie","sweater","shirt","t-shirt","pants","jeans",
    "shorts","skirt","dress","shoe","sneaker","boot","sandals","belt","watch",
    # Vehicles / transport
    "bicycle","motorcycle","scooter","skateboard","car","sports car","sedan","suv","truck",
    "bus","train","tram","airplane","boat","ship","traffic light","stop sign","parking meter",
    # Furniture / rooms
    "chair","stool","couch","sofa","bench","armchair","desk","table","coffee table",
    "dining table","bed","nightstand","cabinet","bookshelf","shelf","drawer","door","window","mirror","lamp",
    "tv","monitor","laptop","keyboard","mouse","remote","cell phone","mobile phone","smartphone","tablet",
    # Kitchen / appliances
    "refrigerator","fridge","freezer","microwave","oven","stove","cooktop","toaster","kettle","coffee maker",
    "blender","sink","dishwasher","faucet","bottle","cup","mug","glass","wine glass","bowl","plate","fork","knife","spoon",
    # Food
    "banana","apple","orange","broccoli","carrot","pizza","hot dog","sandwich","cake","donut","ice cream",
    # Animals
    "dog","cat","bird","horse","sheep","cow","elephant","bear","zebra","giraffe","duck","goose","chicken",
    "fish","shark","whale","dolphin","rabbit","hamster","mouse","rat","squirrel","deer","pig",
    # Outdoors / sports
    "bottle of water","water bottle","ball","soccer ball","basketball","tennis racket","baseball bat","skis","snowboard","kite",
    "surfboard","golf club","frisbee","helmet","backpack","tent","bicycle helmet","traffic cone",
    # Tools / hardware
    "hammer","screwdriver","wrench","pliers","drill","saw","tape measure","ladder","shovel","rake",
    # Office / school
    "book","notebook","pen","pencil","marker","whiteboard","blackboard","backpack","calculator","printer","projector",
    # Bathroom / cleaning
    "toilet","toilet paper","toiletries","sink","soap","towel","bathtub","shower","toothbrush","toothpaste",
    # Signs / devices
    "keyboard","microphone","headphones","earbuds","speaker","router","modem","camera","webcam","tripod",
    # Misc
    "vase","plant","potted plant","flower","clock","scissors","teddy bear","hair drier","toothbrush"
]


def load_labels(path_txt: str | None):
    if not path_txt:
        return DEFAULT_LABELS
    p = Path(path_txt)
    if not p.exists():
        print(f"[ERROR] Labels file not found: {p}", file=sys.stderr)
        sys.exit(1)
    labels = []
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            labels.append(s)
    if not labels:
        print("[WARN] Labels file is empty; using defaults.", file=sys.stderr)
        return DEFAULT_LABELS
    return labels


def load_image(path: str) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def nms_per_class(boxes, scores, iou_thresh=0.5):
    """
    Simple class-agnostic NMS for OWL‑ViT outputs (already class-specific by query label).
    """
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
def detect(image: Image.Image, labels, model_name="google/owlvit-base-patch16", score_thresh=0.25, iou_thresh=0.5, device=None):
    """
    Run OWL‑ViT zero‑shot detection against the given label list.
    Returns: list of dicts with box [x1,y1,x2,y2], score, label
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoProcessor.from_pretrained(model_name)
    model = OwlViTForObjectDetection.from_pretrained(model_name).to(device).eval()

    # OWL‑ViT takes a list-of-lists of strings as queries (batch dimension and candidate set dimension).
    texts = [labels]  # single image batch
    inputs = processor(text=texts, images=image, return_tensors="pt").to(device)

    outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]]).to(device)  # (H, W)

    results = processor.post_process_object_detection(
        outputs=outputs, target_sizes=target_sizes, threshold=score_thresh
    )[0]

    boxes = results["boxes"].detach().cpu().numpy()  # (N,4) in xyxy
    scores = results["scores"].detach().cpu().numpy()
    label_ids = results["labels"].detach().cpu().numpy().astype(int)

    # Apply NMS
    keep = nms_per_class(boxes, scores, iou_thresh=iou_thresh)
    boxes = boxes[keep]; scores = scores[keep]; label_ids = label_ids[keep]

    dets = []
    for b, s, li in zip(boxes, scores, label_ids):
        # Map index to label string
        name = labels[li] if 0 <= li < len(labels) else f"id{li}"
        dets.append({"box": b.tolist(), "score": float(s), "label": name})
    return dets


def draw_detections(image: Image.Image, detections, alpha=0.35):
    """
    Draw boxes and labels on a copy of the image. Returns BGR image (numpy).
    """
    rgb = np.array(image)[:, :, ::-1]  # to BGR
    canvas = rgb.copy()
    rng = np.random.default_rng(12345)

    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det["box"]]
        color = tuple(int(c) for c in rng.integers(0, 255, size=3))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        text = f'{det["label"]} {det["score"]:.2f}'
        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ty = max(y1 - th - bl - 4, 0)
        cv2.rectangle(canvas, (x1, ty), (x1 + tw + 6, ty + th + bl + 6), (0, 0, 0), -1)
        cv2.putText(canvas, text, (x1 + 3, ty + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Blend lightly with original for visibility
    out = (alpha * canvas + (1 - alpha) * rgb).astype(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser(description="Open‑vocabulary object detection with OWL‑ViT (draw boxes + labels)")
    ap.add_argument("--image", required=True, help="Path to input image (HEIC/JPG/PNG)")
    ap.add_argument("--labels", default=None, help="Path to text file with labels (one per line). Defaults to a broad built‑in list.")
    ap.add_argument("--out", default="overlay.png", help="Output image with boxes drawn")
    ap.add_argument("--csv", default="detections.csv", help="CSV to save detections")
    ap.add_argument("--model", default="google/owlvit-base-patch16", help="Hugging Face model id (e.g., google/owlvit-base-patch16, google/owlvit-large-patch14)")
    ap.add_argument("--score", type=float, default=0.25, help="Score threshold (0-1)")
    ap.add_argument("--nms", type=float, default=0.5, help="IOU threshold for NMS (0-1)")
    args = ap.parse_args()

    labels = load_labels(args.labels)
    img = load_image(args.image)

    print(f"[INFO] Running OWL‑ViT ({args.model}) on {args.image} with {len(labels)} labels ...")
    dets = detect(img, labels, model_name=args.model, score_thresh=args.score, iou_thresh=args.nms)

    # Draw
    overlay_bgr = draw_detections(img, dets)
    cv2.imwrite(args.out, overlay_bgr)
    print(f"[DONE] Wrote overlay: {args.out}")

    # CSV
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
