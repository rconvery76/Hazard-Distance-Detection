#!/usr/bin/env python3
"""
heic_to_depth.py

Convert an HEIC (or any Pillow-supported) image into a depth map using MiDaS.
- Loads HEIC via pillow-heif (registers a HEIF opener for PIL).
- Runs MiDaS (small model by default) to estimate depth on CPU or GPU.
- Saves both a 16-bit depth PNG (for downstream work) and an 8-bit preview PNG.

Usage:
    python heic_to_depth.py input.heic --out depth.png --preview depth_preview.png --model MiDaS_small

Dependencies (install once):
    pip install pillow pillow-heif torch torchvision
    # First run will download MiDaS weights via torch.hub (internet required).

Optional:
    # For faster/better quality:
    --model DPT_Hybrid or DPT_Large (requires more VRAM/compute).

Notes:
    - If your iPhone portrait HEIC contains a true depth/disparity auxiliary image, this script will
      *still* run monocular depth. Extracting embedded depth from HEIC is device/format dependent;
      monocular depth works universally.
"""


import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Register HEIC/HEIF opener for Pillow
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception as e:
    print("[WARN] pillow-heif not available or failed to register. HEIC files may not open.", file=sys.stderr)

import torch


def load_image_as_rgb(path: str) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Apple Silicon
        return torch.device("mps")
    return torch.device("cpu")


def load_midas(model_name: str = "MiDaS_small", device: torch.device = torch.device("cpu")):
    """
    Load MiDaS model + transforms from torch.hub.
    model_name in {"MiDaS_small", "DPT_Hybrid", "DPT_Large"}
    """
    # Load model
    midas = torch.hub.load("intel-isl/MiDaS", model_name)
    midas.to(device)
    midas.eval()

    # Load transforms
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_name in ("DPT_Large", "DPT_Hybrid"):
        transform = transforms.dpt_transform
    else:
        transform = transforms.small_transform

    return midas, transform


@torch.inference_mode()
def estimate_depth(img_pil: Image.Image, midas, transform, device: torch.device) -> np.ndarray:
    """
    Returns a single-channel depth array (float32), larger = farther (MiDaS is relative).
    """
    # Preprocess
    img_np = np.array(img_pil, dtype=np.uint8)
    input_batch = transform(img_np).to(device)

    # Predict
    prediction = midas(input_batch)
    # Resize to original resolution
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img_pil.size[::-1],  # (H, W)
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    depth = prediction.detach().cpu().numpy().astype(np.float32)
    return depth


def save_depth_16bit(depth: np.ndarray, out_path: str):
    """
    Save depth as a 16-bit PNG after min-max normalization.
    This preserves more gradation than 8-bit previews.
    """
    d = depth.copy()
    d -= d.min()
    if d.max() > 0:
        d /= d.max()
    d16 = (d * 65535.0).round().astype(np.uint16)
    Image.fromarray(d16, mode="I;16").save(out_path)


def save_preview_8bit(depth: np.ndarray, out_preview: str, invert: bool = True):
    """
    Save an 8-bit preview where brighter = closer (invert=True).
    """
    d = depth.copy()
    d -= d.min()
    if d.max() > 0:
        d /= d.max()
    if invert:
        d = 1.0 - d
    d8 = (d * 255.0).round().astype(np.uint8)
    Image.fromarray(d8, mode="L").save(out_preview)


def parse_args():
    p = argparse.ArgumentParser(description="Convert HEIC image to depth map using MiDaS")
    p.add_argument("input", help="Path to input image (HEIC/HEIF/JPG/PNG/...)")
    p.add_argument("--out", default=None, help="Output 16-bit depth PNG path (default: <stem>_depth16.png)")
    p.add_argument("--preview", default=None, help="Output 8-bit preview PNG path (default: <stem>_depth_preview.png)")
    p.add_argument("--model", default="MiDaS_small", choices=["MiDaS_small", "DPT_Hybrid", "DPT_Large"],
                   help="MiDaS model to use")
    return p.parse_args()


def main():
    args = parse_args()
    in_path = Path(args.input)
    if not in_path.exists():
        print(f"[ERROR] Input not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    out_depth = args.out or str(in_path.with_name(in_path.stem + "_depth16.png"))
    out_preview = args.preview or str(in_path.with_name(in_path.stem + "_depth_preview.png"))

    print("[INFO] Loading image:", in_path)
    img = load_image_as_rgb(str(in_path))

    device = get_device()
    print("[INFO] Using device:", device)

    print(f"[INFO] Loading MiDaS model: {args.model}")
    midas, transform = load_midas(args.model, device)

    print("[INFO] Estimating depth...")
    depth = estimate_depth(img, midas, transform, device)

    print("[INFO] Saving:", out_depth)
    save_depth_16bit(depth, out_depth)

    print("[INFO] Saving preview:", out_preview)
    save_preview_8bit(depth, out_preview, invert=True)

    print("[DONE] Depth maps written.")


if __name__ == "__main__":
    main()
