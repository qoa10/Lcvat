#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
vis_yolo_box_mask.py

Visualize:
  - original image
  - YOLO detection bboxes (from .txt)
  - optional SAM2 mask (grayscale png)

Usage example (PowerShell, after activating .venv310):

    python vis_yolo_box_mask.py `
        --image "F:\Tao\cvat\1.png" `
        --yolo-txt "F:\Tao\cvat\1.txt" `
        --mask "F:\Tao\cvat\0001_mask.png" `
        --out "F:\Tao\cvat\1_vis.png"

If you don't have a mask yet, just omit `--mask`:

    python vis_yolo_box_mask.py `
        --image "F:\Tao\cvat\1.png" `
        --yolo-txt "F:\Tao\cvat\1.txt" `
        --out "F:\Tao\cvat\1_vis_box_only.png"
"""

import argparse
import os
from typing import List, Tuple

import cv2
import numpy as np


def parse_yolo_file(txt_path: str, img_w: int, img_h: int) -> List[Tuple[int, float, float, float, float]]:
    """
    Read ALL lines in a YOLO txt file and convert to pixel XYXY.

    YOLO line: class x_center y_center width height   (all normalized to [0,1])
    Return: list of (class_id, x1, y1, x2, y2)
    """
    if not os.path.isfile(txt_path):
        raise FileNotFoundError(f"YOLO label file not found: {txt_path}")

    boxes = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                print(f"[WARN] Skip invalid line in {txt_path}: {line}")
                continue

            cls_id = int(float(parts[0]))
            x_c = float(parts[1])
            y_c = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])

            x_center = x_c * img_w
            y_center = y_c * img_h
            w_px = w * img_w
            h_px = h * img_h

            x1 = x_center - w_px / 2.0
            y1 = y_center - h_px / 2.0
            x2 = x_center + w_px / 2.0
            y2 = y_center + h_px / 2.0

            boxes.append((cls_id, x1, y1, x2, y2))

    return boxes


def overlay_mask(image_bgr: np.ndarray, mask_path: str, alpha: float = 0.4) -> np.ndarray:
    """
    Overlay a grayscale mask on the image (as colored transparent layer).
    """
    if not os.path.isfile(mask_path):
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Failed to read mask image: {mask_path}")

    mask_resized = cv2.resize(mask, (image_bgr.shape[1], image_bgr.shape[0]))

    # Create a color layer: here we use green channel for mask
    color_layer = np.zeros_like(image_bgr)
    color_layer[:, :, 1] = mask_resized  # G channel

    blended = cv2.addWeighted(image_bgr, 1.0, color_layer, alpha, 0)
    return blended


def main():
    parser = argparse.ArgumentParser(
        description="Visualize YOLO bboxes (and optional mask) on image."
    )
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument(
        "--yolo-txt",
        required=True,
        help="Path to YOLO txt (detection format).",
    )
    parser.add_argument(
        "--mask",
        help="Optional: path to SAM2 mask png (same size or will be resized).",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Path to save visualization image.",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image with cv2: {args.image}")

    h, w = img.shape[:2]

    # Draw mask first (background)
    vis = img.copy()
    if args.mask:
        print(f"[INFO] Overlay mask from {args.mask}")
        vis = overlay_mask(vis, args.mask, alpha=0.4)

    # Draw YOLO boxes
    boxes = parse_yolo_file(args.yolo_txt, w, h)
    print(f"[INFO] Found {len(boxes)} boxes in {args.yolo_txt}")

    for cls_id, x1, y1, x2, y2 in boxes:
        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))
        # green rectangle with thickness 2
        cv2.rectangle(vis, p1, p2, (0, 255, 0), 2)
        cv2.putText(
            vis,
            f"id={cls_id}",
            (p1[0], max(0, p1[1] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            lineType=cv2.LINE_AA,
        )

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    cv2.imwrite(args.out, vis)
    print(f"[INFO] Saved visualization to: {args.out}")


if __name__ == "__main__":
    main()
