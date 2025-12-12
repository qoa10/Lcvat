#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
yolo_sam2_to_yoloseg.py

Batch pipeline:
    YOLO detection labels (.txt with bbox) + images
 -> SAM2 masks (one instance per bbox)
 -> YOLOv8-seg labels (.txt with polygons)

Directory layout example:

    images/
        0001.jpg
        0002.jpg
        ...
    labels_det/       # existing YOLO detection labels
        0001.txt
        0002.txt
        ...
    labels_seg/       # OUTPUT: YOLO-seg labels we generate
        0001.txt
        0002.txt
        ...
    debug_vis/        # optional: visualization with polygons

Usage (PowerShell, after activating .venv310):

    python yolo_sam2_to_yoloseg.py ^
        --images-dir "F:\\crack_dataset\\images" ^
        --det-labels-dir "F:\\crack_dataset\\labels_det" ^
        --seg-labels-dir "F:\\crack_dataset\\labels_seg" ^
        --debug-vis-dir "F:\\crack_dataset\\debug_vis" ^
        --model-id "facebook/sam2-hiera-large"

"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch

from sam2.sam2_image_predictor import SAM2ImagePredictor


# --------- SAM2 wrapper ---------


class SAM2Wrapper:
    def __init__(
        self,
        model_id: str = "facebook/sam2-hiera-large",
        device: str | None = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        print(f"[INFO] torch.cuda.is_available() = {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[INFO] Using device: {self.device} ({torch.cuda.get_device_name(0)})")
        else:
            print("[WARN] CUDA not available, falling back to CPU.")

        print(f"[INFO] Loading SAM2 model: {model_id} on {self.device} ...")
        self.predictor: SAM2ImagePredictor = SAM2ImagePredictor.from_pretrained(
            model_id, device=self.device
        )
        print("[INFO] SAM2 model loaded.")

    def set_image(self, image_bgr: np.ndarray) -> None:
        if image_bgr is None:
            raise ValueError("Input image is None (cv2.imread failed?).")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image_rgb)

    def predict_boxes(
        self,
        boxes_xyxy: List[Tuple[float, float, float, float]],
    ) -> Tuple[List[np.ndarray], List[float]]:
        """
        For each box, run SAM2 and return mask + quality score.

        Returns:
            masks  : list of HxW float arrays in [0,1]
            scores : list of float scores
        """
        masks_out: List[np.ndarray] = []
        scores_out: List[float] = []

        for box in boxes_xyxy:
            box_arr = np.array(box, dtype=np.float32)[None, :]

            with torch.inference_mode():
                masks, ious, _ = self.predictor.predict(
                    box=box_arr,
                    multimask_output=False,
                )

            mask = masks[0]
            score = float(ious.max())
            masks_out.append(mask)
            scores_out.append(score)

        return masks_out, scores_out


# --------- YOLO utils ---------


def read_yolo_det_labels(
    txt_path: Path, img_w: int, img_h: int
) -> List[Tuple[int, float, float, float, float]]:
    """
    Read YOLO detection labels (bbox format) and convert to pixel XYXY.

    Each line:
        class x_center y_center width height   (all normalized [0,1])

    Returns list of:
        (class_id, x1, y1, x2, y2)  in pixels
    """
    if not txt_path.is_file():
        return []

    boxes: List[Tuple[int, float, float, float, float]] = []
    with txt_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                print(f"[WARN] Invalid YOLO line in {txt_path}: {line}")
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


def mask_to_polygon(
    mask: np.ndarray, min_area: float = 10.0, max_points: int = 100
) -> List[Tuple[float, float]]:
    """
    Convert a binary mask (H, W) to a SINGLE polygon (list of (x,y) points).

    - Use the largest contour by area.
    - Optionally simplify / downsample to at most max_points vertices.
    """
    if mask.dtype != np.uint8:
        mask = (mask > 0.5).astype(np.uint8)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return []

    # Choose largest contour
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    largest = contours[0]
    area = cv2.contourArea(largest)
    if area < min_area:
        return []

    # largest: (N,1,2) -> (N,2)
    pts = largest.reshape(-1, 2)

    # Optional: downsample if too many points
    if len(pts) > max_points:
        idx = np.linspace(0, len(pts) - 1, max_points).astype(int)
        pts = pts[idx]

    return [(float(x), float(y)) for x, y in pts]


def write_yoloseg_labels(
    out_path: Path,
    polygons: List[List[Tuple[float, float]]],
    cls_ids: List[int],
    img_w: int,
    img_h: int,
) -> None:
    """
    Write YOLOv8-seg style labels.

    Each instance:
        class x1 y1 x2 y2 ... xN yN
    where x,y are normalized to [0,1].
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    for cls_id, poly in zip(cls_ids, polygons):
        if not poly:
            continue

        coords = []
        for x, y in poly:
            xn = x / img_w
            yn = y / img_h
            coords.extend([xn, yn])

        # Require at least 3 points
        if len(coords) < 6:
            continue

        line = str(cls_id) + " " + " ".join(f"{v:.6f}" for v in coords)
        lines.append(line)

    with out_path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

    print(f"[INFO] Wrote YOLO-seg labels: {out_path}  (instances={len(lines)})")


def draw_debug_vis(
    image_bgr: np.ndarray,
    polygons: List[List[Tuple[float, float]]],
    cls_ids: List[int],
    out_path: Path,
) -> None:
    """
    Draw polygons on image for visual checking.
    """
    vis = image_bgr.copy()

    for cls_id, poly in zip(cls_ids, polygons):
        if not poly:
            continue
        pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
        color = (0, 255, 0)
        cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=2)
        # draw class id near first point
        x0, y0 = int(pts[0, 0, 0]), int(pts[0, 0, 1])
        cv2.putText(
            vis,
            str(cls_id),
            (x0, max(0, y0 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            1,
            lineType=cv2.LINE_AA,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)
    print(f"[INFO] Saved debug vis: {out_path}")


# --------- main loop ---------


def main():
    parser = argparse.ArgumentParser(
        description="Batch: YOLO det -> SAM2 -> YOLOv8-seg polygons."
    )
    parser.add_argument("--images-dir", required=True, help="Dir of input images.")
    parser.add_argument(
        "--det-labels-dir",
        required=True,
        help="Dir of YOLO detection labels (.txt).",
    )
    parser.add_argument(
        "--seg-labels-dir",
        required=True,
        help="Output dir for YOLO-seg labels (.txt).",
    )
    parser.add_argument(
        "--debug-vis-dir",
        help="Optional: output dir for visualization png.",
    )
    parser.add_argument(
        "--model-id",
        default="facebook/sam2-hiera-large",
        help="SAM2 model id (e.g., facebook/sam2-hiera-tiny/small/large).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device, e.g. 'cuda' or 'cpu'. Default: auto.",
    )

    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    det_labels_dir = Path(args.det_labels_dir)
    seg_labels_dir = Path(args.seg_labels_dir)
    debug_vis_dir = Path(args.debug_vis_dir) if args.debug_vis_dir else None

    sam2 = SAM2Wrapper(model_id=args.model_id, device=args.device)

    image_paths = sorted(
        [p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )
    print(f"[INFO] Found {len(image_paths)} images in {images_dir}")

    for img_path in image_paths:
        stem = img_path.stem
        det_txt = det_labels_dir / f"{stem}.txt"
        seg_txt = seg_labels_dir / f"{stem}.txt"
        vis_png = debug_vis_dir / f"{stem}_vis.png" if debug_vis_dir else None

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] Failed to read image: {img_path}, skip.")
            continue
        h, w = img.shape[:2]

        boxes_info = read_yolo_det_labels(det_txt, w, h)
        if not boxes_info:
            print(f"[INFO] No det labels for {stem}, write empty seg txt.")
            seg_txt.parent.mkdir(parents=True, exist_ok=True)
            seg_txt.write_text("")
            continue

        cls_ids = [b[0] for b in boxes_info]
        boxes_xyxy = [(b[1], b[2], b[3], b[4]) for b in boxes_info]

        print(f"[INFO] {stem}: {len(boxes_xyxy)} boxes")

        sam2.set_image(img)
        masks, scores = sam2.predict_boxes(boxes_xyxy)

        for i, sc in enumerate(scores[:5]):
            print(f"    Box {i} SAM2 score = {sc:.4f}")
        if len(scores) > 5:
            print(f"    ... total boxes = {len(scores)}")

        polygons: List[List[Tuple[float, float]]] = []
        for mask in masks:
            poly = mask_to_polygon(mask)
            polygons.append(poly)

        write_yoloseg_labels(seg_txt, polygons, cls_ids, w, h)

        if vis_png is not None:
            draw_debug_vis(img, polygons, cls_ids, vis_png)


if __name__ == "__main__":
    main()
