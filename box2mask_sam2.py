#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
box2mask_sam2.py  (GPU + SAM2 large version)

Use SAM2 to convert YOLO bounding boxes into a segmentation mask.

- Input:
    --image    : path to image
    --yolo-txt : YOLO detection label file (all boxes will be used)
    OR
    --bbox     : single bbox "x1,y1,x2,y2" in pixels

- Output:
    --out      : binary mask .png (0 background, 255 foreground),
                 union of masks for all boxes.

Default:
    model-id = "facebook/sam2-hiera-large"
    device   = "cuda" if available, else "cpu"

Example (YOLO txt, recommended):

    python box2mask_sam2.py ^
        --image "F:\\Tao\\cvat\\1.png" ^
        --yolo-txt "F:\\Tao\\cvat\\1.txt" ^
        --out "F:\\Tao\\cvat\\1_mask_large.png"

Example (single bbox):

    python box2mask_sam2.py ^
        --image "F:\\Tao\\cvat\\1.png" ^
        --bbox "100,120,360,300" ^
        --out "F:\\Tao\\cvat\\1_mask_large.png"
"""

import argparse
import os
from typing import List, Tuple

import cv2
import numpy as np
import torch

from sam2.sam2_image_predictor import SAM2ImagePredictor


# ---------------- SAM2 wrapper ----------------


class SAM2Wrapper:
    """
    Thin wrapper around SAM2ImagePredictor.

    - Load SAM2 from HuggingFace.
    - Given image + list of XYXY boxes, return binary masks.
    """

    def __init__(
        self,
        model_id: str = "facebook/sam2-hiera-large",
        device: str | None = None,
    ) -> None:
        # Choose device: prefer GPU
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        print(f"[INFO] torch.cuda.is_available() = {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[INFO] Using device: {self.device} ({torch.cuda.get_device_name(0)})")
        else:
            print("[WARN] CUDA not available, falling back to CPU.")

        print(f"[INFO] Loading SAM2 model: {model_id} on {self.device} ...")
        # from_pretrained 会把权重下到本地缓存；device 决定推理在哪个设备上
        self.predictor: SAM2ImagePredictor = SAM2ImagePredictor.from_pretrained(
            model_id,
            device=self.device,
        )
        print("[INFO] SAM2 model loaded.")

    def _prepare_image(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Convert OpenCV BGR image to RGB for SAM2.
        """
        if image_bgr is None:
            raise ValueError("Input image is None (cv2.imread failed?).")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return image_rgb

    def boxes_to_masks(
        self,
        image_bgr: np.ndarray,
        boxes_xyxy: List[Tuple[float, float, float, float]],
    ) -> Tuple[List[np.ndarray], List[float]]:
        """
        For each box, run SAM2 and return masks + scores.

        Args:
            image_bgr : HxWx3 BGR image
            boxes_xyxy: list of (x1, y1, x2, y2)

        Returns:
            masks : list of HxW float arrays in [0, 1]
            scores: list of IoU/quality scores
        """
        image_rgb = self._prepare_image(image_bgr)
        self.predictor.set_image(image_rgb)

        masks_out: List[np.ndarray] = []
        scores_out: List[float] = []

        for box in boxes_xyxy:
            box_arr = np.array(box, dtype=np.float32)[None, :]  # (1,4)

            with torch.inference_mode():
                masks, ious, _ = self.predictor.predict(
                    box=box_arr,
                    multimask_output=False,  # one mask per box
                )

            mask = masks[0]         # (H, W)
            score = float(ious.max())

            masks_out.append(mask)
            scores_out.append(score)

        return masks_out, scores_out


# ---------------- bbox utilities ----------------


def parse_bbox(arg: str) -> Tuple[float, float, float, float]:
    """
    Parse bbox string "x1,y1,x2,y2" to float tuple.
    """
    parts = arg.split(",")
    if len(parts) != 4:
        raise ValueError(
            f"Invalid bbox '{arg}'. Expected 'x1,y1,x2,y2' (4 numbers)."
        )
    x1, y1, x2, y2 = [float(p) for p in parts]
    return x1, y1, x2, y2


def parse_yolo_file(
    txt_path: str, img_w: int, img_h: int
) -> List[Tuple[int, float, float, float, float]]:
    """
    Read ALL lines in a YOLO txt file and convert to pixel XYXY.

    YOLO line: class x_center y_center width height  (normalized [0,1])
    Returns: list of (class_id, x1, y1, x2, y2)
    """
    if not os.path.isfile(txt_path):
        raise FileNotFoundError(f"YOLO label file not found: {txt_path}")

    boxes: List[Tuple[int, float, float, float, float]] = []

    with open(txt_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                print(f"[WARN] Skip invalid YOLO line in {txt_path}: {line}")
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


# ---------------- mask saving ----------------


def save_mask(mask: np.ndarray, out_path: str) -> None:
    """
    Save HxW mask (0/1 or float) as 0/255 PNG.
    """
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    m = mask.astype(np.float32)
    m = (m > 0.5).astype(np.uint8) * 255
    cv2.imwrite(out_path, m)
    print(f"[INFO] Saved mask to: {out_path}")


# ---------------- main CLI ----------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SAM2 (large) on one image and YOLO boxes, save union mask."
    )
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument(
        "--bbox",
        help=(
            "Single bbox in 'x1,y1,x2,y2' pixels. "
            "If --yolo-txt is provided, this is ignored."
        ),
    )
    parser.add_argument(
        "--yolo-txt",
        help=(
            "Optional: YOLO txt label file (detection format). "
            "If provided, ALL boxes in this file will be used."
        ),
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Path to output mask .png.",
    )
    parser.add_argument(
        "--model-id",
        default="facebook/sam2-hiera-large",
        help="HuggingFace model id for SAM2 (default: large).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device, e.g. 'cuda' or 'cpu'. Default: auto-detect.",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    # Read image
    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image with cv2: {args.image}")

    h, w = img.shape[:2]

    # Decide where boxes come from
    boxes_xyxy: List[Tuple[float, float, float, float]]

    if args.yolo_txt:
        boxes_info = parse_yolo_file(args.yolo_txt, w, h)
        if not boxes_info:
            raise ValueError(f"No valid boxes found in YOLO file: {args.yolo_txt}")
        boxes_xyxy = [(x1, y1, x2, y2) for (_, x1, y1, x2, y2) in boxes_info]
        print(f"[INFO] Using {len(boxes_xyxy)} boxes from YOLO txt: {args.yolo_txt}")
    else:
        if not args.bbox:
            raise ValueError("Either --yolo-txt or --bbox must be provided.")
        boxes_xyxy = [parse_bbox(args.bbox)]
        print("[INFO] Using single bbox from --bbox argument.")

    print(f"[INFO] First bbox example (x1,y1,x2,y2) = {boxes_xyxy[0]}")

    # Run SAM2
    sam2 = SAM2Wrapper(model_id=args.model_id, device=args.device)
    masks, scores = sam2.boxes_to_masks(img, boxes_xyxy)

    # Log first few scores
    for i, sc in enumerate(scores[:5]):
        print(f"[INFO] Box {i} SAM2 quality score = {sc:.4f}")
    if len(scores) > 5:
        print(f"[INFO] ... total boxes = {len(scores)}")

    # Union all masks into a single mask
    union = np.zeros((h, w), dtype=np.float32)
    for m in masks:
        union = np.maximum(union, m.astype(np.float32))

    save_mask(union, args.out)


if __name__ == "__main__":
    main()
