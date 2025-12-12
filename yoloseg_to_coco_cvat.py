#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
yoloseg_to_coco_cvat.py

Convert YOLOv8-seg polygon labels (.txt) + images
into a single COCO-style JSON file that can be imported into CVAT.

Assumptions:
- Each image has an optional YOLO-seg txt file with the same stem name.
- YOLO-seg format (one instance per line):

    class x1 y1 x2 y2 ... xN yN   (x,y are normalized to [0,1])

Output:
- A COCO JSON with:
    images      : one entry per image
    annotations : one entry per instance (polygon)
    categories  : one entry per class id (class_0, class_1, ...)

Usage (PowerShell, after activating .venv310):

    python yoloseg_to_coco_cvat.py ^
        --images-dir "F:\\crack_dataset\\images" ^
        --seg-labels-dir "F:\\crack_dataset\\labels_seg" ^
        --output-json "F:\\crack_dataset\\coco_sam2_for_cvat.json"
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


def polygon_area(xs: List[float], ys: List[float]) -> float:
    """
    Compute polygon area using the shoelace formula.
    xs, ys are lists of pixel coordinates in order.
    """
    if len(xs) < 3:
        return 0.0
    x = np.array(xs)
    y = np.array(ys)
    return float(0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def main():
    parser = argparse.ArgumentParser(
        description="Convert YOLOv8-seg labels to COCO JSON for CVAT."
    )
    parser.add_argument("--images-dir", required=True, help="Directory of images.")
    parser.add_argument(
        "--seg-labels-dir",
        required=True,
        help="Directory of YOLOv8-seg labels (.txt).",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Path to output COCO JSON.",
    )

    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    seg_labels_dir = Path(args.seg_labels_dir)
    output_json = Path(args.output_json)

    image_paths = sorted(
        [p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )
    print(f"[INFO] Found {len(image_paths)} images in {images_dir}")

    coco_images: List[Dict] = []
    coco_annotations: List[Dict] = []
    class_ids_seen: set[int] = set()

    image_id = 1
    ann_id = 1

    for img_path in image_paths:
        stem = img_path.stem
        seg_txt = seg_labels_dir / f"{stem}.txt"

        # Read image to get size
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] Failed to read image {img_path}, skip.")
            continue
        h, w = img.shape[:2]

        coco_images.append(
            {
                "id": image_id,
                "file_name": img_path.name,
                "width": w,
                "height": h,
            }
        )

        if seg_txt.is_file():
            with seg_txt.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 7:
                        # class + at least 3 points (x,y)*3 -> 1+6 = 7
                        print(f"[WARN] Skip invalid seg line in {seg_txt}: {line}")
                        continue

                    cls_id = int(float(parts[0]))
                    coords = [float(v) for v in parts[1:]]
                    if len(coords) % 2 != 0:
                        print(f"[WARN] Odd number of coords in {seg_txt}: {line}")
                        continue

                    # denormalize to pixel coordinates
                    xs: List[float] = []
                    ys: List[float] = []
                    for xi, yi in zip(coords[0::2], coords[1::2]):
                        xs.append(xi * w)
                        ys.append(yi * h)

                    if len(xs) < 3:
                        continue

                    # segmentation: flatten [x1,y1, x2,y2, ...]
                    seg_flat: List[float] = []
                    for xx, yy in zip(xs, ys):
                        seg_flat.extend([xx, yy])

                    # bbox: [x_min, y_min, width, height]
                    x_min = float(min(xs))
                    y_min = float(min(ys))
                    x_max = float(max(xs))
                    y_max = float(max(ys))
                    bbox_w = x_max - x_min
                    bbox_h = y_max - y_min
                    if bbox_w <= 0 or bbox_h <= 0:
                        continue

                    area = polygon_area(xs, ys)

                    class_ids_seen.add(cls_id)
                    coco_annotations.append(
                        {
                            "id": ann_id,
                            "image_id": image_id,
                            # COCO 通常类别 id 从 1 开始，这里用 cls_id + 1
                            "category_id": cls_id + 1,
                            "segmentation": [seg_flat],
                            "area": float(area),
                            "bbox": [x_min, y_min, bbox_w, bbox_h],
                            "iscrowd": 0,
                        }
                    )
                    ann_id += 1

        else:
            print(f"[INFO] No seg label for {stem}, only image entry is created.")

        image_id += 1

    # Build categories
    categories: List[Dict] = []
    for cid in sorted(class_ids_seen):
        categories.append(
            {
                "id": cid + 1,
                "name": f"class_{cid}",
                "supercategory": "",
            }
        )

    coco = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": categories,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2)

    print(f"[INFO] Wrote COCO JSON to: {output_json}")
    print(f"[INFO] images      : {len(coco_images)}")
    print(f"[INFO] annotations : {len(coco_annotations)}")
    print(f"[INFO] categories  : {len(categories)}")


if __name__ == "__main__":
    main()
