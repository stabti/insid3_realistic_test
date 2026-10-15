"""Convert a binary mask PNG to a COCO 1.0 annotation file for CVAT.

Produces a single-image, single-annotation COCO JSON that CVAT imports via the
"COCO 1.0" format. Two segmentation encodings:

  --seg-format polygon  (default) -> editable polygon(s) in CVAT. Holes are NOT
                                     representable; thin detail may be simplified.
  --seg-format rle                 -> pixel-perfect mask (preserves holes/detail).

Examples
--------
    python mask_to_coco.py my_target_mask.png --image my_target.jpg --out ann.json
    python mask_to_coco.py my_target_mask.png --image my_target.jpg --out ann.json \
        --seg-format rle --label valve

CVAT usage: create a task with the SAME image, then
"Actions -> Upload annotations -> COCO 1.0" and pick this JSON. The category
name (--label) must match (or be added to) a label in the CVAT task.
"""
import argparse
import json
import os

import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as mask_util


def load_binary_mask(path):
    """Load a mask PNG as a bool (H, W) array (any nonzero pixel = foreground)."""
    a = np.array(Image.open(path))
    if a.ndim == 3:  # RGB/RGBA -> use alpha if present, else any channel
        a = a[..., 3] if a.shape[2] == 4 else a.sum(axis=2)
    return a > 0


def mask_to_polygons(mask, min_area):
    """Extract external-contour polygons as COCO [x1,y1,x2,y2,...] lists."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    polys = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        c = c.reshape(-1, 2)
        if len(c) < 3:
            continue
        polys.append(c.flatten().astype(float).tolist())
    return polys


def mask_to_rle(mask):
    """Encode mask as a COCO compressed RLE with a JSON-serializable counts str."""
    rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def main():
    ap = argparse.ArgumentParser(description="Binary mask PNG -> COCO 1.0 JSON for CVAT.")
    ap.add_argument("mask", help="Binary mask PNG.")
    ap.add_argument("--image", help="Image the mask belongs to (for size + file_name).")
    ap.add_argument("--file-name", help="Override file_name recorded in COCO "
                                        "(default: basename of --image, else of mask).")
    ap.add_argument("--out", required=True, help="Output COCO JSON path.")
    ap.add_argument("--label", default="object", help="Category name (default: object).")
    ap.add_argument("--seg-format", choices=["polygon", "rle"], default="polygon")
    ap.add_argument("--min-area", type=float, default=10.0,
                    help="Drop polygon components smaller than this (px). polygon only.")
    args = ap.parse_args()

    mask = load_binary_mask(args.mask)
    h, w = mask.shape

    if args.image is not None:
        iw, ih = Image.open(args.image).size
        if (ih, iw) != (h, w):
            mask = np.array(
                Image.fromarray(mask.astype(np.uint8) * 255).resize((iw, ih), Image.NEAREST)
            ) > 0
            h, w = mask.shape
        file_name = args.file_name or os.path.basename(args.image)
    else:
        file_name = args.file_name or os.path.basename(args.mask)

    if mask.sum() == 0:
        raise SystemExit("Mask is empty (no foreground pixels).")

    # Accurate bbox/area from RLE regardless of chosen segmentation encoding.
    rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    area = float(mask_util.area(rle))
    bbox = [float(v) for v in mask_util.toBbox(rle)]  # [x, y, w, h]

    if args.seg_format == "polygon":
        segmentation = mask_to_polygons(mask, args.min_area)
        if not segmentation:
            raise SystemExit("No polygon >= --min-area; try --seg-format rle or lower --min-area.")
        iscrowd = 0
    else:
        segmentation = mask_to_rle(mask)
        iscrowd = 0

    coco = {
        "licenses": [{"id": 1, "name": "", "url": ""}],
        "info": {"description": "INSID3 mask export", "version": "1.0"},
        "categories": [{"id": 1, "name": args.label, "supercategory": ""}],
        "images": [{"id": 1, "file_name": file_name, "width": w, "height": h,
                    "license": 1, "flickr_url": "", "coco_url": "", "date_captured": 0}],
        "annotations": [{
            "id": 1, "image_id": 1, "category_id": 1,
            "segmentation": segmentation, "area": area,
            "bbox": bbox, "iscrowd": iscrowd,
        }],
    }

    with open(args.out, "w") as f:
        json.dump(coco, f)

    kind = f"{len(segmentation)} polygon(s)" if args.seg_format == "polygon" else "RLE mask"
    print(f"wrote {args.out} | image='{file_name}' {w}x{h} | label='{args.label}' | {kind}")


if __name__ == "__main__":
    main()
