"""Minimal INSID3 test: segment a target image from a reference image + mask.

Defaults to the bundled cat assets and the 'base' backbone (the 'large' weights
are not downloaded in this setup). Override any path or the model size via flags.

Examples
--------
    # bundled cat demo
    python run_minimal.py

    # your own image + mask
    python run_minimal.py --ref my_ref.jpg --ref-mask my_ref_mask.png \
        --target my_target.jpg --out my_pred.png
"""
import argparse
import os

import numpy as np
from PIL import Image

from models import build_insid3
from utils.visualization import visualize_prediction_segmentation as visualize


def main():
    ap = argparse.ArgumentParser(description="Minimal INSID3 in-context segmentation demo.")
    ap.add_argument("--ref", default="assets/ref_cat_image.jpg", help="Reference image.")
    ap.add_argument("--ref-mask", default="assets/ref_cat_mask.png", help="Reference binary mask.")
    ap.add_argument("--target", default="assets/target_cat_image.jpg", help="Target image to segment.")
    ap.add_argument("--out", default="target_cat_pred.png", help="Output visualization path.")
    ap.add_argument("--mask-out", default=None,
                    help="Path for the raw predicted binary mask PNG "
                         "(default: '<out>_mask.png'). Pass '' to skip.")
    ap.add_argument("--model-size", default="base", choices=["small", "base", "large"],
                    help="DINOv3 backbone (default: base; 'large' weights not downloaded).")
    args = ap.parse_args()

    model = build_insid3(model_size=args.model_size)
    model.set_reference(args.ref, args.ref_mask)
    model.set_target(args.target)
    pred = model.segment()

    print(f"pred mask: {tuple(pred.shape)} | foreground={int(pred.sum())} px "
          f"({100 * pred.float().mean():.1f}%)")

    visualize(args.ref, args.ref_mask, args.target, pred, args.out)
    print(f"visualization -> {args.out}")

    mask_out = args.mask_out
    if mask_out is None:
        mask_out = os.path.splitext(args.out)[0] + "_mask.png"
    if mask_out:
        mask_np = (pred.detach().cpu().numpy().astype(np.uint8) * 255)
        Image.fromarray(mask_np, mode="L").save(mask_out)
        print(f"predicted mask -> {mask_out}")


if __name__ == "__main__":
    main()
