"""Generate a binary reference mask for INSID3 using SAM2.

The output is a single-channel (mode 'L') PNG with values {0, 255}, exactly the
format INSID3's `load_mask` expects (any nonzero pixel = foreground).

Examples
--------
Point prompt (headless) -- click one point on the object:
    python make_mask.py my_image.jpg --points 320,240 --out my_mask.png

Multiple points, mixing foreground (1) and background (0) hints:
    python make_mask.py my_image.jpg --points 320,240 400,300 --labels 1 0 --out my_mask.png

Box prompt (x1,y1,x2,y2):
    python make_mask.py my_image.jpg --box 100,80,540,420 --out my_mask.png

Interactive (needs a display): left-click = foreground, right-click = background,
press ENTER to finish, 'u' to undo, 'r' to reset:
    python make_mask.py my_image.jpg --interactive --out my_mask.png

Every run also writes a "<out>_overlay.png" preview so you can eyeball the result.
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

DEFAULT_CKPT = "sam_ckpts/sam2.1_hiera_small.pt"
DEFAULT_CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"


def parse_xy(values):
    """Parse a list of 'X,Y' strings into an (N, 2) float array."""
    pts = []
    for v in values:
        x, y = v.split(",")
        pts.append([float(x), float(y)])
    return np.array(pts, dtype=np.float32)


def build_predictor(cfg, ckpt, device):
    model = build_sam2(cfg, ckpt, device=device)
    return SAM2ImagePredictor(model)


def run_sam(predictor, image, point_coords=None, point_labels=None, box=None, device="cuda"):
    """Return the best (highest-scoring) mask as a bool (H, W) array."""
    predictor.set_image(image)
    ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device == "cuda"
        else torch.no_grad()
    )
    with torch.inference_mode(), ctx:
        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=True,
        )
    best = int(np.argmax(scores))
    return masks[best].astype(bool), float(scores[best])


def save_mask(mask, out_path):
    """Save a bool (H, W) mask as a single-channel {0,255} PNG."""
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(out_path)


def save_overlay(image, mask, out_path):
    """Save an RGB preview with the mask tinted green over the image."""
    ov = image.copy().astype(np.float32)
    green = np.array([0, 255, 0], dtype=np.float32)
    ov[mask] = 0.45 * ov[mask] + 0.55 * green
    Image.fromarray(ov.astype(np.uint8), mode="RGB").save(out_path)


def interactive_points(image):
    """Open a matplotlib window; collect fg/bg clicks. Returns (coords, labels)."""
    import matplotlib.pyplot as plt

    coords, labels = [], []
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title("L-click=fg  R-click=bg  |  u=undo  r=reset  ENTER=done")

    def redraw():
        ax.clear()
        ax.imshow(image)
        ax.set_title("L-click=fg  R-click=bg  |  u=undo  r=reset  ENTER=done")
        for (x, y), lab in zip(coords, labels):
            ax.plot(x, y, marker="*", markersize=14,
                    color="lime" if lab == 1 else "red",
                    markeredgecolor="white")
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax or event.xdata is None:
            return
        coords.append([event.xdata, event.ydata])
        labels.append(1 if event.button == 1 else 0)
        redraw()

    def on_key(event):
        if event.key == "enter":
            plt.close(fig)
        elif event.key == "u" and coords:
            coords.pop(); labels.pop(); redraw()
        elif event.key == "r":
            coords.clear(); labels.clear(); redraw()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()
    if not coords:
        raise SystemExit("No points selected.")
    return np.array(coords, dtype=np.float32), np.array(labels, dtype=np.int32)


def main():
    ap = argparse.ArgumentParser(description="Generate an INSID3 reference mask with SAM2.")
    ap.add_argument("image", help="Path to the reference image.")
    ap.add_argument("--out", required=True, help="Output mask PNG path.")
    ap.add_argument("--points", nargs="+", metavar="X,Y", help="Prompt point(s) 'X,Y'.")
    ap.add_argument("--labels", nargs="+", type=int,
                    help="1=foreground, 0=background per point (default: all 1).")
    ap.add_argument("--box", metavar="X1,Y1,X2,Y2", help="Box prompt.")
    ap.add_argument("--interactive", action="store_true", help="Click points in a GUI window.")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--model-cfg", default=DEFAULT_CFG)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    image = np.array(Image.open(args.image).convert("RGB"))
    predictor = build_predictor(args.model_cfg, args.ckpt, device)

    point_coords = point_labels = box = None
    if args.interactive:
        point_coords, point_labels = interactive_points(image)
    elif args.points:
        point_coords = parse_xy(args.points)
        if args.labels:
            assert len(args.labels) == len(point_coords), "labels must match points"
            point_labels = np.array(args.labels, dtype=np.int32)
        else:
            point_labels = np.ones(len(point_coords), dtype=np.int32)
    elif args.box:
        box = np.array([float(v) for v in args.box.split(",")], dtype=np.float32)
    else:
        raise SystemExit("Provide one of --points, --box, or --interactive.")

    mask, score = run_sam(predictor, image, point_coords, point_labels, box, device)

    save_mask(mask, args.out)
    overlay_path = os.path.splitext(args.out)[0] + "_overlay.png"
    save_overlay(image, mask, overlay_path)

    cover = 100.0 * mask.mean()
    print(f"SAM score={score:.3f} | foreground={int(mask.sum())} px ({cover:.1f}%)")
    print(f"mask    -> {args.out}")
    print(f"overlay -> {overlay_path}  (open this to verify)")


if __name__ == "__main__":
    main()
