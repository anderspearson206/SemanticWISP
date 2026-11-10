"""Split every Mall frame into a grid of sub-frames and label each one person / empty / ambiguous.

The Mall dataset has 13-53 people in every frame, so whole frames are never empty. Tiles of a 3x3 grid
(213x160 px, the WISP 4:3 aspect) give views with zero to a few people, closer to our task: is anyone in view?

Writes (resumable: existing crop files are skipped, the label CSV is always rewritten):
    cropped/frames_<nx>x<ny>/seq_<frame>_r<row>c<col>.jpg
    cropped/labels_<nx>x<ny>.csv   one row per crop, labels from semwisp.mall.crop_labels

Run from the repo root:
    .venv/Scripts/python.exe data/mall_dataset/cropped/make_crops.py --grid 3x3
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from semwisp.mall import CROP_ROOT, frame_path, grid_boxes, crop_labels, load_gt, load_roi  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--grid', default='3x3', help='columns x rows, e.g. 3x3 (9 views) or 3x2, 4x3 (default 3x3)')
    ap.add_argument('--out', type=Path, default=CROP_ROOT, help='output folder (default data/mall_dataset/cropped)')
    ap.add_argument('--quality', type=int, default=95, help='JPEG quality of the saved crops (default 95)')
    ap.add_argument('--max-outside', type=float, default=0.05,
                    help='largest fraction of a crop outside the annotated ROI that may still be labelled empty')
    ap.add_argument('--frames', type=int, default=None, help='only the first N frames (for a quick test)')
    args = ap.parse_args()

    nx, ny = (int(v) for v in args.grid.lower().split('x'))
    tag = f'{nx}x{ny}'
    img_dir = args.out / f'frames_{tag}'
    img_dir.mkdir(parents=True, exist_ok=True)
    points, counts = load_gt()
    roi, _ = load_roi()
    tiles = grid_boxes(nx, ny)
    n = len(points) if args.frames is None else min(args.frames, len(points))

    rows, written = [], 0
    for i in range(1, n + 1):
        im = None
        for r, c, x0, y0, x1, y1 in tiles:
            name = f'seq_{i:06d}_r{r}c{c}.jpg'
            out = img_dir / name
            if not out.exists():
                if im is None:
                    im = Image.open(frame_path(i)).convert('RGB')
                im.crop((x0, y0, x1, y1)).save(out, quality=args.quality)
                written += 1
            lab = crop_labels(points[i - 1], (x0, y0, x1, y1), roi, args.max_outside)
            rows.append(dict(file=f'frames_{tag}/{name}', frame=i, row=r, col=c,
                             x0=x0, y0=y0, x1=x1, y1=y1, frame_count=int(counts[i - 1]), **lab))
        if i % 250 == 0:
            print(f'{i}/{n} frames', flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out / f'labels_{tag}.csv', index=False)
    print(f'{len(df)} crops ({written} newly written) in {img_dir}')
    print('label counts (1 person, 0 empty, -1 ambiguous):', df.label.value_counts().sort_index().to_dict())


if __name__ == '__main__':
    main()
