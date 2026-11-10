"""Mall dataset (Chen et al., CUHK) loading helpers and the crop/label rules used in data/mall_dataset/cropped/.

2000 RGB frames, 640x480, one fixed camera over a shopping-mall walkway. Ground truth is one head point per person
(13-53 people per frame), annotated only inside the region of interest (ROI) mask in perspective_roi.mat.
"""
from pathlib import Path

import numpy as np
import scipy.io as sio

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / 'data' / 'mall_dataset' / 'mall_dataset'
FRAMES = ROOT / 'frames'
CROP_ROOT = REPO / 'data' / 'mall_dataset' / 'cropped'
IMG_W, IMG_H = 640, 480
BLACK_ROWS = 16          # rows 0-15 are a black band above the image content (outside the ROI, but nobody can be there)


def frame_path(i):
    """Path of frame i (1-based, as in the dataset's demo.m)."""
    return FRAMES / f'seq_{i:06d}.jpg'


def load_gt(root=ROOT):
    """Returns (points, counts): points[i] is an (n_i, 2) float array of (x, y) head points of frame i+1."""
    m = sio.loadmat(Path(root) / 'mall_gt.mat')
    points = [f['loc'][0, 0].astype(np.float64) for f in m['frame'][0]]
    counts = m['count'].ravel().astype(int)
    assert all(len(p) == c for p, c in zip(points, counts))
    return points, counts


def load_roi(root=ROOT):
    """Returns (roi, pmap): boolean 480x640 annotation mask and the dataset's perspective map pMapN (varies with y only)."""
    r = sio.loadmat(Path(root) / 'perspective_roi.mat')
    return r['roi'][0, 0]['mask'].astype(bool), r['pMapN']


def person_height(y):
    """Rough standing-person height in pixels for a head point at row y.

    Hand-fitted on frames 970 and 1500 (about 20 px near the top of the image, about 140 px at the bottom). It is
    deliberately on the generous side: it is only used to decide when a crop is safely empty.
    """
    return 0.33 * np.asarray(y, np.float64) + 5


def person_boxes(points):
    """Estimated person boxes (x0, y0, x1, y1) from head points: head near the top, body below it."""
    p = np.asarray(points, np.float64).reshape(-1, 2)
    h = person_height(p[:, 1])
    w = 0.45 * h
    return np.stack([p[:, 0] - w / 2, p[:, 1] - 0.12 * h, p[:, 0] + w / 2, p[:, 1] + 0.88 * h], 1)


def grid_boxes(nx, ny, w=IMG_W, h=IMG_H):
    """Non-overlapping tiles (x0, y0, x1, y1) of an nx-by-ny grid, row-major; edges rounded to whole pixels."""
    xs = np.linspace(0, w, nx + 1).round().astype(int)
    ys = np.linspace(0, h, ny + 1).round().astype(int)
    return [(r, c, xs[c], ys[r], xs[c + 1], ys[r + 1]) for r in range(ny) for c in range(nx)]


def crop_labels(points, box, roi, max_outside=0.05):
    """Labels one crop box=(x0, y0, x1, y1) of a frame with head points `points`.

    n_heads: head points inside the crop.
    n_touch: estimated person boxes that overlap the crop (includes the n_heads people).
    roi_frac: fraction of the crop inside the annotated region (the black band at the top counts as inside).
    label: 1 = a head is inside; 0 = no estimated person box touches the crop and at most `max_outside` of it lies
           outside the ROI (where people are not annotated); -1 = ambiguous (body only, or unannotated area).
    """
    x0, y0, x1, y1 = box
    p = np.asarray(points, np.float64).reshape(-1, 2)
    n_heads = int(((p[:, 0] >= x0) & (p[:, 0] < x1) & (p[:, 1] >= y0) & (p[:, 1] < y1)).sum())
    b = person_boxes(p)
    n_touch = int(((b[:, 0] < x1) & (b[:, 2] > x0) & (b[:, 1] < y1) & (b[:, 3] > y0)).sum())
    m = roi.copy()
    m[:BLACK_ROWS] = True
    roi_frac = float(m[y0:y1, x0:x1].mean())
    if n_heads > 0:
        label = 1
    elif n_touch == 0 and roi_frac >= 1 - max_outside:
        label = 0
    else:
        label = -1
    return dict(n_heads=n_heads, n_touch=n_touch, roi_frac=round(roi_frac, 4), label=label)
