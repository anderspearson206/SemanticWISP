"""PIROPO Room B (conventional camera) loading helpers.

Conventions match data_exploration/piropo_roomB_exploration.ipynb: annotation row i is the i-th JPEG in sorted
file-name order, one (x, y) head point per person in 704x576 pixels, (0, 0) = empty slot.
"""
import re
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / 'data' / 'roomb_piropo' / 'Room B' / 'conv_2B' / 'conv_2B'
ANN_DIR = ROOT / 'Ground_Truth_Annotations'
H, W = 120, 160                      # WISP-camera resolution used throughout the project
ANNOTATED = ['training', 'training_seats', 'test1', 'test2', 'test3']
BAD_THRESH = 15                      # corrupted frame: mean gray level this far from the sequence median


def frame_time(path):
    # file names are capture timestamps, e.g. 2015-11-13T12-14-47.255Z.jpg
    return datetime.strptime(Path(path).stem, '%Y-%m-%dT%H-%M-%S.%fZ').timestamp()


def _natural(name):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', name)]


def load_sequences(root=ROOT, names=None):
    """Returns {name: dict(files, times[, label, n_people, points])} in natural sequence order."""
    root = Path(root)
    seqs = {}
    for d in sorted(root.iterdir(), key=lambda p: _natural(p.name)):
        if not d.is_dir() or d.name == 'Ground_Truth_Annotations':
            continue
        name = d.name.replace('conv2B_', '')
        if names is not None and name not in names:
            continue
        files = sorted(d.glob('*.jpg'))
        s = dict(files=files, times=np.array([frame_time(f) for f in files]))
        ann = root / 'Ground_Truth_Annotations' / f'groundTruth_{d.name}.csv'
        if ann.exists():
            raw = np.loadtxt(ann, delimiter=',', dtype=int, ndmin=2)
            assert len(raw) == len(files) and (raw[:, 0] == np.arange(1, len(raw) + 1)).all(), name
            pts = raw[:, 1:].reshape(len(raw), -1, 2)
            present = (pts != 0).any(-1)
            s['points'] = [p[m] for p, m in zip(pts, present)]
            s['n_people'] = present.sum(1)
            s['label'] = (s['n_people'] > 0).astype(np.uint8)
        seqs[name] = s
    return seqs


def load_gray(path, size=(W, H)):
    """Grayscale frame resized to size=(w, h), decoded at reduced scale via JPEG draft mode."""
    with Image.open(path) as im:
        im.draft('L', size)
        return np.asarray(im.convert('L').resize(size, Image.BILINEAR), np.uint8)


def flag_bad(X):
    """Corrupted (block-shifted) frames in a stack of grayscale frames from one sequence."""
    m = X.reshape(len(X), -1).mean(1)
    return np.abs(m - np.median(m)) > BAD_THRESH


def thumb(X, w, h):
    """Block-average 120x160 frames (any leading dims) down to h x w; w and h must divide 160 and 120."""
    X = np.asarray(X, np.float32)
    return X.reshape(*X.shape[:-2], h, H // h, w, W // w).mean((-3, -1))


def auc(score, y):
    """ROC AUC via the Mann-Whitney U statistic, with average ranks for ties."""
    score, y = np.asarray(score, np.float64), np.asarray(y).astype(bool)
    order = np.argsort(score, kind='mergesort')
    s_sorted = score[order]
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    for v in np.unique(s_sorted[np.r_[False, np.diff(s_sorted) == 0]]):
        ranks[score == v] = ranks[score == v].mean()
    n1, n0 = y.sum(), (~y).sum()
    return (ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
