"""Cached WISP-resolution frames: every PIROPO Room B sequence as one uint8 array, in time order.

Decoding 9,332 JPEGs takes about a minute; the cache (about 180 MB, data/cache/) makes later loads instant.
It is built on first use on whichever machine has the JPEGs, so it never needs to be copied to Hyak.
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import piropo

CACHE = piropo.REPO / 'data' / 'cache' / f'piropo_conv2B_{piropo.W}x{piropo.H}.npz'


def build_cache(path=CACHE):
    seqs, out = piropo.load_sequences(), {}
    with ThreadPoolExecutor(8) as ex:
        for name, s in seqs.items():
            X = np.stack(list(ex.map(piropo.load_gray, s['files'])))
            out[f'{name}/X'], out[f'{name}/t'], out[f'{name}/bad'] = X, s['times'], piropo.flag_bad(X)
            if 'label' in s:
                out[f'{name}/label'], out[f'{name}/n_people'] = s['label'], s['n_people']
            print(f'cached {name}: {len(X)} frames', flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + '.tmp.npz')
    np.savez(tmp, **out)
    tmp.replace(path)


def load_frames(names=None, path=CACHE):
    """{sequence: dict(X uint8 (n, 120, 160), t seconds, bad bool[, label, n_people])}; names=None loads all 13."""
    if not path.exists():
        build_cache(path)
    seqs = {}
    with np.load(path) as z:
        for key in z.files:
            name, field = key.split('/')
            if names is None or name in names:
                seqs.setdefault(name, {})[field] = z[key]
    return seqs if names is None else {n: seqs[n] for n in names}


def load_mall_tile(row=2, col=2, grid='3x3'):
    """One tile of the Mall dataset as a second 'room': dict(X uint8 (2000, 120, 160), t, bad, label).

    Frames are the crops in data/mall_dataset/cropped/ (make_crops.py), grayscale, resized to 160x120. label is
    1 (a head is inside the tile), 0 (nobody touches it) or -1 (ambiguous: use for neither learning nor scoring).
    t is the frame number, not a real time (the video is under 2 frames per second). Cached in data/cache/."""
    import pandas as pd
    from PIL import Image
    from . import mall
    path = CACHE.parent / f'mall_{grid}_r{row}c{col}_{piropo.W}x{piropo.H}.npz'
    if not path.exists():
        d = pd.read_csv(mall.CROP_ROOT / f'labels_{grid}.csv')
        d = d[(d.row == row) & (d.col == col)].sort_values('frame')
        def load(f):
            with Image.open(mall.CROP_ROOT / f) as im:
                return np.asarray(im.convert('L').resize((piropo.W, piropo.H), Image.BILINEAR), np.uint8)
        with ThreadPoolExecutor(8) as ex:
            X = np.stack(list(ex.map(load, d.file)))
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, X=X, t=d.frame.values.astype(float), bad=np.zeros(len(X), bool), label=d.label.values.astype(np.int8))
    with np.load(path) as z:
        return {k: z[k] for k in z.files}
