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
