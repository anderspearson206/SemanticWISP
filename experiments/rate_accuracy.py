"""Shared loading for rate_accuracy.ipynb: every encoder evaluated so far, on one pair of axes.

All points are scored on the same frames' ground truth, PIROPO `test1`-`test3`, and with no tuning on them:
  supervised   thumbnail / JPEG payload + a CNN trained on ~3,000 true labels of this room (thumbnail_sweep.py);
               decision at the decoder's own zero.
  generic      thumbnail / JPEG payload, decoded, + the COCO detector with no room knowledge (coco_detector_eval.py);
               threshold tuned on `training` + `training_seats` (a few minutes of labelled data, no training).
  adaptive     the online loop with no labels at all (detector teacher), first pass, queries from test1-3 only
               (tiered_checks.py, both kinds of check); its own yes/no answers. Bytes include every check.
This module only reads cached results; nothing is run here.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RES = REPO / 'results'
TEST, TRAIN = ['test1', 'test2', 'test3'], ['training', 'training_seats']


def bal_acc(answer, label):
    answer, label = np.asarray(answer, bool), np.asarray(label, bool)
    return float((answer[label].mean() + (~answer[~label]).mean()) / 2)


def auc(score, label):
    from semwisp.piropo import auc as _auc
    return float(_auc(score, label))


def load_npz(path):
    with np.load(path) as z:
        meta = json.loads(str(z['meta'])) if 'meta' in z.files else {}
        n = len(z['label'])
        cols = {k: z[k] for k in z.files if k != 'meta' and z[k].shape == (n,)}     # per-query columns only
    return meta, pd.DataFrame(cols)


def supervised():
    """One row per thumbnail_sweep configuration (no channel loss, no sensor noise)."""
    rows = []
    for f in sorted((RES / 'thumbnail_sweep').glob('*.npz')):
        meta, df = load_npz(f)
        if meta.get('p_loss') or meta.get('read_noise'):
            continue
        df = df[df.label >= 0]
        rows.append(dict(name=f.stem, family='JPEG' if meta['method'] == 'jpeg' else 'thumbnail', method=meta['method'],
                         w=meta['w'], h=meta['h'], bits=meta['bits'], bytes=float(df.up_bytes.mean()),
                         epcs=float(df.up_epcs.mean()), bal_acc=bal_acc(df.score > 0, df.label),
                         auc=auc(df.score, df.label)))
    return pd.DataFrame(rows)


def channel_loss():
    """thumbnail_sweep configurations run with EPC loss, with their loss-free twin."""
    rows = []
    for f in sorted((RES / 'thumbnail_sweep').glob('*_loss*.npz')):
        meta, df = load_npz(f)
        _, clean = load_npz(f.with_name(f.stem.split('_loss')[0] + '.npz'))
        rows.append(dict(name=f.stem.split('_loss')[0], p_loss=meta['p_loss'], bytes=meta['n_bytes'],
                         epcs=float(df.up_epcs.mean()), auc=auc(df.score, df.label), auc_clean=auc(clean.score, clean.label),
                         family='JPEG' if meta['method'] == 'jpeg' else 'thumbnail'))
    return pd.DataFrame(rows)


def generic():
    """COCO detector (320 px Faster R-CNN) on what the WISP could send: thumbnails and JPEG, every 2nd frame."""
    from semwisp.frames import load_frames
    from semwisp.tag.jpeg import JpegTag
    frames = load_frames(TRAIN + TEST)
    rows = []
    for f in sorted((RES / 'coco_detectors').glob('frcnn_mnv3_320_*_s2.npz')):
        cond = f.stem[len('frcnn_mnv3_320_'):-3]
        if cond in ('rgb', 'gray'):
            continue                      # not something a WISP can send
        z = np.load(f)
        def part(names):
            s = np.concatenate([z[f'score_{n}'] for n in names])
            idx = [z[f'idx_{n}'] for n in names]
            y = np.concatenate([frames[n]['label'][i] for n, i in zip(names, idx)])
            ok = ~np.concatenate([frames[n]['bad'][i] for n, i in zip(names, idx)])
            return s[ok], y[ok].astype(bool), [(n, i[~frames[n]['bad'][i]]) for n, i in zip(names, idx)]
        s_tr, y_tr, _ = part(TRAIN)
        s_te, y_te, te_idx = part(TEST)
        thr = max(np.unique(s_tr), key=lambda t: bal_acc(s_tr >= t, y_tr))
        if cond.startswith('jpeg'):
            size, q = cond[4:].split('q')
            w, h = map(int, size.split('x'))
            tag = JpegTag(w, h, int(q))
            nbytes = float(np.mean([len(tag.encode(frames[n]['X'][j])) for n, i in te_idx for j in i]))
            family = 'JPEG'
        else:
            w, h = map(int, cond.split('x'))
            nbytes, family = float(w * h), 'thumbnail'
        rows.append(dict(name=cond, family=family, w=w, h=h, bytes=nbytes, threshold=float(thr),
                         bal_acc=bal_acc(s_te >= thr, y_te), auc=auc(s_te, y_te)))
    return pd.DataFrame(rows)


def adaptive(teacher='detector'):
    """First-pass rows from test1-3 of each adaptive run on the PIROPO stream."""
    rows = []
    for d, check in [('tiered_checks', None)]:                  # holds both kinds of check, bootstrap 200
        for f in sorted((RES / d).glob(f'piropo_{teacher}_*.npz')):
            meta, df = load_npz(f)
            if d == 'tiered_checks' and meta['bootstrap'] != 200:
                continue
            df = df[(df['pass'] == 1) & df.seq.isin(TEST) & (df.label >= 0)]
            learner = meta['learner'] if d == 'tiered_checks' else f"{meta['learner']}_{meta['regions']}"
            rows.append(dict(name=f.stem, source=d, learner=learner, check=check or meta['check'], teacher=teacher,
                             bytes=float(df.up_bytes.mean()), full_rate=float(df.explored.mean()),
                             bal_acc=bal_acc(df.answer, df.label)))
    return pd.DataFrame(rows)


def static_code():
    """Learn once (adaptive_loop.py static_k96): a 12-byte code fitted to the first minutes, never updated."""
    meta, df = load_npz(RES / 'adaptive_loop' / 'static_k96.npz')
    df = df.iloc[:len(df) // 2]          # saved before replays were tagged with a pass; the stream was played twice
    df = df[df.seq.isin(TEST) & (df.label >= 0)]
    return dict(bytes=float(df.up_bytes.mean()), bal_acc=bal_acc(df.answer, df.label))
