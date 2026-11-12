"""Adaptive loop through the simulator (approach category 5): the server starts with full frames, learns a
room-specific code from a pretrained detector's labels, sends it to the tag, and keeps it up to date.

Stream: the five annotated PIROPO sequences in the order they were recorded (training, test3, test1,
training_seats, test2), every 2nd frame, as one continuous deployment of about 30 minutes. Ground truth is used
only to score the result, never to learn (except in the `oracle` configurations).

The stream is replayed `--passes` times without resetting tag or server. Pass 1 is the honest test: everything
is new. Pass 2 shows the same frames again, which is optimistic, and answers a narrower question: once the
server has seen what this room does, how rarely does it need a full frame?

Teacher: Faster R-CNN MobileNetV3 (COCO), applied to the 160x120 grayscale frame the tag would send. Its scores
are read from results/coco_detectors/frcnn_mnv3_320_160x120_s2.npz (run coco_detector_eval.py first).

Configurations (name: what changes):
  adaptive_k<K>   K-bit code, explores and relearns
  static_k<K>     learns once after the bootstrap, then never asks for a full frame again
  oracle_k<K>     adaptive, but the teacher is replaced by the ground-truth labels (ceiling for this code)

Results: results/adaptive_loop/<name>.npz, one per configuration, holding one row per query. Existing files
are skipped.

  python experiments/adaptive_loop.py
  python experiments/adaptive_loop.py --configs adaptive_k96 static_k96 --bootstrap 400
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from semwisp import piropo
from semwisp.frames import load_frames
from semwisp.server import AdaptiveServer, BatchLearner, CachedTeacher
from semwisp.sim import replay
from semwisp.tag import IntervalCodeTag

ORDER = ['training', 'test3', 'test1', 'training_seats', 'test2']      # recording order
STRIDE = 2                                                             # matches the cached teacher scores
TEACHER = piropo.REPO / 'results' / 'coco_detectors' / 'frcnn_mnv3_320_160x120_s2.npz'
DEFAULT = ['adaptive_k96', 'adaptive_k32', 'adaptive_k8', 'static_k96', 'oracle_k96', 'oracle_static_k96']


def summarize(df):
    """Metrics over the queries after the bootstrap (those answered with the code, with or without a check)."""
    d = df[df.used_code]
    code_only = d[~d.explored]
    bal = lambda g: (g.answer[g.label == 1].mean() + (1 - g.answer[g.label == 0]).mean()) / 2
    return dict(n_bootstrap=int((~df.used_code).sum()), n_deployed=len(d),
                accuracy=float((d.answer == d.label).mean()), balanced_accuracy=float(bal(d)),
                auc_code_only=float(piropo.auc(code_only.score, code_only.label)),
                up_bytes_per_query=float(d.up_bytes.mean()), explored_fraction=float(d.explored.mean()),
                reconfigs=int(df.reconfig.sum()), down_bytes_total=float(df.down_bits.sum() / 8),
                balanced_accuracy_by_seq={n: float(bal(g)) for n, g in d.groupby('seq', sort=False)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--configs', nargs='+', default=DEFAULT)
    ap.add_argument('--bootstrap', type=int, default=200, help='labelled full frames before the first code')
    ap.add_argument('--relearn-every', type=int, default=50)
    ap.add_argument('--window', type=int, default=1000)
    ap.add_argument('--passes', type=int, default=2)
    ap.add_argument('--out', type=Path, default=piropo.REPO / 'results' / 'adaptive_loop')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    frames = load_frames(ORDER)
    scores = np.load(TEACHER)
    t, ts, gt = [], [], []
    for n in ORDER:
        assert (scores[f'idx_{n}'] == np.arange(0, len(frames[n]['X']), STRIDE)).all()
        t.append(frames[n]['t'][::STRIDE]); ts.append(scores[f'score_{n}']); gt.append(frames[n]['label'][::STRIDE])
    teachers = dict(detector=CachedTeacher(np.concatenate(t), np.concatenate(ts)),
                    oracle=CachedTeacher(np.concatenate(t), np.concatenate(gt)))

    for name in args.configs:
        out = args.out / f'{name}.npz'
        if out.exists():
            print(f'skip {name} (exists)')
            continue
        kind, k = name.rsplit('_k', 1)
        t0 = time.time()
        server = AdaptiveServer(teachers['oracle' if kind.startswith('oracle') else 'detector'],
                                BatchLearner(k=int(k), window=args.window), bootstrap=args.bootstrap,
                                relearn_every=args.relearn_every, adapt='static' not in kind)
        tag, parts = IntervalCodeTag(), []
        for p in range(1, args.passes + 1):
            parts.append(replay(frames, tag, server, stride=STRIDE, reset=False).df.assign(**{'pass': p}))
        df = pd.concat(parts, ignore_index=True)
        df['teacher'] = [teachers['detector'].table[float(x)] for x in df.t]
        by_pass = {p: summarize(g) for p, g in df.groupby('pass')}
        meta = dict(name=name, kind=kind, k=int(k), bootstrap=args.bootstrap, relearn_every=args.relearn_every,
                    window=args.window, seconds=round(time.time() - t0), by_pass=by_pass)
        code = server.active[0]            # the code in use at the end, for inspection
        np.savez(out, meta=json.dumps(meta), code_cell=code.cell, code_lo=code.lo, code_hi=code.hi,
                 code_weight=server.active[1].coef_[0],
                 **{c: np.array(df[c].tolist(), dtype=str) if c == 'seq' else df[c].values for c in df.columns})
        for p, m in by_pass.items():
            print(f"{name} pass {p}: balanced accuracy {m['balanced_accuracy']:.3f}, code-only AUC "
                  f"{m['auc_code_only']:.3f}, {m['up_bytes_per_query']:.0f} B/query up, explored "
                  f"{m['explored_fraction']:.1%}, {m['reconfigs']} configs ({m['down_bytes_total']:.0f} B down)", flush=True)


if __name__ == '__main__':
    main()
