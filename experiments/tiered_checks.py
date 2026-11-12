"""Tiered checks and readaptation with a detector teacher everywhere.

The adaptive loops of adaptive_loop.py and online_adaptation.py check the tag's code against full frames, and
those checks are almost all of the uplink. semwisp.server.TieredServer checks against a 192-byte thumbnail
first (scored by a room decoder learned online from the teacher's labels) and fetches a full frame only when
the thumbnail cannot settle it; code answers that are themselves uncertain (|logit| < --margin) are also settled
by a thumbnail. This script compares the two kinds of check, and sweeps the bootstrap length, the setting that
readaptation time mostly reflects.

Streams (as in online_adaptation.py)
  piropo   the five annotated PIROPO sequences in recording order, every 2nd frame, replayed twice.
  switch   the tag is moved: PIROPO `training` -> a Mall tile -> PIROPO `training_seats` + `test2`. One pass.
           Now with a detector teacher too (results/coco_detectors/frcnn_mnv3_320_mall_tile_160x120.npz,
           from `coco_detector_eval.py --mall-tile`).
Teachers: `detector` (COCO detector on the 160x120 frame; no labels used anywhere) and `oracle` (ground truth).
Learners: online_pixels (OnlineLearner, 96 single pixels), online_cells (OnlineLearner, whole blocks),
          batch_cells (BatchLearner, the baseline).
Checks:   full (AdaptiveServer) or tiered (TieredServer).

Results: results/tiered_checks/<stream>_<teacher>_<learner>_<check>_b<bootstrap>.npz, one row per query;
existing files are skipped.

  python experiments/tiered_checks.py
  python experiments/tiered_checks.py --only switch_detector
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from semwisp import piropo
from semwisp.frames import load_frames, load_mall_tile
from semwisp.metrics import summary
from semwisp.server import AdaptiveServer, BatchLearner, CachedTeacher, OnlineLearner, TieredServer
from semwisp.sim import replay
from semwisp.tag import IntervalCodeTag
from semwisp.tag.interval import CELLS, lattice, sparse

ORDER = ['training', 'test3', 'test1', 'training_seats', 'test2']
STRIDE = 2
DET = piropo.REPO / 'results' / 'coco_detectors'
LEARNERS = dict(online_pixels=lambda k: OnlineLearner(k=k, table=sparse(lattice(4), 4)),
                online_cells=lambda k: OnlineLearner(k=k, table=CELLS),
                batch_cells=lambda k: BatchLearner(k=k, table=CELLS))
BOOTSTRAPS = [25, 50, 100, 200, 400]
RUNS = ([(s, t, l, c, 200) for s in ('piropo', 'switch') for t in ('detector', 'oracle') for l in LEARNERS
         for c in ('full', 'tiered')]
        + [('switch', t, l, 'tiered', b) for t in ('detector', 'oracle') for l in ('online_pixels', 'batch_cells')
           for b in BOOTSTRAPS if b != 200])


def every(s, k):
    return {f: v[::k] for f, v in s.items()}


def setup():
    """(streams, teachers): streams[name] is a list of passes, each {sequence name: frames}; teachers[name] is a
    CachedTeacher."""
    P = {n: every(s, STRIDE) for n, s in load_frames(ORDER).items()}
    mall = load_mall_tile()
    det, det_mall = np.load(DET / 'frcnn_mnv3_320_160x120_s2.npz'), np.load(DET / 'frcnn_mnv3_320_mall_tile_160x120.npz')
    assert (det_mall['t'] == mall['t']).all()
    t = np.concatenate([P[n]['t'] for n in ORDER] + [mall['t']])
    truth = np.concatenate([P[n]['label'] for n in ORDER] + [mall['label']]).astype(float)
    teachers = dict(oracle=CachedTeacher(t, np.where(truth < 0, 0.45, truth)),     # 0.45: no opinion
                    detector=CachedTeacher(t, np.concatenate([det[f'score_{n}'] for n in ORDER] + [det_mall['score']])))
    streams = dict(piropo=[P] * 2,
                   switch=[{'room A (training)': P['training'], 'room B (mall tile)': mall,
                            'room A, rearranged': P['training_seats'], 'room A, rearranged (test2)': P['test2']}])
    return streams, teachers


def make_server(teacher, learner, check, boot, k=96, margin=2.0):
    L = LEARNERS[learner](k)
    return (TieredServer(teacher, L, bootstrap=boot, code_margin=margin) if check == 'tiered'
            else AdaptiveServer(teacher, L, bootstrap=boot))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', help='run only configurations whose name contains this text')
    ap.add_argument('--k', type=int, default=96, help='bits in the code (96 = one EPC)')
    ap.add_argument('--margin', type=float, default=2.0, help='TieredServer code_margin')
    ap.add_argument('--out', type=Path, default=piropo.REPO / 'results' / 'tiered_checks')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    streams, teachers = setup()
    for stream, teacher, learner, check, boot in RUNS:
        name = f'{stream}_{teacher}_{learner}_{check}_b{boot}'
        out = args.out / f'{name}.npz'
        if args.only and args.only not in name:
            continue
        if out.exists():
            print(f'skip {name} (exists)')
            continue
        t0 = time.time()
        server = make_server(teachers[teacher], learner, check, boot, args.k, args.margin)
        tag, parts = IntervalCodeTag(), []
        for p, seqs in enumerate(streams[stream], 1):
            parts.append(replay(seqs, tag, server, reset=False).df.assign(**{'pass': p}))
        df = pd.concat(parts, ignore_index=True)
        code = server.active[0]
        meta = dict(name=name, stream=stream, teacher=teacher, learner=learner, check=check, bootstrap=boot,
                    k=args.k, margin=args.margin if check == 'tiered' else None, bits=len(code.cell),
                    pixels_read=code.n_pixels, seconds=round(time.time() - t0),
                    by_pass={p: summary(g) for p, g in df.groupby('pass')})
        np.savez(out, meta=json.dumps(meta),
                 **{c: df[c].values if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c])
                    else np.array(df[c].tolist(), dtype=str) for c in df.columns})            # strings: no pickles
        for p, m in meta['by_pass'].items():
            print(f"{name} pass {p}: balanced accuracy {m['balanced_accuracy']:.3f}, {m['up_bytes_per_query']:.0f} B/query, "
                  f"full frame on {m['full_frame_rate']:.1%}, thumbnail on {m.get('thumb_rate', 0):.1%} of queries, "
                  f"{m['reconfigs']} configs; final code {meta['bits']} bits from {meta['pixels_read']} pixels "
                  f"({meta['seconds']}s)", flush=True)


if __name__ == '__main__':
    main()
