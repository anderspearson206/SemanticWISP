"""Online adaptation: how few pixels does the tag need, and how long does the loop take to (re)adapt?

The server learns only from running statistics (semwisp.server.OnlineLearner): nothing is stored and nothing
is refitted, so the same loop runs unchanged in any room. BatchLearner (store frames, refit) is the baseline.

Streams
  piropo   the five annotated PIROPO sequences in recording order, every 2nd frame, replayed twice.
           Teacher: the cached COCO detector on the 160x120 frame, or the ground truth ("oracle").
  switch   the tag is moved: PIROPO `training` -> one tile of the Mall dataset (a different scene) -> PIROPO
           `training_seats` + `test2` (the first room again, with the furniture rearranged). One pass. Oracle
           teacher only, because no detector scores exist for the Mall tile; it measures how fast the loop can
           readapt given correct labels, not how well a detector supervises it.

Candidate regions (what the tag may be told to look at), and the pixels it must read for a 96-bit code
  cells       blocks of the 4x3, 8x6 and 16x12 grids, every pixel of each block (up to 19,200 pixels)
  cells_s5    the same blocks, one pixel per 5x5 patch (at most 768)
  cells_s10   the same blocks, one pixel per 10x10 patch (at most 192)
  pixels      single pixels on a 4-pixel lattice (96: one pixel per bit)

The online learner also detects an abrupt change (empty frames that no longer match its background) and
starts again from full frames; the batch baseline has no such step and just keeps refitting its window.

Results: results/online_adaptation/<stream>_<teacher>_<learner>_<regions>.npz, one row per query; existing
files are skipped. Per-room readaptation times are computed from the rows by semwisp.metrics.readaptation.

  python experiments/online_adaptation.py
  python experiments/online_adaptation.py --only switch
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from semwisp import piropo
from semwisp.frames import load_frames, load_mall_tile
from semwisp.metrics import summary
from semwisp.server import AdaptiveServer, BatchLearner, CachedTeacher, OnlineLearner
from semwisp.sim import replay
from semwisp.tag import IntervalCodeTag
from semwisp.tag.interval import CELLS, lattice, sparse

ORDER = ['training', 'test3', 'test1', 'training_seats', 'test2']
STRIDE = 2
DETECTOR = piropo.REPO / 'results' / 'coco_detectors' / 'frcnn_mnv3_320_160x120_s2.npz'
REGIONS = dict(cells=CELLS, cells_s5=sparse(CELLS, 5), cells_s10=sparse(CELLS, 10), pixels=sparse(lattice(4), 4))
RUNS = ([('piropo', t, 'online', r) for t in ('oracle', 'detector') for r in REGIONS]
        + [('piropo', t, 'batch', 'cells') for t in ('oracle', 'detector')]
        + [('switch', 'oracle', 'online', r) for r in REGIONS] + [('switch', 'oracle', 'batch', 'cells')])


def every(s, k):
    return {f: v[::k] for f, v in s.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', help='run only configurations whose name contains this text')
    ap.add_argument('--k', type=int, default=96, help='bits in the code (96 = one EPC)')
    ap.add_argument('--half-life', type=int, default=200, help='OnlineLearner memory, in labelled frames')
    ap.add_argument('--out', type=Path, default=piropo.REPO / 'results' / 'online_adaptation')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    P = {n: every(s, STRIDE) for n, s in load_frames(ORDER).items()}
    mall = load_mall_tile()
    det = np.load(DETECTOR)
    t = np.concatenate([P[n]['t'] for n in ORDER] + [mall['t']])
    truth = np.concatenate([P[n]['label'] for n in ORDER] + [mall['label']]).astype(float)
    teachers = dict(oracle=CachedTeacher(t, np.where(truth < 0, 0.45, truth)),     # 0.45: no opinion
                    detector=CachedTeacher(np.concatenate([P[n]['t'] for n in ORDER]),
                                           np.concatenate([det[f'score_{n}'] for n in ORDER])))
    streams = dict(piropo=([P] * 2, ORDER),
                   switch=([{'room A (training)': P['training'], 'room B (mall tile)': mall,
                             'room A, rearranged': P['training_seats'], 'room A, rearranged (test2)': P['test2']}],
                           None))

    for stream, teacher, kind, regions in RUNS:
        name = f'{stream}_{teacher}_{kind}_{regions}'
        out = args.out / f'{name}.npz'
        if args.only and args.only not in name:
            continue
        if out.exists():
            print(f'skip {name} (exists)')
            continue
        t0 = time.time()
        learner = (OnlineLearner(k=args.k, table=REGIONS[regions], half_life=args.half_life) if kind == 'online'
                   else BatchLearner(k=args.k, table=REGIONS[regions]))
        server, tag, parts = AdaptiveServer(teachers[teacher], learner), IntervalCodeTag(), []
        for p, seqs in enumerate(streams[stream][0], 1):
            parts.append(replay(seqs, tag, server, reset=False).df.assign(**{'pass': p}))
        df = pd.concat(parts, ignore_index=True)
        code = server.active[0]
        meta = dict(name=name, stream=stream, teacher=teacher, learner=kind, regions=regions, k=args.k,
                    half_life=args.half_life, bits=len(code.cell), pixels_read=code.n_pixels,
                    seconds=round(time.time() - t0), by_pass={p: summary(g) for p, g in df.groupby('pass')})
        np.savez(out, meta=json.dumps(meta), code_rects=code.rects, code_lo=code.lo, code_hi=code.hi,
                 **{c: np.array(df[c].tolist(), dtype=str) if c == 'seq' else df[c].values for c in df.columns})
        for p, m in meta['by_pass'].items():
            print(f"{name} pass {p}: balanced accuracy {m['balanced_accuracy']:.3f}, {m['up_bytes_per_query']:.0f} B/query, "
                  f"full frame on {m['full_frame_rate']:.1%} of queries, {m['reconfigs']} configs; "
                  f"final code {meta['bits']} bits from {meta['pixels_read']} pixels ({meta['seconds']}s)", flush=True)


if __name__ == '__main__':
    main()
