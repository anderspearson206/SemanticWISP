"""Supervised ceiling for the 96-region code: what the code could do in a room if every frame's true label were known.

The adaptive loop learns its code online from a teacher. To say how close it gets ("within X of the supervised code
after N queries"), this fits the same kind of code with true labels on the room's own frames and scores it on
held-out time, with the same arithmetic the tag uses (semwisp.tag.interval.code_bits).

Rooms (frames as the replays see them: every 2nd frame, time order):
  A      room A as recorded: training, test3, test1 (the order of the PIROPO stream)
  A2     room A rearranged: training_seats, test2
  mall   the Mall tile of the switch stream (labels -1 are ambiguous: neither fitted nor scored)
Each room is cut into --folds contiguous time blocks. Each block is scored by a code fitted on the other blocks,
leaving out --gap frames on each side of the held-out block (adjacent frames are near-duplicates).
Corrupted frames (`bad`) are never fitted on; they are scored, as in the replays, and flagged in the results.

Codes (k = 96 regions, as the loop):
  tables   pixels: single pixels on a 4-pixel lattice (1,200 candidates, the online_pixels learner's table);
           cells: whole blocks of 4x3, 8x6 and 16x12 grids (252 candidates, CELLS)
  fitters  batch: semwisp.server.adaptive.learn_interval_code (the BatchLearner's fit, L1 selection);
           online: the OnlineLearner's statistics fed every training frame, without fading or change detection,
           then make() (the method's own selection rule with all the labels it could ever get)

Results: results/supervised_ceiling/<room>_<fitter>_<table>.npz with per-frame seq, idx, t, label, bad, fold, logit
(held-out decoder output; answer = logit > 0); existing files are skipped. Labels are the annotations (what the
replays score against).

  python experiments/supervised_ceiling.py
"""
import argparse, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from semwisp import piropo
from semwisp.frames import load_frames, load_mall_tile
from semwisp.server.adaptive import OnlineLearner, learn_interval_code
from semwisp.tag.interval import CELLS, cell_sums, code_bits, lattice, sparse

STRIDE = 2
ROOMS = {'A': ['training', 'test3', 'test1'], 'A2': ['training_seats', 'test2'], 'mall': ['mall_tile']}
TABLES = {'pixels': sparse(lattice(4), 4), 'cells': CELLS}
OUT = piropo.REPO / 'results' / 'supervised_ceiling'


def room_frames(names):
    """Concatenated frames of a room, in time order: dict(X, t, label, bad, seq, idx)."""
    parts = []
    for n in names:
        s = load_mall_tile() if n == 'mall_tile' else load_frames([n])[n]
        idx = np.arange(0, len(s['X']), 1 if n == 'mall_tile' else STRIDE)    # the replays use every Mall frame
        parts.append(dict(X=s['X'][idx], t=s['t'][idx], label=s['label'][idx].astype(int), bad=s['bad'][idx],
                          seq=np.full(len(idx), n), idx=idx))
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def fit(fitter, sums, frames, y, table, k):
    """(code, decoder) from labelled training frames. frames: the raw frames (the online learner reads them)."""
    if fitter == 'batch':
        return learn_interval_code(sums, y, k, table)
    L = OnlineLearner(k=k, table=table, half_life=1e12, change_frac=2.0)      # no fading, no change detection
    for f, label in zip(frames, y):
        L.observe(f, int(label))
    return L.make()


def run(room, fitter, table_name, folds, gap, k):
    R = room_frames(ROOMS[room])
    table = TABLES[table_name]
    sums = cell_sums(R['X'], table)
    n = len(R['t'])
    fold = np.minimum(np.arange(n) * folds // n, folds - 1)
    logit = np.full(n, np.nan)
    for f in range(folds):
        test = fold == f
        lo, hi = np.flatnonzero(test)[[0, -1]]
        near = (np.arange(n) >= lo - gap) & (np.arange(n) <= hi + gap)
        train = ~near & ~R['bad'] & (R['label'] >= 0)
        y = R['label'][train]
        if y.min() == y.max():
            print(f'  {room} fold {f}: training blocks hold one class only, skipped', flush=True)
            continue
        code, dec = fit(fitter, sums[train], R['X'][train], y, table, k)
        logit[test] = dec.decision_function(code_bits(sums[test], code).astype(float))
    return dict(seq=R['seq'], idx=R['idx'], t=R['t'], label=R['label'], bad=R['bad'], fold=fold, logit=logit)


def balanced_accuracy(answer, label):
    ok = label >= 0
    a, y = answer[ok], label[ok]
    return (a[y == 1].mean() + (1 - a[y == 0]).mean()) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rooms', nargs='+', default=list(ROOMS))
    ap.add_argument('--fitters', nargs='+', default=['batch', 'online'])
    ap.add_argument('--tables', nargs='+', default=list(TABLES))
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--gap', type=int, default=10, help='frames left out on each side of a held-out block')
    ap.add_argument('--k', type=int, default=96)
    ap.add_argument('--out', type=Path, default=OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for room in args.rooms:
        for fitter in args.fitters:
            for table in args.tables:
                out = args.out / f'{room}_{fitter}_{table}.npz'
                if out.exists():
                    print(f'skip {out.name} (exists)', flush=True)
                    continue
                t0 = time.time()
                r = run(room, fitter, table, args.folds, args.gap, args.k)
                tmp = out.with_name(out.stem + '.tmp.npz')
                np.savez(tmp, **r, folds=args.folds, gap=args.gap, k=args.k)
                tmp.replace(out)
                ok = ~np.isnan(r['logit'])
                ans = (r['logit'] > 0).astype(int)
                print(f"{room} {fitter} {table}: balanced accuracy {balanced_accuracy(ans[ok], r['label'][ok]):.3f} "
                      f"(corrupted frames excluded: {balanced_accuracy(ans[ok & ~r['bad']], r['label'][ok & ~r['bad']]):.3f}), "
                      f"AUC {piropo.auc(r['logit'][ok & (r['label'] >= 0)], r['label'][ok & (r['label'] >= 0)]):.3f}, "
                      f"{ok.sum()} of {len(ok)} frames scored ({time.time() - t0:.0f}s)", flush=True)


if __name__ == '__main__':
    main()
