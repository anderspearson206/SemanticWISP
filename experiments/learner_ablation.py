"""Which part of the online learner loses accuracy? Its three parts swapped one at a time against the batch fit's.

supervised_ceiling.py found that the OnlineLearner's code, built from statistics of *every* labelled frame of a room,
is far worse than the batch fit's (learn_interval_code) on the Mall tile and in room A2, so its gap is not only
about labels or adaptation. The code has three parts, and the two learners differ in each:
  intervals  median: each region's range is the median over empty frames +- 3 x 1.4826 x MAD (batch)
             mean:   mean over empty frames +- 3 x 1.25 x mean absolute deviation (online, without fading)
  selection  l1:     L1-penalised logistic regression, the k largest |coefficients| (batch)
             greedy: greedy discriminant gain given the regions already picked (online, OnlineLearner.make)
  decoder    logreg: logistic regression on the selected bits (batch)
             lda:    linear discriminant of the selected bits (online)
All 2 x 2 x 2 combinations, fitted with true labels on the same held-out time blocks as supervised_ceiling.py
(5 contiguous blocks, 10-frame gaps, corrupted frames never fitted on). k = 96.

Results: results/learner_ablation/results.csv (room, table, intervals, selection, decoder, balanced accuracy, AUC);
recomputed in full each run (about a minute).

  python experiments/learner_ablation.py
"""
import sys, warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from semwisp import piropo
from semwisp.server.adaptive import _intervals
from semwisp.tag.interval import cell_sums, code_bits, n_samples
from supervised_ceiling import ROOMS, TABLES, balanced_accuracy, room_frames

OUT = piropo.REPO / 'results' / 'learner_ablation'
K, FOLDS, GAP, RIDGE = 96, 5, 10, 1e-3


def intervals(kind, mean_empty, table):
    if kind == 'median':
        c = np.median(mean_empty, 0)
        return _intervals(c, 3 * 1.4826 * np.median(np.abs(mean_empty - c), 0), table)
    c = mean_empty.mean(0)
    return _intervals(c, 3 * 1.25 * np.abs(mean_empty - c).mean(0), table)


def select(kind, bits, y, k):
    if kind == 'l1':
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            l1 = LogisticRegression(penalty='l1', solver='liblinear', C=0.5, random_state=0).fit(bits, y)
        return np.sort(np.argsort(-np.abs(l1.coef_[0]), kind='stable')[:k])
    # OnlineLearner.make on exact (unfaded) statistics: class means and the pooled within-class covariance
    x = bits.astype(float)
    m = np.stack([x[y == c].mean(0) for c in (0, 1)])
    d = x - m[y]
    S = d.T @ d / len(x) + RIDGE * np.eye(x.shape[1])
    gap = m[1] - m[0]
    r, A, sel, free = gap.copy(), S.copy(), [], np.ones(len(gap), bool)
    while len(sel) < k:
        gain = np.where(free & (A.diagonal() > RIDGE / 2), r ** 2 / np.maximum(A.diagonal(), 1e-12), 0)
        j = int(gain.argmax())
        if gain[j] < 1e-9:
            break
        sel.append(j); free[j] = False
        col = A[:, j] / A[j, j]
        r = r - col * r[j]
        A = A - np.outer(col, A[j])
    return np.sort(sel)


def decoder(kind, bits, y):
    x = bits.astype(float)
    if kind == 'logreg':
        dec = LogisticRegression(C=1.0, max_iter=1000).fit(x, y)
        return lambda b: dec.decision_function(b.astype(float))
    m = np.stack([x[y == c].mean(0) for c in (0, 1)])
    d = x - m[y]
    S = d.T @ d / len(x) + RIDGE * np.eye(x.shape[1])
    w = np.linalg.solve(S, m[1] - m[0])
    c = w @ (m[0] + m[1]) / 2
    return lambda b: b.astype(float) @ w - c


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for room, (tname, table) in product(ROOMS, TABLES.items()):
        R = room_frames(ROOMS[room])
        sums = cell_sums(R['X'], table)
        mean = sums / n_samples(table)
        n = len(R['t'])
        fold = np.minimum(np.arange(n) * FOLDS // n, FOLDS - 1)
        logits = {c: np.full(n, np.nan) for c in product(('median', 'mean'), ('l1', 'greedy'), ('logreg', 'lda'))}
        for f in range(FOLDS):
            test = fold == f
            lo, hi = np.flatnonzero(test)[[0, -1]]
            train = ~((np.arange(n) >= lo - GAP) & (np.arange(n) <= hi + GAP)) & ~R['bad'] & (R['label'] >= 0)
            y = R['label'][train]
            if y.min() == y.max():
                continue
            for iv in ('median', 'mean'):
                full = intervals(iv, mean[train][y == 0], table)
                bits_tr, bits_te = code_bits(sums[train], full), code_bits(sums[test], full)
                for sk in ('l1', 'greedy'):
                    sel = select(sk, bits_tr, y, K)
                    for dk in ('logreg', 'lda'):
                        logits[iv, sk, dk][test] = decoder(dk, bits_tr[:, sel], y)(bits_te[:, sel])
        for (iv, sk, dk), lg in logits.items():
            ok = ~np.isnan(lg) & (R['label'] >= 0)
            rows.append(dict(room=room, table=tname, intervals=iv, selection=sk, decoder=dk,
                             balanced_accuracy=balanced_accuracy((lg[ok] > 0).astype(int), R['label'][ok]),
                             auc=piropo.auc(lg[ok], R['label'][ok])))
        print(f'{room} {tname} done', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'results.csv', index=False)
    pd.set_option('display.width', 200)
    print(df.pivot_table(index=['intervals', 'selection', 'decoder'], columns=['room', 'table'],
                         values='balanced_accuracy').round(3).to_string())


if __name__ == '__main__':
    main()
