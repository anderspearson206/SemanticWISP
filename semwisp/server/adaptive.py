"""Adaptive loop, server side: learn a room-specific code from full frames labelled by a pretrained teacher,
send it to the tag, then keep checking it against occasional full frames.

  bootstrap   every query asks for the full frame; the teacher labels it.
  deployed    every query asks for the compact code. Every `interval`-th query also asks for the full frame
              (a "check"): the teacher's label is compared with the code's answer and passed to the learner.
              The more they disagree, the more often the server checks. After `relearn_every` new labelled
              frames the learner produces a new code and the server reconfigures the tag.

Two learners produce the code:
  OnlineLearner   the method: constant-time update per labelled frame, old evidence fades, no stored frames
                  and no fitting step. It can run indefinitely and follow the room as it changes.
  BatchLearner    the baseline: stores recent labelled frames and refits a model on them.
"""
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression

from .. import bits as B
from ..piropo import H, W
from ..sim import Request, Server
from ..tag.interval import IntervalCode, cell_sums, code_bits, n_samples, CELLS


def _intervals(center, spread, table):
    """An IntervalCode over the whole table: each region's usual range in the empty room."""
    half = np.clip(spread, 5, 20)
    return IntervalCode(np.arange(len(table)), np.clip(np.floor(center - half), 0, 255).astype(np.int64),
                        np.clip(np.ceil(center + half), 0, 255).astype(np.int64), table)


def learn_interval_code(sums, y, k, table=CELLS):
    """Batch fit. sums: (n, len(table)) region sums of labelled frames; y: 0/1. Returns (IntervalCode, decoder).

    Each region's interval is centred on its median over the empty frames, with a half-width of three robust
    standard deviations (kept between 5 and 20 gray levels). An L1-penalised logistic regression then picks
    the k regions whose bits predict the labels best, and a plain logistic regression on those is the decoder."""
    mean = sums / n_samples(table)
    med = np.median(mean[y == 0], 0)
    full = _intervals(med, 3 * 1.4826 * np.median(np.abs(mean[y == 0] - med), 0), table)
    bits = code_bits(sums, full)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        l1 = LogisticRegression(penalty='l1', solver='liblinear', C=0.5, random_state=0).fit(bits, y)
        sel = np.sort(np.argsort(-np.abs(l1.coef_[0]), kind='stable')[:k])
        dec = LogisticRegression(C=1.0, max_iter=1000).fit(bits[:, sel], y)
    return IntervalCode(sel, full.lo[sel], full.hi[sel], table), dec


class BatchLearner:
    """Baseline: keep the last `window` labelled frames and refit on them."""

    def __init__(self, k=96, window=1000, table=CELLS):
        self.k, self.window, self.table = k, window, table
        self.sums, self.labels = [], []

    def observe(self, frame, label):
        self.sums.append(cell_sums(frame, self.table)); self.labels.append(label)
        del self.sums[:-self.window], self.labels[:-self.window]

    def counts(self):
        n1 = int(sum(self.labels))
        return len(self.labels) - n1, n1

    def make(self):
        return learn_interval_code(np.array(self.sums), np.array(self.labels), self.k, self.table)


class Linear:
    """Decoder for the online learner: a weight per bit and an offset."""

    def __init__(self, coef, intercept):
        self.coef_, self.intercept_ = coef[None], intercept

    def decision_function(self, bits):
        return bits @ self.coef_[0] + self.intercept_


def gauss_cross(mu0, s0, mu1, s1):
    """The score between mu0 and mu1 where two equal-weight Gaussians N(mu0, s0) and N(mu1, s1) cross (the
    balanced-error threshold if the classes' decoder outputs are Gaussian); the midpoint if there is none."""
    lo, hi = min(mu0, mu1), max(mu0, mu1)
    if abs(s0 - s1) < 1e-9 * max(s0, s1):
        return (mu0 + mu1) / 2
    a = 1 / s0 ** 2 - 1 / s1 ** 2
    b = -2 * (mu0 / s0 ** 2 - mu1 / s1 ** 2)
    c = mu0 ** 2 / s0 ** 2 - mu1 ** 2 / s1 ** 2 + 2 * np.log(s0 / s1)
    roots = np.roots([a, b, c])
    roots = roots[np.isreal(roots)].real
    inside = roots[(roots > lo) & (roots < hi)]
    return float(inside[0]) if len(inside) else (mu0 + mu1) / 2


class OnlineLearner:
    """Constant-time learning from a stream of labelled frames; no frames are stored and nothing is refitted.

    All it keeps are running statistics, exponentially weighted so that evidence fades with a half-life of
    `half_life` labelled frames and the model follows the room as it changes:
      mu, dev    per candidate region, mean gray level and mean absolute deviation over frames labelled empty.
                 They define the region's usual range, and so its bit: outside the range or not.
      m0, m1     how often each bit is set in empty frames and in person frames.
      S          covariance of the bits around their class means (which bits tend to be set together).

    make() works from those statistics alone. It picks regions one at a time, each time the one that adds the
    most separation between the two classes *given the regions already picked* (so a region that repeats what
    a chosen one says is worth nothing), and the decoder is the linear discriminant of the picked bits.
    Memory and time per labelled frame grow with the square of the number of candidates.

    Decoder options (defaults reproduce the original learner):
      shrink     shrink the covariance towards its diagonal before solving for the decoder weights (0..1). With
                 binary bits that are rarely set, the plain covariance gives such bits huge weights.
      threshold  'midpoint': halfway between the two class means of the decoder output (the original);
                 'gauss': where the two classes' Gaussians of the decoder output cross, with each class's spread
                 from its own running covariance (kept only for this option). Both use running statistics only.
      In experiments/learner_ablation.py (true labels, held-out time blocks) shrink=0.3 with 'gauss' raised the Mall
      tile from 0.89 / 0.86 to 0.96 / 0.95 balanced accuracy (cells / pixels) and room A2 cells from 0.78 to 0.85.

    Change detection: fading is too slow for an abrupt change (the tag is moved, the furniture is rearranged).
    If `change_after` empty-labelled frames in a row have more than `change_frac` of all candidate bits set,
    the background it learned no longer describes the room: it discards its statistics, raises `changed`, and
    starts again from that frame."""

    def __init__(self, k=96, table=CELLS, half_life=200, ridge=1e-3, change_frac=0.3, change_after=3,
                 shrink=0.0, threshold='midpoint'):
        assert len(table) <= 2000, 'the covariance is len(table) squared'
        assert threshold in ('midpoint', 'gauss')
        self.k, self.table, self.ridge = k, table, ridge
        self.shrink, self.threshold = shrink, threshold
        self.change_frac, self.change_after = change_frac, change_after
        self.alpha = 1 - 0.5 ** (1 / half_life)
        self.area = n_samples(table)
        self.changed = False
        self._start()

    def _start(self):
        self.n = [0, 0]
        self.mu = self.dev = None
        self.m = np.zeros((2, len(self.table)))
        self.S = np.zeros((len(self.table), len(self.table)))
        self.Sc = np.zeros((2, len(self.table), len(self.table))) if self.threshold == 'gauss' else None
        self.seen = self.odd = 0                       # labelled frames in m and S; odd-looking empties in a row

    def _full(self):
        return _intervals(self.mu, 3 * 1.25 * self.dev, self.table)       # 1.25 x mean abs. deviation ~ 1 sigma

    def observe(self, frame, label):
        sums = cell_sums(frame, self.table)
        if self.mu is not None:
            x = code_bits(sums, self._full()).astype(float)
            if label == 0:
                self.odd = self.odd + 1 if x.mean() > self.change_frac else 0
                if self.odd >= self.change_after:
                    self._start()
                    self.changed = True
        if self.mu is not None:
            d = x - self.m[label]
            self.m[label] += max(self.alpha, 1 / (self.n[label] + 1)) * d          # plain averages at first
            self.S += max(self.alpha, 1 / (self.seen + 1)) * (np.outer(d, x - self.m[label]) - self.S)
            if self.Sc is not None:                   # per-class covariance, for the 'gauss' threshold
                self.Sc[label] += max(self.alpha, 1 / (self.n[label] + 1)) * (np.outer(d, x - self.m[label]) - self.Sc[label])
            self.seen += 1
        if label == 0:
            mean = sums / self.area
            if self.mu is None:
                self.mu, self.dev = mean.copy(), np.full(len(mean), 2.0)
            else:
                a = max(self.alpha, 1 / (self.n[0] + 1))
                d = mean - self.mu
                self.mu += a * d
                self.dev += a * (np.abs(d) - self.dev)
        self.n[label] += 1

    def counts(self):
        return tuple(self.n)

    def make(self):
        S = self.S + self.ridge * np.eye(len(self.S))
        gap = self.m[1] - self.m[0]
        r, A, sel = gap.copy(), S.copy(), []           # r, A: gap and covariance with the picked regions removed
        free = np.ones(len(r), bool)
        while len(sel) < self.k:
            gain = np.where(free & (A.diagonal() > self.ridge / 2), r ** 2 / np.maximum(A.diagonal(), 1e-12), 0)
            j = int(gain.argmax())
            if gain[j] < 1e-9:                         # nothing left that adds separation
                break
            sel.append(j); free[j] = False
            col = A[:, j] / A[j, j]
            r = r - col * r[j]
            A = A - np.outer(col, A[j])
        sel = np.sort(sel)
        Ss = S[np.ix_(sel, sel)]
        if self.shrink:
            Ss = (1 - self.shrink) * Ss + self.shrink * np.diag(Ss.diagonal() + 0.01)
        w = np.linalg.solve(Ss, gap[sel])
        mu0, mu1 = w @ self.m[0, sel], w @ self.m[1, sel]
        cut = (mu0 + mu1) / 2
        C = self.Sc[:, sel][:, :, sel] if self.Sc is not None else np.stack([S[np.ix_(sel, sel)]] * 2)
        sd = [np.sqrt(max(w @ C[c] @ w, 1e-12)) for c in (0, 1)]
        if self.threshold == 'gauss':
            cut = gauss_cross(mu0, sd[0], mu1, sd[1])
        full = self._full()
        dec = Linear(w, -cut)
        # Per-class statistics of what the server receives, for a server that models them (TemporalServer):
        # the decoder output (mean, sd) and the share of code bits set (mean, sd), each for empty and person.
        dec.out_stats = ((mu0 - cut, sd[0]), (mu1 - cut, sd[1]))
        k = len(sel)
        dec.frac_stats = tuple((float(self.m[c, sel].mean()), float(np.sqrt(max(C[c].sum(), 0)) / k)) for c in (0, 1))
        return IntervalCode(sel, full.lo[sel], full.hi[sel], self.table), dec


class CachedTeacher:
    """Pretrained person detector applied to a full 160x120 frame, looked up from scores computed beforehand,
    keyed by capture time. A score of None means the teacher has no opinion on that frame."""

    def __init__(self, times, scores):
        self.table = dict(zip(np.asarray(times).tolist(), np.asarray(scores, float).tolist()))

    def __call__(self, t, frame):
        return self.table[float(t)]


class AdaptiveServer(Server):
    def __init__(self, teacher, learner, bootstrap=200, relearn_every=50, pos_thr=0.6, neg_thr=0.3,
                 max_interval=64, adapt=True, noise_floor=0.0, full_jpeg=None):
        """teacher(t, frame) -> person score in [0, 1]. A full frame is used as a label only when the teacher
        is confident: score >= pos_thr (person) or <= neg_thr (empty).
        adapt=False: learn once after the bootstrap, then never check or relearn.
        Check schedule: every `interval` queries, interval = 0.25 / disagreement (running rate at which the code
        and the teacher disagree), between 2 and max_interval. noise_floor is subtracted from the disagreement
        first: the share of disagreements expected from the teacher's own errors, which a perfect code would
        still show (default 0: the original schedule).
        full_jpeg: JPEG quality the tag sends full frames at (IntervalCodeTag(full_jpeg=...)); the server decodes
        them, and the teacher and learner see the decoded frame. None: raw frames."""
        self.teacher, self.learner, self.bootstrap, self.relearn_every = teacher, learner, bootstrap, relearn_every
        self.pos_thr, self.neg_thr, self.max_interval, self.adapt = pos_thr, neg_thr, max_interval, adapt
        self.noise_floor = noise_floor
        self.jpeg = None
        if full_jpeg is not None:
            from ..tag.jpeg import JpegTag
            self.jpeg = JpegTag(W, H, full_jpeg)
        self.active = self.pending = None          # (IntervalCode, decoder) in use / learned but not yet sent
        self.new = self.since = 0                  # labelled frames since the last code; queries since a check
        self.disagree, self.interval = 0.25, 2     # running disagreement rate between code and teacher

    def begin(self, t):
        self.t, self.logit, self.full, self.reconfig, self.changed = t, None, None, False, False
        self.plan = []
        if self.pending is not None:
            self.plan.append('config')
        if self.active is None and self.pending is None:
            self.plan.append('full')
        else:
            self.plan.append('code')
            self.since += 1
            if self.adapt and self.since >= self.interval:
                self.plan.append('full')
                self.since = 0

    def request(self, round):
        if round >= len(self.plan):
            return None
        op = self.plan[round]
        if op == 'config':
            return Request('config', dict(code=self.pending[0]), n_bits=self.pending[0].n_bits)
        return Request(op)

    def receive(self, request, rx):
        if request.op == 'config':
            self.active, self.pending, self.reconfig = self.pending, None, True
        elif request.op == 'code':
            code, dec = self.active
            self.logit = float(dec.decision_function(B.unpack(rx.data, 1, len(code.cell))[None].astype(float))[0])
        else:
            frame = self.frame_of(rx)
            self.full = self.teacher(self.t, frame)
            label = 1 if self.full >= self.pos_thr else 0 if self.full <= self.neg_thr else None
            if label is None:
                return
            self.learner.observe(frame, label); self.new += 1
            if getattr(self.learner, 'changed', False):            # the room changed: back to the bootstrap
                self.learner.changed, self.changed = False, True
                self.active = self.pending = None
                self.new, self.disagree, self.interval = 0, 0.25, 2
            elif self.logit is not None:
                self.disagree += 0.1 * (float((self.logit > 0) != label) - self.disagree)
                self.set_interval()

    def frame_of(self, rx):
        """The full frame a 'full' reply carries, decoded if the tag sent it as JPEG."""
        if self.jpeg is None:
            return rx.data.reshape(H, W)
        return self.jpeg.decode(rx.data).reshape(H, W)

    def set_interval(self):
        self.interval = int(np.clip(round(0.25 / max(self.disagree - self.noise_floor, 1e-3)), 2, self.max_interval))

    def decide(self):
        """Score in [0, 1]. When the full frame was fetched this query, the teacher's answer is used."""
        score = self.full if self.full is not None else 1 / (1 + np.exp(-self.logit))
        self.answer = (self.full >= (self.pos_thr + self.neg_thr) / 2) if self.full is not None else self.logit > 0
        n0, n1 = self.learner.counts()
        ready = n0 + n1 >= self.bootstrap if self.active is None else self.adapt and self.new >= self.relearn_every
        if ready and self.pending is None and min(n0, n1) >= 10:
            self.pending = self.learner.make()
            self.new = 0
        return score

    def info(self):
        return dict(answer=int(self.answer), explored=self.full is not None, used_code=self.logit is not None,
                    reconfig=self.reconfig, interval=self.interval, changed=self.changed)
