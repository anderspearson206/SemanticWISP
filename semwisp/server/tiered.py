"""Tiered checks for the adaptive loop: check the 12-byte code against a 192-byte thumbnail first, and fetch the
19,200-byte full frame only when the thumbnail cannot settle the question.

In AdaptiveServer every check is a full frame, and those checks are almost all of the uplink (a full frame is
1,600 times the code). Here a check has up to three steps:

  code    every query: the K-bit interval code (one EPC for K = 96).
  thumb   every `interval`-th query (the same disagreement-driven schedule as AdaptiveServer): a 16x12 8-bit
          thumbnail, scored by ThumbDecoder, a room decoder the server learns online from the teacher's labels.
  full    the full frame for the teacher, only if the thumbnail decoder is not ready yet, is unsure, disagrees
          with the code, or `full_every` thumbnail checks have passed without one (so labels keep coming).
          A code reply with an unusually large share of bits set (the room no longer matches the code, see
          adaptive_loop.ipynb section 4) also goes straight to a full frame, so a change is noticed quickly.

The answer comes from the most informative step reached: the teacher on a full frame, else a confident
thumbnail, else the code.
"""
import numpy as np

from .. import bits as B
from ..piropo import H, W
from ..sim import Request
from ..tag.thumbnail import block_mean
from .adaptive import AdaptiveServer


class ThumbDecoder:
    """Room decoder for w x h thumbnails, learned from labelled frames with running statistics only.

    Features: |thumbnail - empty-room thumbnail|, where the empty-room thumbnail is a running mean over frames
    labelled empty. Decoder: linear discriminant from running class means and a shrunk covariance, with a
    two-parameter logistic calibration fitted online (one gradient step per labelled frame), so that its
    probability can decide whether the thumbnail is trusted. Everything fades with a half-life of `half_life`
    labelled frames; reset() starts again (called when the learner detects a change of room)."""

    def __init__(self, w=16, h=12, half_life=300, shrink=0.2, min_per_class=20, lr=0.05):
        self.w, self.h, self.shrink, self.min_per_class, self.lr = w, h, shrink, min_per_class, lr
        self.alpha = 1 - 0.5 ** (1 / half_life)
        self.reset()

    def reset(self):
        d = self.w * self.h
        self.n, self.bg = [0, 0], None
        self.m, self.S, self.seen = np.zeros((2, d)), np.zeros((d, d)), 0
        self.wv = None
        self.a, self.b = 1.0, 0.0                    # calibration: p = sigmoid(a * score + b)

    def _feat(self, thumb):
        return np.abs(thumb.astype(float).ravel() - self.bg)

    @property
    def ready(self):
        return self.wv is not None and min(self.n) >= self.min_per_class

    def score(self, thumb):
        """Uncalibrated discriminant, or None before the decoder has seen both classes."""
        if self.wv is None or self.bg is None:
            return None
        return float(self._feat(thumb) @ self.wv - self.c)

    def prob(self, thumb):
        s = self.score(thumb)
        return None if s is None else float(1 / (1 + np.exp(-np.clip(self.a * s + self.b, -30, 30))))

    def observe(self, thumb, label):
        x = thumb.astype(float).ravel()
        if self.bg is None:
            if label == 1:
                return                               # no background yet: nothing to compare with
            self.bg = x.copy()
        s = self.score(thumb)
        if s is not None:                            # calibrate on frames the discriminant has not seen yet
            p = 1 / (1 + np.exp(-np.clip(self.a * s + self.b, -30, 30)))
            g = p - label
            self.a -= self.lr * g * s / (1 + abs(s))
            self.b -= self.lr * g
            self.a = max(self.a, 1e-3)
        f = self._feat(thumb)
        d = f - self.m[label]
        self.m[label] += max(self.alpha, 1 / (self.n[label] + 1)) * d
        self.S += max(self.alpha, 1 / (self.seen + 1)) * (np.outer(d, f - self.m[label]) - self.S)
        self.seen += 1
        if label == 0:
            self.bg += max(self.alpha, 1 / (self.n[0] + 1)) * (x - self.bg)
        self.n[label] += 1
        if min(self.n) >= 2:
            S = (1 - self.shrink) * self.S + self.shrink * np.diag(self.S.diagonal() + 1.0)
            self.wv = np.linalg.solve(S, self.m[1] - self.m[0])
            self.c = float(self.wv @ (self.m[0] + self.m[1]) / 2)


class TieredServer(AdaptiveServer):
    def __init__(self, teacher, learner, thumb=(16, 12), confident=0.9, full_every=4, anomaly_frac=0.4,
                 anomaly_gap=4, code_margin=0.0, **kw):
        """thumb: (w, h) of the check thumbnail. confident: the thumbnail decoder's probability (either way) at
        which a thumbnail check is trusted without a full frame. full_every: at most this many trusted thumbnail
        checks in a row before a full frame is fetched anyway. anomaly_frac: a code reply with more than this
        share of its bits set goes straight to a full frame (at most once per `anomaly_gap` queries).
        code_margin: a code answer with |logit| below this also asks for a thumbnail, outside the schedule (a
        progressive query: only uncertain answers pay for more bytes); a confident thumbnail then answers, even
        against the code, and an unsure one leaves the code's answer. Such a thumbnail is not a check: it never
        leads to a full frame. Other arguments as AdaptiveServer."""
        super().__init__(teacher, learner, **kw)
        self.tw, self.th = thumb
        self.dec = ThumbDecoder(*thumb)
        self.confident, self.full_every, self.code_margin = confident, full_every, code_margin
        self.anomaly_frac, self.anomaly_gap = anomaly_frac, anomaly_gap
        self.trusted_run, self.since_anomaly = 0, anomaly_gap

    def begin(self, t):
        """As AdaptiveServer, but a deployed query starts with the code alone; checks are added in receive()."""
        self.t, self.logit, self.full, self.reconfig, self.changed = t, None, None, False, False
        self.plan = ['config'] if self.pending is not None else []
        if self.active is None and self.pending is None:
            self.plan.append('full')
        else:
            self.plan.append('code')
            self.since += 1
        self.p_thumb, self.thumbed, self.anomaly, self.scheduled = None, False, False, False
        self.why = 'bootstrap' if self.plan[-1] == 'full' else ''      # why a full frame was fetched
        self.since_anomaly += 1

    def request(self, round):
        if round >= len(self.plan):
            return None
        op = self.plan[round]
        if op == 'thumb':
            return Request('thumb', dict(w=self.tw, h=self.th))
        return super().request(round)

    def receive(self, request, rx):
        if request.op == 'code':
            super().receive(request, rx)
            if not self.adapt:
                return
            k = len(self.active[0].cell)
            if (B.unpack(rx.data, 1, k).mean() > self.anomaly_frac and self.since_anomaly >= self.anomaly_gap):
                self.plan.append('full')
                self.anomaly, self.since_anomaly, self.since, self.why = True, 0, 0, 'anomaly'
            elif self.since >= self.interval:
                self.plan.append('thumb')
                self.since, self.scheduled = 0, True
            elif abs(self.logit) < self.code_margin and self.dec.ready:
                self.plan.append('thumb')
        elif request.op == 'thumb':
            self.thumbed = True
            thumb = rx.data.reshape(self.th, self.tw)
            p = self.dec.prob(thumb) if self.dec.ready else None
            sure = p is not None and max(p, 1 - p) >= self.confident
            if not self.scheduled:      # settles an uncertain code answer; never escalates (it is not a check)
                self.p_thumb = p if sure else None
            elif sure and (p >= 0.5) == (self.logit > 0) and self.trusted_run < self.full_every:
                self.p_thumb = p
                self.trusted_run += 1
                self.disagree += 0.1 * (0.0 - self.disagree)
                self.interval = int(np.clip(round(0.25 / max(self.disagree, 1e-3)), 2, self.max_interval))
            else:
                self.why = ('unsure' if not sure else 'disagree' if (p >= 0.5) != (self.logit > 0) else 'floor')
                self.p_thumb = p if sure else None
                self.plan.append('full')
        elif request.op == 'full':
            self.trusted_run = 0
            frame = rx.data.reshape(H, W)
            super().receive(request, rx)
            if self.changed:
                self.dec.reset()
            elif self.full >= self.pos_thr or self.full <= self.neg_thr:
                self.dec.observe(block_mean(frame, self.tw, self.th), int(self.full >= self.pos_thr))
        else:
            super().receive(request, rx)

    def decide(self):
        score = super().decide()                     # also the learning step: make a new code when due
        if self.full is None and self.p_thumb is not None:
            self.answer = self.p_thumb >= 0.5        # a trusted thumbnail answers the query
            return self.p_thumb
        return score

    def info(self):
        d = super().info()
        d.update(thumbed=self.thumbed, anomaly=self.anomaly, why=self.why, used_code=self.logit is not None)
        return d
