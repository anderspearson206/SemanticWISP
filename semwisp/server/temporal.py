"""Tiered checks with a temporal model on the server: the server predicts the answer before the tag replies.

Compression and prediction are the same problem: what the server must receive is only what it cannot predict.
TieredServer judges every code reply on its own (|logit| < margin -> thumbnail; > 40% of bits set -> full
frame). But presence persists: in PIROPO the true label changes on 3-5% of consecutive queries, while the code's
own answer flips on 14-23%. TemporalServer keeps a running belief P(person now | everything received so far)
and asks for more bytes only when that belief, not the single reply, is uncertain.

  Belief      a two-state hidden Markov filter. Before each reply the belief moves towards 0.5 by the per-query
              change rate r; each reply then multiplies the odds by its likelihood ratio:
                code       N(logit | person) / N(logit | empty) with the decoder output's two class means and their pooled
                           variance (decoder.out_stats; linear in the logit), clipped to +-llr_cap (they are
                           in-sample fits, so overconfident in the tails);
                thumbnail  p / (1 - p) of the thumbnail decoder's calibrated probability;
                full frame the teacher's label sets the belief to teacher_trust (or 1 - teacher_trust).
  Change rate r is learned per room from pairs of teacher labels g queries apart: P(differ) = (1 - (1 - 2r)^g) / 2,
              maximum likelihood over a grid of r, with fading (half-life `r_half_life` pairs). Teacher errors
              make labels differ more often, so r errs high (less smoothing), the safe side.
  Uncertain   a thumbnail is asked for when the belief after the code reply is below `confident_belief` either
              way (instead of |logit| < margin), and a trusted thumbnail settles it as before.
  Anomaly     a code reply is an anomaly when its share of set bits is unlikely under both classes: above
              mean + anomaly_z * sd for each class (decoder.frac_stats, the learner's per-class bit rates),
              never below `anomaly_min` (the fixed 0.4 trigger of TieredServer fired on ordinary person events in
              the Mall tile, where people set far more bits than in room A).
              A real change persists: a full frame fetched for an anomaly that the previous code reply did not
              share is answered by the teacher but not learned from (hold_glitches). Without this, a corrupted
              frame of room A (test1, flat gray lower half, missed by piropo.flag_bad) fetched once per pass
              added up to three "odd empty frames" and made the learner restart from the corrupted frame.
  Answer      the teacher on a full frame, else the belief (> 0.5).

Each part can be switched off (`smooth`, `learn_anomaly`) for ablation; with both off this is TieredServer.
Needs a decoder with out_stats / frac_stats (OnlineLearner); otherwise it falls back to TieredServer's rules.
"""
import numpy as np

from .. import bits as B
from .tiered import TieredServer


def _logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return float(1 / (1 + np.exp(-np.clip(z, -30, 30))))


class ChangeRate:
    """Per-query probability that the label changes, from pairs of labels g queries apart (grid MLE, fading)."""
    GRID = np.geomspace(1e-3, 0.3, 40)

    def __init__(self, half_life=100, prior=0.02, min_pairs=10):
        self.fade, self.min_pairs, self.prior = 0.5 ** (1 / half_life), min_pairs, prior
        self.reset()

    def reset(self):
        self.ll, self.pairs, self.last = np.zeros(len(self.GRID)), 0, None

    def observe(self, q, label):
        """q: query number; label: 0/1 teacher label at that query."""
        if self.last is not None and q > self.last[0]:
            g = q - self.last[0]
            p = np.clip((1 - (1 - 2 * self.GRID) ** g) / 2, 1e-9, 1 - 1e-9)
            self.ll = self.fade * self.ll + np.log(p if label != self.last[1] else 1 - p)
            self.pairs += 1
        self.last = (q, label)

    @property
    def r(self):
        return float(self.GRID[self.ll.argmax()]) if self.pairs >= self.min_pairs else self.prior


class TemporalServer(TieredServer):
    def __init__(self, teacher, learner, smooth=True, learn_anomaly=True, confident_belief=0.88, llr_cap=6.0,
                 teacher_trust=0.97, r_half_life=100, r_prior=0.02, anomaly_z=4.0, anomaly_min=0.2,
                 hold_glitches=True, **kw):
        """confident_belief: the belief (either way) below which a code reply asks for a thumbnail; 0.88 is
        the probability at TieredServer's |logit| = 2. hold_glitches: a full frame fetched for an anomaly that
        the previous code reply did not share is answered but not learned from (a real change persists).
        Other arguments as TieredServer."""
        super().__init__(teacher, learner, **kw)
        self.smooth, self.learn_anomaly = smooth, learn_anomaly
        self.confident_belief, self.llr_cap, self.teacher_trust = confident_belief, llr_cap, teacher_trust
        self.anomaly_z, self.anomaly_min = anomaly_z, anomaly_min
        self.hold_glitches = hold_glitches
        self.rate = ChangeRate(r_half_life, r_prior)
        self.belief, self.q = 0.5, 0
        self.last_high = self.persistent = self.glitch = False

    # --- belief -------------------------------------------------------------------------------------------
    def _predict(self):
        r = self.rate.r
        self.belief = self.belief * (1 - r) + (1 - self.belief) * r

    def _update(self, llr):
        self.belief = _sigmoid(_logit(self.belief) + llr)

    def _code_llr(self, logit):
        dec = self.active[1]
        # Equal-variance Gaussian ratio: linear in the logit, zero at the decoder's own threshold (logit 0).
        # The two-variance ratio is not monotone: far on the empty side the wider class wins again, which in
        # room A held the belief at "person" on clearly empty replies.
        (m0, s0), (m1, s1) = dec.out_stats
        llr = (m1 - m0) / ((s0 ** 2 + s1 ** 2) / 2) * logit
        return float(np.clip(llr, -self.llr_cap, self.llr_cap))

    def _anomaly_cut(self):
        dec = self.active[1]
        if not self.learn_anomaly or not hasattr(dec, 'frac_stats'):
            return self.anomaly_frac
        return max(self.anomaly_min, max(m + self.anomaly_z * s for m, s in dec.frac_stats))

    @property
    def modelled(self):
        return self.smooth and hasattr(self.active[1], 'out_stats')

    # --- protocol -----------------------------------------------------------------------------------------
    def begin(self, t):
        super().begin(t)
        self.glitch = False
        self.q += 1
        self._predict()

    def receive(self, request, rx):
        if request.op == 'code':
            super(TieredServer, self).receive(request, rx)          # AdaptiveServer: sets self.logit
            if self.modelled:
                self._update(self._code_llr(self.logit))
            if not self.adapt:
                return
            k = len(self.active[0].cell)
            self.frac = B.unpack(rx.data, 1, k).mean()
            unsure = (max(self.belief, 1 - self.belief) < self.confident_belief if self.modelled
                      else abs(self.logit) < self.code_margin)
            high = self.frac > self._anomaly_cut()
            self.persistent, self.last_high = high and self.last_high, high
            if high and self.since_anomaly >= self.anomaly_gap:
                self.plan.append('full')
                self.anomaly, self.since_anomaly, self.since, self.why = True, 0, 0, 'anomaly'
            elif self.since >= self.interval:
                self.plan.append('thumb')
                self.since, self.scheduled = 0, True
            elif unsure and self.dec.ready:
                self.plan.append('thumb')
            return
        if request.op == 'full' and self.anomaly and self.hold_glitches and not self.persistent:
            # A one-query anomaly is more likely a sensor glitch (e.g. a corrupted frame) than a new room: the
            # teacher answers it, but neither the learner nor the thumbnail decoder learns from it.
            self.trusted_run, self.glitch = 0, True
            self.full = self.teacher(self.t, self.frame_of(rx))
            if self.full >= self.pos_thr or self.full <= self.neg_thr:
                self.belief = self.teacher_trust if self.full >= self.pos_thr else 1 - self.teacher_trust
            return
        super().receive(request, rx)
        if request.op == 'thumb' and self.smooth:
            thumb = rx.data.reshape(self.th, self.tw)
            p = self.dec.prob(thumb) if self.dec.ready else None
            if p is not None:
                self._update(float(np.clip(_logit(p), -self.llr_cap, self.llr_cap)))
        elif request.op == 'full':
            if self.changed:
                self.rate.reset()
                self.belief = 0.5
                return
            label = 1 if self.full >= self.pos_thr else 0 if self.full <= self.neg_thr else None
            if label is not None:
                self.rate.observe(self.q, label)
                self.belief = self.teacher_trust if label else 1 - self.teacher_trust

    def decide(self):
        score = super().decide()
        if self.full is None and self.smooth and self.active is not None and self.logit is not None:
            self.answer = self.belief > 0.5
            return self.belief
        return score

    def info(self):
        d = super().info()
        d.update(belief=self.belief, rate=self.rate.r, glitch=self.glitch,
                 logit=np.nan if self.logit is None else self.logit)
        return d
