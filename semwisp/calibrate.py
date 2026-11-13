"""Label cutoffs for a teacher, chosen from its scores alone.

A teacher's person scores on frames from one camera form two modes, empty and person. label_free_cutoffs fits a
two-component Gaussian mixture to the scores and returns the scores at which a frame's component posterior reaches
`confident` (0.99: about 1% wrong labels if the mixture fits). Frames between the two cutoffs get no label.

Fit it on footage the evaluation never uses (for PIROPO: the unannotated sequences), so no label and no evaluated
frame is involved.
"""
import numpy as np
from sklearn.mixture import GaussianMixture


def mixture(scores, seed=0):
    """Two-component 1-D Gaussian mixture; components sorted by mean (0: empty, 1: person)."""
    g = GaussianMixture(2, random_state=seed).fit(np.asarray(scores, float).reshape(-1, 1))
    order = np.argsort(g.means_.ravel())
    return g, order


def empty_mode_cutoff(scores, z=2.33, bw=0.01):
    """Empty cutoff from the empty mode's left side only.

    A detector teacher's errors are one-sided: people it barely sees score in the gap between the modes, while
    empty frames rarely score high. So the gap is not part of the empty mode, and a mixture fit stretches the
    empty component over it. The empty mode's left side is uncontaminated: take its peak (smoothed histogram,
    below the mixture's 50% boundary, exact zeros = no detection left out), estimate its spread from the scores
    below the peak (mirrored, as a half-normal), and cut at peak + z * spread (z = 2.33: 99% one-sided)."""
    x = np.asarray(scores, float)
    x = x[x > 0]
    g, (lo, hi) = mixture(x)
    s = np.linspace(0, 1, 1001)
    boundary = s[np.argmax((g.predict_proba(s.reshape(-1, 1))[:, hi] >= 0.5) & (s >= g.means_.ravel()[lo]))]
    edges = np.arange(0, 1 + bw, bw)
    h, _ = np.histogram(x, edges)
    k = np.exp(-0.5 * (np.arange(-3, 4)) ** 2)
    h = np.convolve(h, k / k.sum(), 'same')
    centres = (edges[:-1] + edges[1:]) / 2
    low = centres < boundary
    peak = centres[low][np.argmax(h[low])]
    below = x[x <= peak]
    spread = np.sqrt(np.mean((below - peak) ** 2))
    return float(peak + z * spread), float(peak), float(spread)


def asymmetric_cutoffs(scores, confident=0.99, z=2.33):
    """(person cutoff, empty cutoff): person from the mixture posterior (label_free_cutoffs), empty from the empty
    mode's uncontaminated left side (empty_mode_cutoff)."""
    return label_free_cutoffs(scores, confident)[0], empty_mode_cutoff(scores, z)[0]


def label_free_cutoffs(scores, confident=0.99, grid=2001):
    """(person cutoff, empty cutoff) as Python floats: score >= person cutoff -> person, <= empty cutoff -> empty.

    Searched only between the two component means: with unequal variances the wider component also wins far out
    on the other side, which says nothing about labels."""
    g, (lo, hi) = mixture(scores)
    m = g.means_.ravel()
    s = np.linspace(m[lo], m[hi], grid)
    post = g.predict_proba(s.reshape(-1, 1))
    # monotone, as in the notebook: every score beyond a cutoff must also be confident
    not_up, not_down = np.flatnonzero(post[:, hi] < confident), np.flatnonzero(post[:, lo] < confident)
    i = not_up[-1] + 1 if len(not_up) else 0
    j = not_down[0] - 1 if len(not_down) else grid - 1
    pos = float(s[min(i, grid - 1)])
    neg = float(s[max(j, 0)])
    return pos, neg
