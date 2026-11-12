"""Configurable tag for the adaptive loop: the server tells it which pixels (or blocks of pixels) to look at
and what counts as unusual for each."""
from dataclasses import dataclass, field

import numpy as np

from .. import bits as B
from ..piropo import H, W
from ..sim import Tag

GRIDS = [(4, 3), (8, 6), (16, 12)]
# Candidate regions as rows of (x0, y0, width, height, sample). The tag reads one pixel per sample x sample
# block of the region (the block's centre), so sample=1 uses every pixel and sample=10 uses 1 pixel in 100.
CELLS = np.array([(gx * (W // w), gy * (H // h), W // w, H // h, 1)
                  for w, h in GRIDS for gy in range(h) for gx in range(w)])      # every block of GRIDS: 252 regions


def sparse(table, sample):
    """The same regions, read from one pixel per sample x sample block. Region corners and sizes must be
    multiples of `sample` (for CELLS: 1, 2, 5 or 10)."""
    assert (table[:, :4] % sample == 0).all()
    return np.column_stack([table[:, :4], np.full(len(table), sample)])


def lattice(block=1):
    """Candidate table of touching block x block regions. lattice(1) is every single pixel (19,200 candidates)."""
    return np.array([(x, y, block, block, 1) for y in range(0, H, block) for x in range(0, W, block)])


def n_samples(table):
    return (table[:, 2] // table[:, 4]) * (table[:, 3] // table[:, 4])


def pixels_read(table):
    """Number of distinct pixels the tag has to read to evaluate every region in `table`."""
    used = np.zeros((H, W), bool)
    for x0, y0, w, h, s in table:
        used[y0 + s // 2:y0 + h:s, x0 + s // 2:x0 + w:s] = True
    return int(used.sum())


@dataclass
class IntervalCode:
    """One bit per chosen region: 1 if its mean gray level is outside [lo, hi]."""
    cell: np.ndarray        # indices into `table`
    lo: np.ndarray          # gray levels 0..255
    hi: np.ndarray
    table: np.ndarray = field(default_factory=lambda: CELLS, repr=False)

    @property
    def rects(self):
        return self.table[self.cell]

    @property
    def n_bits(self):       # size of the instruction on the downlink: region index + two thresholds per bit
        return len(self.cell) * (int(np.ceil(np.log2(len(self.table)))) + 8 + 8)

    @property
    def n_bytes(self):      # size of the reply
        return B.n_bytes(len(self.cell), 1)

    @property
    def n_pixels(self):     # distinct pixels the tag has to read to answer
        return pixels_read(self.rects)


def cell_sums(X, table=CELLS):
    """Sum of the sampled pixels of every region in `table` for frames X (..., 120, 160) -> (..., len(table)),
    exact integers."""
    X = np.asarray(X)
    out = np.empty(X.shape[:-2] + (len(table),), np.int64)
    for s in np.unique(table[:, 4]):
        m = table[:, 4] == s
        sub = X[..., s // 2::s, s // 2::s]
        ii = np.zeros(sub.shape[:-2] + (sub.shape[-2] + 1, sub.shape[-1] + 1), np.int64)
        ii[..., 1:, 1:] = sub.cumsum(-2, dtype=np.int64).cumsum(-1)
        x0, y0 = table[m, 0] // s, table[m, 1] // s
        x1, y1 = x0 + table[m, 2] // s, y0 + table[m, 3] // s
        out[..., m] = ii[..., y1, x1] - ii[..., y0, x1] - ii[..., y1, x0] + ii[..., y0, x0]
    return out


def code_bits(sums, code):
    """The tag's arithmetic, shared with the server's learning code so both sides agree exactly:
    compare the region sum with the thresholds scaled by the number of pixels summed (no division on the tag).
    `sums` holds the sums of all regions of code.table."""
    n = n_samples(code.rects)
    s = sums[..., code.cell]
    return ((s < code.lo * n) | (s > code.hi * n)).astype(np.uint8)


class IntervalCodeTag(Tag):
    """Requests: 'full' -> the raw frame; 'config' -> store a new IntervalCode (kept in FRAM, so it survives
    between queries and sequences); 'code' -> one bit per configured region; 'thumb' (args w, h) -> an 8-bit
    block-mean thumbnail, the cheap check of semwisp.server.TieredServer."""

    def __init__(self):
        self.code = None

    def respond(self, request):
        if request.op == 'full':
            self.ledger.ram_bytes = max(self.ledger.ram_bytes, W)          # stream it out row by row
            return self.frame().ravel()
        if request.op == 'config':
            self.code = request.args['code']
            return np.zeros(0, np.uint8)                                   # acknowledged by the reply itself
        if request.op == 'code':
            c = self.code
            self.ledger.mcu_ops += int(n_samples(c.rects).sum()) + 3 * len(c.cell)   # adds; 2 compares + 1 bit store per region
            self.ledger.ram_bytes = max(self.ledger.ram_bytes, 4 * len(c.cell) + c.n_bytes)
            return B.pack(code_bits(cell_sums(self.frame(), c.table), c), 1)
        if request.op == 'thumb':
            from .thumbnail import block_mean
            w, h = request.args['w'], request.args['h']
            self.ledger.mcu_ops += H * W + 2 * w * h                       # as ThumbnailTag(method='mean')
            self.ledger.ram_bytes = max(self.ledger.ram_bytes, 2 * w + w * h)
            return block_mean(self.frame(), w, h).ravel()
        raise ValueError(request.op)
