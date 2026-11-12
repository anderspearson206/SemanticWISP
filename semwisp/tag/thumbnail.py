"""Thumbnail encoders (and, at 160x120 and 8 bits, the uncompressed baseline). Integer arithmetic only."""
import numpy as np

from .. import bits as B
from ..piropo import H, W
from ..sim import Tag


def block_mean(x, w, h):
    """The tag's integer block average of a 120x160 uint8 frame to h x w (rounded to nearest). The server uses
    the same function on full frames, so both sides compute identical thumbnails."""
    bw, bh = W // w, H // h
    n = bw * bh
    return ((x.reshape(*x.shape[:-2], h, bh, w, bw).sum((-3, -1), dtype=np.uint32) + n // 2) // n).astype(np.uint8)


class ThumbnailTag(Tag):
    """Sends a w x h thumbnail with `bits` bits per pixel (the top bits of each value).

    method='mean'       block average, computed while the frame streams in: one accumulator per output column
    method='subsample'  keeps the centre pixel of each block: almost no computation, but noisier
    """

    def __init__(self, w, h, bits=8, method='mean'):
        assert W % w == 0 and H % h == 0 and 1 <= bits <= 8 and method in ('mean', 'subsample')
        self.w, self.h, self.bits, self.method = w, h, bits, method
        self.n_bytes = B.n_bytes(w * h, bits)

    def respond(self, request):
        x, (bw, bh) = self.frame(), (W // self.w, H // self.h)
        if self.method == 'mean':
            n = bw * bh
            q = block_mean(x, self.w, self.h)
            self.ledger.mcu_ops += x.size + 2 * self.w * self.h         # one add per pixel; divide + shift per output
            acc = self.w * (2 if 255 * n < 65536 else 4)                # one row of accumulators
        else:
            q = x[bh // 2::bh, bw // 2::bw]
            self.ledger.mcu_ops += self.w * self.h
            acc = 0
        self.ledger.ram_bytes = max(self.ledger.ram_bytes, acc + self.n_bytes)
        return B.pack(q.astype(np.uint8) >> (8 - self.bits), self.bits)

    def unpack(self, rx):
        """Server side: received payload -> h x w float image in [0, 1] (each value at the centre of its bin)."""
        code = B.unpack(rx.data, self.bits, self.w * self.h).astype(np.float32)
        return ((code + 0.5) * (256 >> self.bits) / 256).reshape(self.h, self.w)
