"""JPEG baseline: the conventional "compress the picture, recover it at the server" encoder.

The tag block-averages the frame to w x h (as ThumbnailTag) and JPEG-encodes it (grayscale, baseline,
standard Huffman tables, quality q). Only the entropy-coded scan is sent: for a fixed (w, h, q) the header
(quantisation and Huffman tables, frame size) is the same for every frame, so the server keeps a copy and
splices it back on. That is the most generous way to count JPEG's bytes. The payload length varies per frame.

Encoding uses PIL (libjpeg) on the simulation host. The MCU cost is a rough count for a fixed-point 8x8 DCT
and Huffman coding, not a measurement.
"""
import io

import numpy as np
from PIL import Image

from ..piropo import H, W
from ..sim import Tag
from .thumbnail import block_mean

EOI = b'\xff\xd9'


def _encode(img, q):
    buf = io.BytesIO()
    Image.fromarray(img, 'L').save(buf, 'JPEG', quality=q, optimize=False, progressive=False)
    return buf.getvalue()


def _split(jpg):
    """(header up to and including the SOS segment, entropy-coded scan without EOI)."""
    i = jpg.index(b'\xff\xda')
    sos_end = i + 2 + int.from_bytes(jpg[i + 2:i + 4], 'big')
    assert jpg.endswith(EOI)
    return jpg[:sos_end], jpg[sos_end:-2]


class JpegTag(Tag):
    def __init__(self, w, h, quality):
        assert W % w == 0 and H % h == 0 and w >= 8 and h >= 8, 'JPEG works on 8x8 blocks'
        self.w, self.h, self.q = w, h, quality
        self.header = _split(_encode(np.zeros((h, w), np.uint8), quality))[0]

    def encode(self, frame):
        header, scan = _split(_encode(block_mean(frame, self.w, self.h), self.q))
        assert header == self.header
        return scan

    def respond(self, request):
        scan = self.encode(self.frame())
        self.ledger.mcu_ops += H * W + 2 * self.w * self.h + 40 * self.w * self.h    # block mean + DCT/quant/Huffman
        self.ledger.ram_bytes = max(self.ledger.ram_bytes, 2 * self.w + 8 * self.w + len(scan))  # 8 rows + output
        return np.frombuffer(scan, np.uint8)

    def decode(self, data):
        """Server side: scan bytes -> h x w uint8 image. Lost bytes (EPC loss, zeroed by the link) corrupt the
        scan from that point on; what libjpeg cannot decode at all comes back mid-gray."""
        from PIL import ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        try:
            with Image.open(io.BytesIO(self.header + bytes(data) + EOI)) as im:
                img = np.asarray(im.convert('L'), np.uint8)
            if img.shape == (self.h, self.w):
                return img
        except (OSError, SyntaxError, ValueError):
            pass
        return np.full((self.h, self.w), 128, np.uint8)

    def unpack(self, rx):
        """As ThumbnailTag.unpack: h x w float image in [0, 1]."""
        return self.decode(rx.data).astype(np.float32) / 255
