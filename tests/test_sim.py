"""Checks of the simulator plumbing on synthetic frames (no dataset needed).

  .venv/Scripts/python.exe tests/test_sim.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from semwisp import bits, piropo
from semwisp.link import Link
from semwisp.sensor import Sensor
from semwisp.server import OneShotServer
from semwisp.sim import Ledger, replay
from semwisp.tag import ThumbnailTag

rng = np.random.default_rng(0)


def test_bits_roundtrip():
    for b in range(1, 9):
        v = rng.integers(0, 2 ** b, 37).astype(np.uint8)
        p = bits.pack(v, b)
        assert len(p) == bits.n_bytes(37, b) and (bits.unpack(p, b, 37) == v).all(), b


def test_link_chunks():
    link = Link(epc_bytes=12)
    assert link.n_epcs(10) == (1, 12) and link.n_epcs(12) == (1, 12)
    assert link.n_epcs(13) == (2, 11)                      # 1 index byte per EPC
    assert link.n_epcs(19200) == (1920, 10)                # 2 index bytes: more than 256 chunks
    led = Ledger()
    rx = Link(p_loss=0.5, seed=1).uplink(np.full(110, 7, np.uint8), led)
    assert led.up_epcs == 10 and led.up_bytes == 110
    lost = ~rx.ok.reshape(10, 11)
    assert 0 < lost.all(1).sum() < 10 and (lost.all(1) == lost.any(1)).all()   # whole EPCs are lost
    assert (rx.data[~rx.ok] == 0).all() and (rx.data[rx.ok] == 7).all()


def test_thumbnail_matches_reference():
    X = rng.integers(0, 256, (5, piropo.H, piropo.W)).astype(np.uint8)
    seqs = {'s': dict(X=X, t=np.arange(5.0), bad=np.array([0, 0, 1, 0, 0], bool), label=np.ones(5, np.uint8))}
    for w, h in [(160, 120), (16, 12), (4, 3)]:
        tag = ThumbnailTag(w, h, 8, 'mean')
        tr = replay(seqs, tag, OneShotServer(tag.unpack), keep_payloads=True)
        assert list(tr.df.idx) == [0, 1, 3, 4]             # the corrupted frame is skipped
        got = np.stack([tag.unpack(p[0]) for p in tr.payloads]) * 256 - 0.5
        ref = piropo.thumb(X[[0, 1, 3, 4]], w, h)
        assert np.abs(got - ref).max() <= 0.5 + 1e-4, (w, h)   # integer rounding of the float reference
        assert (tr.df.up_bytes == w * h).all() and (tr.df.pixels_read == 19200).all() and (tr.df.rounds == 1).all()
    tag = ThumbnailTag(8, 6, 1, 'subsample')
    tr = replay(seqs, tag, OneShotServer(tag.unpack), keep_payloads=True)
    assert (tr.df.up_bytes == 6).all() and (tr.df.up_epcs == 1).all() and (tr.df.mcu_ops == 48).all()
    assert ((tag.unpack(tr.payloads[0][0]) > 0.5) == (X[0][10::20, 10::20] >= 128)).all()


def test_sensor_noise():
    scene, led = np.full((piropo.H, piropo.W), 100, np.uint8), Ledger()
    assert Sensor().capture(scene, led) is scene and led.pixels_read == 19200
    noisy = Sensor(read_noise=4.0).capture(scene, led).astype(float)
    assert abs(noisy.std() - 4.0) < 0.3 and abs(noisy.mean() - 100) < 0.3


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
