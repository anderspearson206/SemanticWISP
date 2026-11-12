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


def test_adaptive_loop():
    """Bright square = person. The loop must bootstrap on full frames, configure the tag, then mostly use the code."""
    from semwisp.server import AdaptiveServer, BatchLearner, OnlineLearner
    from semwisp.tag.interval import lattice, sparse, pixels_read, CELLS
    from semwisp.tag import IntervalCodeTag
    from semwisp.tag.interval import cell_sums, code_bits
    n = 600
    y = (np.arange(n) // 25) % 2
    X = rng.integers(95, 106, (n, piropo.H, piropo.W)).astype(np.uint8)
    for i in np.flatnonzero(y):                             # the 'person' stands in one of four places
        r, c = [(0, 0), (40, 80), (80, 40), (60, 120)][rng.integers(4)]
        X[i, r:r + 40, c:c + 40] = 220
    seqs = {'s': dict(X=X, t=np.arange(float(n)), bad=np.zeros(n, bool), label=y.astype(np.uint8))}
    from semwisp.sim import Request
    px = lattice(1)
    assert len(px) == 19200 and (cell_sums(X[:2], px) == X[:2].reshape(2, -1)).all()
    for learner, idx_bits in [(BatchLearner(k=16), 8), (OnlineLearner(k=16), 8),
                              (OnlineLearner(k=16, table=sparse(lattice(8), 8)), 9)]:
        tag, server = IntervalCodeTag(), AdaptiveServer(lambda t, frame: float(y[int(t)]), learner, bootstrap=100)
        df = replay(seqs, tag, server, reset=False).df
        boot, dep = df[~df.used_code], df[df.used_code]
        assert len(boot) == 100 and (boot.up_bytes == 19200).all() and df.reconfig.sum() >= 1
        n_bits = len(tag.code.cell)
        assert n_bits <= 16 and df.down_bits[df.reconfig].iloc[-1] == n_bits * (idx_bits + 16)
        code_only = dep[~dep.explored & ~dep.reconfig]      # a reconfiguration adds one acknowledging EPC
        assert (code_only.up_bytes <= 2).all() and (code_only.up_epcs == 1).all()
        assert (dep.answer == dep.label).mean() > 0.95 and dep.explored.mean() < 0.2, type(learner).__name__
        led = Ledger(); tag.new_query(Sensor(), X[-1], led)                 # tag and server arithmetic agree
        reply = bits.unpack(tag.respond(Request('code')), 1, n_bits)
        assert (reply == code_bits(cell_sums(X[-1], tag.code.table), tag.code)).all()
    assert tag.code.n_pixels == len(tag.code.cell) <= 16                   # the pixel code reads one pixel per bit
    assert pixels_read(CELLS) == 19200 and pixels_read(sparse(CELLS, 5)) == 768 and pixels_read(sparse(CELLS, 10)) == 192
    assert (cell_sums(X[0], sparse(CELLS, 10))[-192:] == X[0][5::10, 5::10].ravel()).all()


def test_jpeg_tag():
    """Header-stripped JPEG round-trips through the link, and the payload is the scan only."""
    from semwisp.tag import JpegTag
    X = (np.add.outer(np.arange(piropo.H), np.arange(piropo.W)) % 256).astype(np.uint8)[None].repeat(3, 0)
    seqs = {'s': dict(X=X, t=np.arange(3.0), bad=np.zeros(3, bool), label=np.ones(3, np.uint8))}
    tag = JpegTag(80, 60, 50)
    tr = replay(seqs, tag, OneShotServer(tag.unpack), keep_payloads=True)
    img = tag.decode(tr.payloads[0][0].data)
    from semwisp.tag.thumbnail import block_mean
    assert img.shape == (60, 80) and np.abs(img.astype(int) - block_mean(X[0], 80, 60)).mean() < 4
    assert (tr.df.up_bytes < 80 * 60).all() and tr.payloads[0][0].data[:2].tobytes() != b'\xff\xd8'


def test_tiered_server():
    """Same toy room as test_adaptive_loop: tiered checks must keep accuracy and fetch fewer full frames."""
    from semwisp.server import AdaptiveServer, OnlineLearner, TieredServer
    from semwisp.tag import IntervalCodeTag
    from semwisp.tag.thumbnail import block_mean
    n = 900
    y = (np.arange(n) // 25) % 2
    X = rng.integers(95, 106, (n, piropo.H, piropo.W)).astype(np.uint8)
    for i in np.flatnonzero(y):
        r, c = [(0, 0), (40, 80), (80, 40), (60, 120)][rng.integers(4)]
        X[i, r:r + 40, c:c + 40] = 220
    seqs = {'s': dict(X=X, t=np.arange(float(n)), bad=np.zeros(n, bool), label=y.astype(np.uint8))}
    teacher = lambda t, frame: float(y[int(t)])
    full = replay(seqs, IntervalCodeTag(), AdaptiveServer(teacher, OnlineLearner(k=16), bootstrap=100), reset=False).df
    tag = IntervalCodeTag()
    tiered = replay(seqs, tag, TieredServer(teacher, OnlineLearner(k=16), bootstrap=100, code_margin=2), reset=False).df
    dep = tiered[tiered.used_code]
    assert (dep.answer == dep.label).mean() > 0.95 and tiered.thumbed.any()
    assert tiered.explored.sum() <= full.explored.sum()
    th = dep[dep.thumbed & ~dep.explored & ~dep.reconfig]
    assert ((th.up_bytes == 2 + 192)).all() and (th.rounds == 2).all()      # code (16 bits) + 16x12 thumbnail
    from semwisp.sim import Request
    led = Ledger(); tag.new_query(Sensor(), X[1], led)
    assert (tag.respond(Request('thumb', dict(w=16, h=12))) == block_mean(X[1], 16, 12).ravel()).all()


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
