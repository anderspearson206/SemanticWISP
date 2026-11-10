"""Baseline and thumbnail encoders through the simulator (approach categories 1 and 2).

For each configuration W x H, bits per pixel, method (mean | subsample):
  1. replay the training sequences through tag -> link -> server and collect what the server receives,
  2. train a small CNN decoder on those received payloads,
  3. replay the test sequences with the trained decoder and record score, label and cost counts per query.
160x120 at 8 bits is the uncompressed baseline. Train: `training`, `training_seats`. Test: `test1`-`test3`.

Results go to results/thumbnail_sweep/<config>.npz, one per configuration; existing files are skipped and an
interrupted training resumes from its checkpoint.

  python experiments/thumbnail_sweep.py                              # default grid
  python experiments/thumbnail_sweep.py --configs 16x12:8:mean 8x6:1:subsample --read-noise 4
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from semwisp import piropo
from semwisp.frames import load_frames
from semwisp.link import Link
from semwisp.sensor import Sensor
from semwisp.server import CNNDecoder, OneShotServer
from semwisp.sim import replay
from semwisp.tag import ThumbnailTag

TRAIN, TEST = ['training', 'training_seats'], ['test1', 'test2', 'test3']
GRID = ([f'{w}x{h}:8:mean' for w, h in [(160, 120), (80, 60), (40, 30), (32, 24), (16, 12), (8, 6), (4, 3)]]
        + [f'{w}x{h}:{b}:mean' for w, h in [(32, 24), (16, 12), (8, 6)] for b in (4, 2, 1)]
        + [f'{w}x{h}:8:subsample' for w, h in [(32, 24), (16, 12), (8, 6), (4, 3)]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--configs', nargs='+', default=GRID, help='WxH:bits:method')
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--train-stride', type=int, default=2, help='adjacent frames are near-duplicates')
    ap.add_argument('--read-noise', type=float, default=0.0, help='sensor read noise, std in gray levels')
    ap.add_argument('--p-loss', type=float, default=0.0, help='EPC loss probability')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', type=Path, default=piropo.REPO / 'results' / 'thumbnail_sweep')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    frames = load_frames(TRAIN + TEST)
    suffix = (f'_rn{args.read_noise:g}' if args.read_noise else '') + (f'_loss{args.p_loss:g}' if args.p_loss else '')

    for cfg in args.configs:
        size, bits, method = cfg.split(':')
        w, h = map(int, size.split('x'))
        name = f'{size}_b{bits}_{method}{suffix}'
        out, ckpt = args.out / f'{name}.npz', args.out / f'{name}.ckpt'
        if out.exists():
            print(f'skip {name} (exists)')
            continue
        t0 = time.time()
        tag = ThumbnailTag(w, h, int(bits), method)
        world = dict(link=Link(p_loss=args.p_loss, seed=args.seed),
                     sensor=Sensor(read_noise=args.read_noise, seed=args.seed))
        print(f'{name}: {tag.n_bytes} bytes per query', flush=True)

        tr = replay({n: frames[n] for n in TRAIN}, tag, OneShotServer(tag.unpack), stride=args.train_stride,
                    keep_payloads=True, **world)
        X = np.stack([tag.unpack(p[0]) for p in tr.payloads])
        decoder = CNNDecoder(h, w, seed=args.seed).fit(X, tr.df.label.values, epochs=args.epochs, ckpt=ckpt,
                                                       log=lambda m: print(m, flush=True))

        te = replay({n: frames[n] for n in TEST}, tag, OneShotServer(tag.unpack, decoder), **world).df
        auc = {n: piropo.auc(g.score, g.label) for n, g in te.groupby('seq')}
        auc['pooled'] = piropo.auc(te.score, te.label)
        meta = dict(config=cfg, w=w, h=h, bits=int(bits), method=method, n_bytes=tag.n_bytes, n_train=len(X),
                    epochs=args.epochs, read_noise=args.read_noise, p_loss=args.p_loss, seed=args.seed,
                    device=decoder.device, auc=auc, seconds=round(time.time() - t0))
        np.savez(out, meta=json.dumps(meta), **{c: te[c].values.astype(str) if c == 'seq' else te[c].values for c in te.columns})
        ckpt.unlink(missing_ok=True)
        print(f'  test AUC {", ".join(f"{k} {v:.3f}" for k, v in auc.items())}  ({meta["seconds"]}s) -> {out.name}',
              flush=True)


if __name__ == '__main__':
    main()
