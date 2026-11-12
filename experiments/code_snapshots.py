"""Re-runs tiered_checks.py configurations and records every code the server sends to the tag, for pictures.

The runs in results/tiered_checks/ keep one row per query but only the last code. This script replays the same
configuration (the loop is deterministic: cached teacher scores, no randomness), checks that it reproduces the
stored run query for query, and saves each deployed code with the query it was sent on, together with the
state of the thumbnail decoder at that moment (its weights and empty-room thumbnail).

Results: results/tiered_checks/snapshots/<run name>.npz with arrays, per code i:
  query[i], seq[i]                  the query on which the code was sent, and its sequence
  rects[i] (k, 5), lo[i], hi[i]     the regions (x0, y0, w, h, sample) and their usual gray-level ranges
  coef[i] (k,), intercept[i]        the server-side decoder of the code bits
  thumb_w[i] (h*w,), thumb_bg[i]    ThumbDecoder weights and empty-room thumbnail (NaN before it is ready)

  python experiments/code_snapshots.py                         # the default runs below
  python experiments/code_snapshots.py switch_detector_online_cells_tiered_b200
"""
import json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from semwisp.sim import replay
from semwisp.tag import IntervalCodeTag
from tiered_checks import make_server, setup

OUT = Path(__file__).resolve().parents[1] / 'results' / 'tiered_checks'
DEFAULT = ['switch_detector_online_pixels_tiered_b200', 'piropo_detector_online_pixels_tiered_b200']


def record(server):
    """Wrap server.begin / server.receive to log each config request."""
    log, state = [], dict(q=-1, seq=None)
    begin, receive = server.begin, server.receive

    def begin_(t):
        state['q'] += 1
        begin(t)

    def receive_(request, rx):
        if request.op == 'config':
            code, dec = server.pending
            d = getattr(server, 'dec', None)
            n = 0 if d is None else d.w * d.h
            ready = d is not None and d.wv is not None and d.bg is not None
            log.append(dict(query=state['q'], rects=code.rects.copy(), lo=code.lo.copy(), hi=code.hi.copy(),
                            coef=np.asarray(dec.coef_[0], float).copy(), intercept=float(np.ravel(dec.intercept_)[0]),
                            thumb_w=d.wv.copy() if ready else np.full(n, np.nan),
                            thumb_bg=d.bg.copy() if ready else np.full(n, np.nan)))
        receive(request, rx)

    server.begin, server.receive = begin_, receive_
    return log


def main():
    names = sys.argv[1:] or DEFAULT
    (OUT / 'snapshots').mkdir(exist_ok=True)
    streams, teachers = setup()
    for name in names:
        out = OUT / 'snapshots' / f'{name}.npz'
        if out.exists():
            print(f'skip {name} (exists)')
            continue
        stream, teacher, *rest = name.split('_')
        learner, check, boot = '_'.join(rest[:-2]), rest[-2], int(rest[-1][1:])
        with np.load(OUT / f'{name}.npz') as z:
            meta, stored = json.loads(str(z['meta'])), z['up_bytes']
        server = make_server(teachers[teacher], learner, check, boot, meta['k'], meta['margin'] or 2.0)
        log, tag, dfs = record(server), IntervalCodeTag(), []
        for seqs in streams[stream]:
            dfs.append(replay(seqs, tag, server, reset=False).df)
        up = np.concatenate([d.up_bytes.values for d in dfs])
        seq = np.concatenate([d.seq.values for d in dfs])
        assert len(up) == len(stored) and (up == stored).all(), 'replay does not reproduce the stored run'
        q = np.array([e['query'] for e in log])
        np.savez(out, query=q, seq=np.array(seq[q].tolist(), dtype=str),
                 **{k: np.stack([e[k] for e in log]) for k in ('rects', 'lo', 'hi', 'coef', 'thumb_w', 'thumb_bg')},
                 intercept=np.array([e['intercept'] for e in log]))
        print(f'{name}: {len(log)} codes, reproduces the stored run', flush=True)


if __name__ == '__main__':
    main()
