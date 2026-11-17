"""Long static-room test: does the adaptive loop converge to full-frame accuracy at minimal bytes when the room
does not change?

One room is replayed --passes times in a row with the tag and server state kept (replay(..., reset=False)), so the
loop sees the same room for a long time. Per pass: balanced accuracy, uplink bytes per query, full-frame and
thumbnail rates, and the check interval. The learner is the calibrated 96-pixel online learner with tiered checks.

Hypothesis (CLAUDE.md, 2026-11-13): the steady-state cost is set by the check schedule, not the 12-byte code.
The interval is 0.25 / disagreement (code against teacher); a teacher about 5% wrong keeps the disagreement at or
above ~0.05 even for a perfect code, so the interval stays at about 5 queries, and a forced full frame every 4
trusted thumbnails then costs ~1 kB per query on its own.

Rooms:      A     room A as recorded: training, test3, test1 (every 2nd frame; 2,301 queries per pass)
            mall  the Mall tile (every frame; 2,000 queries per pass; labels -1 are not scored)
Teachers:   fcos (cutoffs from unlabelled footage, as tiered_checks.py) and oracle (true labels, the control)
Schedules:  default   max_interval=64,   full_every=4,  noise_floor=0     (as tiered_checks.py)
            relaxed   max_interval=1024, full_every=16, noise_floor=0
            floor     max_interval=1024, full_every=16, noise_floor=0.05  (discount the teacher's own error rate)
Caveat: passes after the first replay identical frames, which overstates a real static room (lighting drifts).

Results: results/static_room/<room>_<teacher>_<schedule>.npz, one row per query (as tiered_checks.py, plus
`pass` and `interval`), meta with per-pass summaries; existing files are skipped.

  python experiments/static_room.py
  python experiments/static_room.py --only A_fcos
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from semwisp import piropo
from semwisp.metrics import summary
from semwisp.server import OnlineLearner, TieredServer
from semwisp.sim import replay
from semwisp.tag import IntervalCodeTag
from semwisp.tag.interval import lattice, sparse
from tiered_checks import setup

OUT = piropo.REPO / 'results' / 'static_room'
SCHEDULES = {'default': dict(max_interval=64, full_every=4, noise_floor=0.0),
             'relaxed': dict(max_interval=1024, full_every=16, noise_floor=0.0),
             'floor': dict(max_interval=1024, full_every=16, noise_floor=0.05)}
RUNS = ([(room, 'fcos', s) for room in ('A', 'mall') for s in SCHEDULES]
        + [(room, 'oracle', s) for room in ('A', 'mall') for s in ('default', 'floor')])


def rooms():
    streams, teachers = setup()
    P = streams['piropo'][0]
    mall = streams['switch'][0]['room B (mall tile)']
    return {'A': {n: P[n] for n in ('training', 'test3', 'test1')}, 'mall': {'mall tile': mall}}, teachers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--passes', type=int, default=6)
    ap.add_argument('--only', help='run only configurations whose name contains this text')
    ap.add_argument('--k', type=int, default=96)
    ap.add_argument('--margin', type=float, default=2.0, help='TieredServer code_margin (as tiered_checks.py)')
    ap.add_argument('--out', type=Path, default=OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    R, teachers = rooms()
    for room, teacher, sched in RUNS:
        name = f'{room}_{teacher}_{sched}'
        out = args.out / f'{name}.npz'
        if args.only and args.only not in name:
            continue
        if out.exists():
            print(f'skip {name} (exists)', flush=True)
            continue
        t0 = time.time()
        teach, pos, neg = teachers[teacher]
        kw = SCHEDULES[sched]
        L = OnlineLearner(k=args.k, table=sparse(lattice(4), 4), shrink=0.3, threshold='gauss')
        server = TieredServer(teach, L, code_margin=args.margin, pos_thr=pos, neg_thr=neg, bootstrap=200,
                              max_interval=kw['max_interval'], full_every=kw['full_every'], noise_floor=kw['noise_floor'])
        tag, parts = IntervalCodeTag(), []
        for p in range(1, args.passes + 1):
            parts.append(replay(R[room], tag, server, reset=False).df.assign(**{'pass': p}))
            m = summary(parts[-1])
            print(f"{name} pass {p}: balanced accuracy {m['balanced_accuracy']:.3f}, {m['up_bytes_per_query']:.0f} B/query, "
                  f"full frame on {m['full_frame_rate']:.1%}, thumbnail on {m.get('thumb_rate', 0):.1%}, "
                  f"median interval {parts[-1].interval.median():.0f} ({time.time() - t0:.0f}s)", flush=True)
        df = pd.concat(parts, ignore_index=True)
        meta = dict(name=name, room=room, teacher=teacher, schedule=sched, **kw, k=args.k, margin=args.margin,
                    pos_thr=pos, neg_thr=neg, passes=args.passes, seconds=round(time.time() - t0),
                    by_pass={p: summary(g) for p, g in df.groupby('pass')})
        tmp = out.with_name(out.stem + '.tmp.npz')
        np.savez(tmp, meta=json.dumps(meta),
                 **{c: df[c].values if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c])
                    else np.array(df[c].tolist(), dtype=str) for c in df.columns})            # strings: no pickles
        tmp.replace(out)


if __name__ == '__main__':
    main()
