"""Side-by-side video of a settled room: TieredServer against TemporalServer, query by query, on the same frames.

Replays two static_room.py configurations (default: the Mall tile with FCOS, schedules `relaxed` and
`relaxed_temporal`), checks that each replay reproduces its stored run in results/static_room/ query for query,
and renders a window of one pass. The loop is deterministic (cached teacher scores), so the replay is the run.

Panels, top row:
  camera     the frame (what a full-frame tag would send, 19,200 B), the true label, and FCOS's answer on it
  tiered     what TieredServer received this query: black with the code's 96 pixels (filled orange = bit set) if
             only the code came, the 16x12 thumbnail if one was fetched, the full frame if one was fetched;
             its answer (correct or not), bytes this query, running accuracy and bytes per query
  temporal   the same for TemporalServer, plus its belief P(person)
Below, revealed up to the current query:
  answers    one row each: truth, FCOS on the full frame, tiered, temporal (dark = person); the temporal row
             carries its belief as a line
  bytes      cumulative uplink from the window start (log), with full frames and (Mall tile) JPEG q88 for scale

  python experiments/temporal_video.py                                     # Mall tile, pass 4, queries 0-600
  python experiments/temporal_video.py --room A --pass 4 --start 0 --stop 600
Output: results/demo/temporal_<room>_p<pass>_<start>-<stop>.mp4 (GIF if no ffmpeg).
"""
import argparse, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from semwisp import piropo
from semwisp.sim import replay
from semwisp.tag import IntervalCodeTag
from semwisp.tag.interval import cell_sums, code_bits
from semwisp.tag.thumbnail import block_mean
from static_room import OUT as SR, make_server, rooms
from demo_video import ffmpeg_path

RES = piropo.REPO / 'results'
FULL = piropo.W * piropo.H
ORANGE, TEMPORAL, GRAY, INK, MUTED, GOOD, BAD = '#eb6834', '#2a78d6', '#8a8984', '#1f1e1c', '#6b6a66', '#1baf7a', '#e34948'
NAMES = {'relaxed': 'tiered (each reply alone)', 'relaxed_temporal': 'temporal (belief over time)'}


def run(room, teacher, sched, passes, R, teachers):
    """Replay passes 1..passes; per query, the code the tag answered with (None if no code was asked for)."""
    server, tag = make_server(teachers[teacher], sched), IntervalCodeTag()
    used, cur = [], {}
    begin, receive = server.begin, server.receive

    def begin_(t):
        if cur:
            used.append(cur.get('code'))
        cur.clear(); cur['q'] = True
        begin(t)

    def receive_(request, rx):
        if request.op == 'code':
            cur['code'] = server.active[0]
        receive(request, rx)

    server.begin, server.receive = begin_, receive_
    parts = [replay(R[room], tag, server, reset=False).df.assign(**{'pass': p}) for p in range(1, passes + 1)]
    used.append(cur.get('code'))
    df = pd.concat(parts, ignore_index=True)
    stored = np.load(SR / f'{room}_{teacher}_{sched}.npz')
    n = len(df)
    for col in ('answer', 'up_bytes', 't'):
        assert np.array_equal(stored[col][:n], df[col].values), f'{sched}: replay differs from the stored run ({col})'
    df['code'] = used
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--room', default='mall', choices=['mall', 'A'])
    ap.add_argument('--teacher', default='fcos')
    ap.add_argument('--a', default='relaxed', help='left adaptive schedule')
    ap.add_argument('--b', default='relaxed_temporal', help='right adaptive schedule')
    ap.add_argument('--pass', dest='pass_', type=int, default=4)
    ap.add_argument('--start', type=int, default=0, help='first query of the window, counted within the pass')
    ap.add_argument('--stop', type=int, default=600)
    ap.add_argument('--fps', type=int, default=20)
    ap.add_argument('--dpi', type=int, default=90)
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args()

    R, teachers = rooms()
    teach, pos, neg = teachers[args.teacher]
    t0 = time.time()
    runs = {s: run(args.room, args.teacher, s, args.pass_, R, teachers) for s in (args.a, args.b)}
    print(f'replayed and checked both runs ({time.time() - t0:.0f}s)', flush=True)
    W = {s: d[d['pass'] == args.pass_].reset_index(drop=True).iloc[args.start:args.stop].reset_index(drop=True)
         for s, d in runs.items()}
    A, B = W[args.a], W[args.b]
    assert (A.t.values == B.t.values).all()
    n = len(A)
    seqs = R[args.room]
    X = np.stack([seqs[s]['X'][i] for s, i in zip(A.seq, A.idx)])
    lab = A.label.values
    fcos = np.array([teach(t, None) >= (pos + neg) / 2 for t in A.t.values]).astype(int)
    jpeg = None
    if args.room == 'mall':
        z = np.load(RES / 'teachers' / 'fcos_r50@480_jpeg160x120q88_s2' / 'mall_tile.npz')
        jb = dict(zip(z['t'].tolist(), z['bytes'].tolist()))
        jpeg = np.array([jb[t] for t in A.t.values], float)

    def stats(ans, byt):
        ok = (ans == lab) & (lab >= 0)
        return (np.cumsum(ok) / np.maximum(np.cumsum(lab >= 0), 1), np.cumsum(byt) / np.arange(1, n + 1), np.cumsum(byt))
    S = {'full': stats(fcos, np.full(n, FULL)), args.a: stats(A.answer.values, A.up_bytes.values.astype(float)),
         args.b: stats(B.answer.values, B.up_bytes.values.astype(float))}

    # ---- figure
    fig = plt.figure(figsize=(13.5, 8.6), facecolor='white')
    gs = fig.add_gridspec(4, 3, height_ratios=[5, 0.35, 1.3, 2.0], hspace=0.45, wspace=0.08,
                          left=0.06, right=0.98, top=0.9, bottom=0.07)
    head = fig.text(0.06, 0.965, '', fontsize=13, fontweight='bold', color=INK)
    fig.text(0.06, 0.937, f"{'Mall tile' if args.room == 'mall' else 'room A'}, pass {args.pass_} of a static-room "
             f'replay (settled), FCOS as teacher. Same frames, same learner, same check schedule; only the server differs.',
             fontsize=9.5, color=MUTED)
    P = {}
    for j, (key, title, color) in enumerate((('full', 'camera: full frame, FCOS', INK),
                                              (args.a, NAMES.get(args.a, args.a), ORANGE),
                                              (args.b, NAMES.get(args.b, args.b), TEMPORAL))):
        ax = fig.add_subplot(gs[0, j])
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(color); s.set_linewidth(2.5 if key != 'full' else 1)
        ax.set_title(title, loc='left', fontsize=11, color=color, fontweight='bold')
        P[key] = dict(ax=ax, im=ax.imshow(np.zeros((piropo.H, piropo.W)), cmap='gray', vmin=0, vmax=255),
                      sent=ax.text(0.98, 0.975, '', transform=ax.transAxes, ha='right', va='top', fontsize=9,
                                   bbox=dict(fc='white', ec='none', pad=2, alpha=0.9)),
                      mark=ax.text(0.02, 0.975, '', transform=ax.transAxes, ha='left', va='top', fontsize=18,
                                   fontweight='bold', bbox=dict(fc='white', ec='none', pad=2, alpha=0.9)),
                      stat=ax.text(0.0, -0.04, '', transform=ax.transAxes, ha='left', va='top', fontsize=9.5,
                                   family='monospace', color=INK),
                      dots=ax.scatter([], [], s=22) if key != 'full' else None,
                      only=ax.text(0.5, 0.04, '', transform=ax.transAxes, ha='center', va='bottom', fontsize=9,
                                   color='#bdbcb6'))
    belief_txt = P[args.b]['ax'].text(0.02, 0.04, '', transform=P[args.b]['ax'].transAxes, fontsize=9.5,
                                      color='white', bbox=dict(fc=TEMPORAL, ec='none', pad=2.5))

    # answers raster
    axR = fig.add_subplot(gs[2, :])
    rows = [np.where(lab < 0, 0.5, lab), fcos, A.answer.values, B.answer.values]
    raster = np.full((4, n), np.nan)
    rim = axR.imshow(raster, aspect='auto', cmap='Greys', vmin=-0.3, vmax=1.3, interpolation='nearest',
                     extent=(-0.5, n - 0.5, 3.5, -0.5))
    blf, = axR.plot([], [], color=TEMPORAL, lw=1.2)
    axR.set_yticks(range(4), ['truth', 'full frame', 'tiered', 'temporal'], fontsize=9)
    axR.set_xlim(-0.5, n - 0.5); axR.set_xticks([])
    axR.set_title('answers (dark = person); blue line: temporal belief P(person)', loc='left', fontsize=9.5, color=MUTED)
    belief = B.belief.values

    # cumulative bytes
    axC = fig.add_subplot(gs[3, :])
    q = np.arange(n)
    lines = {k: axC.plot([], [], color=c, lw=2, label=l)[0]
             for k, c, l in ((args.a, ORANGE, 'tiered'), (args.b, TEMPORAL, 'temporal'))}
    axC.plot(q, np.cumsum(np.full(n, FULL)) / 1e3, color=INK, lw=1, ls='--', label='full frame every query')
    if jpeg is not None:
        axC.plot(q, np.cumsum(jpeg) / 1e3, color=GRAY, lw=1.2, ls=':', label='JPEG q88 every query')
    axC.set_yscale('log'); axC.set_ylim(0.05, FULL * n / 1e3 * 1.5); axC.set_xlim(0, n)
    axC.set_ylabel('cumulative\nuplink (kB)'); axC.set_xlabel(f'query (from {args.start} in pass {args.pass_})')
    axC.legend(loc='lower right', frameon=False, fontsize=8.5, ncol=4)
    axC.grid(alpha=0.25)
    for ax in (axR, axC):
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    cursor = [axR.axvline(0, color=INK, lw=1), axC.axvline(0, color=INK, lw=1)]

    def received(r, f):
        sent, view = [], np.zeros_like(f)
        if r.used_code:
            sent.append('code')
        if r.thumbed:
            sent.append('thumb'); view = np.kron(block_mean(f, 16, 12), np.ones((10, 10), np.uint8))
        if r.explored:
            sent.append('full'); view = f
        return sent, view

    def draw(i):
        f = X[i]
        truth = {1: 'person', 0: 'empty', -1: 'ambiguous'}[int(lab[i])]
        head.set_text(f'query {args.start + i}   ·   truth: {truth}')
        for key, d in (('full', None), (args.a, A), (args.b, B)):
            p, (acc, mean_b, _) = P[key], S[key]
            if d is None:
                ans, byt = fcos[i], FULL
                p['im'].set_data(f); p['sent'].set_text('full frame  19,200 B')
            else:
                r = d.iloc[i]
                ans, byt = int(r.answer), int(r.up_bytes)
                sent, view = received(r, f)
                p['im'].set_data(view)
                p['sent'].set_text(f"{' + '.join(sent)}  {byt:,} B")
                p['only'].set_text('' if (r.thumbed or r.explored) else 'received: only these 96 bits (12 B)')
                code = r.code
                if code is not None:
                    rects = code.rects
                    bits = code_bits(cell_sums(f, code.table), code)
                    p['dots'].set_offsets(np.c_[rects[:, 0] + rects[:, 4] // 2, rects[:, 1] + rects[:, 4] // 2])
                    on = bits[:, None] == 1
                    p['dots'].set_facecolor(np.where(on, matplotlib.colors.to_rgba_array([ORANGE] * len(bits)), (0, 0, 0, 0)))
                    p['dots'].set_edgecolor(np.where(on, (1, 1, 1, 1), (0.8, 0.8, 0.8, 1)))
                else:
                    p['dots'].set_offsets(np.empty((0, 2)))
            right = lab[i] >= 0 and ans == lab[i]
            p['mark'].set_text('' if lab[i] < 0 else ('✓' if right else '✗'))
            p['mark'].set_color(GOOD if right else BAD)
            p['stat'].set_text(f"answer {'person' if ans else 'empty':6s}   {byt:>6,} B\n"
                               f"running {acc[i]:.3f} correct, {mean_b[i]:>6,.0f} B/query")
        belief_txt.set_text(f'belief P(person) = {belief[i]:.2f}')
        for k, row in enumerate(rows):
            raster[k, i] = row[i]
        rim.set_data(raster)
        blf.set_data(q[:i + 1], 3.45 - 0.9 * belief[:i + 1])
        for k in (args.a, args.b):
            lines[k].set_data(q[:i + 1], S[k][2][:i + 1] / 1e3)
        for c in cursor:
            c.set_xdata([i, i])

    ff = ffmpeg_path()
    out = args.out or (RES / 'demo' / f'temporal_{args.room}_p{args.pass_}_{args.start}-{args.stop}.{"mp4" if ff else "gif"}')
    out.parent.mkdir(parents=True, exist_ok=True)
    if ff:
        plt.rcParams['animation.ffmpeg_path'] = ff
        writer = animation.FFMpegWriter(fps=args.fps, bitrate=3000)
    else:
        writer = animation.PillowWriter(fps=args.fps)
    t0 = time.time()
    with writer.saving(fig, str(out), args.dpi):
        for i in range(n):
            draw(i)
            writer.grab_frame()
            if i % 200 == 0:
                print(f'  {i}/{n} frames ({time.time() - t0:.0f}s)', flush=True)
    draw(n - 1)
    fig.savefig(out.with_suffix('.png'), dpi=args.dpi)          # last frame as a still
    print(f'wrote {out} ({n} frames, {time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
