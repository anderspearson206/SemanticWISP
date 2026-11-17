"""Side-by-side demo video: full frames, a non-adaptive compressed image, and the adaptive loop, query by query.

One video frame per query of a replayed tiered_checks.py run (default: the switch stream, FCOS teacher, calibrated
96-pixel online learner, tiered checks). Panels, left to right:
  full      "send the full frame every query": the 160x120 frame (19,200 bytes) and FCOS's answer on it (person if
            its score >= the midpoint of its two label cutoffs, as the server answers from a full frame)
  jpeg      (if --baseline names scored results) "send a JPEG every query", non-adaptive: the decoded JPEG the
            server receives, its size in bytes, and FCOS's answer on it with the same cutoffs. Scores come from
            `teacher_eval.py --input jpeg160x120qQ` (results/teachers/fcos_r50@480_<baseline>_s2/). The default
            quality 88 averages slightly more bytes per frame than the adaptive loop on this stream.
  adaptive  what the adaptive server actually received for this query. The code's 96 pixels as dots (filled
            orange = bit set: that pixel is outside its usual range; hollow = not set), drawn on the 16x12
            thumbnail if one was also fetched, or on the full frame if one was fetched; black if only the code came.
  top       the query, the room, and the true label
  each      this query's answer (correct or not), its uplink bytes, the running accuracy and running mean bytes
  bottom    running accuracy and running mean uplink bytes per query of all panels, revealed up to the query

Accuracy is the share of correct answers over queries with a label (Mall tile labels -1 are skipped); it is
cumulative from --start (--running clip, the default) or from the stream start (--running stream), with a
100-query rolling mean drawn thin to show the moves.
The codes come from results/tiered_checks/snapshots/<run>.npz (experiments/code_snapshots.py; created if missing).

Output: MP4 if ffmpeg is available (on PATH, or the imageio-ffmpeg package), else an animated GIF (large; use
--step and a short --start/--stop range).

  python experiments/demo_video.py                                  # whole stream
  python experiments/demo_video.py --start 1950 --stop 2400        # around the first move
  python experiments/demo_video.py --baseline none                  # two panels, no JPEG
"""
import argparse, json, re, subprocess, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from semwisp import piropo
from semwisp.calibrate import asymmetric_cutoffs
from semwisp.frames import load_frames, load_mall_tile
from semwisp.tag.interval import IntervalCode, cell_sums, code_bits

RES = piropo.REPO / 'results'
FULL_BYTES = piropo.W * piropo.H
ROOMS = {'room A (training)': 'room A', 'room B (mall tile)': 'Mall tile', 'room A, rearranged': 'room A rearranged',
         'room A, rearranged (test2)': 'room A rearranged'}
BLUE, ORANGE, AQUA, INK, MUTED, GOOD, BAD = '#2a78d6', '#eb6834', '#1baf7a', '#1f1e1c', '#52514e', '#1baf7a', '#e34948'
SEQ_FILES = ['training', 'training_seats', 'test2', 'mall_tile']


def ffmpeg_path():
    if animation.FFMpegWriter.isAvailable():
        return 'ffmpeg'
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


def scores_by_t(folder, field='score'):
    out = {}
    for f in SEQ_FILES:
        d = np.load(folder / f'{f}.npz')
        out.update(zip(d['t'].tolist(), d[field].tolist()))
    return out


def load(run, baseline):
    z = np.load(RES / 'tiered_checks' / f'{run}.npz')
    meta = json.loads(str(z['meta']))
    n = len(z['label'])
    df = pd.DataFrame({k: z[k] for k in z.files if k != 'meta' and z[k].shape == (n,)})
    assert meta['stream'] == 'switch', 'the frame lookup below is written for the switch stream'
    snap_path = RES / 'tiered_checks' / 'snapshots' / f'{run}.npz'
    if not snap_path.exists():
        subprocess.run([sys.executable, str(Path(__file__).with_name('code_snapshots.py')), run], check=True)
    snap = np.load(snap_path)
    P = load_frames(['training', 'training_seats', 'test2'])
    src = {'room A (training)': (P['training'], 2), 'room A, rearranged': (P['training_seats'], 2),
           'room A, rearranged (test2)': (P['test2'], 2), 'room B (mall tile)': (load_mall_tile(), 1)}
    X = np.stack([src[s][0]['X'][int(i) * src[s][1]] for s, i in zip(df.seq, df.idx)])
    assert all(src[s][0]['t'][int(i) * src[s][1]] == t for s, i, t in zip(df.seq, df.idx, df.t))
    # FCOS answers as the server answers from a full frame: person if score >= the midpoint of its label cutoffs
    calib = np.concatenate([np.load(f)['score'] for f in sorted((RES / 'teachers' / 'fcos_r50@480_s4').glob('*.npz'))])
    pos, neg = asymmetric_cutoffs(calib)
    cut = (pos + neg) / 2
    df['full_answer'] = (df.t.map(scores_by_t(RES / 'teachers' / 'fcos_r50@480_s2')) >= cut).astype(int)
    if baseline:
        folder = RES / 'teachers' / f'fcos_r50@480_{baseline}_s2'
        df['jpeg_answer'] = (df.t.map(scores_by_t(folder)) >= cut).astype(int)
        df['jpeg_bytes'] = df.t.map(scores_by_t(folder, 'bytes'))
        assert df.jpeg_bytes.notna().all()
    return df, snap, X, meta


def curves(df, panels, start=0):
    """Running accuracy and mean bytes per panel, accumulated from query `start` (NaN before it)."""
    after = np.arange(len(df)) >= start
    lab = (df.label >= 0).values
    out = {}
    for p in panels:
        ans, byt = {'full': (df.full_answer, np.full(len(df), FULL_BYTES)),
                    'jpeg': (df.get('jpeg_answer'), df.get('jpeg_bytes')),
                    'adaptive': (df.answer, df.up_bytes)}[p]
        ok = ((ans == df.label).values & lab).astype(float)
        byt = np.asarray(byt, float)
        acc = np.cumsum(ok * after) / np.maximum(np.cumsum(lab & after), 1)
        mean_bytes = np.cumsum(byt * after) / np.maximum(np.cumsum(after), 1)
        out[p] = dict(answer=np.asarray(ans), acc=np.where(after, acc, np.nan), bytes=byt,
                      mean_bytes=np.where(after, mean_bytes, np.nan),
                      roll=pd.Series(np.where(lab, ok, np.nan)).rolling(100, min_periods=20).mean().values)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='switch_fcos_online_pixels_cal_tiered_b200')
    ap.add_argument('--baseline', default='jpeg160x120q88',
                    help="non-adaptive panel: an input scored by teacher_eval.py --input (e.g. jpeg160x120q88), or 'none'")
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--stop', type=int, default=None)
    ap.add_argument('--step', type=int, default=1, help='render every n-th query')
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--format', choices=['auto', 'mp4', 'gif'], default='auto')
    ap.add_argument('--dpi', type=int, default=90)
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--running', choices=['clip', 'stream'], default='clip',
                    help='accumulate running accuracy and bytes from --start (clip) or from the stream start')
    args = ap.parse_args()
    baseline = None if args.baseline == 'none' else args.baseline

    df, snap, X, meta = load(args.run, baseline)
    panels = ['full'] + (['jpeg'] if baseline else []) + ['adaptive']
    C = curves(df, panels, args.start if args.running == 'clip' else 0)
    n = len(df)
    stop = min(args.stop or n, n)
    queries = range(args.start, stop, args.step)
    ff = ffmpeg_path()
    fmt = args.format if args.format != 'auto' else ('mp4' if ff else 'gif')
    if fmt == 'mp4' and not ff:
        sys.exit('no ffmpeg: install it, or `pip install imageio-ffmpeg`, or use --format gif')
    suffix = f'_vs_{baseline}' if baseline else ''
    out = args.out or RES / 'demo' / f'{args.run}{suffix}_{args.start}-{stop}.{fmt}'
    out.parent.mkdir(parents=True, exist_ok=True)
    jtag = None
    if baseline:
        from semwisp.tag.jpeg import JpegTag
        m = re.fullmatch(r'jpeg(\d+)x(\d+)q(\d+)', baseline)
        assert m, 'the picture of the baseline panel is drawn for JPEG baselines'
        jtag = JpegTag(int(m[1]), int(m[2]), int(m[3]))

    # figure
    k = len(panels)
    fig = plt.figure(figsize=(6.4 * k + 0.4, 9.2), dpi=args.dpi)
    gs = fig.add_gridspec(4, k, height_ratios=[0.25, 3.0, 1.1, 1.1], hspace=0.38, wspace=0.08,
                          left=0.06, right=0.98, top=0.97, bottom=0.06)
    head = fig.add_subplot(gs[0, :]); head.axis('off')
    head_txt = head.text(0.5, 0.5, '', ha='center', va='center', fontsize=13, color=INK)
    style = {'full': ('send the full frame every query', ORANGE),
             'jpeg': (f'send a JPEG every query (quality {jtag.q if jtag else ""}, not adaptive)', AQUA),
             'adaptive': (f"adaptive: 96-pixel code, tiered checks ({meta['teacher']} teacher)", BLUE)}
    A = {}
    for j, p in enumerate(panels):
        ax = fig.add_subplot(gs[1, j])
        title, c = style[p]
        im = ax.imshow(X[0] if p != 'adaptive' else np.zeros_like(X[0]), cmap='gray', vmin=0, vmax=255)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(c); s.set_linewidth(3)
        ax.set_title(title, fontsize=11, color=c, fontweight='bold', loc='left')
        A[p] = dict(ax=ax, im=im,
                    sent=ax.text(0.99, 0.98, '', transform=ax.transAxes, ha='right', va='top', fontsize=10.5,
                                 color='white', fontweight='bold', bbox=dict(fc='#000000aa', ec='none', pad=3)),
                    stat=ax.text(0, -0.03, '', transform=ax.transAxes, ha='left', va='top', fontsize=10,
                                 family='monospace', color=INK),
                    mark=ax.text(0.015, 0.975, '', transform=ax.transAxes, ha='left', va='top', fontsize=20,
                                 fontweight='bold', bbox=dict(fc='white', ec='none', pad=2, alpha=0.9)))
    dots = A['adaptive']['ax'].scatter([], [], s=24)
    only = A['adaptive']['ax'].text(0.5, 0.03, '', transform=A['adaptive']['ax'].transAxes, ha='center', va='bottom',
                                    fontsize=9.5, color='#bdbcb6')

    axA, axB = fig.add_subplot(gs[2, :]), fig.add_subplot(gs[3, :])
    q = np.arange(n)
    lines = {}
    label = {'full': 'full frame', 'jpeg': f'JPEG q{jtag.q if jtag else ""}', 'adaptive': 'adaptive'}
    for p in panels:
        c = style[p][1]
        lines[p, 'acc'], = axA.plot([], [], color=c, lw=2, label=label[p])
        lines[p, 'roll'], = axA.plot([], [], color=c, lw=0.8, alpha=0.35)
        lines[p, 'bytes'], = axB.plot([], [], color=c, lw=2)
    room_of = df.seq.map(ROOMS).values
    bounds = [0] + [i for i in range(1, n) if room_of[i] != room_of[i - 1]] + [n]
    for ax in (axA, axB):
        ax.set_xlim(args.start, stop)
        for kk, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
            ax.axvspan(a, b, color=['#f3f2ee', '#e9f0fb'][kk % 2], zorder=0)
            if ax is axA and b > args.start and a < stop:
                ax.text(max(a, args.start) + (stop - args.start) * 0.005, 1.0, room_of[a], fontsize=8.5,
                        color=MUTED, va='top')
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    cursorA, cursorB = axA.axvline(args.start, color=INK, lw=1), axB.axvline(args.start, color=INK, lw=1)
    axA.set_ylim(0, 1.04); axA.set_ylabel('running\naccuracy')
    axA.legend(loc='lower left', frameon=False, fontsize=9, ncol=3)
    axA.text(0.995, 0.03, 'thin: last 100 queries', transform=axA.transAxes, ha='right', fontsize=8, color=MUTED)
    axB.set_yscale('log'); axB.set_ylim(8, 30000); axB.set_ylabel('running mean\nbytes / query')
    axB.set_xlabel('query (one per frame)')

    def draw(i):
        r = df.iloc[i]
        f = X[i]
        truth = {1: 'person', 0: 'empty', -1: 'ambiguous'}[int(r.label)]
        head_txt.set_text(f'query {i}   ·   {room_of[i]}   ·   truth: {truth}')
        A['full']['im'].set_data(f)
        A['full']['sent'].set_text('full frame  19,200 B')
        if jtag:
            scan = jtag.encode(f)
            A['jpeg']['im'].set_data(jtag.decode(np.frombuffer(scan, np.uint8)))
            A['jpeg']['sent'].set_text(f'JPEG  {len(scan):,} B')
        # what the adaptive server received
        sent, view = [], np.zeros_like(f)
        if r.used_code:
            sent.append('code')
        if r.thumbed:
            sent.append('thumb')
            view = np.kron(piropo.thumb(f, 16, 12), np.ones((10, 10))).astype(np.uint8)
        if r.explored:
            sent.append('full')
            view = f
        A['adaptive']['im'].set_data(view)
        only.set_text('' if (r.thumbed or r.explored) else 'the server receives only these 96 bits')
        s = np.searchsorted(snap['query'], i, side='right') - 1
        if r.used_code and s >= 0:
            rects = snap['rects'][s]
            bits = code_bits(cell_sums(f, rects), IntervalCode(np.arange(len(rects)), snap['lo'][s], snap['hi'][s], rects))
            dots.set_offsets(np.c_[rects[:, 0] + rects[:, 4] // 2, rects[:, 1] + rects[:, 4] // 2])
            dots.set_facecolor(np.where(bits[:, None] == 1, matplotlib.colors.to_rgba_array([ORANGE] * len(bits)),
                                        (0, 0, 0, 0)))
            dots.set_edgecolor(np.where(bits[:, None] == 1, (1, 1, 1, 1), (0.85, 0.85, 0.85, 1)))
        else:
            dots.set_offsets(np.empty((0, 2)))
        A['adaptive']['sent'].set_text(f"{' + '.join(sent)}  {int(r.up_bytes):,} B" + ('  · new code' if r.reconfig else ''))
        for p in panels:
            c = C[p]
            ans = c['answer'][i]
            right = r.label >= 0 and ans == r.label
            A[p]['mark'].set_text('' if r.label < 0 else ('✓' if right else '✗'))
            A[p]['mark'].set_color(GOOD if right else BAD)
            A[p]['stat'].set_text(f"answer: {'person' if ans else 'empty':7s} this query: {c['bytes'][i]:>6,.0f} B\n"
                                  f"running: {c['acc'][i]:.3f} correct, {c['mean_bytes'][i]:>6,.0f} B/query")
            lines[p, 'acc'].set_data(q[args.start:i + 1], c['acc'][args.start:i + 1])
            lines[p, 'roll'].set_data(q[args.start:i + 1], c['roll'][args.start:i + 1])
            lines[p, 'bytes'].set_data(q[args.start:i + 1], c['mean_bytes'][args.start:i + 1])
        cursorA.set_xdata([i, i]); cursorB.set_xdata([i, i])

    if fmt == 'mp4':
        plt.rcParams['animation.ffmpeg_path'] = ff
        writer = animation.FFMpegWriter(fps=args.fps, bitrate=3000)
    else:
        writer = animation.PillowWriter(fps=args.fps)
    t0 = time.time()
    with writer.saving(fig, str(out), args.dpi):
        for j, i in enumerate(queries):
            draw(i)
            writer.grab_frame()
            if j % 200 == 0:
                print(f'  {j}/{len(queries)} frames ({time.time() - t0:.0f}s)', flush=True)
    print(f'wrote {out} ({len(queries)} frames, {time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
