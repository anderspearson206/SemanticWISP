"""Candidate teachers for the adaptive loop: COCO detectors scored on exactly what the server receives.

The loop's teacher sees only the full frame the tag sends: grayscale 160x120 (the cached frames of
semwisp.frames, the same arrays the replays use). A teacher here is a torchvision detector plus an input scale
(the detector's internal resize, min_size) and optionally a horizontal-flip average. Per frame the score is the
highest 'person' confidence (0 if none); the box of that detection is kept for pictures.

The current teacher is frcnn_mnv3_320@320 (results/coco_detectors/frcnn_mnv3_320_160x120_s2.npz).

Results: results/teachers/<teacher>_s<stride>/<sequence>.npz with keys t, idx, score, box (x1, y1, x2, y2 in
160x120 pixels); one file per sequence, skipped if it exists, so the script can be interrupted and rerun.
Sequence 'mall_tile' is the Mall tile of the switch stream (every frame; --mall). A sequence in progress is
checkpointed to <sequence>.part.npz every --chunk frames and resumed from there.

  python experiments/teacher_eval.py --stride 8 --teachers frcnn_mnv3_320@320 frcnn_r50v2@480 retina_r50v2@480
  python experiments/teacher_eval.py --stride 2 --mall --batch 2 --teachers fcos_r50@480
"""
import argparse, sys, time
from pathlib import Path

import numpy as np
import torch
from torchvision.models import detection as D

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from semwisp import piropo
from semwisp.frames import load_frames, load_mall_tile

MODELS = {   # name -> (constructor, weights)
    'frcnn_mnv3_320': (D.fasterrcnn_mobilenet_v3_large_320_fpn, D.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT),
    'frcnn_mnv3': (D.fasterrcnn_mobilenet_v3_large_fpn, D.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT),
    'frcnn_r50v2': (D.fasterrcnn_resnet50_fpn_v2, D.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT),
    'retina_r50v2': (D.retinanet_resnet50_fpn_v2, D.RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT),
    'fcos_r50': (D.fcos_resnet50_fpn, D.FCOS_ResNet50_FPN_Weights.DEFAULT),
    'kprcnn_r50': (D.keypointrcnn_resnet50_fpn, D.KeypointRCNN_ResNet50_FPN_Weights.DEFAULT),
}
ONE_STAGE = {'retina_r50v2', 'fcos_r50'}
PERSON = 1   # COCO category id in torchvision detection models
OUT = piropo.REPO / 'results' / 'teachers'


def parse(teacher):
    """'model@size[+flip]' -> (model, size, flip)."""
    name, flip = (teacher[:-5], True) if teacher.endswith('+flip') else (teacher, False)
    model, size = name.split('@')
    return model, int(size), flip


def build(model, size, device):
    fn, weights = MODELS[model]
    # one-stage detectors call their score floor score_thresh; torchvision silently ignores box_score_thresh there
    # (FCOS and RetinaNet results scored before 2026-11-13 have floors 0.2 and 0.05: scores below are 0)
    thresh = {'score_thresh' if model in ONE_STAGE else 'box_score_thresh': 0.01}
    # 160x120 frames: min_size scales the short side to `size`; max_size must not bind for a 4:3 frame
    return fn(weights=weights, min_size=size, max_size=size * 4 // 3 + 1, **thresh).eval().to(device)


def best_person(det):
    keep = det['labels'] == PERSON
    if not keep.any():
        return 0.0, np.zeros(4, np.float32)
    s, b = det['scores'][keep], det['boxes'][keep]
    i = int(s.argmax())
    return float(s[i]), b[i].cpu().numpy().astype(np.float32)


def score(model, X, flip, device, batch=8):
    """X uint8 (n, 120, 160) -> scores (n,), boxes (n, 4). flip: mean of the frame's and its mirror's score."""
    n = len(X)
    scores, boxes = np.zeros(n, np.float32), np.zeros((n, 4), np.float32)
    with torch.inference_mode():
        for b in range(0, n, batch):
            xs = [torch.from_numpy(np.repeat(x[None], 3, 0)).float().div(255).to(device) for x in X[b:b + batch]]
            for k, det in enumerate(model(xs)):
                scores[b + k], boxes[b + k] = best_person(det)
            if flip:
                for k, det in enumerate(model([x.flip(-1) for x in xs])):
                    scores[b + k] = (scores[b + k] + best_person(det)[0]) / 2
    return scores, boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--teachers', nargs='+', required=True, help='model@size[+flip], model one of ' + ', '.join(MODELS))
    ap.add_argument('--stride', type=int, default=2, help='score every n-th frame of each annotated sequence')
    ap.add_argument('--seqs', nargs='+', default=piropo.ANNOTATED)
    ap.add_argument('--mall', action='store_true', help='also score every frame of the Mall tile')
    ap.add_argument('--out', type=Path, default=OUT)
    ap.add_argument('--batch', type=int, default=8, help='frames per forward pass (lower it for memory)')
    ap.add_argument('--chunk', type=int, default=200, help='checkpoint a sequence every n frames')
    args = ap.parse_args()
    for t in args.teachers:
        assert parse(t)[0] in MODELS, t

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    frames = {n: s for n, s in load_frames(args.seqs).items()}
    jobs = {n: (np.arange(0, len(s['X']), args.stride), s) for n, s in frames.items()}
    if args.mall:
        m = load_mall_tile()
        jobs['mall_tile'] = (np.arange(len(m['X'])), m)

    for teacher in args.teachers:
        model_name, size, flip = parse(teacher)
        folder = args.out / f'{teacher}_s{args.stride}'
        folder.mkdir(parents=True, exist_ok=True)
        model = None
        for name, (idx, s) in jobs.items():
            out = folder / f'{name}.npz'
            if out.exists():
                print(f'skip {teacher} {name} (exists)', flush=True)
                continue
            model = model or build(model_name, size, device)
            part = out.with_name(out.stem + '.part.npz')
            sc, bx = np.zeros(len(idx), np.float32), np.zeros((len(idx), 4), np.float32)
            done = 0
            if part.exists():
                with np.load(part) as z:
                    done = len(z['score'])
                    sc[:done], bx[:done] = z['score'], z['box']
                print(f'resume {teacher} {name} at {done}/{len(idx)}', flush=True)
            t0, n0 = time.time(), done
            while done < len(idx):
                end = min(done + args.chunk, len(idx))
                sc[done:end], bx[done:end] = score(model, s['X'][idx[done:end]], flip, device, args.batch)
                done = end
                tmp = out.with_name(out.stem + '.tmp.npz')
                if done < len(idx):
                    np.savez(tmp, score=sc[:done], box=bx[:done])
                    tmp.replace(part)
                    print(f'  {teacher} {name}: {done}/{len(idx)}', flush=True)
                else:
                    np.savez(tmp, t=s['t'][idx], idx=idx, score=sc, box=bx)
                    tmp.replace(out)
                    part.unlink(missing_ok=True)
            n = max(len(idx) - n0, 1)
            print(f'{teacher} {name}: {len(idx)} frames, {time.time() - t0:.0f}s '
                  f'({(time.time() - t0) / n:.2f} s/frame)', flush=True)


if __name__ == '__main__':
    main()
