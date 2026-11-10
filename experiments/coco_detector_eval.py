"""Score PIROPO Room B frames with COCO-pretrained torchvision detectors.

Per frame, the score is the highest 'person' confidence among the detections (0 if none). Each input condition
simulates what the server would receive:
  rgb      full-resolution colour frame (reference: what a normal camera sees)
  gray     full-resolution grayscale
  WxH      WISP view: grayscale 160x120 (bilinear), block-averaged to W x H; the detector upsamples it itself

Results go to results/coco_detectors/<model>_<cond>_s<stride>.npz, one file per (model, condition); existing files
are skipped, so the script can be interrupted and rerun.

  python experiments/coco_detector_eval.py --model frcnn_mnv3_320 --stride 2 --conds rgb gray 160x120 80x60 32x24 16x12
"""
import argparse, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.models import detection as D

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from semwisp import piropo

MODELS = {   # name -> (constructor, weights); box_score_thresh is lowered so most frames get a nonzero score
    'frcnn_mnv3_320': (D.fasterrcnn_mobilenet_v3_large_320_fpn, D.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT),
    'frcnn_mnv3': (D.fasterrcnn_mobilenet_v3_large_fpn, D.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT),
    'frcnn_r50v2': (D.fasterrcnn_resnet50_fpn_v2, D.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT),
}
PERSON = 1   # COCO category id in torchvision detection models


def make_input(path, cond):
    with Image.open(path) as im:
        if cond == 'rgb':
            a = np.asarray(im.convert('RGB'))
        elif cond == 'gray':
            a = np.repeat(np.asarray(im.convert('L'))[..., None], 3, -1)
        else:
            w, h = map(int, cond.split('x'))
            g = np.asarray(im.convert('L').resize((piropo.W, piropo.H), Image.BILINEAR), np.uint8)
            a = np.repeat(piropo.thumb(g, w, h).round().astype(np.uint8)[..., None], 3, -1)
    return torch.from_numpy(np.array(a)).permute(2, 0, 1).float() / 255


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='frcnn_mnv3_320', choices=MODELS)
    ap.add_argument('--conds', nargs='+', default=['rgb', 'gray', '160x120', '80x60', '32x24', '16x12'])
    ap.add_argument('--stride', type=int, default=2, help='score every n-th frame (adjacent frames are near-duplicates)')
    ap.add_argument('--seqs', nargs='+', default=piropo.ANNOTATED)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--out', type=Path, default=piropo.REPO / 'results' / 'coco_detectors')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fn, weights = MODELS[args.model]
    model = fn(weights=weights, box_score_thresh=0.01).eval().to(device)
    seqs = piropo.load_sequences(names=args.seqs)
    args.out.mkdir(parents=True, exist_ok=True)

    for cond in args.conds:
        out = args.out / f'{args.model}_{cond}_s{args.stride}.npz'
        if out.exists():
            print(f'skip {out.name} (exists)')
            continue
        res, t0 = {}, time.time()
        for name, s in seqs.items():
            idx = np.arange(0, len(s['files']), args.stride)
            scores = np.zeros(len(idx), np.float32)
            with ThreadPoolExecutor(4) as ex, torch.inference_mode():
                for b in range(0, len(idx), args.batch):
                    xs = list(ex.map(lambda i: make_input(s['files'][i], cond), idx[b:b + args.batch]))
                    for k, det in enumerate(model([x.to(device) for x in xs])):
                        p = det['scores'][det['labels'] == PERSON]
                        scores[b + k] = p.max().item() if len(p) else 0.0
            res[f'idx_{name}'], res[f'score_{name}'] = idx, scores
            print(f'{args.model} {cond} {name}: {len(idx)} frames, {time.time() - t0:.0f}s elapsed', flush=True)
        np.savez(out, **res)
        print(f'wrote {out}')


if __name__ == '__main__':
    main()
