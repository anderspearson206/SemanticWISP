"""End-to-end check of the training pipeline, and a template for real training scripts.

Trains a tiny CNN on 32x24 thumbnails of a strided subset of PIROPO (train: `training`, test: `test1`) for a
few epochs. It exercises everything a real run needs: data loading through semwisp, device selection,
a checkpoint after every epoch, resuming after a requeue, and a result file in results/.

  python cluster/smoke_test.py                       # laptop (CPU), about a minute
  bash cluster/submit.sh TIME=10:00 cluster/smoke_test.py    # on Hyak
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from semwisp import piropo


def load(name, stride):
    s = piropo.load_sequences(names=[name])[name]
    X = np.stack([piropo.load_gray(f) for f in s['files'][::stride]])
    keep = ~piropo.flag_bad(X)
    return (torch.from_numpy(piropo.thumb(X[keep], 32, 24) / 255.0).unsqueeze(1),
            torch.from_numpy(s['label'][::stride][keep].astype(np.float32)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--stride', type=int, default=10)
    ap.add_argument('--out', type=Path, default=piropo.REPO / 'results' / '_smoke')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    result, ckpt = args.out / 'result.json', args.out / 'ckpt.pt'
    if result.exists():
        print(f'skip: {result} exists')
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'torch {torch.__version__}, device {device}'
          + (f' ({torch.cuda.get_device_name(0)})' if device == 'cuda' else ''), flush=True)
    t0 = time.time()
    (Xtr, ytr), (Xte, yte) = load('training', args.stride), load('test1', args.stride)
    print(f'loaded {len(Xtr)} train / {len(Xte)} test frames in {time.time() - t0:.0f}s', flush=True)

    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                          nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
                          nn.Flatten(), nn.Linear(16, 1)).to(device)
    opt = torch.optim.Adam(model.parameters(), 3e-3)
    start = 0
    if ckpt.exists():                      # a requeued job restarts the script: continue where it stopped
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state['model']); opt.load_state_dict(state['opt']); start = state['epoch']
        print(f'resumed from epoch {start}')

    Xtr, ytr, Xte = Xtr.to(device), ytr.to(device), Xte.to(device)
    for epoch in range(start, args.epochs):
        model.train()
        for idx in torch.randperm(len(Xtr), device=device).split(64):
            loss = nn.functional.binary_cross_entropy_with_logits(model(Xtr[idx]).squeeze(1), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.inference_mode():
            auc = piropo.auc(model(Xte).squeeze(1).cpu().numpy(), yte.numpy())
        print(f'epoch {epoch + 1}/{args.epochs}: last loss {loss.item():.3f}, test1 AUC {auc:.3f}', flush=True)
        tmp = ckpt.with_suffix('.tmp')     # write then rename, so a job killed mid-save leaves the old checkpoint
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), epoch=epoch + 1), tmp)
        tmp.replace(ckpt)

    result.write_text(json.dumps(dict(device=device, torch=torch.__version__, epochs=args.epochs,
                                      n_train=len(Xtr), n_test=len(Xte), test1_auc=float(auc))))
    print(f'wrote {result}')


if __name__ == '__main__':
    main()
