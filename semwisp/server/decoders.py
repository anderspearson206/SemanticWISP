"""Server-side decoders: received image-like payload -> person score."""
from pathlib import Path

import numpy as np
import torch
from torch import nn


def _net(h, w, width=16):
    """Small CNN for any input from 3x4 up to 120x160: halve the resolution while it has at least 12 rows,
    then pool to 3x4 so coarse position survives."""
    layers, c_in, c = [], 1, width
    for _ in range(4):
        layers += [nn.Conv2d(c_in, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU()]
        if h >= 12:
            layers.append(nn.MaxPool2d(2))
            h, w = h // 2, w // 2
        c_in, c = c, min(2 * c, 64)
    return nn.Sequential(*layers, nn.AdaptiveAvgPool2d((3, 4)), nn.Flatten(), nn.Dropout(0.2), nn.Linear(c_in * 12, 1))


class CNNDecoder:
    def __init__(self, h, w, device=None, seed=0):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(seed)
        self.net = _net(h, w).to(self.device)

    def fit(self, X, y, epochs=12, batch=128, lr=2e-3, ckpt=None, log=print):
        """X: (n, h, w) floats in [0, 1]; y: 0/1. With ckpt (a path), training resumes after an interruption."""
        X = torch.as_tensor(np.asarray(X, np.float32)).unsqueeze(1).to(self.device)
        y = torch.as_tensor(np.asarray(y, np.float32)).to(self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr, weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * -(-len(X) // batch))
        start, ckpt = 0, Path(ckpt) if ckpt else None
        if ckpt and ckpt.exists():
            st = torch.load(ckpt, map_location=self.device)
            self.net.load_state_dict(st['net']); opt.load_state_dict(st['opt']); sched.load_state_dict(st['sched'])
            start = st['epoch']
            log(f'  resumed from epoch {start}')
        for epoch in range(start, epochs):
            self.net.train()
            total = 0.0
            for idx in torch.randperm(len(X), device=self.device).split(batch):
                loss = nn.functional.binary_cross_entropy_with_logits(self.net(X[idx]).squeeze(1), y[idx])
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
                total += loss.item() * len(idx)
            log(f'  epoch {epoch + 1}/{epochs}: train loss {total / len(X):.4f}')
            if ckpt:
                tmp = ckpt.with_suffix('.tmp')
                torch.save(dict(net=self.net.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                                epoch=epoch + 1), tmp)
                tmp.replace(ckpt)
        self.net.eval()
        return self

    @torch.inference_mode()
    def score(self, x):
        """Logit for one received image (h, w)."""
        return self.net(torch.as_tensor(x, dtype=torch.float32, device=self.device)[None, None]).item()
