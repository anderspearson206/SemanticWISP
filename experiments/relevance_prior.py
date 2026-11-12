"""Where should the tag look? Pixel-importance priors from one empty-room frame and a query, before any person
has been seen.

A large pretrained model on the server looks at one baseline frame of the empty room and says which pixels
matter for a query. Priors (each a 120x160 map):
  grad_person   input-gradient relevance of the COCO detector's "person" evidence (SmoothGrad): how much a
                small change of each pixel would move the person score
  grad_chair    the same for "chair": a different query on the same frame and model
  vit_attn      a supervised ViT-B/16 (ImageNet): last-layer attention from the class token to the patches
  dino_attn     DINO ViT-S/8 (self-supervised; the model known for clean object attention maps), the same
  edges         Sobel gradient magnitude of the frame: a control with no learned model at all
  uniform       a regular lattice; random: uniformly random pixels
They are scored three ways:
  maps     against where people really are (evaluation only, uses the labels): `heads`, a smoothed density of
           annotated head points, and `discrim`, the per-pixel AUC of |frame - baseline| between person and
           empty frames (where a person's presence changes the pixel).
  zero     a 96-pixel code with no labels at all: bit = pixel more than 15 gray levels from the baseline,
           score = bits set; AUC on the room's labelled frames.
  online   the adaptive loop (AdaptiveServer + OnlineLearner, detector teacher, PIROPO stream, one pass) with the
           tag's candidate pixels restricted to the prior's top 300, against the full 4-pixel lattice (1,200
           candidates), for several bootstrap lengths: does a prior let the loop start from fewer labels?

Rooms: `A` (layout of `training`, `test1`, `test3`) and `A2` (rearranged: `training_seats`, `test2`). The
baseline is the room's first frame that is labelled empty, that the detector calls empty with confidence
(score <= 0.3, the adaptive loop's threshold) and that is not corrupted.

Results: results/relevance_prior/maps.npz (the priors and empirical maps), zero.csv, online_<prior>_b<n>.npz.
Existing files are skipped.

  python experiments/relevance_prior.py --stage maps
  python experiments/relevance_prior.py --stage zero
  python experiments/relevance_prior.py --stage online
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from semwisp import piropo
from semwisp.frames import load_frames
from semwisp.piropo import H, W

ROOMS = dict(A=['training', 'test1', 'test3'], A2=['training_seats', 'test2'])
DET = piropo.REPO / 'results' / 'coco_detectors' / 'frcnn_mnv3_320_160x120_s2.npz'
PERSON, CHAIR = 1, 62            # COCO category ids in torchvision detection models
# DINO ViT-S/8 as published by Meta on Hugging Face (dl.fbaipublicfiles.com served an expired certificate)
DINO_URL = 'https://huggingface.co/facebook/dino-vits8/resolve/main/pytorch_model.bin'
PRIORS = ['grad_person', 'grad_chair', 'vit_attn', 'dino_attn', 'edges', 'uniform', 'random']


def baseline(frames, det, seq):
    s, sc = frames[seq], det[f'score_{seq}']
    idx = det[f'idx_{seq}']
    for i, score in zip(idx, sc):
        # flag_bad misses corrupted frames whose lower part is filled with one gray level: reject flat rows too
        flat = (s['X'][i].std(1) < 2).mean() > 0.05
        if s['label'][i] == 0 and not s['bad'][i] and score <= 0.3 and not flat:
            return s['X'][i], int(i)
    raise ValueError(seq)


# ---------------------------------------------------------------- priors

def grad_map(model, frame, cls, n=16, noise=8.0, seed=0):
    """SmoothGrad of the detector's evidence for class `cls`: the log-sum-exp over all region proposals of the
    log-probability of `cls` (smooth, defined even when nothing is detected), differentiated w.r.t. the pixels."""
    captured = {}
    hook = model.roi_heads.box_predictor.register_forward_hook(lambda m, i, o: captured.__setitem__('logits', o[0]))
    g = torch.Generator().manual_seed(seed)
    total = torch.zeros(H, W)
    for _ in range(n):
        x = (torch.as_tensor(frame, dtype=torch.float32) + noise * torch.randn(H, W, generator=g)).clamp(0, 255) / 255
        x.requires_grad_(True)
        model([x[None].expand(3, H, W)])
        lp = torch.log_softmax(captured['logits'], 1)[:, cls]
        torch.logsumexp(lp, 0).backward()
        total += x.grad.abs()
    hook.remove()
    return (total / n).numpy()


def vit_attention(frame):
    """torchvision ViT-B/16 (ImageNet-1k): mean over heads of the last layer's class-token attention."""
    from torchvision.models import vit_b_16, ViT_B_16_Weights
    from torch.nn import functional as F
    m = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1).eval()
    att = {}
    blk = m.encoder.layers[-1]
    orig = blk.self_attention.forward
    def fwd(q, k, v, **kw):
        kw.update(need_weights=True, average_attn_weights=True)
        out, w = orig(q, k, v, **kw)
        att['w'] = w
        return out, w
    blk.self_attention.forward = fwd
    x = torch.as_tensor(frame, dtype=torch.float32)[None, None].expand(1, 3, H, W) / 255
    x = F.interpolate(x, (224, 224), mode='bilinear', align_corners=False)
    x = (x - torch.tensor([0.485, 0.456, 0.406])[:, None, None]) / torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    with torch.no_grad():
        m(x)
    a = att['w'][0, 0, 1:].reshape(1, 1, 14, 14)
    return F.interpolate(a, (H, W), mode='bilinear', align_corners=False)[0, 0].numpy()


class _Block(torch.nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.norm1, self.norm2 = torch.nn.LayerNorm(d, eps=1e-6), torch.nn.LayerNorm(d, eps=1e-6)
        self.qkv, self.proj, self.heads = torch.nn.Linear(d, 3 * d), torch.nn.Linear(d, d), heads
        self.fc1, self.fc2 = torch.nn.Linear(d, 4 * d), torch.nn.Linear(4 * d, d)

    def forward(self, x):
        B, N, D = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(B, N, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        a = (q @ k.transpose(-2, -1) * (D // self.heads) ** -0.5).softmax(-1)
        self.attn = a
        x = x + self.proj((a @ v).transpose(1, 2).reshape(B, N, D))
        return x + self.fc2(torch.nn.functional.gelu(self.fc1(self.norm2(x))))


def dino_attention(frame, cache):
    """DINO ViT-S/8: mean over heads of the last layer's class-token attention, at the native 160x120 input
    (20x15 patches; position embeddings interpolated). A minimal ViT so that only the weights are downloaded."""
    from torch.nn import functional as F
    path = cache / 'dino_vits8_hf.bin'
    if not path.exists():
        torch.hub.download_url_to_file(DINO_URL, str(path))
    raw = torch.load(path, map_location='cpu', weights_only=True)          # tensors only, no code
    sd = {'cls_token': raw['embeddings.cls_token'], 'pos_embed': raw['embeddings.position_embeddings']}
    d, heads, p = 384, 6, 8
    patch = torch.nn.Conv2d(3, d, p, p)
    patch.weight.data = raw['embeddings.patch_embeddings.projection.weight']
    patch.bias.data = raw['embeddings.patch_embeddings.projection.bias']
    blocks = []
    for i in range(12):
        L = lambda name: raw[f'encoder.layer.{i}.{name}']
        b = _Block(d, heads)
        b.load_state_dict({
            'norm1.weight': L('layernorm_before.weight'), 'norm1.bias': L('layernorm_before.bias'),
            'norm2.weight': L('layernorm_after.weight'), 'norm2.bias': L('layernorm_after.bias'),
            'qkv.weight': torch.cat([L(f'attention.attention.{n}.weight') for n in ('query', 'key', 'value')]),
            'qkv.bias': torch.cat([L(f'attention.attention.{n}.bias') for n in ('query', 'key', 'value')]),
            'proj.weight': L('attention.output.dense.weight'), 'proj.bias': L('attention.output.dense.bias'),
            'fc1.weight': L('intermediate.dense.weight'), 'fc1.bias': L('intermediate.dense.bias'),
            'fc2.weight': L('output.dense.weight'), 'fc2.bias': L('output.dense.bias')})
        blocks.append(b.eval())
    x = torch.as_tensor(frame, dtype=torch.float32)[None, None].expand(1, 3, H, W) / 255
    x = (x - torch.tensor([0.485, 0.456, 0.406])[:, None, None]) / torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    with torch.no_grad():
        t = patch(x).flatten(2).transpose(1, 2)                                  # (1, 300, 384)
        pos = sd['pos_embed']
        grid = pos[:, 1:].reshape(1, 28, 28, d).permute(0, 3, 1, 2)
        grid = F.interpolate(grid, (H // p, W // p), mode='bicubic', align_corners=False).flatten(2).transpose(1, 2)
        t = torch.cat([sd['cls_token'] + pos[:, :1], t + grid], 1)
        for b in blocks:
            t = b(t)
    a = blocks[-1].attn[0, :, 0, 1:].mean(0).reshape(1, 1, H // p, W // p)
    return F.interpolate(a, (H, W), mode='bilinear', align_corners=False)[0, 0].numpy()


def edges(frame):
    x = frame.astype(float)
    gx, gy = np.zeros_like(x), np.zeros_like(x)
    gx[:, 1:-1], gy[1:-1] = x[:, 2:] - x[:, :-2], x[2:] - x[:-2]
    return np.hypot(gx, gy)


def blur(m, sigma):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(m, sigma)


# ---------------------------------------------------------------- empirical maps (labels; evaluation only)

def head_density(seqs, sigma=4.0):
    raw = piropo.load_sequences(names=seqs)
    m = np.zeros((H, W))
    for n in seqs:
        for pts in raw[n]['points']:
            for x, y in pts:
                c, r = int(np.clip(x * W / 704, 0, W - 1)), int(np.clip(y * H / 576, 0, H - 1))
                m[r, c] += 1
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(m, sigma)


def discrim(frames, seqs, base, stride=2):
    """Per-pixel AUC of |frame - baseline| between person and empty frames, folded so 0.5 = uninformative."""
    X = np.concatenate([frames[n]['X'][::stride][~frames[n]['bad'][::stride]] for n in seqs]).astype(np.int16)
    y = np.concatenate([frames[n]['label'][::stride][~frames[n]['bad'][::stride]] for n in seqs]).astype(bool)
    d = np.abs(X - base.astype(np.int16)).reshape(len(X), -1)
    # AUC per pixel via ranks (Mann-Whitney), vectorised
    from scipy.stats import rankdata
    r = rankdata(d, axis=0)
    n1, n0 = y.sum(), (~y).sum()
    auc = (r[y].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)
    return auc.reshape(H, W)


# ---------------------------------------------------------------- selecting pixels

def pick(m, k, spacing=6, seed=0):
    """k pixels from a map, greedily by value, at least `spacing` apart (Chebyshev); 'uniform'/'random' by name."""
    if isinstance(m, str):
        if m == 'uniform':
            gy, gx = np.meshgrid(np.linspace(4, H - 5, int(np.sqrt(k * H / W))).round(),
                                 np.linspace(4, W - 5, int(np.ceil(k / int(np.sqrt(k * H / W))))).round(), indexing='ij')
            return np.column_stack([gy.ravel(), gx.ravel()]).astype(int)[:k]
        rng = np.random.default_rng(seed)
        return np.column_stack([rng.integers(0, H, k), rng.integers(0, W, k)])
    order = np.argsort(-m, axis=None)
    taken = np.zeros((H, W), bool)
    out = []
    for f in order:
        r, c = divmod(int(f), W)
        if taken[r, c]:
            continue
        out.append((r, c))
        taken[max(0, r - spacing + 1):r + spacing, max(0, c - spacing + 1):c + spacing] = True
        if len(out) == k:
            break
    return np.array(out)


# ---------------------------------------------------------------- stages

def stage_maps(args, frames, det):
    out = args.out / 'maps.npz'
    if out.exists():
        print('skip maps (exists)'); return
    from torchvision.models import detection as D
    model = D.fasterrcnn_mobilenet_v3_large_320_fpn(weights=D.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT,
                                                    box_score_thresh=0.01).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    res = {}
    for room, seqs in ROOMS.items():
        t0 = time.time()
        base, i = baseline(frames, det, seqs[0])
        res[f'{room}/baseline'], res[f'{room}/baseline_idx'] = base, i
        res[f'{room}/grad_person'] = blur(grad_map(model, base, PERSON), 2)
        res[f'{room}/grad_chair'] = blur(grad_map(model, base, CHAIR), 2)
        res[f'{room}/vit_attn'] = vit_attention(base)
        res[f'{room}/dino_attn'] = dino_attention(base, piropo.REPO / 'data' / 'cache' / 'weights')
        res[f'{room}/edges'] = blur(edges(base), 2)
        res[f'{room}/heads'] = head_density(seqs)
        res[f'{room}/discrim'] = discrim(frames, seqs, base)
        print(f'maps {room}: baseline {seqs[0]}#{i} ({time.time() - t0:.0f}s)', flush=True)
    np.savez(out, **res)


def stage_zero(args, frames, det):
    out = args.out / 'zero.csv'
    if out.exists():
        print('skip zero (exists)'); return
    M = np.load(args.out / 'maps.npz')
    rows = []
    for room, seqs in ROOMS.items():
        base = M[f'{room}/baseline'].astype(np.int16)
        X = np.concatenate([frames[n]['X'][::2][~frames[n]['bad'][::2]] for n in seqs]).astype(np.int16)
        y = np.concatenate([frames[n]['label'][::2][~frames[n]['bad'][::2]] for n in seqs])
        oracle = M[f'{room}/discrim']
        for prior in PRIORS + ['discrim']:
            for seed in (range(5) if prior == 'random' else [0]):
                px = pick(prior if prior in ('uniform', 'random') else M[f'{room}/{prior}'], args.k, seed=seed)
                bits = np.abs(X[:, px[:, 0], px[:, 1]] - base[px[:, 0], px[:, 1]]) > args.tol
                rows.append(dict(room=room, prior=prior, seed=seed, auc=piropo.auc(bits.sum(1), y),
                                 discrim_at_pixels=float(oracle[px[:, 0], px[:, 1]].mean()),
                                 heads_at_pixels=float(M[f'{room}/heads'][px[:, 0], px[:, 1]].mean())))
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(df.groupby(['room', 'prior'])[['auc', 'discrim_at_pixels', 'heads_at_pixels']].mean().round(3).to_string())


def stage_online(args, frames, det):
    from semwisp.metrics import summary
    from semwisp.server import AdaptiveServer, CachedTeacher, OnlineLearner
    from semwisp.sim import replay
    from semwisp.tag import IntervalCodeTag
    from semwisp.tag.interval import lattice, sparse
    M = np.load(args.out / 'maps.npz')
    ORDER = ['training', 'test3', 'test1', 'training_seats', 'test2']
    P = {n: {f: v[::2] for f, v in frames[n].items()} for n in ORDER}
    teacher = CachedTeacher(np.concatenate([P[n]['t'] for n in ORDER]), np.concatenate([det[f'score_{n}'] for n in ORDER]))
    for prior in ['lattice'] + PRIORS:
        for boot in args.bootstraps:
            out = args.out / f'online_{prior}_b{boot}.npz'
            if out.exists():
                print(f'skip {out.name} (exists)'); continue
            if prior == 'lattice':
                table = sparse(lattice(4), 4)
            else:
                px = pick(prior if prior in ('uniform', 'random') else M[f'A/{prior}'], args.candidates, spacing=3)
                table = np.array([(c, r, 1, 1, 1) for r, c in px])
            t0 = time.time()
            server = AdaptiveServer(teacher, OnlineLearner(k=args.k, table=table), bootstrap=boot)
            df = replay(P, IntervalCodeTag(), server, reset=False).df
            meta = dict(prior=prior, bootstrap=boot, candidates=len(table), seconds=round(time.time() - t0),
                        summary=summary(df))
            np.savez(out, meta=json.dumps(meta),
                     **{c: np.array(df[c].tolist(), dtype=str) if c == 'seq' else df[c].values for c in df.columns})
            s = meta['summary']
            print(f"online {prior} b{boot}: balanced accuracy {s['balanced_accuracy']:.3f}, {s['up_bytes_per_query']:.0f} B/query, "
                  f"full frames {s['full_frame_rate']:.1%} ({meta['seconds']}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['maps', 'zero', 'online'])
    ap.add_argument('--k', type=int, default=96, help='pixels in the code (96 bits = one EPC)')
    ap.add_argument('--tol', type=int, default=15, help='zero-label code: gray levels from the baseline that set a bit')
    ap.add_argument('--candidates', type=int, default=300, help='online stage: candidate pixels taken from a prior')
    ap.add_argument('--bootstraps', type=int, nargs='+', default=[25, 50, 200])
    ap.add_argument('--out', type=Path, default=piropo.REPO / 'results' / 'relevance_prior')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    frames, det = load_frames(sum(ROOMS.values(), [])), np.load(DET)
    dict(maps=stage_maps, zero=stage_zero, online=stage_online)[args.stage](args, frames, det)


if __name__ == '__main__':
    main()
