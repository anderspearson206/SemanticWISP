# CLAUDE.md

Research repo: semantic / task-oriented communication for battery-free (WISP) camera person detection. See `README.md` for the project goals and layout.

## Environment

- Windows laptop, **CPU only** (torch/torchvision are `+cpu` builds). Use the venv interpreter: `.venv/Scripts/python.exe` (Python 3.13; numpy, pandas, scikit-learn, Pillow, matplotlib, torch, torchvision, jupyter).
- Locally: exploration, small models and quick checks only. Large training runs go to **Hyak** (the user's compute cluster); the pipeline is in `cluster/` (`cluster/README.md`; `cluster\hyak.ps1 push|submit|status|pull`), written while Hyak was down and not yet run against it. Hyak jobs are requeued without notice, so training scripts must checkpoint every epoch and resume (pattern: `cluster/smoke_test.py`). Keep training code device-agnostic (`cuda` if available, else `cpu`) and runnable as a script with paths and configs passed in, so it can move to the cluster.
- Colab (GPU) was used for the early proof-of-concept notebooks in `colab_notebooks/`.
- Run a notebook headless from its own directory (paths inside are relative to it):
  `cd data_exploration && ../.venv/Scripts/jupyter.exe nbconvert --to notebook --execute --inplace <nb>.ipynb`
- Not a git repo yet.

## Data conventions (PIROPO Room B, conventional camera)

- Root: `data/roomb_piropo/Room B/conv_2B/conv_2B/` (note the space in `Room B` and the doubled `conv_2B`).
- Sequence folders `conv2B_<name>`; annotations `Ground_Truth_Annotations/groundTruth_conv2B_<name>.csv` exist only for `training`, `training_seats`, `test1`, `test2`, `test3`.
- CSV row `i` (1-based) is the i-th JPEG in **sorted file-name order** (file names are timestamps). Columns: `frame, x1, y1[, x2, y2]`, one head point per person in 704×576 pixels; `(0, 0)` means an empty slot. `test2` has up to 2 people, the rest at most 1. A few points have negative x (person partly out of frame).
- About 43 corrupted frames (block-shifted JPEGs) are labelled 0 regardless of content. Flag them by mean gray level more than 15 from the sequence median, and exclude them.
- Annotations miss people: about 8% of empty-labelled `training` frames (1% in `training_seats`) get a COCO-detector person score of 0.9 or higher and show visible people. Treat measured AUC as capped by label noise; full-resolution detector scores (`results/coco_detectors/`) can serve as pseudo-labels.
- Split by sequence or by contiguous time block. Adjacent frames are near-duplicates.
- `training_bg` = empty room; the layout differs in `training_seats`, so a fixed background does not transfer.

## Mall dataset

- `data/mall_dataset/mall_dataset/frames/seq_%06d.jpg` (1-based), 640×480; `mall_gt.mat` head points are only annotated inside the ROI (`perspective_roi.mat`), so the top-corner shop fronts hold unannotated people and mannequins.
- Presence data comes from tiles: `data/mall_dataset/cropped/labels_<grid>.csv` with label 1 / 0 / -1 (ambiguous: a body reaches in without its head, or the tile is outside the ROI). Use only 0 and 1.

## Conventions shared with the Colab experiments

- WISP input: grayscale **160×120** (`H, W = 120, 160`), bilinear resize from native resolution.
- Payload budgets in bytes at 8 bits per value: 10, 40, 160, 640, 2560 (roughly 1 EPC up to many). Thumbnail grids must divide 160×120 exactly (4×3, 8×6, 16×12, 32×24, 40×30, 80×60).
- Prefer threshold-free metrics (AUC) or per-sequence results: the labelled sequences are 55–80% positive.

## Code layout

- `semwisp/` holds shared helpers (`piropo.py`: `load_sequences`, `load_gray`, `flag_bad`, `thumb`, `auc`; `mall.py`: Mall dataset `load_gt`, `load_roi`, `grid_boxes`, `crop_labels`). New code should import from here rather than copy notebook code. The exploration notebook predates it and still has its own copies.
- WISP simulator (`semwisp/sim.py`): `replay(seqs, tag, server, link, sensor)` runs one query per frame in time order. The server sends `Request`s, the tag answers each with real payload bytes, and a `Ledger` counts what the query cost (rounds, downlink bits, payload bytes, EPCs, pixels read, MCU operations, RAM). New methods are a `Tag` subclass in `semwisp/tag/` and/or a `Server` subclass in `semwisp/server/`; tag code is integer-only so it can be ported to C or Verilog. Frames come from `semwisp.frames.load_frames` (cached in `data/cache/`). `semwisp/cost.py` energy constants are unmeasured (`None`): do not report energy until they are filled in. Plumbing tests: `.venv/Scripts/python.exe tests/test_sim.py`.
- Experiments: a resumable script in `experiments/` writes to `results/<experiment>/` (one file per configuration; skip if it exists), and a notebook next to it analyses the cached results. Run scripts from the repo root, e.g. `.venv/Scripts/python.exe experiments/coco_detector_eval.py --help`.
- Notebooks add the repo root to `sys.path` (`sys.path.insert(0, str(Path('..').resolve()))`) to import `semwisp`.

## Working style

- Notebooks: markdown cells explain *why* and state findings; keep findings consistent with the actual outputs after re-running.
- Never commit or push (global rule); leave changes for the user.
