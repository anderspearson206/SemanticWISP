# CLAUDE.md

Research repo: semantic / task-oriented communication for battery-free (WISP) camera person detection. See `README.md` for the project goals and layout.

## Environment

- Windows laptop, **CPU only** (torch/torchvision are `+cpu` builds). Use the venv interpreter: `.venv/Scripts/python.exe` (Python 3.13; numpy, pandas, scikit-learn, Pillow, matplotlib, torch, torchvision, jupyter).
- Locally: exploration, small models and quick checks only. Large training runs go to **Hyak** (the user's compute cluster); the pipeline is in `cluster/` (`cluster/README.md`; `cluster\hyak.ps1 push|submit|status|pull`), written while Hyak was down and not yet run against it. Hyak jobs are requeued without notice, so training scripts must checkpoint every epoch and resume (pattern: `cluster/smoke_test.py`). Keep training code device-agnostic (`cuda` if available, else `cpu`) and runnable as a script with paths and configs passed in, so it can move to the cluster.
- Colab (GPU) was used for the early proof-of-concept notebooks in `colab_notebooks/`.
- Run a notebook headless from its own directory (paths inside are relative to it):
  `cd data_exploration && ../.venv/Scripts/jupyter.exe nbconvert --to notebook --execute --inplace <nb>.ipynb`
- 16 GB RAM: Claude Code kills background jobs when memory runs low. Run ResNet-50 detectors with small batches.

## Data conventions (PIROPO Room B, conventional camera)

- Root: `data/roomb_piropo/Room B/conv_2B/conv_2B/` (note the space in `Room B` and the doubled `conv_2B`).
- Sequence folders `conv2B_<name>`; annotations `Ground_Truth_Annotations/groundTruth_conv2B_<name>.csv` exist only for `training`, `training_seats`, `test1`, `test2`, `test3`.
- CSV row `i` (1-based) is the i-th JPEG in **sorted file-name order** (file names are timestamps). Columns: `frame, x1, y1[, x2, y2]`, one head point per person in 704×576 pixels; `(0, 0)` means an empty slot. `test2` has up to 2 people, the rest at most 1. A few points have negative x (person partly out of frame).
- About 43 corrupted frames (block-shifted JPEGs) are labelled 0 regardless of content. Flag them by mean gray level more than 15 from the sequence median, and exclude them. The mean-level flag misses some: about 10 frames in the annotated sequences have their lower part filled with one gray level (many rows with std < 2) and are not flagged, and mildly darkened frames slip under the 15-level threshold.
- Annotations miss people: about 8% of empty-labelled `training` frames (1% in `training_seats`) get a COCO-detector person score of 0.9 or higher and show visible people. Treat measured AUC as capped by label noise; full-resolution detector scores (`results/coco_detectors/`) can serve as pseudo-labels.
- Split by sequence or by contiguous time block. Adjacent frames are near-duplicates.
- `training_bg` = empty room; the layout differs in `training_seats`, so a fixed background does not transfer. It is also about 25 gray levels off from the empty frames of every annotated sequence (lighting), so take an empty-room reference from the sequence itself.

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
- Project direction (user, 2026-11-10): the method must adapt online to any room with no offline retraining; the claims to support are that the tag needs only certain pixels, and how long readaptation takes. `OnlineLearner` is the method; `BatchLearner` (store frames, refit) is a baseline.
- Adaptive loop: `semwisp/server/adaptive.py` (`AdaptiveServer(teacher, learner)`, `OnlineLearner`, `BatchLearner`, `CachedTeacher`) with `semwisp/tag/interval.py` (`IntervalCodeTag`: ops `full`, `config`, `code`, `thumb`; candidate-region tables `CELLS`, `sparse(table, sample)`, `lattice(block)`, rows are `x0, y0, w, h, sample`). `semwisp/metrics.py`: `balanced_accuracy`, `summary`, `readaptation`. `semwisp.frames.load_mall_tile()` gives one Mall tile as a second scene (labels 1/0/-1). Run it with `replay(..., reset=False)` so tag and server state persist across sequences. The teacher is the cached COCO detector on 160×120 frames (`results/coco_detectors/frcnn_mnv3_320_160x120_s2.npz`, every 2nd frame of the annotated sequences); it is only about 0.86 balanced-accurate, and PIROPO's sequences are staged scenarios, so first-pass results understate and replayed passes overstate what a real room would give.
- Tiered checks: `semwisp/server/tiered.py` (`TieredServer`, `ThumbDecoder`) checks the code against a 16×12 thumbnail (tag op `thumb`) before fetching a full frame. JPEG baseline: `semwisp.tag.JpegTag` (header not counted; also a `jpeg` method in `thumbnail_sweep.py` and `jpegWxHqQ` conditions in `coco_detector_eval.py`). Every code a run deployed (for pictures): `experiments/code_snapshots.py` replays a `tiered_checks.py` run, checks it reproduces the stored one, and saves the codes to `results/tiered_checks/snapshots/` (kept out of the run folder's top level, which the notebook globs). Detector scores for the Mall tile: `results/coco_detectors/frcnn_mnv3_320_mall_tile_160x120.npz` (`coco_detector_eval.py --mall-tile`). DINO ViT-S/8 weights: `data/cache/weights/dino_vits8_hf.bin` (Hugging Face `facebook/dino-vits8`; `dl.fbaipublicfiles.com` served an expired certificate).
- pandas 3 stores strings with their own dtype, not `object`: when saving a DataFrame to npz, convert non-numeric columns with `np.array(col.tolist(), dtype=str)` or they are pickled.
- Experiments: a resumable script in `experiments/` writes to `results/<experiment>/` (one file per configuration; skip if it exists), and a notebook next to it analyses the cached results. Run scripts from the repo root, e.g. `.venv/Scripts/python.exe experiments/coco_detector_eval.py --help`.
- Notebooks add the repo root to `sys.path` (`sys.path.insert(0, str(Path('..').resolve()))`) to import `semwisp`.

## Teacher screen (2026-11-12, `experiments/teacher_eval.py`)

- Candidate teachers are torchvision COCO detectors given `model@min_size[+flip]`. They score the cached 160×120 gray frames, the same arrays the replays use. Results go to `results/teachers/<teacher>_s<stride>/<sequence>.npz` (`t, idx, score, box`; `mall_tile` with `--mall`). The log is `results/teachers/screen_s8.log`.
- Screened on every 8th frame of the annotated sequences, with corrupted frames excluded. Done: `frcnn_mnv3_320@320` (the current teacher), `frcnn_mnv3_320@640`, `frcnn_mnv3@800`, `fcos_r50@480`, `retina_r50v2@480`, `kprcnn_r50@480`. Not done: `frcnn_r50v2@480` and `kprcnn_r50@800` (the run was killed for low memory). CPU cost is 0.11 s/frame for the current teacher, 0.8–1.1 for FCOS and RetinaNet, and 1.7 for Keypoint R-CNN.
- The current teacher's errors are one-sided. Its person labels (≥ 0.6) are 2% wrong (0.2% once the annotations' missed people are counted as person). About **32% of its empty labels (≤ 0.3) show a person**: at 160×120 it often gives a visible person 0.2–0.3. A stricter empty cutoff helps (≤ 0.05: 6% wrong, but only 49% of frames get a label).
- Ranking: AUC against labels where the full-resolution detector's ≥ 0.9 frames count as person, and the share of frames a teacher can label with ≤ 1% wrong labels (cutoffs set from the labels, so this is for ranking only):
  - Room A (`training`, `test1`, `test3`): current 0.917 / 65%, FCOS 0.992 / 88%, RetinaNet 0.990 / 70%, Keypoint R-CNN 0.996 / 95%.
  - Room A2 (`training_seats`, `test2`, mostly seated and half-hidden people): current 0.922 / 45%, FCOS 0.959 / 51%, RetinaNet 0.945 / 47%, Keypoint R-CNN 0.906 / 31%.
- **FCOS at 480 is the only candidate better than the current teacher in both rooms.** Keypoint R-CNN is best in A and worst in A2. Upscaling the current model's input (`frcnn_mnv3_320@640`) hurts (AUC 0.844), and `frcnn_mnv3@800` is better (0.935) but can't give clean empty labels at any cutoff.
- Score scales differ by model (FCOS mostly 0.2–0.8), so the loop's fixed cutoffs (`pos_thr=0.6, neg_thr=0.3` in `AdaptiveServer`) don't carry over between teachers.
- The comparison scripts were scratch work and are not in the repo yet.

## Next steps

1. Move the teacher comparison into `experiments/teacher_eval.ipynb`. Count empty labels per room, and stop person and empty cutoffs from overlapping.
2. Score `fcos_r50@480` on every 2nd frame plus the Mall tile (`--stride 2 --mall`, about 5,400 frames, about 1.5 h CPU). First add a `--batch` option and use a small batch, for memory.
3. Choose FCOS cutoffs without labels (e.g. from its score distribution, or asymmetric with a strict empty cutoff). Then rerun the `tiered_checks.py` streams with FCOS as a third teacher. Expect fewer *unsure* full frames and a higher detector-taught ceiling.
4. A supervised ceiling for the 96-pixel code: a batch fit with true labels on each room's own frames, scored on held-out time blocks. Adaptive results can then be stated as "within X of the supervised code after N queries, at Y bytes/query".
5. A bound test for the pixel-prior idea (no new models). Use the head-annotation density as a perfect "where can a person appear" prior, combine it with per-pixel noise from a few empty frames, and build a no-label 96-pixel code. Near the label-using best code's 0.95 means an affordance prior (depth and floor/seat segmentation, or generative insertion of the object into the empty frame) is worth building. Near the uniform lattice (0.81–0.88) means it isn't. Attention maps fit queries about objects already in the frame (door open, seat occupied) better than "where will X appear".
6. Optional: finish `frcnn_r50v2@480` and `kprcnn_r50@800` (same command, resumes; small batch).

## Working style

- Notebooks: markdown cells explain *why* and state findings; keep findings consistent with the actual outputs after re-running.
- Never commit or push (global rule); leave changes for the user.
