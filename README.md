# SemanticWISP

Semantic (task-oriented) communication for battery-free person detection on a WISP.

A WISP (Wireless Identification and Sensing Platform) with a camera is powered and read entirely by an RFID reader, so every operation and every transmitted byte costs harvested energy. Instead of sending images, the tag sends a tiny **task-oriented encoding** (a few bytes, ideally a single EPC) and a server-side decoder answers one question: *is there a person in the room?*

## Research directions

- **Tag-side encoders**, from cheapest to most capable:
  - plain downsampling (block averaging / max pooling to a thumbnail),
  - reading out only selected camera pixels, so the sensor never captures the full frame,
  - small fixed or learned filters on the MCU,
  - a small quantized CNN on a low-power FPGA (e.g. Lattice iCE40UP5K).
- **Server-side decoders**: pretrained classifiers adapted to whatever the tag sends.
- **Room-specific online learning**: a sensor stays in one room, so the server can gradually learn an encoding that only has to work *for that room* (a known background and the places people actually stand), and shrink the payload to a few pixels.

## Repository layout

| Path | Contents |
|---|---|
| `colab_notebooks/` | Proof-of-concept notebooks run in Google Colab. `wisp_semantic_epc_feasibility_v5_1` compares tag encoders (thumbnails, Haar/energy features, ternary heads, iCE40UP5K CNNs with a resource/latency/energy fit model) at 10–2560 byte budgets, trained on COCO Visual Wake Words at 160×120 grayscale. |
| `data/` | Datasets (not tracked; see below). |
| `data_exploration/` | Notebooks that explore the datasets. `piropo_roomB_exploration.ipynb`: inventory, annotations, corrupted frames, WISP-resolution thumbnails, and room-specific background/pixel-selection baselines. |
| `semwisp/` | Shared Python code. `piropo.py`: sequence/label loading, corrupted-frame flag, thumbnails, AUC. `mall.py`: Mall dataset loading and crop labels. `frames.py`: cached 160×120 frames. The WISP simulator: `sim.py` (query loop between a tag and a server, with a cost ledger), `sensor.py`, `link.py` (EPC packing and loss), `cost.py` (energy model, constants still to be measured), `tag/` (tag-side encoders), `server/` (decoders and query policies). |
| `tests/` | Simulator plumbing checks on synthetic frames. |
| `experiments/` | Scripts that produce results (runnable locally or on a cluster) and notebooks that analyse them. `coco_detector_eval.py` + `.ipynb`: COCO-pretrained detectors on PIROPO at full resolution down to 16×12 thumbnails. `thumbnail_sweep.py` + `.ipynb`: uncompressed baseline and thumbnails (resolution, bit depth, averaging or subsampling) through the simulator, with a decoder trained on this room. |
| `results/` | Outputs of the experiment scripts (cached scores, logs). |
| `cluster/` | Hyak pipeline: push code, submit Slurm jobs, pull results (`cluster/README.md`). |
| `papers/` | Reference papers (WISP/WISPCam, semantic communication, deep JSCC, backscatter, EPC Gen2). |

## Data

[PIROPO](https://www.gti.ssr.upm.es/data/piropodatabase) (People in Indoor ROoms with Perspective and Omnidirectional cameras), Room B:

- `data/roomb_piropo/Room B/conv_2B/conv_2B/`: conventional camera, extracted. 13 sequences, 9,332 JPEGs at 704×576. Five of them (`training`, `training_seats`, `test1`–`test3`) have per-frame head-point annotations in `Ground_Truth_Annotations/`.
- `data/roomb_piropo/Room B/omni_1B.zip`: omnidirectional camera, not yet extracted.

Things to know before training on it (details in the exploration notebook):

- About 4–6 fps with irregular timing, and people are present in long runs, so split **by sequence**, never by random frame.
- About 43 frames are corrupted JPEGs (blocks shifted and wrapped around). They are labelled "no person" even when someone is visible, so drop them.
- The annotations miss some people: about 8% of the frames labelled empty in `training` contain a clearly visible person, according to a COCO detector and spot checks.
- The room layout changes between sequences (door, chairs), so a fixed background model breaks on `training_seats`.

[Mall dataset](https://personal.ie.cuhk.edu.hk/~ccloy/downloads_mall_dataset.html) (crowd counting, 2000 frames at 640×480, 13–53 people per frame, head points inside an ROI):

- `data/mall_dataset/mall_exploration.ipynb`: inventory, annotations, ROI, perspective, temporal correlation.
- `data/mall_dataset/cropped/`: `make_crops.py` splits each frame into a grid (default 3×3, 213×160 tiles) and labels every tile person / empty / ambiguous (`labels_3x3.csv`, crops in `frames_3x3/`); `view_crops.ipynb` browses them. Whole frames are never empty, but about 7% of 3×3 tiles are.

## Setup

A local virtual environment lives in `.venv/` (Python 3.13, with numpy, pandas, scikit-learn, Pillow, matplotlib, torch, torchvision and Jupyter). The laptop has no GPU, so local work is limited to exploration and small models. Large training runs go to the Hyak cluster: see `cluster/README.md`.

```powershell
.venv\Scripts\Activate.ps1
jupyter lab
```

The Colab notebooks expect Colab plus Google Drive (`USE_DRIVE = True`) and download COCO annotations themselves.
