# lateral_detection

Lateral-only irrigation line detection from blueprint plans.

The dataset is a Roboflow COCO keypoint export
(`keypoints.v56-lat_only_baseline.coco`) in which every annotation is a
single two-point chord. Long laterals are represented as many chained
chords whose endpoints meet at shared vertices. We reconstruct polylines
by merging coincident endpoints, then rasterize them into a binary
segmentation mask used as the training target.

## Layout

```
lateral_detection/
├── configs/
│   ├── base.yaml                       # paths, polyline + rasterize params
│   └── train.yaml                      # model, loss, optimizer, schedule
├── credentials/                        # GCP service-account JSON (gitignored)
│   └── inference-428300-7af7f5da75dc.json
├── data/
│   ├── coco_loader.py                  # parse COCO export → ImageRecord + Chord
│   ├── polyline_builder.py             # chords → polylines (endpoint merging)
│   ├── rasterize.py                    # polylines → binary mask
│   ├── augmentation.py                 # flips + 90° rotations (line-safe)
│   └── dataset.py                      # tile-based PyTorch Dataset + collate
├── models/
│   └── unet.py                         # smp.Unet baseline + DeepUNet two-stream
├── training/
│   ├── losses.py                       # BCE+Dice and Focal+Dice
│   └── trainer.py                      # PyTorch train/val loop + viz + ckpt
├── symbols/
│   └── call_symbol_localizer.py        # Vertex AI client: symbol localization + classification
├── notebooks/
│   ├── 01_explore_labels.ipynb         # dataset / polyline / mask visual check
│   ├── 02_symbol_detection_demo.ipynb  # symbol localizer endpoint smoke-test
│   └── 03_inspect_training_tile.ipynb  # one-pass training-tile sanity check
├── results/                            # (gitignored) outputs
│   ├── data_debug/                     #   rendered label overlays
│   └── symbols/                        #   cached symbol-endpoint responses
├── runs/                               # (gitignored) training runs land here
├── train.py                            # CLI entrypoint for training
├── requirements.txt
└── README.md
```

## Setup

Two-step install because torch's CUDA wheel needs to match the host's driver
(see `scripts/install_torch.sh` for the auto-detection logic).

```bash
# On macOS, system Python is exposed as `python3`. Use it only to bootstrap
# the venv — once activated, the venv provides plain `python` / `pip`.
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel

# Step 1: install torch + torchvision for THIS host.
# Auto-detects driver CUDA via nvidia-smi (cu128 for the 5090 server,
# cu124 for the rtx6000 server, CPU on macOS).
bash scripts/install_torch.sh

# Step 2: install everything else (pure-Python deps).
pip install -r requirements.txt

# Strip notebook outputs from git so diffs stay small.
nbstripout --install
```

> **Troubleshooting** — `command not found: python` after the first line means
> your shell only has `python3`. The venv must be created with `python3 -m
> venv .venv`; inside the activated venv, `python` works. If `pip install`
> still says "command not found" after activation, verify the venv exists
> with `ls .venv/bin/`.

### Deploying to a Linux GPU server

The codebase is portable: every path is built with `pathlib.Path`, no absolute
paths anywhere, line-ending agnostic. The two-step install above works on any
GPU server without edits — `install_torch.sh` reads `nvidia-smi` and selects
the matching wheel index. Two server-specific tips after that:

```bash
# 1) Increase DataLoader workers to match the box's CPU count
#    (edit configs/train.yaml: training.num_workers).
# 2) Use tmux so SSH disconnects don't kill training:
#    tmux new -s train
#    bash scripts/run_v2_ladder_parallel.sh        # or `python train.py ...`
#    # Ctrl-B then D to detach
```

Directory layout on the server must mirror local — `datasets/` sits *next to*
`lateral_detection/`, since `configs/base.yaml` uses the relative path
`../datasets/keypoints.v56-lat_only_baseline.coco`:

```
<some-parent>/
├── datasets/
│   └── keypoints.v56-lat_only_baseline.coco/...
└── lateral_detection/
    ├── train.py
    ...
```

If you want a different absolute layout on the server, override
`data.dataset_root` in `configs/train.yaml` (a fully-qualified `/data/...`
path works fine — the loader treats it as absolute when it starts with `/`).

## Credentials (for the symbol localizer endpoint)

The symbol-localizer client needs a GCP service-account JSON. The expected location is:

```
lateral_detection/credentials/inference-428300-7af7f5da75dc.json
```

The `credentials/` folder and `inference-*.json` glob are both in `.gitignore`, so the key is never committed. If your credentials file lives somewhere else, either set the `GOOGLE_APPLICATION_CREDENTIALS` env var to point at it or pass `--credentials-file` on the CLI:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/credentials/inference-428300-7af7f5da75dc.json"

# CLI invocation
python -m symbols.call_symbol_localizer \
    --image datasets/keypoints.v56-lat_only_baseline.coco/test/<some>.jpg \
    --out-dir results/symbols/test_0
```

For exploration, open `notebooks/02_symbol_detection_demo.ipynb` — it caches the raw responses under `results/symbols/<split>_id<NNN>/` so you don't pay for the call twice.

## Training

Inspect first, then train:

```bash
# 1. Sanity check what the trainer is about to see (image+mask tiles).
jupyter lab notebooks/03_inspect_training_tile.ipynb

# 2. Kick off training. Uses configs/base.yaml + configs/train.yaml.
python train.py
# or override defaults on the command line:
python train.py --device cuda:0 --save-dir runs/v1_resnet34_bcedice
```

Outputs land under `runs/<save_dir>/`:

| File | Contents |
|---|---|
| `best.pth` | Model state dict at the epoch with the highest val Dice |
| `last.pth` | Most recent epoch's state dict |
| `history.json` | Per-epoch losses + metrics |
| `history.png` | Loss and metric curves |
| `val_viz/epoch###_sample##.jpg` | Image / GT / pred / probability heatmap, per epoch |

Most knobs you'd tweak live in `configs/train.yaml`:

- **`model.encoder`** — swap `resnet34` for `resnet50` etc. for more capacity.
- **`loss.name`** — `bce_dice` (default) or `focal_dice`.
- **`loss.bce_pos_weight`** — bump above 1.0 if Dice alone undershoots.
- **`data.train_mode`** — `random` (positive-centered, default) or
  `pos_only_grid` (deterministic).
- **`augmentation.*_prob`** — turn flips/rotations on/off.

## Exploring the dataset

Open `notebooks/01_explore_labels.ipynb` in JupyterLab. The notebook
loads each split, builds polylines, rasterizes a mask, and renders a
four-panel comparison per image:

1. raw image
2. chord endpoints (raw annotations)
3. reconstructed polylines (colored per polyline)
4. rasterized GT mask (red overlay)

Tweak `merge_radius` and `thickness` in `configs/base.yaml` (or directly
in the notebook) and rerun the visualization cells to see the effect.

## Data assumptions

- Image resolution is preserved at native scale (typical plans are
  7200×10800 and larger).
- `category_id == 1` ("lateral") is the only foreground class.
- Each annotation is a 2-keypoint chord, COCO format
  `keypoints = [x1, y1, v1, x2, y2, v2]`.
- Polylines are reconstructed by merging endpoints within `merge_radius`
  pixels (default 10 px) and walking degree-2 chains.
