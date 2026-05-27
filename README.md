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
│   └── base.yaml                       # paths, polyline + rasterize params
├── credentials/                        # GCP service-account JSON (gitignored)
│   └── inference-428300-7af7f5da75dc.json
├── data/
│   ├── coco_loader.py                  # parse COCO export → ImageRecord + Chord
│   ├── polyline_builder.py             # chords → polylines (endpoint merging)
│   └── rasterize.py                    # polylines → binary mask
├── symbols/
│   └── call_symbol_localizer.py        # Vertex AI client: symbol localization + classification
├── notebooks/
│   ├── 01_explore_labels.ipynb         # visual sanity-check of the dataset
│   └── 02_symbol_detection_demo.ipynb  # smoke-test the symbol localizer endpoint
├── results/                            # (gitignored) outputs
│   ├── data_debug/                     #   rendered label overlays
│   └── symbols/                        #   cached symbol-endpoint responses
├── requirements.txt
└── README.md
```

## Setup

```bash
# On macOS, system Python is exposed as `python3`. Use it only to bootstrap
# the virtual environment — once activated, the venv provides plain
# `python` / `pip`.
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# Strip notebook outputs from git so diffs stay small.
nbstripout --install
```

> **Troubleshooting** — `command not found: python` after the first line means
> your shell only has `python3`. The venv must be created with `python3 -m
> venv .venv`; inside the activated venv, `python` works. If `pip install`
> still says "command not found" after activation, verify the venv exists
> with `ls .venv/bin/`.

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
