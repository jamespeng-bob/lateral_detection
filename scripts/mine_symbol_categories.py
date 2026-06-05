"""Mine symbol-category structure from the training set via embeddings.

The Vertex AI symbol localizer returns bounding boxes but no class label.
The "classifier" endpoint actually returns a 128-d embedding per crop, not
a class. So we discover classes empirically:

  Phase A1+A2  cache    for each image, call localize() to get bboxes,
                        then classify() to get embeddings; compute per-symbol
                        ``near_gt`` (is this symbol within D_near px of a GT
                        lateral line? — training set only).
  Phase A3     cluster  k-means on L2-normalized training embeddings into K
                        groups; for each cluster, P(near_gt | cluster).
  Phase A4     label    cluster category by thresholds: P > p_irr → IRRIGATION,
                        P < p_call → CALLOUT, else AMBIGUOUS. Render a sanity-
                        check grid of example crops (one row per cluster).

The per-image cache is JSON, one file per image. Resumable: an existing
cache file is skipped. Embeddings are kept in cache so re-clustering with
a different K is fast.

Outputs:
  <cache-dir>/<split>/<stem>.json            per-image localize+classify cache
  <out-dir>/symbol_clusters.json             cluster centroids + category lookup
  <out-dir>/symbol_clusters_meta.csv         human-readable per-cluster stats
  <out-dir>/cluster_examples.jpg             sanity-check grid (look before B)

Usage
-----
    python -m scripts.mine_symbol_categories \\
        --splits train valid test \\
        --cache-dir results/symbols_cache \\
        --out-dir   results/symbol_categories \\
        --k 30 \\
        --d-near 30

Set --k 0 to pick K automatically via silhouette score over candidate Ks.
Set --max-images N to process at most N images per split (debugging).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.coco_loader import load_split
from data.polyline_builder import build_polylines
from data.rasterize import rasterize_polylines
from symbols.call_symbol_localizer import (
    DEFAULT_CLASSIFICATION_ENDPOINT_ID,
    DEFAULT_GCS_BUCKET,
    DEFAULT_LOCALIZATION_ENDPOINT_ID,
    DEFAULT_LOCATION,
    DEFAULT_PROJECT_ID,
    IsolatedSymbolClient,
    _resolve_credentials_file,
)
from train import merge_configs

Image.MAX_IMAGE_PIXELS = None

CLASSIFY_BATCH_SIZE = 64   # crop_ids per classify() call


# ---------------------------------------------------------------------------
# Phase A1 + A2: per-image cache
# ---------------------------------------------------------------------------


def _near_gt_lookup(gt_mask: np.ndarray | None, d_near: int) -> np.ndarray | None:
    """Distance-transform of (1 - GT). lookup[y,x] = distance to nearest GT px.

    None when there's no GT (e.g., empty annotation). At lookup time, a
    symbol is near_gt if its bbox center has lookup value < d_near.
    """
    if gt_mask is None or not gt_mask.any():
        return None
    # cv2.distanceTransform wants 0 = foreground for its source convention;
    # we want distance FROM each background pixel TO the nearest foreground
    # pixel, so we invert: foreground = pixel == 0 in `inv`.
    inv = (gt_mask == 0).astype(np.uint8)
    return cv2.distanceTransform(inv, cv2.DIST_L2, 3)


def cache_one_image(
    client: IsolatedSymbolClient,
    image_path: Path,
    gt_mask: np.ndarray | None,
    d_near: int,
    out_json: Path,
) -> int:
    """Localize + classify one image, save to JSON. Returns symbol count.

    Skips and returns existing count if out_json already exists.
    """
    if out_json.exists():
        try:
            cached = json.loads(out_json.read_text())
            return len(cached.get("symbols", []))
        except Exception:
            pass  # malformed cache — re-run

    try:
        dets = client.localize(str(image_path))
    except Exception as exc:
        print(f"  [warn] localize failed for {image_path.name}: {exc}")
        return 0

    if not dets:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps({"image": image_path.name, "symbols": []}))
        return 0

    crop_ids = [str(d.get("id", "")) for d in dets]
    embeddings_by_id: dict[str, list[float]] = {}
    for i in range(0, len(crop_ids), CLASSIFY_BATCH_SIZE):
        batch = [cid for cid in crop_ids[i : i + CLASSIFY_BATCH_SIZE] if cid]
        if not batch:
            continue
        try:
            results = client.classify(batch)
        except Exception as exc:
            print(f"  [warn] classify batch failed for {image_path.name}: {exc}")
            continue
        for cid, res in zip(batch, results):
            if isinstance(res, dict) and "embedding" in res:
                embeddings_by_id[cid] = [float(v) for v in res["embedding"]]

    dist_lookup = _near_gt_lookup(gt_mask, d_near)
    H = gt_mask.shape[0] if gt_mask is not None else None
    W = gt_mask.shape[1] if gt_mask is not None else None

    symbols = []
    for d in dets:
        cid = str(d.get("id", ""))
        emb = embeddings_by_id.get(cid)
        if emb is None:
            continue
        x1, y1, x2, y2 = (
            float(d["x1"]),
            float(d["y1"]),
            float(d["x2"]),
            float(d["y2"]),
        )
        near_gt: bool | None = None
        if dist_lookup is not None:
            # bbox-OVERLAP distance: a symbol is near_gt iff any pixel inside
            # its bbox is within d_near of a GT line pixel. The previous
            # bbox-CENTER variant was systematically biased against large
            # symbols (valves, sprinklers): a 50px valve sitting on a lateral
            # has its center ~25px from the line even though the bbox contains
            # the line. Center-distance vs the same d_near would label such
            # a valve as "not near GT" → it polluted the "ambiguous" clusters
            # and the classifier never learned to recognize valves as
            # irrigation. See scripts/recompute_near_gt.py for the rationale.
            x1i = max(0, min(W - 1, int(x1)))
            y1i = max(0, min(H - 1, int(y1)))
            x2i = max(0, min(W - 1, int(x2)))
            y2i = max(0, min(H - 1, int(y2)))
            if x2i >= x1i and y2i >= y1i:
                patch = dist_lookup[y1i : y2i + 1, x1i : x2i + 1]
                near_gt = bool(patch.size > 0 and patch.min() < d_near)
            else:
                near_gt = False
        symbols.append({
            "id":   cid,
            "x1":   x1, "y1": y1, "x2": x2, "y2": y2,
            "conf": float(d.get("conf", 0.0) or 0.0),
            "embedding": emb,
            "near_gt":   near_gt,
        })

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"image": image_path.name, "symbols": symbols}))
    return len(symbols)


# ---------------------------------------------------------------------------
# Phase A3 + A4: load training symbols, cluster, categorize
# ---------------------------------------------------------------------------


def _load_training_symbols(
    cache_root: Path,
    splits_for_training: list[str],
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Return (X, y, refs).

    X         : (N, 128) embeddings, L2-normalized.
    y         : (N,) int  near_gt {0, 1}.
    refs      : N dicts with {split, image, x1, y1, x2, y2} so we can recover
                the source crop for the cluster-example viz.
    """
    embeddings: list[list[float]] = []
    near:       list[int] = []
    refs:       list[dict] = []
    for split in splits_for_training:
        split_dir = cache_root / split
        if not split_dir.is_dir():
            continue
        for jp in sorted(split_dir.glob("*.json")):
            data = json.loads(jp.read_text())
            for s in data["symbols"]:
                if s.get("near_gt") is None:
                    continue
                embeddings.append(s["embedding"])
                near.append(1 if s["near_gt"] else 0)
                refs.append({
                    "split":   split,
                    "image":   data["image"],
                    "x1": s["x1"], "y1": s["y1"], "x2": s["x2"], "y2": s["y2"],
                })
    if not embeddings:
        raise RuntimeError("No training embeddings found in cache; run cache phase first.")
    X = np.asarray(embeddings, dtype=np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    y = np.asarray(near, dtype=np.int32)
    return X, y, refs


def _pick_K_by_silhouette(X: np.ndarray, candidates: list[int]) -> int:
    """Return the K with the highest silhouette score (sub-sampled for speed)."""
    best_k, best_score = candidates[0], -1.0
    for k_try in candidates:
        km = KMeans(n_clusters=k_try, n_init=10, random_state=0)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels, sample_size=min(5000, len(X)), random_state=0)
        print(f"    K={k_try:>3}: silhouette={score:.4f}")
        if score > best_score:
            best_k, best_score = k_try, score
    return best_k


def cluster_and_categorize(
    X: np.ndarray,
    y: np.ndarray,
    K: int,
    p_irr_thresh: float,
    p_call_thresh: float,
) -> tuple[list[dict], np.ndarray]:
    """Returns (clusters_meta, per_sample_labels)."""
    if K == 0:
        print("  picking K via silhouette score...")
        K = _pick_K_by_silhouette(X, candidates=[15, 20, 30, 40, 50])
        print(f"  chose K={K}")
    km = KMeans(n_clusters=K, n_init=10, random_state=0)
    labels = km.fit_predict(X)
    clusters: list[dict] = []
    for c in range(K):
        mask = labels == c
        n = int(mask.sum())
        p_near = float(y[mask].mean()) if n else 0.0
        if   p_near > p_irr_thresh:  cat = "irrigation"
        elif p_near < p_call_thresh: cat = "callout"
        else:                        cat = "ambiguous"
        clusters.append({
            "cluster_id":  c,
            "centroid":    km.cluster_centers_[c].tolist(),
            "n_members":   n,
            "p_near_gt":   p_near,
            "category":    cat,
        })
    return clusters, labels


# ---------------------------------------------------------------------------
# Sanity-check grid: one row per cluster, example crops per row
# ---------------------------------------------------------------------------


def render_cluster_examples(
    clusters: list[dict],
    refs: list[dict],
    sample_labels: np.ndarray,
    dataset_root: Path,
    cfg: dict,
    out_path: Path,
    *,
    crops_per_cluster: int = 6,
    crop_size: int = 96,
    rng_seed: int = 0,
) -> None:
    """Render one row per cluster, ``crops_per_cluster`` example crops per row.

    Each row is annotated with: cluster_id, category, P(near_gt), n_members.
    Categories color-code the row's border (green=irrigation, red=callout,
    yellow=ambiguous).
    """
    rng = np.random.default_rng(rng_seed)
    split_dir_keys = {"train": "train_dir", "valid": "valid_dir", "test": "test_dir"}

    # Sort clusters by category then by P(near_gt) descending — makes the
    # image read top-down: irrigation rows first, then ambiguous, then callout.
    cat_order = {"irrigation": 0, "ambiguous": 1, "callout": 2}
    clusters_sorted = sorted(
        clusters, key=lambda c: (cat_order[c["category"]], -c["p_near_gt"])
    )

    label_band = 240  # px wide; left side for cluster metadata text
    row_h      = crop_size + 6
    width      = label_band + crops_per_cluster * (crop_size + 6) + 6
    height     = row_h * len(clusters_sorted) + 6
    canvas = np.full((height, width, 3), 32, dtype=np.uint8)  # dark grey bg

    cat_colors = {
        "irrigation": (0,   255, 0),    # green
        "ambiguous":  (200, 200, 0),    # yellow
        "callout":    (0,   0,   255),  # red
    }

    # Cache PIL images to avoid re-opening the same source many times.
    img_cache: dict[str, Image.Image] = {}

    for row_idx, c in enumerate(clusters_sorted):
        y_top = 3 + row_idx * row_h
        col_color = cat_colors[c["category"]]
        # Left border strip color-codes the category.
        canvas[y_top : y_top + crop_size, 0:6] = col_color
        # Metadata text on the left band
        text = [
            f"cluster {c['cluster_id']:>2}  [{c['category']}]",
            f"P(near GT) = {c['p_near_gt']:.2f}",
            f"n = {c['n_members']}",
        ]
        for i, line in enumerate(text):
            cv2.putText(
                canvas, line, (12, y_top + 22 + 20 * i),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (240, 240, 240), 1, cv2.LINE_AA,
            )

        # Pick crops_per_cluster random members of this cluster
        member_idxs = np.where(sample_labels == c["cluster_id"])[0]
        if len(member_idxs) == 0:
            continue
        chosen = rng.choice(
            member_idxs, size=min(crops_per_cluster, len(member_idxs)), replace=False,
        )
        for col_idx, idx in enumerate(chosen):
            ref = refs[idx]
            split_dir = dataset_root / cfg["data"][split_dir_keys[ref["split"]]]
            img_path = split_dir / ref["image"]
            cache_key = str(img_path)
            if cache_key not in img_cache:
                if not img_path.is_file():
                    continue
                img_cache[cache_key] = Image.open(img_path).convert("RGB")
                # Limit cache to ~20 images to bound memory
                if len(img_cache) > 20:
                    img_cache.pop(next(iter(img_cache)))
            src = img_cache[cache_key]
            W_img, H_img = src.size
            # Pad bbox slightly so we see context around the symbol
            pad = 4
            x1 = max(0, int(ref["x1"]) - pad)
            y1 = max(0, int(ref["y1"]) - pad)
            x2 = min(W_img, int(ref["x2"]) + pad)
            y2 = min(H_img, int(ref["y2"]) + pad)
            crop = src.crop((x1, y1, x2, y2))
            crop.thumbnail((crop_size, crop_size), Image.LANCZOS)
            crop_arr = np.asarray(crop, dtype=np.uint8)
            ch, cw = crop_arr.shape[:2]
            # Centre the (possibly smaller) thumb inside its crop_size cell.
            x_off = label_band + 6 + col_idx * (crop_size + 6) + (crop_size - cw) // 2
            y_off = y_top + (crop_size - ch) // 2
            canvas[y_off : y_off + ch, x_off : x_off + cw] = crop_arr

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_JPEG_QUALITY), 90])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--config",      default="configs/train.yaml")
    parser.add_argument("--overlay",     default="configs/train_v2b_6k.yaml")
    parser.add_argument("--splits",      nargs="+", default=["train", "valid", "test"])
    parser.add_argument("--cache-dir",   default="results/symbols_cache")
    parser.add_argument("--out-dir",     default="results/symbol_categories")
    parser.add_argument("--k", type=int, default=30,
                        help="K for k-means. 0 = auto-pick via silhouette.")
    parser.add_argument("--d-near", type=int, default=30,
                        help="Pixel distance for a symbol to count as 'near GT line'.")
    parser.add_argument("--p-irr-thresh",  type=float, default=0.7,
                        help="Cluster category 'irrigation' if P(near_gt) > this.")
    parser.add_argument("--p-call-thresh", type=float, default=0.2,
                        help="Cluster category 'callout' if P(near_gt) < this.")
    parser.add_argument("--max-images", type=int, default=0,
                        help="0 = all; otherwise process at most N images per split.")
    parser.add_argument("--skip-cache", action="store_true",
                        help="Skip the API-call phase; only re-cluster the existing cache.")
    args = parser.parse_args()

    cfg = merge_configs(yaml.safe_load(open(args.base_config)),
                        yaml.safe_load(open(args.config)))
    cfg = merge_configs(cfg, yaml.safe_load(open(args.overlay)))

    cache_root = Path(args.cache_dir)
    out_root   = Path(args.out_dir); out_root.mkdir(parents=True, exist_ok=True)
    dataset_root = (Path.cwd() / cfg["data"]["dataset_root"]).resolve()
    lateral_cid  = int(cfg["lateral_category_id"])
    merge_radius = float(cfg["polyline"]["merge_radius"])
    thickness    = int(cfg["rasterize"]["thickness"])
    split_dir_keys = {"train": "train_dir", "valid": "valid_dir", "test": "test_dir"}

    # ---- Phase A1+A2: cache ----------------------------------------------
    if not args.skip_cache:
        client = IsolatedSymbolClient(
            credentials_file=_resolve_credentials_file(""),
            project_id=DEFAULT_PROJECT_ID, location=DEFAULT_LOCATION,
            localization_endpoint_id=DEFAULT_LOCALIZATION_ENDPOINT_ID,
            classification_endpoint_id=DEFAULT_CLASSIFICATION_ENDPOINT_ID,
            gcs_bucket=DEFAULT_GCS_BUCKET,
            nms_iou_thresh=0.5, conf_thresh=0.3,
        )
        for split in args.splits:
            if split not in split_dir_keys:
                continue
            split_dir = dataset_root / cfg["data"][split_dir_keys[split]]
            if not split_dir.is_dir():
                continue
            images, chords_by_image = load_split(split_dir, category_id=lateral_cid)
            items = sorted(images.items())
            if args.max_images > 0:
                items = items[: args.max_images]
            total_symbols = 0
            for image_id, record in tqdm(items, desc=f"cache {split}"):
                chords = chords_by_image.get(image_id, [])
                polylines = build_polylines(chords, merge_radius=merge_radius)
                gt_mask = (
                    rasterize_polylines(
                        polylines=polylines,
                        height=record.height,
                        width=record.width,
                        thickness=thickness,
                    )
                    if polylines else None
                )
                cache_path = cache_root / split / f"{Path(record.file_name).stem}.json"
                n = cache_one_image(
                    client, record.path, gt_mask, args.d_near, cache_path,
                )
                total_symbols += n
            print(f"  [{split}] cached {total_symbols} symbols across {len(items)} images")
            client.cleanup_uploads()
    else:
        print("[mine] --skip-cache: skipping API phase, using existing cache only")

    # ---- Phase A3+A4: cluster + categorize -------------------------------
    print("\n=== loading training symbols ===")
    X, y, refs = _load_training_symbols(cache_root, ["train"])
    print(f"  {len(X)} training symbols; near_gt rate = {y.mean():.3f}")

    print(f"\n=== clustering (K={args.k}) ===")
    clusters, labels = cluster_and_categorize(
        X, y, K=args.k,
        p_irr_thresh=args.p_irr_thresh,
        p_call_thresh=args.p_call_thresh,
    )
    cat_counts = {"irrigation": 0, "ambiguous": 0, "callout": 0}
    for c in clusters:
        cat_counts[c["category"]] += 1
    print(f"  clusters: {cat_counts['irrigation']} irrigation, "
          f"{cat_counts['ambiguous']} ambiguous, {cat_counts['callout']} callout")

    # Save artifacts
    clusters_path = out_root / "symbol_clusters.json"
    clusters_path.write_text(json.dumps({
        "k": len(clusters),
        "d_near": args.d_near,
        "p_irr_thresh": args.p_irr_thresh,
        "p_call_thresh": args.p_call_thresh,
        "clusters": clusters,
    }, indent=2))

    meta_path = out_root / "symbol_clusters_meta.csv"
    with open(meta_path, "w") as f:
        f.write("cluster_id,category,p_near_gt,n_members\n")
        for c in sorted(clusters, key=lambda x: (
            {"irrigation": 0, "ambiguous": 1, "callout": 2}[x["category"]],
            -x["p_near_gt"],
        )):
            f.write(f"{c['cluster_id']},{c['category']},"
                    f"{c['p_near_gt']:.4f},{c['n_members']}\n")

    # Sanity-check grid
    examples_path = out_root / "cluster_examples.jpg"
    print(f"\n=== rendering cluster_examples.jpg ===")
    render_cluster_examples(
        clusters=clusters,
        refs=refs,
        sample_labels=labels,
        dataset_root=dataset_root,
        cfg=cfg,
        out_path=examples_path,
    )

    print(f"\n[mine] wrote {clusters_path}")
    print(f"[mine] wrote {meta_path}")
    print(f"[mine] wrote {examples_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
