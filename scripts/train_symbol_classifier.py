"""Train per-symbol classifier on cached embeddings + bbox features.

Reads the cache produced by ``scripts/mine_symbol_categories.py`` and
trains a binary classifier that predicts ``P(near_gt | symbol)``. Used
at inference by ``inference/symbol_filter.py``.

Why per-symbol instead of per-cluster: the cluster-averaging approach
(see mine_symbol_categories.py) puts visually similar but semantically
different symbols in the same bucket, so ~60% of clusters fall into
"ambiguous" (P between 0.2 and 0.7) and the filter can't act on them.
A direct classifier evaluates each symbol on its own merits and lets
bbox features (width, height, aspect ratio, area) help disambiguate.

Models tried (in order of preference):
  1. Logistic regression with class_weight='balanced'  ← interpretable, fast
  2. MLP (one hidden layer of 64 units, ReLU)         ← if LR AUC is low

Reports AUC on training + validation (val symbols held out from
training, used for honest evaluation). Saves the winning pipeline to
``--out`` as a joblib bundle consumed by SymbolClassifier.

Usage
-----
    python -m scripts.train_symbol_classifier \\
        --cache-dir results/symbols_cache \\
        --train-splits train \\
        --val-splits   valid \\
        --out          results/symbol_categories/symbol_classifier.joblib
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from inference.symbol_filter import build_features


def _load_split_symbols(cache_root: Path, splits: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (embeddings, bboxes, y) across the given cached splits."""
    embeddings: list[list[float]] = []
    bboxes:     list[list[float]] = []
    labels:     list[int]         = []
    for split in splits:
        split_dir = cache_root / split
        if not split_dir.is_dir():
            print(f"  [warn] no cache for split {split!r} at {split_dir}")
            continue
        for jp in sorted(split_dir.glob("*.json")):
            data = json.loads(jp.read_text())
            for s in data["symbols"]:
                if s.get("near_gt") is None:
                    continue  # not a training-labeled symbol
                embeddings.append(s["embedding"])
                bboxes.append([s["x1"], s["y1"], s["x2"], s["y2"]])
                labels.append(1 if s["near_gt"] else 0)
    emb = np.asarray(embeddings, dtype=np.float32)
    bx  = np.asarray(bboxes,     dtype=np.float32)
    y   = np.asarray(labels,     dtype=np.int32)
    return emb, bx, y


def _build_pipeline(name: str) -> Pipeline:
    if name == "logreg":
        return Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf",    LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                C=1.0,
                solver="lbfgs",
                n_jobs=-1,
            )),
        ])
    if name == "mlp":
        return Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf",    MLPClassifier(
                hidden_layer_sizes=(64,),
                activation="relu",
                solver="adam",
                max_iter=200,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=0,
            )),
        ])
    raise ValueError(f"unknown model name {name!r}")


def _evaluate(pipeline: Pipeline, X: np.ndarray, y: np.ndarray, label: str) -> float:
    if len(np.unique(y)) < 2:
        print(f"  {label}: only one class present, AUC undefined")
        return float("nan")
    p = pipeline.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, p)
    print(f"  {label}: n={len(y)}, AUC = {auc:.4f}")
    # Also report what fraction would be flagged at the conservative thresholds
    p_irr  = float((p >= 0.85).mean())
    p_call = float((p <= 0.15).mean())
    p_amb  = 1.0 - p_irr - p_call
    print(f"        @thresholds 0.85/0.15: "
          f"irr={p_irr:.3f}  amb={p_amb:.3f}  call={p_call:.3f}")
    return auc


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--cache-dir",    default="results/symbols_cache")
    parser.add_argument("--train-splits", nargs="+", default=["train"])
    parser.add_argument("--val-splits",   nargs="+", default=["valid"])
    parser.add_argument("--out",          default="results/symbol_categories/symbol_classifier.joblib")
    parser.add_argument("--models",       nargs="+", default=["logreg", "mlp"],
                        help="Models to try, in order; the highest val AUC wins.")
    args = parser.parse_args()

    cache_root = Path(args.cache_dir)
    out_path   = Path(args.out)

    print("=== loading training symbols ===")
    emb_tr, bx_tr, y_tr = _load_split_symbols(cache_root, args.train_splits)
    print(f"  loaded {len(emb_tr)} symbols, near_gt rate = {y_tr.mean():.3f}")
    X_tr = build_features(emb_tr, bx_tr)
    print(f"  feature matrix: {X_tr.shape}")

    print("\n=== loading val symbols ===")
    emb_vl, bx_vl, y_vl = _load_split_symbols(cache_root, args.val_splits)
    print(f"  loaded {len(emb_vl)} symbols, near_gt rate = {y_vl.mean():.3f}")
    X_vl = build_features(emb_vl, bx_vl) if len(emb_vl) > 0 else None

    best_auc_val = -1.0
    best_pipeline: Pipeline | None = None
    best_name: str = ""
    best_auc_train = float("nan")

    for name in args.models:
        print(f"\n=== model: {name} ===")
        pipe = _build_pipeline(name)
        pipe.fit(X_tr, y_tr)
        auc_tr = _evaluate(pipe, X_tr, y_tr, "train")
        auc_vl = _evaluate(pipe, X_vl, y_vl, "val") if X_vl is not None else float("nan")
        if not np.isnan(auc_vl) and auc_vl > best_auc_val:
            best_auc_val   = auc_vl
            best_auc_train = auc_tr
            best_pipeline  = pipe
            best_name      = name

    if best_pipeline is None:
        print("\n[train] no valid model with val AUC; nothing to save.")
        return 1

    print(f"\n=== best model: {best_name}  val AUC = {best_auc_val:.4f} ===")

    # Sanity-check: precision/recall on val at the conservative thresholds
    if X_vl is not None and len(np.unique(y_vl)) > 1:
        p_vl = best_pipeline.predict_proba(X_vl)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_vl, p_vl)
        # Find precision at recall>=0.5 and recall at precision>=0.9
        for target in (0.5, 0.8, 0.9):
            mask = recall >= target
            if mask.any():
                p_at = float(precision[mask].max())
                print(f"  precision @ recall>={target:.1f}: {p_at:.3f}")
        for target in (0.8, 0.9, 0.95):
            mask = precision >= target
            if mask.any():
                r_at = float(recall[mask].max())
                print(f"  recall    @ precision>={target:.2f}: {r_at:.3f}")

    bundle = {
        "pipeline":   best_pipeline,
        "n_features": X_tr.shape[1],
        "auc_train":  best_auc_train,
        "auc_val":    best_auc_val,
        "trained_on": ",".join(args.train_splits),
        "feature_layout": "L2(emb_128) + [w, h, log_aspect, log_area, log_w+log_h]",
        "model_name": best_name,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, str(out_path))
    print(f"\n[train] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
