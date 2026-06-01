"""train.py — lateral binary segmentation trainer entrypoint.

Usage
-----
    python train.py
    python train.py --config configs/train.yaml --device cuda:0

    # Layer a per-experiment overlay (only the differences from train.yaml):
    python train.py --overlay configs/train_v2a.yaml --device cuda:0

By default it reads ``configs/base.yaml`` and overlays ``configs/train.yaml``.
``--overlay`` applies a third layer on top of ``--config`` so each experiment
only needs to declare the keys it changes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

# Make repo modules importable when running the file directly.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.augmentation import TileAugmenter
from data.dataset import TileDataset, collate_tile_samples, worker_init_fn
from models.unet import build_model
from training.losses import build_loss
from training.trainer import SegTrainer, TrainerConfig


def merge_configs(base: dict, override: dict) -> dict:
    """Shallow recursive merge: nested dicts are merged one level deep."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = {**out[key], **value}
        else:
            out[key] = value
    return out


def _resolve_device(name: str) -> str:
    if name.startswith("cuda") and not torch.cuda.is_available():
        print(f"[train] CUDA not available; falling back from {name!r} to 'cpu'.")
        return "cpu"
    if name == "mps" and not getattr(torch.backends, "mps", None) or (
        name == "mps" and not torch.backends.mps.is_available()
    ):
        print("[train] MPS not available; falling back to 'cpu'.")
        return "cpu"
    return name


def build_datasets(cfg: dict) -> tuple[TileDataset, TileDataset]:
    dataset_root = (Path.cwd() / cfg["data"]["dataset_root"]).resolve()
    train_dir = dataset_root / cfg["data"]["train_dir"]
    val_dir   = dataset_root / cfg["data"]["valid_dir"]
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Train dir not found: {train_dir}")
    if not val_dir.is_dir():
        raise FileNotFoundError(f"Val dir not found: {val_dir}")

    aug_cfg = cfg.get("augmentation", {})
    aug = (
        TileAugmenter(
            hflip_prob=float(aug_cfg.get("hflip_prob",     0.5)),
            vflip_prob=float(aug_cfg.get("vflip_prob",     0.5)),
            rotate_90_prob=float(aug_cfg.get("rotate_90_prob", 0.5)),
        )
        if aug_cfg.get("enabled", True)
        else None
    )

    norm = cfg["normalization"]
    common_kwargs = dict(
        tile_size=int(cfg["data"]["tile_size"]),
        stride=int(cfg["data"]["stride"]),
        merge_radius=float(cfg["polyline"]["merge_radius"]),
        thickness=int(cfg["rasterize"]["thickness"]),
        mean=tuple(norm["mean"]),
        std=tuple(norm["std"]),
    )

    train_ds = TileDataset(
        split_dir=train_dir,
        mode=cfg["data"]["train_mode"],
        augmenter=aug,
        samples_per_epoch_per_image=int(
            cfg["data"].get("samples_per_epoch_per_image", 8)
        ),
        **common_kwargs,
    )
    val_ds = TileDataset(
        split_dir=val_dir,
        mode=cfg["data"]["val_mode"],
        augmenter=None,
        **common_kwargs,
    )
    return train_ds, val_ds


def main() -> int:
    parser = argparse.ArgumentParser(description="Lateral segmentation trainer.")
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument(
        "--overlay",
        default=None,
        help="Optional extra config layer (e.g. configs/train_v2a.yaml) applied "
             "on top of --config. Lets each experiment declare only its diffs.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override training.device (e.g. cuda:0, cpu, mps).",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Override training.save_dir.",
    )
    args = parser.parse_args()

    base_cfg  = yaml.safe_load(open(args.base_config))
    train_cfg = yaml.safe_load(open(args.config))
    cfg = merge_configs(base_cfg, train_cfg)
    if args.overlay is not None:
        overlay_cfg = yaml.safe_load(open(args.overlay))
        cfg = merge_configs(cfg, overlay_cfg)

    if args.save_dir is not None:
        cfg["training"]["save_dir"] = args.save_dir

    device = _resolve_device(args.device or cfg["training"].get("device", "cuda"))

    train_ds, val_ds = build_datasets(cfg)
    print(f"[train] train tiles: {len(train_ds)},  val tiles: {len(val_ds)}")

    bs = int(cfg["training"]["batch_size"])
    nw = int(cfg["training"]["num_workers"])
    persistent = nw > 0  # avoid re-forking workers every epoch (cheap win)
    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True,  num_workers=nw,
        pin_memory=(device != "cpu"),
        collate_fn=collate_tile_samples,
        worker_init_fn=worker_init_fn,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False, num_workers=nw,
        pin_memory=(device != "cpu"),
        collate_fn=collate_tile_samples,
        worker_init_fn=worker_init_fn,
        persistent_workers=persistent,
    )

    model = build_model(cfg["model"])
    criterion = build_loss(cfg["loss"])

    trainer_cfg = TrainerConfig(
        save_dir=cfg["training"]["save_dir"],
        epochs=int(cfg["training"]["epochs"]),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
        grad_clip=float(cfg["training"]["grad_clip"]),
        log_interval=int(cfg["training"]["log_interval"]),
        val_viz_count=int(cfg["training"]["val_viz_count"]),
        device=device,
        best_metric=str(cfg["training"]["best_metric"]),
    )

    print(
        f"[train] device={device}  model={cfg['model']['name']}/{cfg['model']['encoder']}  "
        f"loss={cfg['loss']['name']}  save_dir={trainer_cfg.save_dir}"
    )

    trainer = SegTrainer(
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        config=trainer_cfg,
        normalization=(cfg["normalization"]["mean"], cfg["normalization"]["std"]),
    )
    trainer.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
