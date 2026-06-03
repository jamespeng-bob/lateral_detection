"""train.py — lateral binary segmentation trainer entrypoint.

Single-GPU and DDP both supported through the same script.

Single-GPU
----------
    python train.py
    python train.py --config configs/train.yaml --device cuda:0
    python train.py --overlay configs/train_v3a.yaml --device cuda:0

DDP (across N GPUs on one host)
-------------------------------
    torchrun --nproc-per-node=2 --master-port=29500 train.py \
        --overlay configs/train_v2a.yaml

`torchrun` sets RANK / LOCAL_RANK / WORLD_SIZE in the environment; we
detect those and set up the process group automatically. With DDP,
`training.batch_size` in the config is interpreted as PER-GPU; effective
global batch = `batch_size * world_size`.

Config layering
---------------
``--base-config`` → ``--config`` → ``--overlay``. Later layers override
earlier ones; each layer can omit any field it doesn't change.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow the cached allocator to grow segments on demand instead of clamping
# at the largest pre-allocated block. With dense U-Net activations at 1024^2
# this routinely buys us 1–3 GB of usable VRAM by avoiding fragmentation.
# Must be set BEFORE the first `import torch` allocator interaction.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# Make repo modules importable when running the file directly.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.augmentation import TileAugmenter
from data.dataset import TileDataset, collate_tile_samples, worker_init_fn
from models.unet import build_model
from training.losses import build_loss
from training.trainer import SegTrainer, TrainerConfig


# ---------------------------------------------------------------------------
# Config layering
# ---------------------------------------------------------------------------


def merge_configs(base: dict, override: dict) -> dict:
    """Shallow recursive merge: nested dicts are merged one level deep."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = {**out[key], **value}
        else:
            out[key] = value
    return out


# ---------------------------------------------------------------------------
# Device + DDP setup
# ---------------------------------------------------------------------------


def _torchrun_launched() -> bool:
    """True when this process was started by torchrun (DDP)."""
    return all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))


def setup_ddp() -> tuple[int, int, int, str]:
    """Initialize the NCCL process group.

    Returns ``(local_rank, global_rank, world_size, device)``. Must only be
    called when ``_torchrun_launched()`` is True.
    """
    local_rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        backend = "nccl"
        device = f"cuda:{local_rank}"
    else:
        backend = "gloo"
        device = "cpu"
    dist.init_process_group(backend=backend)
    return local_rank, dist.get_rank(), dist.get_world_size(), device


def _resolve_device(name: str) -> str:
    """Resolve --device for single-GPU runs (DDP overrides this)."""
    if name.startswith("cuda") and not torch.cuda.is_available():
        print(f"[train] CUDA not available; falling back from {name!r} to 'cpu'.")
        return "cpu"
    if name == "mps":
        mps_ok = (
            getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()
        )
        if not mps_ok:
            print("[train] MPS not available; falling back to 'cpu'.")
            return "cpu"
    return name


# ---------------------------------------------------------------------------
# Dataset / loader construction (DDP-aware)
# ---------------------------------------------------------------------------


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


def build_loaders(
    train_ds: TileDataset,
    val_ds: TileDataset,
    cfg: dict,
    device: str,
    world_size: int,
    global_rank: int,
) -> tuple[DataLoader, DataLoader]:
    bs = int(cfg["training"]["batch_size"])
    nw = int(cfg["training"]["num_workers"])
    persistent = nw > 0
    pin_memory = device != "cpu"

    if world_size > 1:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=global_rank, shuffle=True,
            drop_last=False,
        )
        val_sampler = DistributedSampler(
            val_ds,   num_replicas=world_size, rank=global_rank, shuffle=False,
            drop_last=False,
        )
        train_shuffle = False  # DistributedSampler handles shuffling
    else:
        train_sampler = None
        val_sampler   = None
        train_shuffle = True

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=train_shuffle, sampler=train_sampler,
        num_workers=nw, pin_memory=pin_memory,
        collate_fn=collate_tile_samples,
        worker_init_fn=worker_init_fn,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False, sampler=val_sampler,
        num_workers=nw, pin_memory=pin_memory,
        collate_fn=collate_tile_samples,
        worker_init_fn=worker_init_fn,
        persistent_workers=persistent,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


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
        help="Override training.device (e.g. cuda:0, cpu, mps). Ignored under "
             "torchrun (DDP picks up LOCAL_RANK).",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Override training.save_dir.",
    )
    args = parser.parse_args()

    # ── Load + merge configs ─────────────────────────────────────────────
    base_cfg  = yaml.safe_load(open(args.base_config))
    train_cfg = yaml.safe_load(open(args.config))
    cfg = merge_configs(base_cfg, train_cfg)
    if args.overlay is not None:
        overlay_cfg = yaml.safe_load(open(args.overlay))
        cfg = merge_configs(cfg, overlay_cfg)
    if args.save_dir is not None:
        cfg["training"]["save_dir"] = args.save_dir

    # ── Device / DDP setup ───────────────────────────────────────────────
    if _torchrun_launched():
        local_rank, global_rank, world_size, device = setup_ddp()
    else:
        device = _resolve_device(args.device or cfg["training"].get("device", "cuda"))
        local_rank, global_rank, world_size = 0, 0, 1

    is_rank_zero = (global_rank == 0)

    # ── Datasets + loaders ───────────────────────────────────────────────
    train_ds, val_ds = build_datasets(cfg)
    train_loader, val_loader = build_loaders(
        train_ds, val_ds, cfg, device=device,
        world_size=world_size, global_rank=global_rank,
    )
    if is_rank_zero:
        per_gpu = int(cfg["training"]["batch_size"])
        eff_bs  = per_gpu * world_size
        print(f"[train] world_size={world_size}  per-GPU bs={per_gpu}  "
              f"effective bs={eff_bs}")
        print(f"[train] train tiles: {len(train_ds)},  val tiles: {len(val_ds)}")

    # ── Model + loss ─────────────────────────────────────────────────────
    model = build_model(cfg["model"]).to(device)

    if world_size > 1:
        # SyncBN sounds like "always the right thing under DDP", but for
        # encoders with many small BN layers (EfficientNet's depthwise +
        # squeeze-excitation, MobileNet, etc.) it can degrade training
        # noticeably at small per-rank batch sizes — the cross-rank
        # variance estimate becomes noisy on thin tensors, and EfficientNet
        # in particular is famously BN-sensitive. v2a-ddp lost ~20 dice
        # points relative to its single-GPU baseline when SyncBN was on;
        # without it (per-rank BN on bs=4 like the original single-GPU
        # run) the same encoder trained cleanly. So: opt-in, default on
        # for backward compatibility, override to false for any encoder
        # that struggles. Encoders that use LayerNorm everywhere (MiT /
        # SegFormer, Swin, ConvNeXt) are unaffected either way because
        # convert_sync_batchnorm finds nothing to convert.
        use_sync_bn = bool(cfg["training"].get("sync_batch_norm", True))
        if use_sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if is_rank_zero:
            print(f"[train] DDP: sync_batch_norm={use_sync_bn}")
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=False,
        )

    criterion = build_loss(cfg["loss"])

    # ── Trainer config ──────────────────────────────────────────────────
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

    if is_rank_zero:
        print(
            f"[train] device={device}  model={cfg['model']['name']}/{cfg['model']['encoder']}  "
            f"loss={cfg['loss']['name']}  save_dir={trainer_cfg.save_dir}"
        )

    # ── Run ──────────────────────────────────────────────────────────────
    trainer = SegTrainer(
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        config=trainer_cfg,
        normalization=(cfg["normalization"]["mean"], cfg["normalization"]["std"]),
    )
    trainer.run()

    # ── Cleanup ──────────────────────────────────────────────────────────
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
