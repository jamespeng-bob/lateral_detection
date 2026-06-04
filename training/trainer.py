"""Train/val loop for binary lateral segmentation.

Single-GPU and DDP both supported through the same class. When DDP is
active (``torch.distributed.is_initialized()``):

- All ranks run the forward/backward train loop on their data shard.
- All ranks run the validation loop on their val shard.
- Metric accumulators all-reduce / all-gather inside ``compute()`` so the
  reported numbers are the *global* val metric over the whole val set.
- Only rank 0 prints, writes ``history.{json,png}``, writes checkpoints,
  and writes ``val_viz`` images. Other ranks are silent.

Single-GPU runs work unchanged — every DDP check short-circuits.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .metrics import (
    BinaryClDiceAccumulator,
    BinaryDiceAccumulator,
    BinaryIoUAccumulator,
    LengthRatioAccumulator,
)


@dataclass
class TrainerConfig:
    save_dir: str
    epochs: int = 80
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    log_interval: int = 20
    val_viz_count: int = 4
    device: str = "cuda"
    best_metric: str = "dice"   # 'dice' or 'loss'

    # LR schedule. 'constant' = no scheduler (use config.lr throughout, matches
    # the v1/v2 baselines). 'cosine' = linear warmup for `warmup_epochs`, then
    # cosine anneal from config.lr to config.lr * cosine_min_lr_ratio over the
    # remaining epochs. Pass `warmup_epochs=0` to skip warmup.
    lr_schedule: str = "constant"   # 'constant' | 'cosine'
    warmup_epochs: int = 0
    cosine_min_lr_ratio: float = 0.01


# ---------------------------------------------------------------------------
# DDP helpers (mirrors metrics.py — defined here too so the trainer doesn't
# need to import private helpers)
# ---------------------------------------------------------------------------


def _ddp_active() -> bool:
    return dist.is_available() and dist.is_initialized()


def _is_rank_zero() -> bool:
    return (not _ddp_active()) or dist.get_rank() == 0


def _world_size() -> int:
    return dist.get_world_size() if _ddp_active() else 1


def _all_reduce_mean(value: float) -> float:
    """Mean ``value`` across all ranks (no-op in single-GPU mode)."""
    if not _ddp_active():
        return float(value)
    device = torch.device("cuda", torch.cuda.current_device())
    t = torch.tensor([float(value)], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float((t / dist.get_world_size()).item())


def _build_lr_scheduler(
    optimizer: optim.Optimizer,
    schedule: str,
    total_epochs: int,
    warmup_epochs: int,
    cosine_min_lr_ratio: float,
) -> optim.lr_scheduler._LRScheduler | None:
    """Build a per-epoch LR scheduler. None for 'constant' (no-op).

    Cosine: linear warmup from 0 to base_lr over ``warmup_epochs``, then
    cosine anneal from base_lr to ``base_lr * cosine_min_lr_ratio`` over
    the remaining ``total_epochs - warmup_epochs`` epochs. ``epoch`` arg
    of ``lr_lambda`` is 0-indexed (matches PyTorch's LambdaLR convention).
    """
    if schedule == "constant":
        return None
    if schedule != "cosine":
        raise ValueError(f"Unknown lr_schedule: {schedule!r}")

    def lr_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            # Linear warmup: epoch 0 → lr * 1/warmup, epoch (warmup-1) → lr * 1.
            return float(epoch + 1) / float(warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return cosine_min_lr_ratio + (1.0 - cosine_min_lr_ratio) * cosine

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class SegTrainer:
    """Train a binary segmentation model end-to-end (single GPU or DDP)."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainerConfig,
        normalization: tuple[list[float], list[float]] = (
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ) -> None:
        # Caller is responsible for moving model to the right device + DDP wrap.
        self.model = model
        self.criterion = criterion.to(config.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = config.device

        # Use the raw model for parameter collection (DDP wraps it once).
        params = (
            self.model.module.parameters() if hasattr(self.model, "module")
            else self.model.parameters()
        )
        self.optimizer = optim.AdamW(
            [p for p in params if p.requires_grad],
            lr=float(config.lr),
            weight_decay=float(config.weight_decay),
        )

        # Optional LR scheduler. Stepped once per epoch (not per batch).
        self.lr_scheduler = _build_lr_scheduler(
            self.optimizer,
            schedule=config.lr_schedule,
            total_epochs=config.epochs,
            warmup_epochs=config.warmup_epochs,
            cosine_min_lr_ratio=config.cosine_min_lr_ratio,
        )

        self.save_dir = Path(config.save_dir)
        if _is_rank_zero():
            self.save_dir.mkdir(parents=True, exist_ok=True)

        self.mean = np.array(normalization[0], dtype=np.float32).reshape(3, 1, 1)
        self.std  = np.array(normalization[1], dtype=np.float32).reshape(3, 1, 1)

        self.history: dict[str, list[float]] = {
            "train_total":     [],
            "train_dice_loss": [],
            "val_total":       [],
            "val_dice_loss":   [],
            "val_dice":        [],
            "val_iou":         [],
            "val_cldice":      [],
            "val_length_ratio_pixel_mean":   [],
            "val_length_ratio_pixel_median": [],
            "val_length_ratio_pixel_p25":    [],
            "val_length_ratio_pixel_p75":    [],
            "val_length_ratio_skel_mean":    [],
            "val_length_ratio_skel_median":  [],
            "val_length_ratio_skel_p25":     [],
            "val_length_ratio_skel_p75":     [],
            "lr":              [],
        }
        self.best_val_dice = -1.0
        self.best_val_loss = float("inf")

        # All val accumulators are reset at the start of every val epoch.
        self._dice_acc   = BinaryDiceAccumulator(threshold=0.5)
        self._iou_acc    = BinaryIoUAccumulator(threshold=0.5)
        self._cldice_acc = BinaryClDiceAccumulator(threshold=0.5)
        self._length_acc = LengthRatioAccumulator(threshold=0.5)

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    def _to_device(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            batch["image"].to(self.device, non_blocking=True),
            batch["mask"].to(self.device,  non_blocking=True),
        )

    def _set_epoch_on_sampler(self, loader: DataLoader, epoch: int) -> None:
        """Bump the DistributedSampler's epoch so shuffling differs each epoch."""
        sampler = getattr(loader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

    # ------------------------------------------------------------------
    # Train / val epochs
    # ------------------------------------------------------------------

    def train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        self._set_epoch_on_sampler(self.train_loader, epoch)
        totals = {"total": 0.0, "dice_loss": 0.0}
        n = 0
        pbar = tqdm(self.train_loader, desc="train", leave=False, disable=not _is_rank_zero())
        for batch in pbar:
            images, masks = self._to_device(batch)
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(images)
            losses = self.criterion(logits, masks)
            losses["total"].backward()
            if self.config.grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                )
            self.optimizer.step()

            totals["total"]     += float(losses["total"].item())
            totals["dice_loss"] += float(losses["dice"].item())
            n += 1
            if n % self.config.log_interval == 0 and _is_rank_zero():
                pbar.set_postfix(
                    total=f"{totals['total']/n:.4f}",
                    dice=f"{totals['dice_loss']/n:.4f}",
                )
        n = max(n, 1)
        # All-reduce the per-rank running averages so the printed/logged
        # number is the global (not per-shard) mean.
        return {
            "total":     _all_reduce_mean(totals["total"] / n),
            "dice_loss": _all_reduce_mean(totals["dice_loss"] / n),
        }

    @torch.no_grad()
    def val_epoch(self, epoch: int) -> dict[str, float]:
        self.model.eval()
        self._set_epoch_on_sampler(self.val_loader, epoch)

        totals = {"total": 0.0, "dice_loss": 0.0}
        n = 0
        viz_saved = 0
        viz_dir = self.save_dir / "val_viz"
        if _is_rank_zero():
            viz_dir.mkdir(parents=True, exist_ok=True)

        for acc in (self._dice_acc, self._iou_acc, self._cldice_acc, self._length_acc):
            acc.reset()

        pbar = tqdm(self.val_loader, desc="val", leave=False, disable=not _is_rank_zero())
        for batch in pbar:
            images, masks = self._to_device(batch)
            logits = self.model(images)
            losses = self.criterion(logits, masks)

            totals["total"]     += float(losses["total"].item())
            totals["dice_loss"] += float(losses["dice"].item())
            n += 1

            # Accumulator metrics — counts are global after compute()'s
            # all-reduce, so they're equivalent to single-GPU bs=N*world.
            self._dice_acc.update(logits, masks)
            self._iou_acc.update(logits, masks)
            self._cldice_acc.update(logits, masks)
            self._length_acc.update(logits, masks)

            # Only rank 0 saves val viz so we don't write conflicting JPEGs.
            if _is_rank_zero() and viz_saved < self.config.val_viz_count:
                self._save_viz_panel(
                    images.detach().cpu(),
                    masks.detach().cpu(),
                    logits.detach().cpu(),
                    viz_dir=viz_dir,
                    epoch=epoch,
                    start_idx=viz_saved,
                    max_save=self.config.val_viz_count - viz_saved,
                )
                viz_saved += images.shape[0]

        n = max(n, 1)
        out = {
            "total":     _all_reduce_mean(totals["total"]     / n),
            "dice_loss": _all_reduce_mean(totals["dice_loss"] / n),
            "dice":      self._dice_acc.compute(),
            "iou":       self._iou_acc.compute(),
            "cldice":    self._cldice_acc.compute(),
        }
        out.update(self._length_acc.compute())
        return out

    # ------------------------------------------------------------------
    # Visualization (rank 0 only)
    # ------------------------------------------------------------------

    def _denormalize(self, img_t: torch.Tensor) -> np.ndarray:
        img = img_t.numpy() * self.std + self.mean
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        return img.transpose(1, 2, 0)

    def _save_viz_panel(
        self,
        images: torch.Tensor,
        masks:  torch.Tensor,
        logits: torch.Tensor,
        viz_dir: Path,
        epoch: int,
        start_idx: int,
        max_save: int,
    ) -> None:
        probs = torch.sigmoid(logits).numpy()
        for b in range(min(images.shape[0], max_save)):
            img_rgb = self._denormalize(images[b])
            gt   = (masks[b, 0].numpy() >= 0.5).astype(np.uint8) * 255
            pred = (probs[b, 0] >= 0.5).astype(np.uint8) * 255
            prob_u8 = (np.clip(probs[b, 0], 0, 1) * 255).astype(np.uint8)
            prob_color = cv2.cvtColor(
                cv2.applyColorMap(prob_u8, cv2.COLORMAP_JET),
                cv2.COLOR_BGR2RGB,
            )

            def to_rgb(m_2d: np.ndarray) -> np.ndarray:
                return cv2.cvtColor(m_2d, cv2.COLOR_GRAY2RGB)

            row = np.concatenate(
                [img_rgb, to_rgb(gt), to_rgb(pred), prob_color],
                axis=1,
            )
            out_path = viz_dir / f"epoch{epoch:03d}_sample{start_idx + b:02d}.jpg"
            cv2.imwrite(str(out_path), cv2.cvtColor(row, cv2.COLOR_RGB2BGR))

    # ------------------------------------------------------------------
    # History + checkpoints (rank 0 only)
    # ------------------------------------------------------------------

    def _update_history(self, train_metrics: dict, val_metrics: dict, lr: float = float("nan")) -> None:
        self.history["train_total"].append(train_metrics["total"])
        self.history["train_dice_loss"].append(train_metrics["dice_loss"])
        self.history["val_total"].append(val_metrics["total"])
        self.history["val_dice_loss"].append(val_metrics["dice_loss"])
        self.history["val_dice"].append(val_metrics["dice"])
        self.history["val_iou"].append(val_metrics["iou"])
        self.history["val_cldice"].append(val_metrics.get("cldice", float("nan")))
        for key in (
            "length_ratio_pixel_mean", "length_ratio_pixel_median",
            "length_ratio_pixel_p25",  "length_ratio_pixel_p75",
            "length_ratio_skel_mean",  "length_ratio_skel_median",
            "length_ratio_skel_p25",   "length_ratio_skel_p75",
        ):
            self.history[f"val_{key}"].append(val_metrics.get(key, float("nan")))
        self.history["lr"].append(float(lr))

    def _save_history(self) -> None:
        (self.save_dir / "history.json").write_text(json.dumps(self.history, indent=2))

    def _save_plots(self) -> None:
        epochs = list(range(1, len(self.history["train_total"]) + 1))
        fig, axes = plt.subplots(1, 4, figsize=(24, 4))

        axes[0].plot(epochs, self.history["train_total"], label="train", marker="o", markersize=3)
        axes[0].plot(epochs, self.history["val_total"],   label="val",   marker="s", markersize=3)
        axes[0].set_title("total loss"); axes[0].set_xlabel("epoch"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs, self.history["train_dice_loss"], label="train", marker="o", markersize=3)
        axes[1].plot(epochs, self.history["val_dice_loss"],   label="val",   marker="s", markersize=3)
        axes[1].set_title("dice loss"); axes[1].set_xlabel("epoch"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

        axes[2].plot(epochs, self.history["val_dice"],   label="dice",   marker="o", markersize=3)
        axes[2].plot(epochs, self.history["val_iou"],    label="iou",    marker="s", markersize=3)
        axes[2].plot(epochs, self.history["val_cldice"], label="clDice", marker="^", markersize=3)
        axes[2].set_title("val region metrics")
        axes[2].set_xlabel("epoch"); axes[2].legend(); axes[2].set_ylim(0, 1); axes[2].grid(True, alpha=0.3)

        ax = axes[3]
        for name, color in (("pixel", "C0"), ("skel", "C1")):
            mean = self.history[f"val_length_ratio_{name}_mean"]
            p25  = self.history[f"val_length_ratio_{name}_p25"]
            p75  = self.history[f"val_length_ratio_{name}_p75"]
            ax.plot(epochs, mean, label=f"{name} mean", color=color, marker="o", markersize=3)
            ax.fill_between(epochs, p25, p75, color=color, alpha=0.15, label=f"{name} p25–p75")
        ax.axhline(1.0, color="k", linestyle="--", linewidth=1, alpha=0.5, label="ideal = 1.0")
        ax.set_title("val length ratio (pred / GT)")
        ax.set_xlabel("epoch"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 2.0)

        fig.tight_layout()
        fig.savefig(self.save_dir / "history.png", dpi=120)
        plt.close(fig)

    def _maybe_save_best(self, epoch: int, val_metrics: dict) -> bool:
        improved = False
        if self.config.best_metric == "dice":
            if val_metrics["dice"] > self.best_val_dice:
                self.best_val_dice = val_metrics["dice"]
                improved = True
        else:
            if val_metrics["total"] < self.best_val_loss:
                self.best_val_loss = val_metrics["total"]
                improved = True
        if improved:
            # Unwrap DDP before saving so the checkpoint loads cleanly
            # under both DDP and single-GPU later.
            state_dict = (
                self.model.module.state_dict() if hasattr(self.model, "module")
                else self.model.state_dict()
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": state_dict,
                    "val_dice":  val_metrics["dice"],
                    "val_iou":   val_metrics["iou"],
                    "val_total": val_metrics["total"],
                },
                self.save_dir / "best.pth",
            )
        return improved

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        for epoch in range(1, self.config.epochs + 1):
            # Notify the criterion of the current epoch so composite losses
            # can ramp aux weights through their warmup. Plain BCEDice /
            # FocalDice don't have set_epoch — that's fine, no-op.
            if hasattr(self.criterion, "set_epoch"):
                self.criterion.set_epoch(epoch)

            # Current LR (for logging). On the first epoch this is config.lr
            # unmodified; under cosine+warmup it ramps up then decays.
            current_lr = float(self.optimizer.param_groups[0]["lr"])

            if _is_rank_zero():
                print(f"\n=== epoch {epoch}/{self.config.epochs}   lr={current_lr:.2e} ===")
            train_metrics = self.train_epoch(epoch)
            val_metrics   = self.val_epoch(epoch)

            if _is_rank_zero():
                print(
                    f"  train: total={train_metrics['total']:.4f}  "
                    f"dice_loss={train_metrics['dice_loss']:.4f}"
                )
                print(
                    f"  val:   total={val_metrics['total']:.4f}  "
                    f"dice_loss={val_metrics['dice_loss']:.4f}  "
                    f"dice={val_metrics['dice']:.4f}  iou={val_metrics['iou']:.4f}  "
                    f"cldice={val_metrics['cldice']:.4f}"
                )
                print(
                    f"  len:   pixel={val_metrics['length_ratio_pixel_mean']:.3f} "
                    f"(median {val_metrics['length_ratio_pixel_median']:.3f})  "
                    f"skeleton={val_metrics['length_ratio_skel_mean']:.3f} "
                    f"(median {val_metrics['length_ratio_skel_median']:.3f})"
                )

            # Checkpoint + history writes ONLY on rank 0 to avoid races.
            if _is_rank_zero():
                improved = self._maybe_save_best(epoch, val_metrics)
                if improved:
                    print(f"  saved best → {self.save_dir / 'best.pth'}")
                state_dict = (
                    self.model.module.state_dict() if hasattr(self.model, "module")
                    else self.model.state_dict()
                )
                torch.save(
                    {"epoch": epoch, "model_state_dict": state_dict},
                    self.save_dir / "last.pth",
                )
                self._update_history(train_metrics, val_metrics, lr=current_lr)
                self._save_history()
                self._save_plots()

            # Step LR scheduler AFTER the epoch is finished (so the lr we
            # logged was the lr actually used for this epoch's gradients).
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # Barrier so all ranks start the next epoch together (and so
            # rank 0's checkpoint write fully completes before any rank
            # might try to read it in a subsequent run).
            if _ddp_active():
                dist.barrier()
