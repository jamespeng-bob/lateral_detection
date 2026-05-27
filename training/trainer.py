"""Vanilla-PyTorch training loop for binary lateral segmentation.

Responsibilities
----------------
- Optimization with AdamW + grad clipping
- Epoch loop with tqdm progress
- Per-epoch validation with Dice + IoU metrics
- Best-by-val-Dice checkpoint + last-epoch checkpoint
- Per-epoch matplotlib loss curve + JSON history
- Per-epoch validation visualization (image / GT / pred / probability)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


class SegTrainer:
    """Train a binary segmentation model end-to-end."""

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
        self.model = model.to(config.device)
        self.criterion = criterion.to(config.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = config.device

        self.optimizer = optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=float(config.lr),
            weight_decay=float(config.weight_decay),
        )

        self.save_dir = Path(config.save_dir)
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
        }
        self.best_val_dice = -1.0
        self.best_val_loss = float("inf")

    # ------------------------------------------------------------------
    # Metric helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _dice_metric(
        logits: torch.Tensor,
        target: torch.Tensor,
        threshold: float = 0.5,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        prob = (torch.sigmoid(logits) >= threshold).float()
        target = (target >= 0.5).float()
        inter = (prob * target).sum()
        denom = prob.sum() + target.sum()
        return (2.0 * inter + eps) / (denom + eps)

    @staticmethod
    def _iou_metric(
        logits: torch.Tensor,
        target: torch.Tensor,
        threshold: float = 0.5,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        prob = (torch.sigmoid(logits) >= threshold).float()
        target = (target >= 0.5).float()
        inter = (prob * target).sum()
        union = prob.sum() + target.sum() - inter
        return (inter + eps) / (union + eps)

    def _to_device(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            batch["image"].to(self.device, non_blocking=True),
            batch["mask"].to(self.device,  non_blocking=True),
        )

    # ------------------------------------------------------------------
    # Train / val epochs
    # ------------------------------------------------------------------

    def train_epoch(self) -> dict[str, float]:
        self.model.train()
        totals = {"total": 0.0, "dice_loss": 0.0}
        n = 0
        pbar = tqdm(self.train_loader, desc="train", leave=False)
        for batch in pbar:
            images, masks = self._to_device(batch)
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(images)
            losses = self.criterion(logits, masks)
            losses["total"].backward()
            if self.config.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()

            totals["total"]     += float(losses["total"].item())
            totals["dice_loss"] += float(losses["dice"].item())
            n += 1
            if n % self.config.log_interval == 0:
                pbar.set_postfix(
                    total=f"{totals['total']/n:.4f}",
                    dice=f"{totals['dice_loss']/n:.4f}",
                )
        n = max(n, 1)
        return {"total": totals["total"] / n, "dice_loss": totals["dice_loss"] / n}

    @torch.no_grad()
    def val_epoch(self, epoch: int) -> dict[str, float]:
        self.model.eval()
        totals = {"total": 0.0, "dice_loss": 0.0, "dice": 0.0, "iou": 0.0}
        n = 0
        viz_saved = 0
        viz_dir = self.save_dir / "val_viz"
        viz_dir.mkdir(parents=True, exist_ok=True)

        pbar = tqdm(self.val_loader, desc="val", leave=False)
        for batch in pbar:
            images, masks = self._to_device(batch)
            logits = self.model(images)
            losses = self.criterion(logits, masks)
            dice = self._dice_metric(logits, masks)
            iou  = self._iou_metric(logits, masks)

            totals["total"]     += float(losses["total"].item())
            totals["dice_loss"] += float(losses["dice"].item())
            totals["dice"]      += float(dice.item())
            totals["iou"]       += float(iou.item())
            n += 1

            if viz_saved < self.config.val_viz_count:
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

            pbar.set_postfix(
                dice=f"{totals['dice']/n:.4f}",
                iou=f"{totals['iou']/n:.4f}",
            )

        n = max(n, 1)
        return {
            "total":     totals["total"]     / n,
            "dice_loss": totals["dice_loss"] / n,
            "dice":      totals["dice"]      / n,
            "iou":       totals["iou"]       / n,
        }

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def _denormalize(self, img_t: torch.Tensor) -> np.ndarray:
        img = img_t.numpy() * self.std + self.mean
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        return img.transpose(1, 2, 0)  # → [H, W, 3] RGB

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
    # History + checkpoints
    # ------------------------------------------------------------------

    def _update_history(self, train_metrics: dict, val_metrics: dict) -> None:
        self.history["train_total"].append(train_metrics["total"])
        self.history["train_dice_loss"].append(train_metrics["dice_loss"])
        self.history["val_total"].append(val_metrics["total"])
        self.history["val_dice_loss"].append(val_metrics["dice_loss"])
        self.history["val_dice"].append(val_metrics["dice"])
        self.history["val_iou"].append(val_metrics["iou"])

    def _save_history(self) -> None:
        (self.save_dir / "history.json").write_text(json.dumps(self.history, indent=2))

    def _save_plots(self) -> None:
        epochs = list(range(1, len(self.history["train_total"]) + 1))
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))

        axes[0].plot(epochs, self.history["train_total"], label="train", marker="o", markersize=3)
        axes[0].plot(epochs, self.history["val_total"],   label="val",   marker="s", markersize=3)
        axes[0].set_title("total loss"); axes[0].set_xlabel("epoch"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs, self.history["train_dice_loss"], label="train", marker="o", markersize=3)
        axes[1].plot(epochs, self.history["val_dice_loss"],   label="val",   marker="s", markersize=3)
        axes[1].set_title("dice loss"); axes[1].set_xlabel("epoch"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

        axes[2].plot(epochs, self.history["val_dice"], label="dice", marker="o", markersize=3)
        axes[2].plot(epochs, self.history["val_iou"],  label="iou",  marker="s", markersize=3)
        axes[2].set_title("val metrics"); axes[2].set_xlabel("epoch"); axes[2].legend()
        axes[2].set_ylim(0, 1); axes[2].grid(True, alpha=0.3)

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
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
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
            print(f"\n=== epoch {epoch}/{self.config.epochs} ===")
            train_metrics = self.train_epoch()
            val_metrics   = self.val_epoch(epoch)

            print(
                f"  train: total={train_metrics['total']:.4f}  "
                f"dice_loss={train_metrics['dice_loss']:.4f}"
            )
            print(
                f"  val:   total={val_metrics['total']:.4f}  "
                f"dice_loss={val_metrics['dice_loss']:.4f}  "
                f"dice={val_metrics['dice']:.4f}  iou={val_metrics['iou']:.4f}"
            )

            improved = self._maybe_save_best(epoch, val_metrics)
            if improved:
                print(f"  saved best → {self.save_dir / 'best.pth'}")

            torch.save(
                {"epoch": epoch, "model_state_dict": self.model.state_dict()},
                self.save_dir / "last.pth",
            )

            self._update_history(train_metrics, val_metrics)
            self._save_history()
            self._save_plots()
