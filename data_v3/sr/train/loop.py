"""
sr/train/loop.py

Training and evaluation loops for SR V3.
"""

from __future__ import annotations

import csv
import dataclasses
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from sr.model.loss import SoftHeatmapLossV3
from sr.train import metrics


# ---------------------------------------------------------------------------
# Train one epoch
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    cfg,    # TrainingConfig
    epoch: int,
) -> dict:
    """Train one epoch with AMP and gradient clipping.

    Returns dict: {loss, bce, peak_mse, zone_recall_10px}
    """
    model.train()
    total_loss = 0.0
    total_bce = 0.0
    total_peak_mse = 0.0
    total_recall = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        zone_lists = batch["zones"]

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda" if device.type == "cuda" else "cpu"):
            logits = model(images)
            loss_dict = loss_fn(logits, targets)
            loss = loss_dict["loss"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        # Metrics (no grad needed)
        with torch.no_grad():
            recall_dict = metrics.zone_recall_at_k_px(
                logits.detach(), targets.detach(), zone_lists
            )

        total_loss += loss.item()
        total_bce += loss_dict["bce"].item()
        total_peak_mse += loss_dict["peak_mse"].item()
        total_recall += recall_dict["zone_recall_at_10px"]
        num_batches += 1

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            recall=f"{recall_dict['zone_recall_at_10px']:.3f}",
        )

    denom = max(num_batches, 1)
    return {
        "loss": total_loss / denom,
        "bce": total_bce / denom,
        "peak_mse": total_peak_mse / denom,
        "zone_recall_10px": total_recall / denom,
    }


# ---------------------------------------------------------------------------
# Eval one epoch
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    save_worst_dir: Optional[Path] = None,
) -> dict:
    """Eval one epoch. Optionally saves worst-recall example image IDs to disk.

    Returns dict: {loss, bce, peak_mse, zone_recall_10px, false_peak_rate,
                   per_channel_recall, per_channel_mae_px}
    """
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    all_zones: list[list] = []
    all_scenarios: list[str] = []
    all_ids: list[str] = []
    total_loss = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [eval]", leave=False)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda" if device.type == "cuda" else "cpu"):
            logits = model(images)
            loss_dict = loss_fn(logits, targets)

        total_loss += loss_dict["loss"].item()
        all_logits.append(logits.cpu())
        all_targets.append(targets.cpu())
        all_zones.extend(batch["zones"])
        all_scenarios.extend(batch["scenario"])
        all_ids.extend(batch["id"])

    all_logits_cat = torch.cat(all_logits, dim=0)
    all_targets_cat = torch.cat(all_targets, dim=0)

    metrics_dict = metrics.compute_batch_metrics(
        all_logits_cat, all_targets_cat, all_zones, all_scenarios
    )
    metrics_dict["loss"] = total_loss / max(len(loader), 1)

    # Derive bce/peak_mse from the aggregated targets for consistency
    # (these aren't recomputed here — callers get them from train loop)
    metrics_dict.setdefault("bce", float("nan"))
    metrics_dict.setdefault("peak_mse", float("nan"))

    # Find and save worst-recall examples
    if save_worst_dir is not None:
        worst_ids = metrics.find_worst_recall_examples(
            all_logits_cat, all_zones, all_ids, n=20
        )
        save_worst_dir.mkdir(exist_ok=True, parents=True)
        (save_worst_dir / f"worst_recall_epoch_{epoch}.txt").write_text("\n".join(worst_ids))

    return metrics_dict


# ---------------------------------------------------------------------------
# Full training run
# ---------------------------------------------------------------------------

def run(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg,            # SRConfig
    device: torch.device,
    output_dir: Path,
) -> dict:
    """Full training run.

    Returns the final val metrics dict.
    Saves checkpoints to output_dir/checkpoints/.
    Saves training history to output_dir/training_history_v3.csv.
    """
    tcfg = cfg.training

    # ------------------------------------------------------------------
    # Optimizer, scheduler, loss, scaler
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tcfg.lr_initial,
        weight_decay=tcfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=tcfg.cosine_T0,
        T_mult=tcfg.cosine_T_mult,
    )
    loss_fn = SoftHeatmapLossV3(tcfg).to(device)
    scaler = torch.amp.GradScaler("cuda" if device.type == "cuda" else "cpu")

    # ------------------------------------------------------------------
    # Optionally compile model (torch >= 2.0)
    # ------------------------------------------------------------------
    try:
        if int(torch.__version__.split(".")[0]) >= 2:
            model = torch.compile(model)
            print("torch.compile enabled.")
    except Exception as exc:
        print(f"torch.compile skipped: {exc}")

    # ------------------------------------------------------------------
    # Output directories / CSV
    # ------------------------------------------------------------------
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    worst_dir = output_dir / "worst_recall"

    csv_path = output_dir / "training_history_v3.csv"
    csv_columns = [
        "epoch",
        "train_loss", "val_loss",
        "train_bce", "val_bce",
        "train_peak_mse", "val_peak_mse",
        "train_recall", "val_recall",
        "val_false_peak_rate",
        "lr",
    ]

    csv_file = csv_path.open("w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
    csv_writer.writeheader()

    # ------------------------------------------------------------------
    # Training state
    # ------------------------------------------------------------------
    history: list[dict] = []
    best_val_loss = float("inf")
    # Track the last 3 epoch checkpoints for rotation
    epoch_ckpt_paths: list[Path] = []

    final_val_metrics: dict = {}

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    num_epochs = tcfg.num_epochs

    for epoch in range(1, num_epochs + 1):
        # Train
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            scaler=scaler,
            device=device,
            cfg=tcfg,
            epoch=epoch,
        )

        # Eval
        val_metrics = eval_one_epoch(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch,
            save_worst_dir=worst_dir,
        )

        # Capture the LR used this epoch before advancing the schedule
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # ------------------------------------------------------------------
        # Logging
        # ------------------------------------------------------------------
        print(
            f"Epoch {epoch}/{num_epochs} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"recall={val_metrics.get('zone_recall_10px', 0.0):.3f} | "
            f"lr={current_lr:.2e}"
        )

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "train_bce": train_metrics.get("bce", float("nan")),
            "val_bce": val_metrics.get("bce", float("nan")),
            "train_peak_mse": train_metrics.get("peak_mse", float("nan")),
            "val_peak_mse": val_metrics.get("peak_mse", float("nan")),
            "train_recall": train_metrics.get("zone_recall_10px", 0.0),
            "val_recall": val_metrics.get("zone_recall_10px", 0.0),
            "val_false_peak_rate": val_metrics.get("false_peak_rate", float("nan")),
            "lr": current_lr,
        }
        csv_writer.writerow(row)
        csv_file.flush()
        history.append(row)

        # ------------------------------------------------------------------
        # Checkpoint helpers
        # ------------------------------------------------------------------
        def _state_dict() -> dict:
            """Extract state dict, handling compiled models."""
            if hasattr(model, "_orig_mod"):
                return model._orig_mod.state_dict()
            return model.state_dict()

        def _save_checkpoint(path: Path) -> None:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": _state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_metrics["loss"],
                    "train_history": history,
                    "config": dataclasses.asdict(cfg),
                },
                path,
            )

        # Best checkpoint
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            _save_checkpoint(ckpt_dir / "best.pt")

        # Periodic epoch checkpoint (every 5 epochs, keep last 3)
        if epoch % 5 == 0:
            epoch_path = ckpt_dir / f"epoch_{epoch:04d}.pt"
            _save_checkpoint(epoch_path)
            epoch_ckpt_paths.append(epoch_path)
            # Remove oldest if more than 3 checkpoints stored
            while len(epoch_ckpt_paths) > 3:
                old_path = epoch_ckpt_paths.pop(0)
                try:
                    old_path.unlink()
                except FileNotFoundError:
                    pass

        final_val_metrics = val_metrics

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    csv_file.close()

    return final_val_metrics
