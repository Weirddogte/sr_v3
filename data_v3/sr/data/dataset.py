from __future__ import annotations
import json
import random
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from sr.config import SRConfig, ZoneLabel, ZoneRole
from sr.data.labels import compute_heatmap_targets_batch, price_to_pixel_y


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _zone_from_dict(d: dict) -> ZoneLabel:
    return ZoneLabel(
        zone_id=d["zone_id"],
        role=ZoneRole(d["role"]),
        low_price=d["low_price"],
        high_price=d["high_price"],
        center_price=d["center_price"],
        touch_count=d["touch_count"],
        strength=d["strength"],
        is_active=d["is_active"],
        center_y=d["center_y"],
        height_px=d["height_px"],
        confluence_bonus=d.get("confluence_bonus", 0.0),
    )


def _stratified_split(
    records: list[dict],
    train_frac: float,
    val_frac: float,
) -> tuple[list[int], list[int], list[int]]:
    """Return (train_indices, val_indices, test_indices) stratified by scenario."""
    scenario_groups: dict[str, list[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        scenario_groups[rec["scenario"]].append(i)

    train_idx, val_idx, test_idx = [], [], []
    for scenario, indices in scenario_groups.items():
        n = len(indices)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        train_idx.extend(indices[:n_train])
        val_idx.extend(indices[n_train : n_train + n_val])
        test_idx.extend(indices[n_train + n_val :])

    return train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class V3SupportResistanceDataset(Dataset):
    """PyTorch Dataset for V3 support/resistance detection with precomputed heatmap targets."""

    def __init__(
        self,
        data_dir: Path,
        cfg: SRConfig,
        device: torch.device,
        split: str = "train",  # "train", "val", or "test"
        precomputed_targets: Optional[torch.Tensor] = None,
    ):
        data_dir = Path(data_dir)
        labels_path = data_dir / "labels_v3.jsonl"

        if not data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {data_dir}\n"
                "Run the data generation pipeline first (sr.data.generator) to populate this directory."
            )
        if not labels_path.exists():
            raise FileNotFoundError(
                f"Labels file not found: {labels_path}\n"
                "Run the data generation pipeline first to produce labels_v3.jsonl."
            )

        self.cfg = cfg
        self.split = split
        self.images_dir = data_dir / "images"

        # ------------------------------------------------------------------
        # 1. Load & parse JSONL records
        # ------------------------------------------------------------------
        raw_records: list[dict] = []
        with labels_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw_records.append(json.loads(line))

        self.records = raw_records

        # Reconstruct ZoneLabel objects per record
        self.zone_lists: list[list[ZoneLabel]] = []
        for rec in self.records:
            zones = [_zone_from_dict(z) for z in rec.get("zones", [])]
            self.zone_lists.append(zones)

        # ------------------------------------------------------------------
        # 2. Stratified split by scenario
        # ------------------------------------------------------------------
        train_idx, val_idx, test_idx = _stratified_split(
            self.records,
            train_frac=cfg.dataset.train_frac,
            val_frac=cfg.dataset.val_frac,
        )

        split_map = {"train": train_idx, "val": val_idx, "test": test_idx}
        if split not in split_map:
            raise ValueError(f"Unknown split '{split}'. Expected one of: train, val, test.")
        self.indices: list[int] = split_map[split]

        # ------------------------------------------------------------------
        # 3. Precompute / cache heatmap targets  [N_split, 5, H]
        # ------------------------------------------------------------------
        N = len(self.indices)
        H = cfg.generator.image_height

        if precomputed_targets is not None:
            self.targets = precomputed_targets[self.indices]
        else:
            # Compute on GPU for speed, store on CPU so DataLoader workers can access them.
            # (Worker processes are forked and cannot use the parent's CUDA context.)
            split_records = [self.records[i] for i in self.indices]
            split_zones = [self.zone_lists[i] for i in self.indices]

            print(f"[dataset] Precomputing heatmap targets for {split} split ({N} examples)…")
            self.targets = compute_heatmap_targets_batch(
                zone_lists=split_zones,
                current_prices=[r["current_price"] for r in split_records],
                price_ranges=[tuple(r["price_range"]) for r in split_records],
                img_height=cfg.generator.image_height,
                device=device,
            ).cpu()  # move to CPU — workers cannot access GPU tensors across process boundaries

        bytes_mb = self.targets.numel() * 4 / 1024 ** 2
        print(f"[dataset] Targets on CPU ({bytes_mb:.1f} MB).")

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        record_idx = self.indices[idx]
        record = self.records[record_idx]

        # Load image
        img_path = self.images_dir / f"{record['id']}.jpg"
        image = Image.open(img_path).convert("RGB")
        image = TF.to_tensor(image)  # [3, H, W], float32 in [0, 1]

        # Get precomputed target [5, H] (always CPU)
        target = self.targets[idx]

        zones = self.zone_lists[record_idx]

        # Horizontal flip augmentation (train only)
        if (
            self.split == "train"
            and random.random() < self.cfg.generator.horizontal_flip_probability
        ):
            image = TF.hflip(image)
            # Swap support (ch 0) and resistance (ch 1) channels
            target = target.clone()
            target[[0, 1]] = target[[1, 0]]
            # Flip along H dimension (price axis reverses)
            target = target.flip(dims=[1])

        return {
            "image": image,        # [3, H, W]
            "target": target,      # [5, H]
            "id": record["id"],
            "scenario": record["scenario"],
            "num_zones": len(zones),
            "zones": zones,        # list[ZoneLabel] — needed for metrics
        }


# ---------------------------------------------------------------------------
# Collate functions
# ---------------------------------------------------------------------------

def mixup_collate_fn(batch: list[dict], alpha: float = 0.3) -> dict:
    """Custom collate with mixup augmentation applied to ~50% of batch pairs."""
    images = torch.stack([b["image"] for b in batch])
    targets = torch.stack([b["target"] for b in batch])
    ids = [b["id"] for b in batch]
    scenarios = [b["scenario"] for b in batch]
    zone_lists = [b["zones"] for b in batch]
    num_zones = [b["num_zones"] for b in batch]

    B = len(batch)
    if alpha > 0 and B >= 2:
        num_mixup_pairs = B // 4  # each pair affects 2 examples
        for _ in range(num_mixup_pairs):
            i, j = random.sample(range(B), 2)
            lam = float(np.random.beta(alpha, alpha))
            images[i] = lam * images[i] + (1 - lam) * images[j]
            targets[i] = lam * targets[i] + (1 - lam) * targets[j]
            # Use majority example's zones
            if lam < 0.5:
                zone_lists[i] = zone_lists[j]

    return {
        "image": images,
        "target": targets,
        "id": ids,
        "scenario": scenarios,
        "zones": zone_lists,
        "num_zones": num_zones,
    }


def eval_collate_fn(batch: list[dict]) -> dict:
    """Collate for val/test — no mixup, handles non-tensorable zone lists."""
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "target": torch.stack([b["target"] for b in batch]),
        "id": [b["id"] for b in batch],
        "scenario": [b["scenario"] for b in batch],
        "zones": [b["zones"] for b in batch],
        "num_zones": [b["num_zones"] for b in batch],
    }


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloaders(
    data_dir: Path,
    cfg: SRConfig,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, val, test DataLoaders. Returns (train_loader, val_loader, test_loader)."""
    train_ds = V3SupportResistanceDataset(data_dir, cfg, device, split="train")
    val_ds = V3SupportResistanceDataset(data_dir, cfg, device, split="val")
    test_ds = V3SupportResistanceDataset(data_dir, cfg, device, split="test")

    _mixup_fn = partial(mixup_collate_fn, alpha=cfg.training.mixup_alpha)

    # multiprocessing_context='spawn' starts clean worker processes that do not
    # inherit the parent's CUDA context, avoiding cudaErrorInitializationError
    # on Linux (Colab default fork would corrupt the CUDA driver state).
    # sr/__init__.py re-adds the package root to sys.path so workers can import.
    _ctx = "spawn"
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=2,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=_mixup_fn,
        multiprocessing_context=_ctx,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size * 2,
        shuffle=False,
        num_workers=2,
        persistent_workers=True,
        collate_fn=eval_collate_fn,
        multiprocessing_context=_ctx,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.training.batch_size * 2,
        shuffle=False,
        num_workers=2,
        persistent_workers=True,
        collate_fn=eval_collate_fn,
        multiprocessing_context=_ctx,
    )

    print(
        f"[dataset] Train: {len(train_ds)} examples | "
        f"Val: {len(val_ds)} | Test: {len(test_ds)}"
    )

    return train_loader, val_loader, test_loader
