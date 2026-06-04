from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from sr.config import TrainingConfig


class SoftHeatmapLossV3(nn.Module):
    """
    Weighted BCE + peak MSE loss for 5-channel heatmap prediction.

    Expects raw logits from the network (no sigmoid applied beforehand).

    Channels:
        0 – support
        1 – resistance
        2 – active zone
        3 – historical zone
        4 – proximity
    """

    def __init__(self, cfg: TrainingConfig) -> None:
        super().__init__()

        # Per-channel positive weights as a [5] buffer (moved to device automatically).
        # Each weight amplifies the loss on positive pixels for the corresponding channel.
        pos_weights = torch.tensor(
            [
                cfg.loss_positive_weight_support,     # 12.0
                cfg.loss_positive_weight_resistance,  # 12.0
                cfg.loss_positive_weight_active,      # 10.0
                cfg.loss_positive_weight_historical,  #  6.0
                cfg.loss_positive_weight_proximity,   #  8.0
            ],
            dtype=torch.float32,
        )
        # Register as a buffer so it moves with .to(device) / .cuda()
        self.register_buffer("pos_weights", pos_weights)

        # Weight for the peak MSE term
        self.peak_mse_weight: float = cfg.loss_peak_mse_weight  # 0.30

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute the combined loss.

        Args:
            logits:  Raw network output, shape [B, 5, H].
            targets: Soft ground-truth heatmap, shape [B, 5, H], values in [0, 1].

        Returns:
            A dict with keys:
                "loss"     – total scalar loss (differentiable)
                "bce"      – weighted BCE component (detached)
                "peak_mse" – peak MSE component (detached)
        """
        # --- Weighted BCE ---
        # Compute un-reduced BCE for every element: [B, 5, H]
        bce_per_elem = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        # Build per-element weight mask equivalent to BCEWithLogitsLoss pos_weight:
        #   weight = 1.0 + (pos_weight - 1.0) * target
        # When target ≈ 1 → weight = pos_weight; when target ≈ 0 → weight = 1.0
        pos_weight: torch.Tensor = self.pos_weights  # type: ignore[assignment]
        # Broadcast pos_weight [5] → [1, 5, 1] for [B, 5, H]
        weight_mask = 1.0 + (pos_weight[None, :, None] - 1.0) * targets
        weighted_bce = (bce_per_elem * weight_mask).mean()

        # --- Peak MSE (channels 0, 1, 2 only: support, resistance, active) ---
        preds_sigmoid = torch.sigmoid(logits[:, :3, :])  # [B, 3, H]
        targets_top3  = targets[:, :3, :]                # [B, 3, H]
        peak_mask     = targets_top3 > 0.35              # bool mask

        if peak_mask.any():
            peak_mse = F.mse_loss(
                preds_sigmoid[peak_mask],
                targets_top3[peak_mask],
            )
        else:
            peak_mse = torch.tensor(0.0, device=logits.device)

        # --- Combined loss ---
        total_loss = weighted_bce + self.peak_mse_weight * peak_mse

        return {
            "loss":     total_loss,
            "bce":      weighted_bce.detach(),
            "peak_mse": peak_mse.detach(),
        }
