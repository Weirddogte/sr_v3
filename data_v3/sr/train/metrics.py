"""
sr/train/metrics.py

All metric computation for SR V3 training and evaluation.

Peak detection uses F.max_pool1d (vectorized batch op) rather than a Python
for-loop over 512 pixels.  The old per-pixel loop was called ~40 000 times per
eval epoch, taking ~100 s.  The vectorized path processes all examples and
channels in a single kernel call, reducing metrics to < 1 s per epoch.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_CHANNEL_FOR_ROLE: dict[str, int] = {
    "confirmed_support":           0,
    "confirmed_resistance":        1,
    "active_zone":                 2,
    "watch_support":               0,
    "watch_resistance":            1,
    "historical_zone":             3,
    "weak_zone":                   3,
    "general_level":               3,
}


def _role_to_channel(role) -> int:
    """Map a ZoneRole enum or string to a primary channel index (0-4)."""
    role_str = role.value if hasattr(role, "value") else str(role)
    return _CHANNEL_FOR_ROLE.get(role_str, 2)


# ---------------------------------------------------------------------------
# Vectorized peak detection  (core primitive)
# ---------------------------------------------------------------------------

def batch_find_peaks(
    heatmap: torch.Tensor,
    threshold: float = 0.10,
    min_distance: int = 5,
) -> torch.Tensor:
    """Vectorized local-maximum detection over a batch of 1-D heatmaps.

    Uses a single ``F.max_pool1d`` call to find the local maximum in a sliding
    window of width ``2*min_distance+1``.  A position is a peak when its value
    equals the local maximum AND exceeds ``threshold``.

    Args:
        heatmap:      [B, H] float tensor (values in [0, 1], on CPU).
        threshold:    Minimum value to qualify as a peak.
        min_distance: Minimum pixels between peaks.

    Returns:
        [B, H] bool tensor — True at peak positions.
    """
    B, H = heatmap.shape
    kernel = 2 * min_distance + 1

    x = heatmap.unsqueeze(1)                                          # [B, 1, H]
    padded = F.pad(x, (min_distance, min_distance), mode="replicate") # [B, 1, H+2k]
    local_max = F.max_pool1d(padded, kernel_size=kernel, stride=1)    # [B, 1, H]
    local_max = local_max.squeeze(1)                                   # [B, H]

    is_peak = (heatmap >= local_max - 1e-6) & (heatmap > threshold)
    return is_peak  # [B, H] bool


# ---------------------------------------------------------------------------
# Single-heatmap wrapper  (used by inference code)
# ---------------------------------------------------------------------------

def find_peaks_1d(
    heatmap: torch.Tensor,   # [H] single channel for one example
    threshold: float = 0.10,
    min_distance: int = 5,
) -> list[int]:
    """Find local maxima in a 1-D heatmap above threshold.

    Thin wrapper around :func:`batch_find_peaks` so that inference code that
    calls this function one heatmap at a time still benefits from the fast
    implementation.
    """
    h = heatmap.float().cpu()
    if h.dim() != 1:
        h = h.squeeze()
    mask = batch_find_peaks(h.unsqueeze(0), threshold=threshold,
                            min_distance=min_distance).squeeze(0)  # [H] bool
    indices = mask.nonzero(as_tuple=False).squeeze(-1)
    if indices.dim() == 0:
        return [int(indices.item())] if mask.any() else []
    return indices.tolist()


# ---------------------------------------------------------------------------
# Zone recall
# ---------------------------------------------------------------------------

def zone_recall_at_k_px(
    pred_logits: torch.Tensor,  # [B, 5, H]
    targets: torch.Tensor,      # [B, 5, H]  (unused — kept for API compat)
    zone_lists: list[list],
    k_px: int = 10,
    threshold: float = 0.10,
) -> dict:
    """Fraction of GT zones with a predicted peak within *k_px* pixels.

    Peak detection is vectorized over all B×5 heatmaps in one pass.

    Returns:
        dict with keys ``zone_recall_at_10px`` (float) and
        ``per_channel_recall`` (list[float] × 5).
    """
    pred = torch.sigmoid(pred_logits.float().cpu())  # [B, 5, H]
    B, C, H = pred.shape

    # Detect peaks for every (example, channel) pair at once.
    pred_flat = pred.reshape(B * C, H)                         # [B*5, H]
    is_peak = batch_find_peaks(pred_flat, threshold=threshold) # [B*5, H] bool
    is_peak = is_peak.reshape(B, C, H)                         # [B, 5, H]

    per_channel_found = [0] * 5
    per_channel_total = [0] * 5

    for i in range(B):
        for zone in (zone_lists[i] or []):
            ch     = _role_to_channel(getattr(zone, "role", "active_zone"))
            center = int(getattr(zone, "center_y", 0))
            lo     = max(0, center - k_px)
            hi     = min(H, center + k_px + 1)
            found  = bool(is_peak[i, ch, lo:hi].any())
            per_channel_found[ch] += int(found)
            per_channel_total[ch] += 1

    total_found = sum(per_channel_found)
    total_zones = sum(per_channel_total)
    return {
        "zone_recall_at_10px": total_found / max(total_zones, 1),
        "per_channel_recall":  [
            per_channel_found[c] / max(per_channel_total[c], 1)
            for c in range(5)
        ],
    }


# ---------------------------------------------------------------------------
# False peak rate
# ---------------------------------------------------------------------------

def false_peak_rate(
    pred_logits: torch.Tensor,  # [B, 5, H]
    zone_lists: list[list],
    threshold: float = 0.10,
    gt_window_px: int = 20,
) -> float:
    """Fraction of predicted peaks that have no matching GT zone within *gt_window_px*."""
    pred = torch.sigmoid(pred_logits.float().cpu())  # [B, 5, H]
    B, C, H = pred.shape

    pred_flat = pred.reshape(B * C, H)
    is_peak   = batch_find_peaks(pred_flat, threshold=threshold).reshape(B, C, H)

    total_peaks = 0
    false_peaks = 0

    for i in range(B):
        gt_centers: dict[int, list[int]] = {c: [] for c in range(5)}
        for zone in (zone_lists[i] or []):
            ch = _role_to_channel(getattr(zone, "role", "active_zone"))
            gt_centers[ch].append(int(getattr(zone, "center_y", 0)))

        for ch in range(5):
            indices = is_peak[i, ch].nonzero(as_tuple=False).squeeze(-1)
            if indices.dim() == 0:
                peaks = [int(indices.item())] if is_peak[i, ch].any() else []
            else:
                peaks = indices.tolist()
            for p in peaks:
                total_peaks += 1
                if not any(abs(p - gt) <= gt_window_px for gt in gt_centers[ch]):
                    false_peaks += 1

    return false_peaks / max(total_peaks, 1)


# ---------------------------------------------------------------------------
# Per-channel peak MAE
# ---------------------------------------------------------------------------

def per_channel_peak_mae(
    pred_logits: torch.Tensor,  # [B, 5, H]
    targets: torch.Tensor,      # [B, 5, H]
    threshold: float = 0.10,
) -> list[float]:
    """Mean pixel distance between predicted peaks and nearest GT peaks, per channel."""
    pred = torch.sigmoid(pred_logits.float().cpu())  # [B, 5, H]
    tgt  = targets.float().cpu()
    B, C, H = pred.shape

    pred_flat   = pred.reshape(B * C, H)
    tgt_flat    = tgt.reshape(B * C, H)
    is_peak_pred = batch_find_peaks(pred_flat, threshold=threshold).reshape(B, C, H)
    is_peak_tgt  = batch_find_peaks(tgt_flat,  threshold=threshold).reshape(B, C, H)

    channel_distances: list[list[float]] = [[] for _ in range(5)]

    for i in range(B):
        for ch in range(5):
            def _peaks(mask_ch):
                idx = mask_ch.nonzero(as_tuple=False).squeeze(-1)
                if idx.dim() == 0:
                    return [int(idx.item())] if mask_ch.any() else []
                return idx.tolist()

            pred_peaks = _peaks(is_peak_pred[i, ch])
            gt_peaks   = _peaks(is_peak_tgt[i, ch])

            if not pred_peaks or not gt_peaks:
                continue
            for pp in pred_peaks:
                channel_distances[ch].append(float(min(abs(pp - gp) for gp in gt_peaks)))

    return [
        sum(d) / len(d) if d else float("nan")
        for d in channel_distances
    ]


# ---------------------------------------------------------------------------
# Per-scenario recall
# ---------------------------------------------------------------------------

def per_scenario_recall(
    pred_logits: torch.Tensor,  # [N, 5, H]
    zone_lists: list[list],
    scenarios: list[str],
    k_px: int = 10,
) -> dict[str, float]:
    """Zone recall broken out by scenario."""
    pred = torch.sigmoid(pred_logits.float().cpu())
    N, C, H = pred.shape

    pred_flat = pred.reshape(N * C, H)
    is_peak   = batch_find_peaks(pred_flat).reshape(N, C, H)

    scenario_found: dict[str, int] = {}
    scenario_total: dict[str, int] = {}

    for i in range(N):
        sc = (scenarios[i] or "unknown")
        scenario_found.setdefault(sc, 0)
        scenario_total.setdefault(sc, 0)
        for zone in (zone_lists[i] or []):
            ch     = _role_to_channel(getattr(zone, "role", "active_zone"))
            center = int(getattr(zone, "center_y", 0))
            lo     = max(0, center - k_px)
            hi     = min(H, center + k_px + 1)
            scenario_found[sc] += int(is_peak[i, ch, lo:hi].any())
            scenario_total[sc] += 1

    return {s: scenario_found[s] / max(scenario_total[s], 1) for s in scenario_found}


# ---------------------------------------------------------------------------
# Batch metrics (called by eval_one_epoch)
# ---------------------------------------------------------------------------

def compute_batch_metrics(
    pred_logits: torch.Tensor,
    targets: torch.Tensor,
    zone_lists: list[list],
    scenarios: list[str],
) -> dict:
    """Compute all metrics for a batch.  Returns dict of metric_name → value.

    All peak detection is vectorized; this typically completes in < 1 s even
    for the full 2 250-example validation set.
    """
    recall_dict = zone_recall_at_k_px(pred_logits, targets, zone_lists, k_px=10)
    fpr         = false_peak_rate(pred_logits, zone_lists)
    mae_list    = per_channel_peak_mae(pred_logits, targets)

    return {
        "zone_recall_10px":         recall_dict["zone_recall_at_10px"],
        "false_peak_rate":          fpr,
        "per_channel_mae_px":       mae_list,
        "per_channel_recall_10px":  recall_dict["per_channel_recall"],
    }


# ---------------------------------------------------------------------------
# Worst recall examples
# ---------------------------------------------------------------------------

def find_worst_recall_examples(
    pred_logits: torch.Tensor,  # [N, 5, H]
    zone_lists: list[list],
    image_ids: list[str],
    n: int = 20,
) -> list[str]:
    """Return IDs of the *n* examples with lowest per-example zone recall."""
    pred = torch.sigmoid(pred_logits.float().cpu())
    N, C, H = pred.shape

    pred_flat = pred.reshape(N * C, H)
    is_peak   = batch_find_peaks(pred_flat).reshape(N, C, H)

    per_example_recall: list[tuple[float, str]] = []

    for i in range(N):
        zones = zone_lists[i] or []
        if not zones:
            per_example_recall.append((1.0, image_ids[i]))
            continue

        found = 0
        for zone in zones:
            ch     = _role_to_channel(getattr(zone, "role", "active_zone"))
            center = int(getattr(zone, "center_y", 0))
            lo     = max(0, center - 10)
            hi     = min(H, center + 11)
            if is_peak[i, ch, lo:hi].any():
                found += 1
        per_example_recall.append((found / len(zones), image_ids[i]))

    per_example_recall.sort(key=lambda x: x[0])
    return [img_id for _, img_id in per_example_recall[:n]]
