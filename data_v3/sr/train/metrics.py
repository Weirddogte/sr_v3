"""
sr/train/metrics.py

All metric computation for SR V3 training and evaluation.
"""

from __future__ import annotations

import math
from typing import Optional

import torch


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
# Peak detection
# ---------------------------------------------------------------------------

def find_peaks_1d(
    heatmap: torch.Tensor,  # [H] single channel for one example
    threshold: float = 0.10,
    min_distance: int = 5,
) -> list[int]:
    """Find local maxima in 1D heatmap above threshold.

    A position is a peak if it is above *threshold* AND strictly greater than
    every neighbour within *min_distance* on both sides (clamped at boundaries).
    """
    H = heatmap.shape[0]
    peaks: list[int] = []

    heatmap_np = heatmap.float().cpu()

    for i in range(H):
        val = heatmap_np[i].item()
        if val < threshold:
            continue
        lo = max(0, i - min_distance)
        hi = min(H, i + min_distance + 1)
        window = heatmap_np[lo:hi]
        if val >= window.max().item() and (window == val).nonzero(as_tuple=False)[0].item() == (i - lo):
            peaks.append(i)

    return peaks


# ---------------------------------------------------------------------------
# Zone recall
# ---------------------------------------------------------------------------

def zone_recall_at_k_px(
    pred_logits: torch.Tensor,  # [B, 5, H]
    targets: torch.Tensor,      # [B, 5, H]  (unused here but kept for API consistency)
    zone_lists: list[list],     # list of ZoneLabel lists per example
    k_px: int = 10,
    threshold: float = 0.10,
) -> dict:
    """Compute zone recall: fraction of GT zones with a predicted peak within k_px.

    Returns dict with:
      - zone_recall_at_10px: float (mean over all examples and zones)
      - per_channel_recall: list[float] of length 5
    """
    pred = torch.sigmoid(pred_logits)  # [B, 5, H]
    B = pred.shape[0]

    per_channel_found = [0] * 5
    per_channel_total = [0] * 5

    for i in range(B):
        zones = zone_lists[i] if zone_lists[i] is not None else []
        for zone in zones:
            ch = _role_to_channel(zone.role) if hasattr(zone, "role") else 2
            center = int(zone.center_y) if hasattr(zone, "center_y") else 0

            peaks = find_peaks_1d(pred[i, ch], threshold=threshold)
            found = any(abs(p - center) <= k_px for p in peaks)

            per_channel_found[ch] += int(found)
            per_channel_total[ch] += 1

    total_found = sum(per_channel_found)
    total_zones = sum(per_channel_total)

    overall_recall = total_found / total_zones if total_zones > 0 else 0.0

    per_channel_recall = [
        per_channel_found[c] / per_channel_total[c] if per_channel_total[c] > 0 else 0.0
        for c in range(5)
    ]

    return {
        "zone_recall_at_10px": overall_recall,
        "per_channel_recall": per_channel_recall,
    }


# ---------------------------------------------------------------------------
# False peak rate
# ---------------------------------------------------------------------------

def false_peak_rate(
    pred_logits: torch.Tensor,  # [B, 5, H]
    zone_lists: list[list],     # ZoneLabel lists per example
    threshold: float = 0.10,
    gt_window_px: int = 20,
) -> float:
    """Fraction of predicted peaks that have no matching GT zone within gt_window_px.

    Measures hallucination: how often the model fires where there is no GT zone.
    """
    pred = torch.sigmoid(pred_logits)  # [B, 5, H]
    B = pred.shape[0]

    total_peaks = 0
    false_peaks = 0

    for i in range(B):
        zones = zone_lists[i] if zone_lists[i] is not None else []

        # Build a lookup: channel -> list of GT center positions
        gt_centers: dict[int, list[int]] = {c: [] for c in range(5)}
        for zone in zones:
            ch = _role_to_channel(zone.role) if hasattr(zone, "role") else 2
            center = int(zone.center_y) if hasattr(zone, "center_y") else 0
            gt_centers[ch].append(center)

        for ch in range(5):
            peaks = find_peaks_1d(pred[i, ch], threshold=threshold)
            for p in peaks:
                total_peaks += 1
                matched = any(abs(p - gt) <= gt_window_px for gt in gt_centers[ch])
                if not matched:
                    false_peaks += 1

    return false_peaks / total_peaks if total_peaks > 0 else 0.0


# ---------------------------------------------------------------------------
# Per-channel peak MAE
# ---------------------------------------------------------------------------

def per_channel_peak_mae(
    pred_logits: torch.Tensor,  # [B, 5, H]
    targets: torch.Tensor,      # [B, 5, H]
    threshold: float = 0.10,
) -> list[float]:
    """For each channel, mean pixel distance between predicted peak and nearest GT peak.

    Returns list of 5 floats (nan if no peaks in either pred or GT for that channel).
    """
    pred = torch.sigmoid(pred_logits)  # [B, 5, H]
    B = pred.shape[0]

    channel_distances: list[list[float]] = [[] for _ in range(5)]

    for i in range(B):
        for ch in range(5):
            pred_peaks = find_peaks_1d(pred[i, ch], threshold=threshold)
            gt_peaks = find_peaks_1d(targets[i, ch], threshold=threshold)

            if not pred_peaks or not gt_peaks:
                continue

            for pp in pred_peaks:
                nearest = min(abs(pp - gp) for gp in gt_peaks)
                channel_distances[ch].append(float(nearest))

    return [
        float(sum(d) / len(d)) if d else float("nan")
        for d in channel_distances
    ]


# ---------------------------------------------------------------------------
# Per-scenario recall
# ---------------------------------------------------------------------------

def per_scenario_recall(
    pred_logits: torch.Tensor,  # [N, 5, H]
    zone_lists: list[list],     # ZoneLabel lists
    scenarios: list[str],
    k_px: int = 10,
) -> dict[str, float]:
    """Zone recall broken out by scenario."""
    pred = torch.sigmoid(pred_logits)
    N = pred.shape[0]

    scenario_found: dict[str, int] = {}
    scenario_total: dict[str, int] = {}

    for i in range(N):
        scenario = scenarios[i] if scenarios[i] is not None else "unknown"
        zones = zone_lists[i] if zone_lists[i] is not None else []

        if scenario not in scenario_found:
            scenario_found[scenario] = 0
            scenario_total[scenario] = 0

        for zone in zones:
            ch = _role_to_channel(zone.role) if hasattr(zone, "role") else 2
            center = int(zone.center_y) if hasattr(zone, "center_y") else 0

            peaks = find_peaks_1d(pred[i, ch])
            found = any(abs(p - center) <= k_px for p in peaks)

            scenario_found[scenario] += int(found)
            scenario_total[scenario] += 1

    return {
        s: scenario_found[s] / scenario_total[s] if scenario_total[s] > 0 else 0.0
        for s in scenario_found
    }


# ---------------------------------------------------------------------------
# Batch metrics
# ---------------------------------------------------------------------------

def compute_batch_metrics(
    pred_logits: torch.Tensor,
    targets: torch.Tensor,
    zone_lists: list[list],
    scenarios: list[str],
) -> dict:
    """Compute all metrics for a batch. Returns dict of metric_name -> value."""
    recall_dict = zone_recall_at_k_px(pred_logits, targets, zone_lists, k_px=10)
    fpr = false_peak_rate(pred_logits, zone_lists)
    mae_list = per_channel_peak_mae(pred_logits, targets)

    return {
        "zone_recall_10px": recall_dict["zone_recall_at_10px"],
        "false_peak_rate": fpr,
        "per_channel_mae_px": mae_list,
        "per_channel_recall_10px": recall_dict["per_channel_recall"],
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
    """Return IDs of n examples with lowest per-example zone recall."""
    pred = torch.sigmoid(pred_logits)
    N = pred.shape[0]

    per_example_recall: list[tuple[float, str]] = []

    for i in range(N):
        zones = zone_lists[i] if zone_lists[i] is not None else []
        if not zones:
            # No GT zones -> treat as perfect (don't penalise examples with no labels)
            per_example_recall.append((1.0, image_ids[i]))
            continue

        found = 0
        for zone in zones:
            ch = _role_to_channel(zone.role) if hasattr(zone, "role") else 2
            center = int(zone.center_y) if hasattr(zone, "center_y") else 0
            peaks = find_peaks_1d(pred[i, ch])
            if any(abs(p - center) <= 10 for p in peaks):
                found += 1

        recall = found / len(zones)
        per_example_recall.append((recall, image_ids[i]))

    # Sort ascending by recall (worst first)
    per_example_recall.sort(key=lambda x: x[0])
    return [img_id for _, img_id in per_example_recall[:n]]
