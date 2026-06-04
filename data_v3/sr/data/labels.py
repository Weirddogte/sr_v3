from __future__ import annotations

import math

import numpy as np
import torch

from sr.config import ZoneLabel, ZoneRole, SRConfig


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHANNEL_NAMES: list[str] = ["support", "resistance", "active", "historical", "proximity"]

# Maps each ZoneRole to a dict of {channel_index: base_amplitude}.
# Channel indices: 0=support, 1=resistance, 2=active, 3=historical, 4=proximity
ROLE_CHANNEL_MAP: dict[ZoneRole, dict[int, float]] = {
    ZoneRole.confirmed_support:    {0: 1.00, 3: 0.40},
    ZoneRole.confirmed_resistance: {1: 1.00, 3: 0.40},
    ZoneRole.active_zone:          {2: 1.00, 3: 0.60},
    ZoneRole.watch_support:        {0: 1.00, 3: 0.30},
    ZoneRole.watch_resistance:     {1: 1.00, 3: 0.30},
    ZoneRole.historical_zone:      {3: 1.00},
    ZoneRole.weak_zone:            {3: 0.60},
    ZoneRole.general_level:        {3: 0.80},
}


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def price_to_pixel_y(
    price: float,
    price_min: float,
    price_max: float,
    img_height: int,
) -> int:
    """Convert price to image row (y=0 at top = high price).

    Parameters
    ----------
    price:
        The price value to convert.
    price_min:
        The minimum price in the visible range.
    price_max:
        The maximum price in the visible range.
    img_height:
        Total number of pixel rows in the image.

    Returns
    -------
    int
        Pixel row index in [0, img_height - 1].
    """
    pct = (price_max - price) / (price_max - price_min)
    return int(np.clip(pct * (img_height - 1), 0, img_height - 1))


def pixel_y_to_price(
    y: int,
    price_min: float,
    price_max: float,
    img_height: int,
) -> float:
    """Convert image row back to price.

    Parameters
    ----------
    y:
        Pixel row index (0 = top of image = highest price).
    price_min:
        The minimum price in the visible range.
    price_max:
        The maximum price in the visible range.
    img_height:
        Total number of pixel rows in the image.

    Returns
    -------
    float
        Reconstructed price value.
    """
    pct = y / (img_height - 1)
    return price_max - pct * (price_max - price_min)


# ---------------------------------------------------------------------------
# Confluence scoring
# ---------------------------------------------------------------------------

def compute_confluence_scores(
    zones: list[ZoneLabel],
    max_zone_width_pct: float,
) -> list[ZoneLabel]:
    """Add ``confluence_bonus`` to zones whose centres are close to each other.

    Any two zones whose centre prices are within
    ``3 * max_zone_width_pct * mean_center_price`` of each other each receive
    an additional ``0.15`` confluence bonus, capped at ``1.0``.

    Parameters
    ----------
    zones:
        List of ``ZoneLabel`` objects to score. The objects are mutated
        in-place and also returned.
    max_zone_width_pct:
        The maximum zone width as a fraction of price, taken from
        ``GeneratorConfig.zone_width_pct_max``.

    Returns
    -------
    list[ZoneLabel]
        The same list with updated ``confluence_bonus`` values.
    """
    if len(zones) < 2:
        return zones

    centers = np.array([z.center_price for z in zones], dtype=float)

    for i in range(len(zones)):
        for j in range(i + 1, len(zones)):
            mean_center = (centers[i] + centers[j]) / 2.0
            threshold = 3.0 * max_zone_width_pct * mean_center
            if abs(centers[i] - centers[j]) <= threshold:
                zones[i].confluence_bonus = min(1.0, zones[i].confluence_bonus + 0.15)
                zones[j].confluence_bonus = min(1.0, zones[j].confluence_bonus + 0.15)

    return zones


# ---------------------------------------------------------------------------
# GPU-accelerated batch heatmap generation
# ---------------------------------------------------------------------------

def compute_heatmap_targets_batch(
    zone_lists: list[list[ZoneLabel]],
    current_prices: list[float],
    price_ranges: list[tuple[float, float]],
    img_height: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate five-channel 1-D heatmap targets for a batch of examples.

    For each example the function places 1-D Gaussian bumps along the height
    axis (pixel rows) for every zone.  Channel 4 (proximity) receives a single
    Gaussian centred on the current price.

    The output is composed with ``torch.maximum`` so that overlapping Gaussians
    do not sum beyond 1.0 — only the strongest signal at each pixel row is kept.
    All values are clamped to [0.0, 1.0].

    Parameters
    ----------
    zone_lists:
        One list of ``ZoneLabel`` objects per batch example.
    current_prices:
        Current price for each example in the batch.
    price_ranges:
        ``(price_min, price_max)`` visible range per example.
    img_height:
        Number of pixel rows (height) — typically 512.
    device:
        PyTorch device on which to create the output tensor.

    Returns
    -------
    torch.Tensor
        Float32 tensor of shape ``[B, 5, H]`` with values in ``[0, 1]``.
    """
    B = len(zone_lists)
    num_channels = len(CHANNEL_NAMES)  # 5

    output = torch.zeros(B, num_channels, img_height, dtype=torch.float32, device=device)

    # Pre-build a shared row-index tensor to avoid re-allocating per zone
    rows = torch.arange(img_height, device=device, dtype=torch.float32)

    for b in range(B):
        zones = zone_lists[b]
        price_min, price_max = price_ranges[b]

        # ------------------------------------------------------------------ #
        # Place a Gaussian for every zone
        # ------------------------------------------------------------------ #
        for zone in zones:
            # Sigma: tighter for high-conviction zones
            if zone.is_high_conviction:
                sigma = max(2.0, zone.height_px * 0.40)
            else:
                sigma = max(3.0, zone.height_px * 0.70)

            center_y = float(zone.center_y)

            # Channel contributions for this zone role
            channel_weights = ROLE_CHANNEL_MAP.get(zone.role, {})

            for ch, base_amp in channel_weights.items():
                # Scale base amplitude by zone strength, then add confluence bonus
                amplitude = base_amp * zone.strength + zone.confluence_bonus
                amplitude = min(1.0, amplitude)

                gauss = amplitude * torch.exp(
                    -0.5 * ((rows - center_y) / sigma) ** 2
                )
                output[b, ch] = torch.maximum(output[b, ch], gauss)

        # ------------------------------------------------------------------ #
        # Channel 4 — proximity to current price
        # ------------------------------------------------------------------ #
        current_y = float(
            price_to_pixel_y(current_prices[b], price_min, price_max, img_height)
        )
        proximity_sigma = 15.0
        proximity_gauss = 1.0 * torch.exp(
            -0.5 * ((rows - current_y) / proximity_sigma) ** 2
        )
        output[b, 4] = torch.maximum(output[b, 4], proximity_gauss)

    # Final clamp to guarantee [0, 1]
    output.clamp_(0.0, 1.0)
    return output
