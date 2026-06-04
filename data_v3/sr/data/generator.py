"""
GPU-accelerated batch OHLC generation with zone magnet effects.

Generates the full SR V3 dataset: OHLC CSVs, chart images, and a labels_v3.jsonl file.
"""

from __future__ import annotations

import json
import math
import multiprocessing
import os
import random
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from sr.config import SRConfig, ZoneLabel, ZoneRole, ScenarioWeights


# ---------------------------------------------------------------------------
# Public data structure
# ---------------------------------------------------------------------------

@dataclass
class GeneratedExample:
    example_id: str
    scenario: str
    num_candles: int
    ohlc: np.ndarray          # shape [T, 4] float32  (open, high, low, close)
    zones: list[ZoneLabel]
    price_range: tuple[float, float]   # (min_price, max_price) visible in chart
    current_price: float               # last close price


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def price_to_pixel_y(price: float, price_min: float, price_max: float, img_height: int) -> int:
    """Convert price to image row (y=0 at top = high price)."""
    if price_max == price_min:
        return img_height // 2
    pct = (price_max - price) / (price_max - price_min)
    return int(np.clip(pct * (img_height - 1), 0, img_height - 1))


def _make_zone_id() -> str:
    return str(uuid.uuid4())[:8]


# ---------------------------------------------------------------------------
# Scenario-specific zone generation  (CPU, per-example loop is fine here)
# ---------------------------------------------------------------------------

SCENARIO_NAMES: list[str] = [
    "range_multi_touch",
    "breakout_retest",
    "breakdown_retest",
    "failed_breakout",
    "failed_breakdown",
    "role_flip_support_to_resistance",
    "role_flip_resistance_to_support",
    "trend_with_shelves",
    "messy_no_clear_level",
    "single_sided_support",
    "single_sided_resistance",
    "double_top",
    "double_bottom",
]


def _sample_zone(
    role: ZoneRole,
    center_price: float,
    half_width: float,
    touch_count: int,
    is_active: bool,
    strength: float,
    confluence_bonus: float,
    price_min: float,
    price_max: float,
    img_height: int,
) -> ZoneLabel:
    low_price = max(price_min * 0.95, center_price - half_width)
    high_price = min(price_max * 1.05, center_price + half_width)
    center_y = price_to_pixel_y(center_price, price_min, price_max, img_height)
    low_y = price_to_pixel_y(high_price, price_min, price_max, img_height)   # high price -> lower row index
    high_y = price_to_pixel_y(low_price, price_min, price_max, img_height)
    height_px = max(1, high_y - low_y)
    return ZoneLabel(
        zone_id=_make_zone_id(),
        role=role,
        low_price=float(low_price),
        high_price=float(high_price),
        center_price=float(center_price),
        touch_count=touch_count,
        strength=float(np.clip(strength, 0.0, 1.0)),
        is_active=is_active,
        center_y=center_y,
        height_px=height_px,
        confluence_bonus=float(np.clip(confluence_bonus, 0.0, 1.0)),
    )


def _generate_zones_for_scenario(
    scenario: str,
    base_price: float,
    price_range_pct: float,
    cfg_gen,           # GeneratorConfig
    rng_np: np.random.Generator,
) -> list[ZoneLabel]:
    """Generate ZoneLabel objects for a single example given its scenario."""

    price_span = base_price * price_range_pct
    price_min = base_price - price_span / 2
    price_max = base_price + price_span / 2
    img_h = cfg_gen.image_height

    def rand_half_width() -> float:
        w_pct = rng_np.uniform(cfg_gen.zone_width_pct_min, cfg_gen.zone_width_pct_max)
        return base_price * w_pct / 2

    def rand_level(lo_frac=0.1, hi_frac=0.9) -> float:
        return float(price_min + rng_np.uniform(lo_frac, hi_frac) * (price_max - price_min))

    zones: list[ZoneLabel] = []

    if scenario == "range_multi_touch":
        n = int(rng_np.integers(2, 5))
        fracs = sorted(rng_np.uniform(0.15, 0.85, size=n))
        for frac in fracs:
            cp = price_min + frac * (price_max - price_min)
            tc = int(rng_np.integers(3, 8))
            role = ZoneRole.confirmed_support if frac < 0.5 else ZoneRole.confirmed_resistance
            zones.append(_sample_zone(role, cp, rand_half_width(), tc, True,
                                      rng_np.uniform(0.55, 0.95), 0.0,
                                      price_min, price_max, img_h))

    elif scenario == "breakout_retest":
        n = int(rng_np.integers(1, 3))
        for _ in range(n):
            cp = rand_level(0.3, 0.7)
            tc = int(rng_np.integers(2, 6))
            # was resistance, now acts as support after breakout
            role = ZoneRole.confirmed_support
            zones.append(_sample_zone(role, cp, rand_half_width(), tc, True,
                                      rng_np.uniform(0.5, 0.9), 0.0,
                                      price_min, price_max, img_h))

    elif scenario == "breakdown_retest":
        n = int(rng_np.integers(1, 3))
        for _ in range(n):
            cp = rand_level(0.3, 0.7)
            tc = int(rng_np.integers(2, 6))
            role = ZoneRole.confirmed_resistance
            zones.append(_sample_zone(role, cp, rand_half_width(), tc, True,
                                      rng_np.uniform(0.5, 0.9), 0.0,
                                      price_min, price_max, img_h))

    elif scenario == "failed_breakout":
        n = int(rng_np.integers(1, 3))
        for _ in range(n):
            cp = rand_level(0.5, 0.85)
            tc = int(rng_np.integers(2, 5))
            role = ZoneRole.confirmed_resistance
            zones.append(_sample_zone(role, cp, rand_half_width(), tc, True,
                                      rng_np.uniform(0.55, 0.9), 0.0,
                                      price_min, price_max, img_h))

    elif scenario == "failed_breakdown":
        n = int(rng_np.integers(1, 3))
        for _ in range(n):
            cp = rand_level(0.15, 0.5)
            tc = int(rng_np.integers(2, 5))
            role = ZoneRole.confirmed_support
            zones.append(_sample_zone(role, cp, rand_half_width(), tc, True,
                                      rng_np.uniform(0.55, 0.9), 0.0,
                                      price_min, price_max, img_h))

    elif scenario == "role_flip_support_to_resistance":
        cp = rand_level(0.3, 0.65)
        tc = int(rng_np.integers(3, 7))
        zones.append(_sample_zone(ZoneRole.confirmed_resistance, cp, rand_half_width(),
                                  tc, True, rng_np.uniform(0.6, 0.95), 0.05,
                                  price_min, price_max, img_h))

    elif scenario == "role_flip_resistance_to_support":
        cp = rand_level(0.35, 0.7)
        tc = int(rng_np.integers(3, 7))
        zones.append(_sample_zone(ZoneRole.confirmed_support, cp, rand_half_width(),
                                  tc, True, rng_np.uniform(0.6, 0.95), 0.05,
                                  price_min, price_max, img_h))

    elif scenario == "trend_with_shelves":
        n = int(rng_np.integers(2, 4))
        uptrend = rng_np.random() > 0.5
        fracs = sorted(rng_np.uniform(0.15, 0.85, size=n))
        for i, frac in enumerate(fracs):
            cp = price_min + frac * (price_max - price_min)
            role = ZoneRole.watch_support if uptrend else ZoneRole.watch_resistance
            tc = int(rng_np.integers(2, 5))
            zones.append(_sample_zone(role, cp, rand_half_width(), tc,
                                      i == len(fracs) - 1,
                                      rng_np.uniform(0.4, 0.8), 0.0,
                                      price_min, price_max, img_h))

    elif scenario == "messy_no_clear_level":
        if rng_np.random() < 0.5:
            cp = rand_level()
            zones.append(_sample_zone(ZoneRole.weak_zone, cp, rand_half_width(),
                                      int(rng_np.integers(1, 3)), False,
                                      rng_np.uniform(0.1, 0.45), 0.0,
                                      price_min, price_max, img_h))

    elif scenario == "single_sided_support":
        n = int(rng_np.integers(1, 3))
        for _ in range(n):
            cp = rand_level(0.1, 0.45)
            tc = int(rng_np.integers(2, 6))
            zones.append(_sample_zone(ZoneRole.confirmed_support, cp, rand_half_width(),
                                      tc, True, rng_np.uniform(0.5, 0.9), 0.0,
                                      price_min, price_max, img_h))

    elif scenario == "single_sided_resistance":
        n = int(rng_np.integers(1, 3))
        for _ in range(n):
            cp = rand_level(0.55, 0.9)
            tc = int(rng_np.integers(2, 6))
            zones.append(_sample_zone(ZoneRole.confirmed_resistance, cp, rand_half_width(),
                                      tc, True, rng_np.uniform(0.5, 0.9), 0.0,
                                      price_min, price_max, img_h))

    elif scenario == "double_top":
        # Resistance at a high level with exactly 2 touches close together
        cp_res = rand_level(0.65, 0.90)
        # zone width is slightly larger to capture double-touch
        hw = base_price * cfg_gen.zone_width_pct_max
        zones.append(_sample_zone(ZoneRole.confirmed_resistance, cp_res, hw,
                                  2, True, rng_np.uniform(0.6, 0.9), 0.0,
                                  price_min, price_max, img_h))
        # Neckline / mid-level support
        cp_sup = rand_level(0.3, 0.55)
        zones.append(_sample_zone(ZoneRole.watch_support, cp_sup, rand_half_width(),
                                  int(rng_np.integers(1, 4)), False,
                                  rng_np.uniform(0.35, 0.65), 0.0,
                                  price_min, price_max, img_h))

    elif scenario == "double_bottom":
        cp_sup = rand_level(0.10, 0.35)
        hw = base_price * cfg_gen.zone_width_pct_max
        zones.append(_sample_zone(ZoneRole.confirmed_support, cp_sup, hw,
                                  2, True, rng_np.uniform(0.6, 0.9), 0.0,
                                  price_min, price_max, img_h))
        cp_res = rand_level(0.45, 0.70)
        zones.append(_sample_zone(ZoneRole.watch_resistance, cp_res, rand_half_width(),
                                  int(rng_np.integers(1, 4)), False,
                                  rng_np.uniform(0.35, 0.65), 0.0,
                                  price_min, price_max, img_h))

    return zones


# ---------------------------------------------------------------------------
# GPU batch generation
# ---------------------------------------------------------------------------

def _generate_batch_gpu(
    batch_size: int,
    cfg: SRConfig,
    device: torch.device,
    rng: torch.Generator,
) -> list[GeneratedExample]:
    """Generate `batch_size` examples on GPU; returns list of GeneratedExample."""

    gen = cfg.generator
    scenario_names = cfg.scenarios.names()
    scenario_weights_list = cfg.scenarios.weights()

    with torch.no_grad():

        # ------------------------------------------------------------------ #
        # 1. Sample scenario indices
        # ------------------------------------------------------------------ #
        weights_t = torch.tensor(scenario_weights_list, dtype=torch.float32, device=device)
        scenario_indices = torch.multinomial(
            weights_t.unsqueeze(0).expand(batch_size, -1),
            num_samples=1,
            replacement=True,
            generator=rng,
        ).squeeze(1)   # [N]

        # ------------------------------------------------------------------ #
        # 2. Sample candle counts
        # ------------------------------------------------------------------ #
        candle_counts = torch.randint(
            gen.num_candles_min,
            gen.num_candles_max + 1,
            (batch_size,),
            generator=rng,
            device=device,
        )  # [N]
        max_T = int(candle_counts.max().item())
        lengths = candle_counts  # alias for clarity

        # ------------------------------------------------------------------ #
        # 3. Sample price frames
        # ------------------------------------------------------------------ #
        base_prices = (
            torch.rand(batch_size, generator=rng, device=device)
            * (gen.max_price - gen.min_price)
            + gen.min_price
        )  # [N]

        price_range_pcts = (
            torch.rand(batch_size, generator=rng, device=device)
            * (0.25 - 0.05)
            + 0.05
        )  # [N]  uniform(0.05, 0.25)

        price_spans = base_prices * price_range_pcts          # [N]
        price_mins = base_prices - price_spans / 2             # [N]
        price_maxs = base_prices + price_spans / 2             # [N]

        # ------------------------------------------------------------------ #
        # 4. Generate zones per example (CPU loop over scenarios)
        # ------------------------------------------------------------------ #
        scenario_indices_cpu = scenario_indices.cpu().numpy()
        base_prices_cpu = base_prices.cpu().numpy()
        price_range_pcts_cpu = price_range_pcts.cpu().numpy()
        lengths_cpu = lengths.cpu().numpy()
        price_mins_cpu = price_mins.cpu().numpy()
        price_maxs_cpu = price_maxs.cpu().numpy()

        # One numpy rng per call (reproducible from torch rng state hash)
        seed_val = int(torch.randint(0, 2**31 - 1, (1,), generator=rng, device=device).item())
        np_rng = np.random.default_rng(seed_val)

        all_zones: list[list[ZoneLabel]] = []
        for i in range(batch_size):
            sc_name = scenario_names[int(scenario_indices_cpu[i])]
            zones_i = _generate_zones_for_scenario(
                sc_name,
                float(base_prices_cpu[i]),
                float(price_range_pcts_cpu[i]),
                gen,
                np_rng,
            )
            all_zones.append(zones_i)

        # ------------------------------------------------------------------ #
        # 5. Build piecewise-linear close price series on GPU
        # ------------------------------------------------------------------ #
        # Each example gets a close series of length max_T (padded).
        # We construct ~8 anchor points distributed over [0, T-1] per example
        # that are biased to visit zone levels.

        NUM_ANCHORS = 8
        # Anchor t-positions: evenly spaced in [0, max_T-1]
        anchor_t = torch.linspace(0, max_T - 1, NUM_ANCHORS, device=device)   # [A]

        # Random noise for anchor prices: [N, A] uniform in [price_min, price_max]
        rand_anchor = torch.rand(batch_size, NUM_ANCHORS, generator=rng, device=device)  # [N, A]
        anchor_prices = (
            price_mins.unsqueeze(1) + rand_anchor * price_spans.unsqueeze(1)
        )  # [N, A]

        # Build time index [N, max_T]
        t_idx = torch.arange(max_T, device=device).unsqueeze(0).expand(batch_size, -1).float()  # [N, T]

        # Piecewise linear interpolation for each example
        # For each time step, find which segment it belongs to and interpolate
        close_series = torch.zeros(batch_size, max_T, device=device, dtype=torch.float32)
        for a in range(NUM_ANCHORS - 1):
            t0 = anchor_t[a]
            t1 = anchor_t[a + 1]
            mask = (t_idx >= t0) & (t_idx <= t1)           # [N, T] bool
            alpha = torch.clamp((t_idx - t0) / (t1 - t0 + 1e-8), 0.0, 1.0)  # [N, T]
            seg_price = (
                anchor_prices[:, a].unsqueeze(1) * (1 - alpha)
                + anchor_prices[:, a + 1].unsqueeze(1) * alpha
            )  # [N, T]
            close_series = torch.where(mask, seg_price, close_series)

        # Add small random walk noise
        noise_scale = price_spans * gen.wick_noise * 0.5
        rw_noise = torch.randn(batch_size, max_T, generator=rng, device=device)
        close_series = close_series + rw_noise * noise_scale.unsqueeze(1)
        close_series = torch.clamp(close_series, price_mins.unsqueeze(1), price_maxs.unsqueeze(1))

        # ------------------------------------------------------------------ #
        # 6. Zone magnet effect on GPU
        # ------------------------------------------------------------------ #
        # We approximate ATR per example as mean(high-low) later; here we
        # use price_span / num_candles * 5 as a proxy before OHLC exists.
        atr_proxy = price_spans / lengths.float() * 5.0   # [N]
        magnet_radius = atr_proxy * gen.zone_magnet_strength  # [N]

        for i in range(batch_size):
            if not all_zones[i]:
                continue
            T_i = int(lengths_cpu[i])
            rad = float(magnet_radius[i].item())
            for z in all_zones[i]:
                ctr = z.center_price
                is_sup = z.role in {
                    ZoneRole.confirmed_support,
                    ZoneRole.watch_support,
                    ZoneRole.active_zone,
                }
                # Find candle indices close to zone
                prices_i = close_series[i, :T_i]
                dist = torch.abs(prices_i - ctr)
                near_mask = dist < rad

                snap_frac = float(np_rng.uniform(0.60, 0.80))
                # Keep only snap_frac proportion of nearby candles
                near_indices = near_mask.nonzero(as_tuple=True)[0]
                if near_indices.numel() == 0:
                    continue
                num_snap = max(1, int(snap_frac * near_indices.numel()))
                perm = torch.randperm(near_indices.numel(), generator=rng, device=device)
                snap_indices = near_indices[perm[:num_snap]]

                snap_noise = torch.randn(num_snap, generator=rng, device=device) * rad * 0.05
                if is_sup:
                    # Pull close toward zone center (support: don't let it drop too far)
                    snapped = torch.full((num_snap,), z.low_price, device=device) + torch.abs(snap_noise)
                    snapped = torch.clamp(snapped, z.low_price, z.high_price)
                else:
                    # Pull close toward zone center (resistance)
                    snapped = torch.full((num_snap,), z.high_price, device=device) - torch.abs(snap_noise)
                    snapped = torch.clamp(snapped, z.low_price, z.high_price)

                close_series[i, snap_indices] = snapped

        # Clamp again after magnet
        close_series = torch.clamp(close_series, price_mins.unsqueeze(1), price_maxs.unsqueeze(1))

        # ------------------------------------------------------------------ #
        # 7. Build OHLC from close series on GPU
        # ------------------------------------------------------------------ #
        # open[t] = close[t-1] (with gap probability)
        open_series = torch.zeros_like(close_series)
        open_series[:, 0] = close_series[:, 0]
        open_series[:, 1:] = close_series[:, :-1]

        # Gap: add a jump to open for a fraction of candles
        gap_mask = torch.rand(batch_size, max_T, generator=rng, device=device) < gen.gap_probability
        gap_dir = torch.sign(torch.randn(batch_size, max_T, generator=rng, device=device))
        gap_size = torch.rand(batch_size, max_T, generator=rng, device=device) * price_spans.unsqueeze(1) * 0.015
        open_series = open_series + gap_mask.float() * gap_dir * gap_size

        # Body noise
        body_noise_open = torch.randn(batch_size, max_T, generator=rng, device=device) * \
                          (base_prices.unsqueeze(1) * gen.body_noise)
        body_noise_close = torch.randn(batch_size, max_T, generator=rng, device=device) * \
                           (base_prices.unsqueeze(1) * gen.body_noise)
        open_series = open_series + body_noise_open
        close_series_noisy = close_series + body_noise_close

        # Wick noise (always positive outward)
        wick_scale = base_prices.unsqueeze(1) * gen.wick_noise
        wick_high = torch.abs(torch.randn(batch_size, max_T, generator=rng, device=device)) * wick_scale
        wick_low = torch.abs(torch.randn(batch_size, max_T, generator=rng, device=device)) * wick_scale

        high_series = torch.maximum(open_series, close_series_noisy) + wick_high
        low_series = torch.minimum(open_series, close_series_noisy) - wick_low

        # Clamp all to price bounds (with small buffer)
        p_min_exp = price_mins.unsqueeze(1) * 0.97
        p_max_exp = price_maxs.unsqueeze(1) * 1.03
        high_series = torch.clamp(high_series, p_min_exp, p_max_exp)
        low_series = torch.clamp(low_series, p_min_exp, p_max_exp)
        open_series = torch.clamp(open_series, p_min_exp, p_max_exp)
        close_series_noisy = torch.clamp(close_series_noisy, p_min_exp, p_max_exp)

        # Ensure OHLC consistency
        high_series = torch.maximum(high_series, torch.maximum(open_series, close_series_noisy))
        low_series = torch.minimum(low_series, torch.minimum(open_series, close_series_noisy))

        # False pierce: a small fraction of candles pierce a nearby zone boundary slightly
        # We apply this as a post-process nudge to high or low
        false_pierce_mask = (
            torch.rand(batch_size, max_T, generator=rng, device=device)
            < gen.false_pierce_probability
        )
        # Random pierce direction (up or down)
        pierce_dir = (torch.rand(batch_size, max_T, generator=rng, device=device) > 0.5).float() * 2 - 1
        pierce_amount = (
            torch.rand(batch_size, max_T, generator=rng, device=device)
            * price_spans.unsqueeze(1)
            * 0.008
        )
        false_pierce_up = false_pierce_mask & (pierce_dir > 0)
        false_pierce_dn = false_pierce_mask & (pierce_dir < 0)
        high_series = high_series + false_pierce_up.float() * pierce_amount
        low_series = low_series - false_pierce_dn.float() * pierce_amount

        # Final clamp
        high_series = torch.clamp(high_series, p_min_exp, p_max_exp)
        low_series = torch.clamp(low_series, p_min_exp, p_max_exp)

        # ------------------------------------------------------------------ #
        # 8. Move to CPU numpy and build GeneratedExample list
        # ------------------------------------------------------------------ #
        open_np = open_series.cpu().numpy().astype(np.float32)
        high_np = high_series.cpu().numpy().astype(np.float32)
        low_np = low_series.cpu().numpy().astype(np.float32)
        close_np = close_series_noisy.cpu().numpy().astype(np.float32)

    examples: list[GeneratedExample] = []
    for i in range(batch_size):
        T_i = int(lengths_cpu[i])
        ohlc_i = np.stack(
            [open_np[i, :T_i], high_np[i, :T_i], low_np[i, :T_i], close_np[i, :T_i]],
            axis=1,
        )  # [T_i, 4]

        sc_name = scenario_names[int(scenario_indices_cpu[i])]
        p_min = float(price_mins_cpu[i])
        p_max = float(price_maxs_cpu[i])
        current_price = float(close_np[i, T_i - 1])

        ex = GeneratedExample(
            example_id=str(uuid.uuid4()).replace("-", "")[:16],
            scenario=sc_name,
            num_candles=T_i,
            ohlc=ohlc_i,
            zones=all_zones[i],
            price_range=(p_min, p_max),
            current_price=current_price,
        )
        examples.append(ex)

    return examples


# ---------------------------------------------------------------------------
# Rendering worker (runs in subprocess)
# ---------------------------------------------------------------------------

def _render_worker(args: tuple) -> None:
    """Subprocess worker: render chart image and optionally apply JPEG artifact."""
    ex, images_dir, gen_cfg = args
    img_path = images_dir / f"{ex.example_id}.png"

    dark_theme = random.random() < gen_cfg.dark_theme_probability
    draw_grid = random.random() < gen_cfg.grid_probability
    draw_axes = random.random() < gen_cfg.axis_labels_probability

    try:
        from sr.data.renderer import render_chart
        render_chart(
            ex.ohlc,
            gen_cfg,
            str(img_path),
            dark_theme=dark_theme,
            draw_grid=draw_grid,
            draw_axes=draw_axes,
        )
    except ImportError:
        # Renderer not yet available; write a placeholder so pipeline continues
        pass

    # Optional JPEG degradation
    if random.random() < gen_cfg.jpeg_artifact_probability and img_path.exists():
        try:
            from PIL import Image
            import io
            img = Image.open(img_path)
            quality = random.randint(72, 88)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
            img.save(img_path)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# JSONL serialisation
# ---------------------------------------------------------------------------

def _example_to_record(ex: GeneratedExample) -> dict:
    return {
        "id": ex.example_id,
        "scenario": ex.scenario,
        "num_candles": ex.num_candles,
        "price_range": list(ex.price_range),
        "current_price": ex.current_price,
        "zones": [
            {
                "zone_id": z.zone_id,
                "role": z.role.value,
                "low_price": z.low_price,
                "high_price": z.high_price,
                "center_price": z.center_price,
                "touch_count": z.touch_count,
                "strength": z.strength,
                "is_active": z.is_active,
                "center_y": z.center_y,
                "height_px": z.height_px,
                "confluence_bonus": z.confluence_bonus,
            }
            for z in ex.zones
        ],
    }


# ---------------------------------------------------------------------------
# Dataset orchestration
# ---------------------------------------------------------------------------

def generate_dataset(cfg: SRConfig, output_dir: Path, device: torch.device) -> None:
    """Generate the full dataset. Saves OHLCs, images, and labels_v3.jsonl."""

    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    ohlc_dir = output_dir / "ohlc"
    images_dir.mkdir(parents=True, exist_ok=True)
    ohlc_dir.mkdir(parents=True, exist_ok=True)

    all_examples: list[GeneratedExample] = []
    generation_batch_size = 256
    num_batches = math.ceil(cfg.dataset.num_examples / generation_batch_size)

    rng = torch.Generator(device=device)
    rng.seed()   # seed from system entropy

    print(f"Generating {cfg.dataset.num_examples} examples on {device} "
          f"in {num_batches} batches of up to {generation_batch_size} ...")

    for batch_idx in tqdm(range(num_batches), desc="Generating OHLC batches"):
        actual_size = min(
            generation_batch_size,
            cfg.dataset.num_examples - batch_idx * generation_batch_size,
        )
        batch = _generate_batch_gpu(actual_size, cfg, device, rng)
        all_examples.extend(batch)

    # ------------------------------------------------------------------ #
    # Parallel rendering
    # ------------------------------------------------------------------ #
    render_args = [(ex, images_dir, cfg.generator) for ex in all_examples]
    num_workers = min(8, os.cpu_count() or 4)
    print(f"\nRendering {len(all_examples)} charts with {num_workers} workers ...")

    with multiprocessing.Pool(num_workers) as pool:
        list(
            tqdm(
                pool.imap(_render_worker, render_args, chunksize=16),
                total=len(all_examples),
                desc="Rendering images",
            )
        )

    # ------------------------------------------------------------------ #
    # Save OHLC CSVs
    # ------------------------------------------------------------------ #
    for ex in tqdm(all_examples, desc="Saving OHLC"):
        df = pd.DataFrame(ex.ohlc, columns=["open", "high", "low", "close"])
        df.to_csv(ohlc_dir / f"{ex.example_id}.csv", index=False)

    # ------------------------------------------------------------------ #
    # Save labels JSONL
    # ------------------------------------------------------------------ #
    labels_path = output_dir / "labels_v3.jsonl"
    with open(labels_path, "w") as f:
        for ex in all_examples:
            record = _example_to_record(ex)
            f.write(json.dumps(record) + "\n")

    print(f"\nLabels saved to {labels_path}")

    # ------------------------------------------------------------------ #
    # Scenario distribution report
    # ------------------------------------------------------------------ #
    from collections import Counter
    scenario_counts = Counter(ex.scenario for ex in all_examples)
    print("\nScenario distribution:")
    for scenario, count in sorted(scenario_counts.items()):
        print(f"  {scenario}: {count} ({count / len(all_examples) * 100:.1f}%)")

    print(f"\nDataset complete: {len(all_examples)} examples in {output_dir}")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate SR V3 dataset")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/generated"),
        help="Directory to write images, OHLCs, and labels_v3.jsonl",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=None,
        help="Override cfg.dataset.num_examples",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="torch device (cuda / cpu)",
    )
    args = parser.parse_args()

    _cfg = SRConfig()
    if args.num_examples is not None:
        _cfg.dataset.num_examples = args.num_examples

    _device = torch.device(args.device)
    generate_dataset(_cfg, args.output_dir, _device)
