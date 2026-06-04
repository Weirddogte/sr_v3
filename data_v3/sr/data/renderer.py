from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sr.config import ZoneLabel, ZoneRole


# ---------------------------------------------------------------------------
# Theme palettes
# ---------------------------------------------------------------------------

_DARK = {
    "bg":         "#1a1a2e",
    "up":         "#26a69a",
    "down":       "#ef5350",
    "wick_up":    "#4db6ac",
    "wick_down":  "#ef9a9a",
    "grid":       "gray",
}

_LIGHT = {
    "bg":         "#ffffff",
    "up":         "#26a69a",
    "down":       "#ef5350",
    "wick_up":    "#26a69a",
    "wick_down":  "#ef5350",
    "grid":       "gray",
}

# Zone overlay colours keyed by ZoneRole
_ZONE_COLORS: dict[ZoneRole, tuple[str, float]] = {
    ZoneRole.confirmed_support:    ("#26a69a", 0.25),
    ZoneRole.watch_support:        ("#26a69a", 0.25),
    ZoneRole.confirmed_resistance: ("#ef5350", 0.25),
    ZoneRole.watch_resistance:     ("#ef5350", 0.25),
    ZoneRole.active_zone:          ("#ffd700", 0.30),
    ZoneRole.historical_zone:      ("#888888", 0.15),
    ZoneRole.weak_zone:            ("#888888", 0.15),
    ZoneRole.general_level:        ("#888888", 0.15),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _price_range(ohlc: np.ndarray, padding: float = 0.05) -> tuple[float, float]:
    """Return (y_min, y_max) with percentage padding applied."""
    lo = float(ohlc[:, 2].min())   # column 2 = low
    hi = float(ohlc[:, 1].max())   # column 1 = high
    span = hi - lo
    pad = span * padding
    return lo - pad, hi + pad


def _build_figure(
    gen_cfg,
    dark_theme: bool,
) -> tuple[plt.Figure, plt.Axes, dict]:
    """Create a figure/axes pair at the configured pixel size."""
    dpi = 100
    width_in  = gen_cfg.image_width  / dpi
    height_in = gen_cfg.image_height / dpi

    theme = _DARK if dark_theme else _LIGHT

    fig = plt.figure(figsize=(width_in, height_in), dpi=dpi)
    fig.patch.set_facecolor(theme["bg"])
    ax = fig.add_subplot(111)
    ax.set_facecolor(theme["bg"])
    fig.set_size_inches(width_in, height_in)

    return fig, ax, theme


def _strip_axes(ax: plt.Axes) -> None:
    """Remove all axis decoration."""
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_candles(
    ax: plt.Axes,
    ohlc: np.ndarray,
    theme: dict,
) -> None:
    """Draw bodies and wicks for every candle."""
    T = len(ohlc)
    x = np.arange(T, dtype=float)

    # Candle width: leave a tiny gap between neighbours (5 % of slot)
    width = 0.90

    for i in range(T):
        o, h, l, c = float(ohlc[i, 0]), float(ohlc[i, 1]), float(ohlc[i, 2]), float(ohlc[i, 3])
        up = c >= o
        body_color = theme["up"] if up else theme["down"]
        wick_color = theme["wick_up"] if up else theme["wick_down"]

        # Wick
        ax.plot([x[i], x[i]], [l, h], color=wick_color, linewidth=0.8, zorder=1)

        # Body
        body_bottom = min(o, c)
        body_height = abs(c - o) or 1e-6  # avoid zero-height body
        rect = mpatches.Rectangle(
            (x[i] - width / 2, body_bottom),
            width,
            body_height,
            linewidth=0,
            facecolor=body_color,
            zorder=2,
        )
        ax.add_patch(rect)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_chart(
    ohlc: np.ndarray,
    gen_cfg,
    output_path: str,
    dark_theme: bool = False,
    draw_grid: bool = True,
    draw_axes: bool = False,
) -> None:
    """Render a plain candlestick chart and save it as a PNG.

    Parameters
    ----------
    ohlc:
        Array of shape [T, 4] with columns [open, high, low, close].
    gen_cfg:
        ``GeneratorConfig`` supplying ``image_width`` / ``image_height``.
    output_path:
        Destination file path (PNG).
    dark_theme:
        Use the dark colour palette when True.
    draw_grid:
        Draw faint horizontal grid lines when True.
    draw_axes:
        Show right-side price tick labels when True.
    """
    fig, ax, theme = _build_figure(gen_cfg, dark_theme)
    T = len(ohlc)
    y_min, y_max = _price_range(ohlc)

    _draw_candles(ax, ohlc, theme)

    ax.set_xlim(-0.5, T - 0.5)
    ax.set_ylim(y_min, y_max)

    if draw_grid:
        ax.yaxis.grid(True, alpha=0.15, color=theme["grid"], zorder=0)
        ax.set_axisbelow(True)
    else:
        ax.yaxis.grid(False)

    if draw_axes:
        # Show price ticks on the right side only
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:.2f}"))
        ax.tick_params(right=True, labelright=True, left=False, labelleft=False,
                       bottom=False, labelbottom=False, colors=theme["grid"])
        for spine in ("top", "left", "bottom"):
            ax.spines[spine].set_visible(False)
        ax.spines["right"].set_visible(True)
        ax.spines["right"].set_color(theme["grid"])
        ax.spines["right"].set_alpha(0.4)
    else:
        _strip_axes(ax)

    fig.tight_layout(pad=0)
    fig.savefig(output_path, dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def render_chart_with_zones(
    ohlc: np.ndarray,
    zones: list,
    gen_cfg,
    output_path: str,
    dark_theme: bool = False,
) -> None:
    """Render a candlestick chart with zone overlays and save it as a PNG.

    Parameters
    ----------
    ohlc:
        Array of shape [T, 4] with columns [open, high, low, close].
    zones:
        List of ``ZoneLabel`` instances to overlay.
    gen_cfg:
        ``GeneratorConfig`` supplying ``image_width`` / ``image_height``.
    output_path:
        Destination file path (PNG).
    dark_theme:
        Use the dark colour palette when True.
    """
    fig, ax, theme = _build_figure(gen_cfg, dark_theme)
    T = len(ohlc)
    y_min, y_max = _price_range(ohlc)

    _draw_candles(ax, ohlc, theme)

    ax.set_xlim(-0.5, T - 0.5)
    ax.set_ylim(y_min, y_max)
    _strip_axes(ax)

    # Draw zone rectangles — span the full x width of the chart
    x_left  = -0.5
    x_width = T  # from -0.5 to T-0.5

    for zone in zones:
        color, alpha = _ZONE_COLORS.get(zone.role, ("#888888", 0.15))
        rect_height = zone.high_price - zone.low_price
        if rect_height <= 0:
            continue
        rect = mpatches.Rectangle(
            (x_left, zone.low_price),
            x_width,
            rect_height,
            linewidth=0,
            facecolor=color,
            alpha=alpha,
            zorder=3,
        )
        ax.add_patch(rect)

    fig.tight_layout(pad=0)
    fig.savefig(output_path, dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
