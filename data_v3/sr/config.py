from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


@dataclass
class GeneratorConfig:
    num_candles_min: int = 80
    num_candles_max: int = 220
    image_width: int = 768
    image_height: int = 512
    min_price: float = 20.0
    max_price: float = 500.0
    zone_width_pct_min: float = 0.003
    zone_width_pct_max: float = 0.020
    wick_noise: float = 0.010
    body_noise: float = 0.008
    zone_magnet_strength: float = 0.40
    false_pierce_probability: float = 0.22
    gap_probability: float = 0.06
    dark_theme_probability: float = 0.50
    grid_probability: float = 0.55
    axis_labels_probability: float = 0.20
    jpeg_artifact_probability: float = 0.15
    horizontal_flip_probability: float = 0.50


@dataclass
class ScenarioWeights:
    range_multi_touch: float = 0.14
    breakout_retest: float = 0.10
    breakdown_retest: float = 0.10
    failed_breakout: float = 0.10
    failed_breakdown: float = 0.10
    role_flip_support_to_resistance: float = 0.08
    role_flip_resistance_to_support: float = 0.08
    trend_with_shelves: float = 0.08
    messy_no_clear_level: float = 0.08
    single_sided_support: float = 0.06
    single_sided_resistance: float = 0.06
    double_top: float = 0.06
    double_bottom: float = 0.06

    def as_dict(self) -> dict:
        return {
            "range_multi_touch": self.range_multi_touch,
            "breakout_retest": self.breakout_retest,
            "breakdown_retest": self.breakdown_retest,
            "failed_breakout": self.failed_breakout,
            "failed_breakdown": self.failed_breakdown,
            "role_flip_support_to_resistance": self.role_flip_support_to_resistance,
            "role_flip_resistance_to_support": self.role_flip_resistance_to_support,
            "trend_with_shelves": self.trend_with_shelves,
            "messy_no_clear_level": self.messy_no_clear_level,
            "single_sided_support": self.single_sided_support,
            "single_sided_resistance": self.single_sided_resistance,
            "double_top": self.double_top,
            "double_bottom": self.double_bottom,
        }

    def names(self) -> list[str]:
        return list(self.as_dict().keys())

    def weights(self) -> list[float]:
        return list(self.as_dict().values())


@dataclass
class DatasetConfig:
    num_examples: int = 15000
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15


@dataclass
class ModelConfig:
    in_channels: int = 3
    out_channels: int = 5
    base_channels: int = 48
    output_height: int = 512
    dropout: float = 0.05
    width_attn_reduction: int = 1


@dataclass
class TrainingConfig:
    batch_size: int = 96
    num_epochs: int = 40
    lr_initial: float = 8e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 2.0
    cosine_T0: int = 5
    cosine_T_mult: int = 2
    mixup_alpha: float = 0.3
    loss_positive_weight_support: float = 12.0
    loss_positive_weight_resistance: float = 12.0
    loss_positive_weight_active: float = 10.0
    loss_positive_weight_historical: float = 6.0
    loss_positive_weight_proximity: float = 8.0
    loss_peak_mse_weight: float = 0.30

    def channel_pos_weights(self) -> list[float]:
        return [
            self.loss_positive_weight_support,
            self.loss_positive_weight_resistance,
            self.loss_positive_weight_active,
            self.loss_positive_weight_historical,
            self.loss_positive_weight_proximity,
        ]


@dataclass
class InferenceConfig:
    sensitivity_profiles: dict = field(
        default_factory=lambda: {
            "strict": {
                "min_score": 0.55,
                "min_touch": 5,
            },
            "balanced": {
                "min_score": 0.35,
                "min_touch": 3,
            },
            "sensitive": {
                "min_score": 0.22,
                "min_touch": 2,
            },
        }
    )
    ohlc_min_touch_count: int = 3
    ohlc_min_score: float = 0.22
    zone_staleness_recency_threshold: float = 0.25
    zone_staleness_touch_threshold: int = 3


@dataclass
class SRConfig:
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    scenarios: ScenarioWeights = field(default_factory=ScenarioWeights)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


class ZoneRole(Enum):
    confirmed_support = "confirmed_support"
    confirmed_resistance = "confirmed_resistance"
    active_zone = "active_zone"
    watch_support = "watch_support"
    watch_resistance = "watch_resistance"
    historical_zone = "historical_zone"
    weak_zone = "weak_zone"
    general_level = "general_level"


@dataclass
class ZoneLabel:
    zone_id: str
    role: ZoneRole
    low_price: float
    high_price: float
    center_price: float
    touch_count: int
    strength: float
    is_active: bool
    center_y: int
    height_px: int
    confluence_bonus: float = 0.0

    @property
    def channel_weights(self) -> dict:
        mapping = {
            ZoneRole.confirmed_support:          {0: 1.00, 3: 0.40},
            ZoneRole.confirmed_resistance:       {1: 1.00, 3: 0.40},
            ZoneRole.active_zone:                {2: 1.00, 3: 0.60},
            ZoneRole.watch_support:              {0: 1.00, 3: 0.30},
            ZoneRole.watch_resistance:           {1: 1.00, 3: 0.30},
            ZoneRole.historical_zone:            {3: 1.00},
            ZoneRole.weak_zone:                  {3: 0.60},
            ZoneRole.general_level:              {3: 0.80},
        }
        return mapping[self.role]

    @property
    def is_high_conviction(self) -> bool:
        return (
            self.touch_count >= 5
            and self.role in {ZoneRole.confirmed_support, ZoneRole.confirmed_resistance}
        )
