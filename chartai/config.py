from __future__ import annotations

import os
from dataclasses import dataclass, fields


@dataclass
class Config:
    # --- Candle extraction (OpenCV HSV: hue 0-180) ---
    green_hue_min: int = 35
    green_hue_max: int = 95
    red_hue_low: int = 10
    red_hue_high: int = 170
    min_saturation: int = 70
    min_value: int = 70
    min_candle_height: int = 3
    max_candle_width_frac: float = 0.03
    min_candles: int = 20

    # --- Structure ---
    atr_period: int = 14
    swing_lookback: int = 3
    trend_threshold_atr: float = 1.0

    # --- FVG / liquidity ---
    min_fvg_atr: float = 0.25
    max_fvg_fill: float = 0.75
    equal_level_atr: float = 0.15

    # --- Strategy ---
    min_bias_score: int = 2
    sl_buffer_atr: float = 0.25
    min_risk_atr: float = 0.5
    max_risk_atr: float = 6.0
    min_rr: float = 1.5
    max_targets: int = 3
    rr_multiples: tuple = (2.0, 3.0)

    @classmethod
    def from_env(cls) -> "Config":
        cfg = cls()
        for f in fields(cfg):
            default = getattr(cfg, f.name)
            if isinstance(default, bool) or not isinstance(default, (int, float)):
                continue
            raw = os.getenv(f"CHARTAI_{f.name.upper()}")
            if raw is not None:
                setattr(cfg, f.name, type(default)(raw))
        return cfg
