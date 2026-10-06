from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import Config
from .models import Candle


@dataclass
class PixelCandle:
    x: float
    y_high: float
    y_low: float
    y_body_top: float
    y_body_bottom: float
    bullish: bool


class PriceScale:
    """Linear pixel-y <-> price mapping."""

    def __init__(self, y1: float, p1: float, y2: float, p2: float):
        if y1 == y2 or p1 == p2:
            raise ValueError("Calibration points must be distinct.")
        self.y1 = float(y1)
        self.p1 = float(p1)
        self.slope = (float(p2) - float(p1)) / (float(y2) - float(y1))

    def price(self, y: float) -> float:
        return self.p1 + (y - self.y1) * self.slope

    def y(self, price: float) -> float:
        return self.y1 + (price - self.p1) / self.slope


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def _masks(bgr: np.ndarray, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s, v = cfg.min_saturation, cfg.min_value
    green = cv2.inRange(
        hsv, np.array([cfg.green_hue_min, s, v]), np.array([cfg.green_hue_max, 255, 255])
    )
    red = cv2.inRange(
        hsv, np.array([0, s, v]), np.array([cfg.red_hue_low, 255, 255])
    ) | cv2.inRange(
        hsv, np.array([cfg.red_hue_high, s, v]), np.array([180, 255, 255])
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, kernel)
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, kernel)
    return green, red


def _components(
    mask: np.ndarray, bullish: bool, cfg: Config, max_w: float
) -> List[PixelCandle]:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out: List[PixelCandle] = []
    for i in range(1, n):
        x, y, w, h, _area = (int(v) for v in stats[i])
        if h < cfg.min_candle_height or w > max_w:
            continue
        sub = labels[y : y + h, x : x + w] == i
        row_w = sub.sum(axis=1)
        mx = int(row_w.max())
        if mx <= 2:  # doji / thin candle: no distinguishable body
            body_top = body_bot = y + h / 2.0
        else:
            rows = np.where(row_w >= 0.6 * mx)[0]
            body_top = float(y + rows.min())
            body_bot = float(y + rows.max() + 1)
        out.append(
            PixelCandle(
                x=x + w / 2.0,
                y_high=float(y),
                y_low=float(y + h),
                y_body_top=body_top,
                y_body_bottom=body_bot,
                bullish=bullish,
            )
        )
    return out


def extract_pixel_candles(
    bgr: np.ndarray, cfg: Config, roi: Optional[Sequence[int]] = None
) -> List[PixelCandle]:
    ox = oy = 0
    work = bgr
    if roi:
        x, y, w, h = (int(v) for v in roi)
        work = bgr[y : y + h, x : x + w]
        ox, oy = x, y
    green, red = _masks(work, cfg)
    max_w = max(6.0, cfg.max_candle_width_frac * work.shape[1])
    found = _components(green, True, cfg, max_w) + _components(red, False, cfg, max_w)
    for pc in found:
        pc.x += ox
        pc.y_high += oy
        pc.y_low += oy
        pc.y_body_top += oy
        pc.y_body_bottom += oy
    found.sort(key=lambda p: p.x)

    deduped: List[PixelCandle] = []
    for pc in found:
        if deduped and abs(pc.x - deduped[-1].x) < 1.5:
            if (pc.y_low - pc.y_high) > (deduped[-1].y_low - deduped[-1].y_high):
                deduped[-1] = pc
            continue
        deduped.append(pc)

    if len(deduped) < cfg.min_candles:
        raise ValueError(
            f"Only {len(deduped)} candles detected (need {cfg.min_candles}). "
            "Check candle colors/hue config, use --roi, or zoom the chart."
        )
    return deduped


def build_scale(
    pcs: List[PixelCandle],
    calib: Optional[Sequence[float]] = None,
    price_range: Optional[Sequence[float]] = None,
) -> Tuple[PriceScale, bool]:
    """Returns (scale, calibrated). Uncalibrated charts use normalized 0-100 units."""
    if calib:
        y1, p1, y2, p2 = calib
        return PriceScale(y1, p1, y2, p2), True
    top = min(pc.y_high for pc in pcs)
    bottom = max(pc.y_low for pc in pcs)
    if price_range:
        hi, lo = max(price_range), min(price_range)
        return PriceScale(top, hi, bottom, lo), True
    return PriceScale(top, 100.0, bottom, 0.0), False


def to_candles(pcs: List[PixelCandle], scale: PriceScale) -> List[Candle]:
    candles: List[Candle] = []
    for i, pc in enumerate(pcs):
        high = scale.price(pc.y_high)
        low = scale.price(pc.y_low)
        a, b = scale.price(pc.y_body_top), scale.price(pc.y_body_bottom)
        body_hi, body_lo = max(a, b), min(a, b)
        if pc.bullish:
            o, c = body_lo, body_hi
        else:
            o, c = body_hi, body_lo
        candles.append(
            Candle(idx=i, open=float(o), high=float(max(high, low)), low=float(min(high, low)), close=float(c))
        )
    return candles
