from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .config import Config
from .models import FVG, Candle, LiquidityLevel, StructureEvent, Swing


def compute_atr(candles: List[Candle], period: int) -> float:
    trs = []
    for i, c in enumerate(candles):
        if i == 0:
            trs.append(c.high - c.low)
        else:
            pc = candles[i - 1].close
            trs.append(max(c.high - c.low, abs(c.high - pc), abs(c.low - pc)))
    return float(np.mean(trs[-period:]))


def find_swings(candles: List[Candle], n: int) -> List[Swing]:
    swings: List[Swing] = []
    for i in range(n, len(candles) - n):
        hi, lo = candles[i].high, candles[i].low
        left_h = [c.high for c in candles[i - n : i]]
        right_h = [c.high for c in candles[i + 1 : i + n + 1]]
        left_l = [c.low for c in candles[i - n : i]]
        right_l = [c.low for c in candles[i + 1 : i + n + 1]]
        if hi >= max(left_h) and hi > max(right_h):
            swings.append(Swing(i, hi, "high"))
        if lo <= min(left_l) and lo < min(right_l):
            swings.append(Swing(i, lo, "low"))
    return swings


def detect_structure(
    candles: List[Candle], swings: List[Swing], n: int
) -> Tuple[List[StructureEvent], str]:
    """Close beyond the last confirmed swing = BOS (with trend) or CHoCH (against)."""
    events: List[StructureEvent] = []
    trend = "neutral"
    last_high = last_low = None
    pending = sorted(swings, key=lambda s: s.idx)
    p = 0
    for i, c in enumerate(candles):
        while p < len(pending) and pending[p].idx + n <= i:
            s = pending[p]
            p += 1
            if s.kind == "high":
                last_high = s
            else:
                last_low = s
        if last_high is not None and c.close > last_high.price:
            kind = "CHoCH" if trend == "bearish" else "BOS"
            events.append(StructureEvent(i, kind, "bullish", last_high.price))
            trend = "bullish"
            last_high = None
        elif last_low is not None and c.close < last_low.price:
            kind = "CHoCH" if trend == "bullish" else "BOS"
            events.append(StructureEvent(i, kind, "bearish", last_low.price))
            trend = "bearish"
            last_low = None
    return events, trend


def trend_bias(candles: List[Candle], atr_val: float, cfg: Config) -> Dict[str, str]:
    """Regression slope over the last 25% / 50% / 100% of candles, in ATR units."""
    out: Dict[str, str] = {}
    for name, frac in (("short", 0.25), ("medium", 0.5), ("long", 1.0)):
        k = min(len(candles), max(5, int(len(candles) * frac)))
        y = np.array([c.close for c in candles[-k:]])
        slope = np.polyfit(np.arange(k), y, 1)[0]
        move = float(slope * (k - 1) / atr_val) if atr_val > 0 else 0.0
        if move > cfg.trend_threshold_atr:
            out[name] = "bullish"
        elif move < -cfg.trend_threshold_atr:
            out[name] = "bearish"
        else:
            out[name] = "ranging"
    return out


def detect_fvgs(candles: List[Candle], atr_val: float, cfg: Config) -> List[FVG]:
    fvgs: List[FVG] = []
    min_size = cfg.min_fvg_atr * atr_val
    for i in range(1, len(candles) - 1):
        a, c = candles[i - 1], candles[i + 1]
        later = candles[i + 2 :]
        if c.low > a.high:  # bullish imbalance
            bottom, top = a.high, c.low
            size = top - bottom
            if size < min_size:
                continue
            filled = 0.0
            if later:
                filled = (top - min(x.low for x in later)) / size
            fvgs.append(FVG(i, "bullish", top, bottom, float(min(max(filled, 0.0), 1.0))))
        elif c.high < a.low:  # bearish imbalance
            bottom, top = c.high, a.low
            size = top - bottom
            if size < min_size:
                continue
            filled = 0.0
            if later:
                filled = (max(x.high for x in later) - bottom) / size
            fvgs.append(FVG(i, "bearish", top, bottom, float(min(max(filled, 0.0), 1.0))))
    return fvgs


def _cluster(points: List[Swing], tol: float) -> List[List[Swing]]:
    groups: List[List[Swing]] = []
    for s in sorted(points, key=lambda s: s.price):
        if groups and abs(s.price - groups[-1][-1].price) <= tol:
            groups[-1].append(s)
        else:
            groups.append([s])
    return groups


def map_liquidity(
    candles: List[Candle], swings: List[Swing], atr_val: float, cfg: Config
) -> List[LiquidityLevel]:
    tol = cfg.equal_level_atr * atr_val
    levels: List[LiquidityLevel] = []
    for kind, side in (("high", "buy-side"), ("low", "sell-side")):
        pts = [s for s in swings if s.kind == kind]
        for g in _cluster(pts, tol):
            last = max(s.idx for s in g)
            later = candles[last + 1 :]
            if kind == "high":
                price = max(s.price for s in g)
                swept = any(c.high > price for c in later)
                label = "Equal Highs" if len(g) >= 2 else "Swing High"
            else:
                price = min(s.price for s in g)
                swept = any(c.low < price for c in later)
                label = "Equal Lows" if len(g) >= 2 else "Swing Low"
            levels.append(LiquidityLevel(float(price), side, label, bool(swept), last))

    hi_i = max(range(len(candles)), key=lambda i: candles[i].high)
    lo_i = min(range(len(candles)), key=lambda i: candles[i].low)
    for price, side, label, idx in (
        (candles[hi_i].high, "buy-side", "Session High", hi_i),
        (candles[lo_i].low, "sell-side", "Session Low", lo_i),
    ):
        match = next(
            (l for l in levels if l.side == side and abs(l.price - price) < 1e-9), None
        )
        if match:
            match.label = label
        else:
            levels.append(LiquidityLevel(float(price), side, label, False, idx))
    return levels
