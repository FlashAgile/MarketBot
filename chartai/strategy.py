from __future__ import annotations

from typing import Dict, List

from .config import Config
from .models import FVG, Candle, LiquidityLevel, StructureEvent, Swing, TradePlan

_SIGN = {"bullish": 1, "bearish": -1, "ranging": 0}


def bias_score(bias: Dict[str, str], events: List[StructureEvent]) -> int:
    score = sum(_SIGN[v] for v in bias.values())
    if events:
        score += 2 * _SIGN[events[-1].direction]
    return score


def build_plan(
    candles: List[Candle],
    swings: List[Swing],
    fvgs: List[FVG],
    liquidity: List[LiquidityLevel],
    events: List[StructureEvent],
    bias: Dict[str, str],
    atr_val: float,
    cfg: Config,
) -> TradePlan:
    price = candles[-1].close
    score = bias_score(bias, events)

    if score >= cfg.min_bias_score:
        s, direction, kind = 1, "long", "bullish"
    elif score <= -cfg.min_bias_score:
        s, direction, kind = -1, "short", "bearish"
    else:
        return TradePlan(
            bias_score=score,
            notes=[f"Mixed signals (bias score {score}); stand aside."],
        )

    zones = [
        f
        for f in fvgs
        if f.kind == kind
        and f.filled_pct < cfg.max_fvg_fill
        and ((f.mid < price) if s == 1 else (f.mid > price))
    ]
    if not zones:
        return TradePlan(
            direction="none",
            bias_score=score,
            notes=[f"{direction.title()} bias but no active {kind} FVG to trade from; wait for a pullback."],
        )
    zone = max(zones, key=lambda f: f.mid) if s == 1 else min(zones, key=lambda f: f.mid)

    notes: List[str] = []
    entry = zone.mid
    buf = cfg.sl_buffer_atr * atr_val

    # Stop: beyond nearest swing that sits outside the zone (invalidation level)
    if s == 1:
        cands = [w.price for w in swings if w.kind == "low" and w.price < zone.bottom]
        sl = (max(cands) if cands else zone.bottom - atr_val) - buf
    else:
        cands = [w.price for w in swings if w.kind == "high" and w.price > zone.top]
        sl = (min(cands) if cands else zone.top + atr_val) + buf
    risk = abs(entry - sl)

    if risk > cfg.max_risk_atr * atr_val:
        sl = (zone.bottom - buf) if s == 1 else (zone.top + buf)
        risk = abs(entry - sl)
        notes.append("Swing stop too wide; stop placed just beyond the FVG instead.")
    if risk < cfg.min_risk_atr * atr_val:
        risk = cfg.min_risk_atr * atr_val
        sl = entry - s * risk
        notes.append("Stop widened to the minimum ATR-based risk.")

    # Targets: unswept liquidity beyond entry, nearest first, filtered by min R:R
    pools = sorted(
        (
            l
            for l in liquidity
            if not l.swept and ((l.price > entry) if s == 1 else (l.price < entry))
        ),
        key=lambda l: abs(l.price - entry),
    )
    targets: List[dict] = []
    for l in pools:
        rr = abs(l.price - entry) / risk
        if rr >= cfg.min_rr:
            targets.append({"price": float(l.price), "rr": round(rr, 2), "source": l.label})
        if len(targets) >= cfg.max_targets:
            break
    liquidity_targets = bool(targets)
    if not targets:
        for m in cfg.rr_multiples:
            targets.append({"price": float(entry + s * m * risk), "rr": float(m), "source": f"{m}R multiple"})
        notes.append("No unswept liquidity with sufficient R:R; using R-multiple targets.")

    factors = {
        "trend aligned (>=2 horizons)": sum(1 for v in bias.values() if _SIGN[v] == s) >= 2,
        "last structure event aligned": bool(events) and _SIGN[events[-1].direction] == s,
        "FVG largely unmitigated (<25% filled)": zone.filled_pct < 0.25,
        "target at real liquidity pool": liquidity_targets,
    }
    confluence = sum(factors.values())
    notes.extend(f"{'+' if ok else '-'} {name}" for name, ok in factors.items())

    return TradePlan(
        direction=direction,
        bias_score=score,
        entry_zone=(float(zone.bottom), float(zone.top)),
        entry=float(entry),
        stop_loss=float(sl),
        risk_per_unit=float(risk),
        targets=targets,
        confluence=confluence,
        notes=notes,
    )
