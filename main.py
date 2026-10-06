#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict

from chartai.config import Config
from chartai.preprocess import (
    build_scale,
    extract_pixel_candles,
    load_image,
    to_candles,
)
from chartai.strategy import build_plan
from chartai.structure import (
    compute_atr,
    detect_fvgs,
    detect_structure,
    find_swings,
    map_liquidity,
    trend_bias,
)
from chartai.visualize import annotate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Candlestick chart image analyzer")
    p.add_argument("image", help="Path to chart image (PNG/JPG)")
    p.add_argument("--out-dir", default="output")
    p.add_argument("--price-range", nargs=2, type=float, metavar=("HIGH", "LOW"),
                   help="Price at the highest wick and the lowest wick")
    p.add_argument("--calib", nargs=4, type=float, metavar=("Y1", "P1", "Y2", "P2"),
                   help="Two (pixel_y, price) calibration points")
    p.add_argument("--roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"),
                   help="Crop region containing only candles")
    p.add_argument("--swing-n", type=int, help="Swing lookback (candles each side)")
    return p.parse_args()


def r(v, nd=4):
    return None if v is None else round(float(v), nd)


def main() -> None:
    args = parse_args()
    cfg = Config.from_env()
    if args.swing_n:
        cfg.swing_lookback = args.swing_n

    bgr = load_image(args.image)
    pcs = extract_pixel_candles(bgr, cfg, args.roi)
    scale, calibrated = build_scale(pcs, args.calib, args.price_range)
    candles = to_candles(pcs, scale)

    atr_val = compute_atr(candles, cfg.atr_period)
    swings = find_swings(candles, cfg.swing_lookback)
    events, structure_trend = detect_structure(candles, swings, cfg.swing_lookback)
    bias = trend_bias(candles, atr_val, cfg)
    fvgs = detect_fvgs(candles, atr_val, cfg)
    liquidity = map_liquidity(candles, swings, atr_val, cfg)
    plan = build_plan(candles, swings, fvgs, liquidity, events, bias, atr_val, cfg)

    os.makedirs(args.out_dir, exist_ok=True)
    report = {
        "calibrated": calibrated,
        "units": "price" if calibrated else "normalized_0_100",
        "candles_detected": len(candles),
        "last_close": r(candles[-1].close),
        "atr": r(atr_val),
        "trend_bias": bias,
        "structure_trend": structure_trend,
        "structure_events": [asdict(e) for e in events[-5:]],
        "swings": [asdict(s) for s in swings],
        "fvgs": [asdict(f) | {"mid": r(f.mid)} for f in fvgs],
        "liquidity": [asdict(l) for l in liquidity],
        "trade_plan": asdict(plan),
    }
    json_path = os.path.join(args.out_dir, "analysis.json")
    img_path = os.path.join(args.out_dir, "annotated.png")
    with open(json_path, "w") as fh:
        json.dump(report, fh, indent=2)
    annotate(bgr, pcs, scale, swings, fvgs, liquidity, plan, img_path)

    print(f"Candles: {len(candles)} | Units: {report['units']} | Last close: {r(candles[-1].close, 2)}")
    print(f"Trend bias: {bias} | Structure: {structure_trend}")
    if events:
        e = events[-1]
        print(f"Last event: {e.direction} {e.kind} @ {r(e.level, 2)} (candle {e.idx})")
    active = [f for f in fvgs if f.filled_pct < cfg.max_fvg_fill]
    print(f"FVGs: {len(fvgs)} total, {len(active)} active")
    print(f"Liquidity levels: {sum(1 for l in liquidity if not l.swept)} unswept")
    print("-" * 50)
    if plan.direction == "none":
        print("NO TRADE:", "; ".join(plan.notes))
    else:
        print(f"{plan.direction.upper()} | confluence {plan.confluence}/4")
        print(f"Entry zone: {r(plan.entry_zone[0], 2)} - {r(plan.entry_zone[1], 2)} (limit @ {r(plan.entry, 2)})")
        print(f"Stop loss:  {r(plan.stop_loss, 2)}  (risk {r(plan.risk_per_unit, 2)})")
        for i, t in enumerate(plan.targets, 1):
            print(f"TP{i}: {r(t['price'], 2)}  ({t['rr']}R, {t['source']})")
        for n in plan.notes:
            print("  ", n)
    print(f"\nSaved: {img_path}, {json_path}")


if __name__ == "__main__":
    main()
