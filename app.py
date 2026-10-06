from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict

import cv2
import numpy as np
import streamlit as st

from chartai.config import Config
from chartai.preprocess import build_scale, extract_pixel_candles, to_candles
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


def demo_chart(seed: int = 1, drift: float = 2.0) -> np.ndarray:
    """Draw a synthetic green/red candlestick chart (BGR image) for demo purposes."""
    rng = np.random.default_rng(seed)
    n, w, h = 90, 1100, 620
    price = 5100.0
    data = []
    for _ in range(n):
        o = price
        move = rng.normal(drift, 6)
        if rng.random() < 0.12:
            move += abs(rng.uniform(15, 30)) * (1 if drift >= 0 else -1)
        c = o + move
        hi = max(o, c) + rng.uniform(1, 7)
        lo = min(o, c) - rng.uniform(1, 7)
        data.append((o, hi, lo, c))
        price = c
    pmax = max(d[1] for d in data) + 10
    pmin = min(d[2] for d in data) - 10
    top, bot = 40, h - 40

    def y(p: float) -> int:
        return int(top + (pmax - p) / (pmax - pmin) * (bot - top))

    img = np.full((h, w, 3), (30, 25, 22), np.uint8)
    for g in range(6):
        yy = int(top + g * (bot - top) / 5)
        cv2.line(img, (0, yy), (w - 1, yy), (70, 70, 70), 1)
    for i, (o, hi, lo, c) in enumerate(data):
        cx = 30 + i * 11
        col = (80, 190, 40) if c >= o else (60, 60, 230)
        cv2.line(img, (cx, y(hi)), (cx, y(lo)), col, 1)
        yt, yb = y(max(o, c)), y(min(o, c))
        cv2.rectangle(img, (cx - 3, yt), (cx + 3, max(yb, yt + 1)), col, -1)
    return img


def rnd(v, nd: int = 2):
    return None if v is None else round(float(v), nd)


def analyze(bgr, cfg: Config, roi=None, price_range=None) -> dict:
    """Run the full pipeline on a BGR image and return results plus annotated PNG bytes."""
    pcs = extract_pixel_candles(bgr, cfg, roi)
    scale, calibrated = build_scale(pcs, None, price_range)
    candles = to_candles(pcs, scale)

    atr_val = compute_atr(candles, cfg.atr_period)
    swings = find_swings(candles, cfg.swing_lookback)
    events, structure_trend = detect_structure(candles, swings, cfg.swing_lookback)
    bias = trend_bias(candles, atr_val, cfg)
    fvgs = detect_fvgs(candles, atr_val, cfg)
    liquidity = map_liquidity(candles, swings, atr_val, cfg)
    plan = build_plan(candles, swings, fvgs, liquidity, events, bias, atr_val, cfg)

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "annotated.png")
        annotate(bgr, pcs, scale, swings, fvgs, liquidity, plan, path)
        with open(path, "rb") as fh:
            png = fh.read()

    report = {
        "calibrated": calibrated,
        "units": "price" if calibrated else "normalized_0_100",
        "candles_detected": len(candles),
        "last_close": rnd(candles[-1].close, 4),
        "atr": rnd(atr_val, 4),
        "trend_bias": bias,
        "structure_trend": structure_trend,
        "structure_events": [asdict(e) for e in events[-5:]],
        "swings": [asdict(s) for s in swings],
        "fvgs": [asdict(f) | {"mid": rnd(f.mid, 4)} for f in fvgs],
        "liquidity": [asdict(item) for item in liquidity],
        "trade_plan": asdict(plan),
    }
    return {
        "png": png,
        "report": report,
        "plan": plan,
        "events": events,
        "fvgs": fvgs,
        "liquidity": liquidity,
        "bias": bias,
        "structure_trend": structure_trend,
        "calibrated": calibrated,
        "n": len(candles),
        "last_close": candles[-1].close,
        "cfg": cfg,
    }


def show_results(res: dict) -> None:
    plan, report, cfg = res["plan"], res["report"], res["cfg"]
    units = "price" if res["calibrated"] else "normalized 0-100 units"

    st.image(res["png"], caption="Annotated chart")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Candles detected", res["n"])
    c2.metric("Last close", rnd(res["last_close"]))
    c3.metric("Structure", res["structure_trend"].title())
    c4.metric("Units", "Price" if res["calibrated"] else "0-100")

    st.subheader("Trend bias")
    b1, b2, b3 = st.columns(3)
    for col, name in zip((b1, b2, b3), ("short", "medium", "long")):
        col.metric(name.title() + "-term", res["bias"][name].title())

    if res["events"]:
        e = res["events"][-1]
        st.caption(f"Last structure event: {e.direction} {e.kind} at {rnd(e.level)} (candle {e.idx})")

    st.subheader("Trade plan")
    if plan.direction == "none":
        st.warning("No trade: " + " ".join(plan.notes))
    else:
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Direction", plan.direction.upper())
        d2.metric("Entry zone", f"{rnd(plan.entry_zone[0])} - {rnd(plan.entry_zone[1])}")
        d3.metric("Stop loss", rnd(plan.stop_loss))
        d4.metric("Confluence", f"{plan.confluence}/4")
        st.caption(f"Limit entry at {rnd(plan.entry)} | risk per unit {rnd(plan.risk_per_unit)} | {units}")
        st.table(
            [
                {"Target": f"TP{i}", "Price": rnd(t["price"]), "R:R": t["rr"], "Source": t["source"]}
                for i, t in enumerate(plan.targets, 1)
            ]
        )
        for note in plan.notes:
            st.write("- " + note)

    active = [f for f in res["fvgs"] if f.filled_pct < cfg.max_fvg_fill]
    with st.expander(f"Fair Value Gaps ({len(res['fvgs'])} total, {len(active)} active)"):
        if res["fvgs"]:
            st.dataframe(
                [
                    {
                        "Candle": f.idx,
                        "Type": f.kind,
                        "Bottom": rnd(f.bottom),
                        "Top": rnd(f.top),
                        "Filled %": round(f.filled_pct * 100),
                    }
                    for f in res["fvgs"]
                ]
            )
        else:
            st.write("None found.")

    with st.expander("Liquidity levels"):
        st.dataframe(
            [
                {
                    "Price": rnd(item.price),
                    "Side": item.side,
                    "Label": item.label,
                    "Swept": item.swept,
                }
                for item in sorted(res["liquidity"], key=lambda x: -x.price)
            ]
        )

    d1, d2 = st.columns(2)
    d1.download_button("Download annotated image", res["png"], "annotated.png", "image/png")
    d2.download_button(
        "Download JSON report", json.dumps(report, indent=2), "analysis.json", "application/json"
    )


def main() -> None:
    st.set_page_config(page_title="MarketBot Chart Analyzer", layout="wide")
    st.title("MarketBot Chart Analyzer")
    st.caption(
        "Upload a candlestick chart screenshot (green = bullish, red = bearish). "
        "Educational tool, not financial advice. Prices read from an image are approximate."
    )

    uploaded = st.file_uploader("Chart image", type=["png", "jpg", "jpeg"])
    if st.button("Try a demo chart"):
        ok, buf = cv2.imencode(".png", demo_chart())
        st.session_state["demo_png"] = buf.tobytes()

    data = uploaded.getvalue() if uploaded else st.session_state.get("demo_png")
    if data is None:
        st.info("Upload a chart or tap 'Try a demo chart' to see it work.")
        return

    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        st.error("Could not read that file as an image. Try a PNG or JPG.")
        return
    height, width = bgr.shape[:2]

    with st.sidebar:
        st.header("Options")
        use_range = st.checkbox("I know the price range")
        price_range = None
        if use_range:
            hi = st.number_input("Price at the highest wick", value=100.0)
            lo = st.number_input("Price at the lowest wick", value=0.0)
            if hi != lo:
                price_range = (hi, lo)
        use_roi = st.checkbox("Crop to candle area")
        roi = None
        if use_roi:
            st.caption(f"Image size: {width} x {height}")
            rx = st.number_input("X", 0, width - 1, 0)
            ry = st.number_input("Y", 0, height - 1, 0)
            rw = st.number_input("Width", 1, width, width)
            rh = st.number_input("Height", 1, height, height)
            roi = (int(rx), int(ry), int(min(rw, width - rx)), int(min(rh, height - ry)))
        with st.expander("Advanced"):
            cfg = Config.from_env()
            cfg.min_candles = st.slider("Minimum candles", 5, 100, cfg.min_candles)
            cfg.swing_lookback = st.slider("Swing lookback", 1, 8, cfg.swing_lookback)
            cfg.min_saturation = st.slider("Color saturation cutoff", 20, 200, cfg.min_saturation)
            cfg.min_rr = st.slider("Minimum reward:risk", 1.0, 5.0, float(cfg.min_rr), 0.1)

    try:
        res = analyze(bgr, cfg, roi, price_range)
    except (ValueError, FileNotFoundError) as exc:
        st.error(str(exc))
        return
    show_results(res)


main()
