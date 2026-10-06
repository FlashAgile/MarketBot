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
from chartai.strategy import bias_score, build_plan
from chartai.structure import (
    compute_atr,
    detect_fvgs,
    detect_structure,
    find_swings,
    map_liquidity,
    trend_bias,
)
from chartai.visualize import annotate

HTF = {"key": "htf", "title": "1-hour chart", "label": "1 hour", "minutes": 60}
LTF = {"key": "ltf", "title": "1-minute chart", "label": "1 minute", "minutes": 1}
SIM_PATHS = 3000
SIGN = {"bullish": 1, "bearish": -1, "ranging": 0}


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def fmt_minutes(m: float) -> str:
    if m < 60:
        return f"{round(m):d} min"
    for limit, size, name in ((1440, 60, "hour"), (10080, 1440, "day"), (None, 10080, "week")):
        if limit is None or m < limit:
            v = round(m / size, 1)
            return f"{v:g} {name}{'' if v == 1 else 's'}"
    return f"{m:g} min"


def rnd(v, nd: int = 2):
    return None if v is None else round(float(v), nd)


def pct(x: float) -> str:
    """Percent string that never claims certainty from a simulation."""
    if x > 0.995:
        return ">99%"
    if x < 0.005:
        return "<1%"
    return f"{round(100 * x)}%"


def bias_windows(n: int) -> dict:
    """Candle counts behind the short / medium / long trend bias (mirrors chartai.structure)."""
    return {
        name: min(n, max(5, int(n * frac)))
        for name, frac in (("short", 0.25), ("medium", 0.5), ("long", 1.0))
    }


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


# ----------------------------------------------------------------------------
# Chart analysis (one chart)
# ----------------------------------------------------------------------------
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
        "swings": swings,
        "candles": candles,
        "atr": atr_val,
        "bias": bias,
        "structure_trend": structure_trend,
        "calibrated": calibrated,
        "n": len(candles),
        "last_close": candles[-1].close,
        "cfg": cfg,
    }


# ----------------------------------------------------------------------------
# Combining the two timeframes
# ----------------------------------------------------------------------------
def combine(res_h: dict, res_l: dict) -> dict:
    """1-hour chart sets the bias; the 1-minute chart must confirm it and supplies the entry."""
    cfg = res_h["cfg"]
    h_score = bias_score(res_h["bias"], res_h["events"])
    l_score = bias_score(res_l["bias"], res_l["events"])
    thr = cfg.min_bias_score
    h_dir = "long" if h_score >= thr else "short" if h_score <= -thr else "neutral"
    l_dir = "long" if l_score >= thr else "short" if l_score <= -thr else "neutral"
    plan = res_l["plan"]

    htf_factors: dict = {}
    if h_dir == "neutral":
        status = "stand_aside"
        headline = f"No trade: the 1-hour chart gives no clear bias (score {h_score})."
        detail = "Wait for the 1-hour trend and structure to line up."
    elif l_dir not in ("neutral", h_dir):
        status = "conflict"
        headline = (
            f"Timeframes disagree: the 1-hour chart points {h_dir}, "
            f"the 1-minute chart points {l_dir}. Stand aside."
        )
        detail = "A trade against the 1-hour bias is not offered."
    elif plan.direction == "none":
        status = "wait"
        if l_dir == "neutral":
            headline = f"1-hour bias is {h_dir}, but the 1-minute chart is mixed (score {l_score}) and has no entry yet."
        else:
            headline = f"The 1-hour and 1-minute charts agree ({h_dir}), but there is no entry zone yet."
        detail = " ".join(plan.notes)
    else:
        status = "aligned"
        word = "up" if h_dir == "long" else "down"
        headline = f"Aligned {h_dir.upper()}: the 1-hour and 1-minute charts both point {word}."
        detail = ""
        s = 1 if h_dir == "long" else -1
        htf_factors = {
            "1H trend aligned (at least 2 horizons)": sum(1 for v in res_h["bias"].values() if SIGN[v] == s) >= 2,
            "1H last structure event aligned": bool(res_h["events"]) and SIGN[res_h["events"][-1].direction] == s,
        }
    confluence = (plan.confluence + sum(htf_factors.values())) if status == "aligned" else 0
    return {
        "status": status,
        "headline": headline,
        "detail": detail,
        "h_dir": h_dir,
        "l_dir": l_dir,
        "h_score": h_score,
        "l_score": l_score,
        "htf_factors": htf_factors,
        "confluence": confluence,
        "confluence_max": 6,
    }


# ----------------------------------------------------------------------------
# Possible-outcome simulation (runs on the 1-minute chart)
# ----------------------------------------------------------------------------
def key_levels(res: dict) -> dict:
    """Nearest unswept liquidity above and below the last close (ATR fallback)."""
    last, atr = res["last_close"], res["atr"]
    liq = [item for item in res["liquidity"] if not item.swept]
    above = sorted((item for item in liq if item.price > last), key=lambda item: item.price)
    below = sorted((item for item in liq if item.price < last), key=lambda item: -item.price)
    up = (above[0].price, above[0].label) if above else (last + 2 * atr, "2 ATR above")
    dn = (below[0].price, below[0].label) if below else (last - 2 * atr, "2 ATR below")
    return {"up": up, "dn": dn}


def simulate_paths(candles, start: float, horizon: int, drift: bool, n: int = SIM_PATHS, seed: int = 42) -> dict:
    """Bootstrap candle moves (close change plus wicks) from the chart itself."""
    close = np.array([c.close for c in candles])
    high = np.array([c.high for c in candles])
    low = np.array([c.low for c in candles])
    d = np.diff(close)
    up_wick = np.maximum(high[1:] - close[1:], 0.0)
    dn_wick = np.maximum(close[1:] - low[1:], 0.0)
    if not drift:
        d = d - d.mean()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, horizon))
    closes = start + np.cumsum(d[idx], axis=1)
    return {"close": closes, "high": closes + up_wick[idx], "low": closes - dn_wick[idx]}


def _first(mask: np.ndarray):
    hit = mask.any(axis=1)
    idx = np.where(hit, mask.argmax(axis=1), mask.shape[1])
    return hit, idx


def race(paths: dict, target: float, stop: float, long: bool = True) -> dict:
    """Which barrier is touched first. If both hit in one candle, the stop counts first."""
    hi, lo = paths["high"], paths["low"]
    if long:
        t_hit, t_idx = _first(hi >= target)
        s_hit, s_idx = _first(lo <= stop)
    else:
        t_hit, t_idx = _first(lo <= target)
        s_hit, s_idx = _first(hi >= stop)
    stop_first = s_hit & (~t_hit | (s_idx <= t_idx))
    target_first = t_hit & (~s_hit | (t_idx < s_idx))
    neither = ~(stop_first | target_first)
    return {
        "target": float(target_first.mean()),
        "stop": float(stop_first.mean()),
        "neither": float(neither.mean()),
        "target_median": float(np.median(t_idx[target_first] + 1)) if target_first.any() else None,
        "stop_median": float(np.median(s_idx[stop_first] + 1)) if stop_first.any() else None,
    }


def touch(paths: dict, level: float, from_above: bool) -> dict:
    mask = (paths["low"] <= level) if from_above else (paths["high"] >= level)
    hit, idx = _first(mask)
    return {
        "p": float(hit.mean()),
        "median": float(np.median(idx[hit] + 1)) if hit.any() else None,
    }


def outcomes(res: dict, horizon: int, use_plan: bool) -> dict:
    candles, last, plan = res["candles"], res["last_close"], res["plan"]
    lv = key_levels(res)
    results = {}
    for name, drift in (("No drift", False), ("Recent drift continues", True)):
        p0 = simulate_paths(candles, last, horizon, drift)
        r = {
            "race": race(p0, lv["up"][0], lv["dn"][0], True),
            "closes_up": float((p0["close"][:, -1] > last).mean()),
        }
        if use_plan and plan.direction != "none":
            long = plan.direction == "long"
            zone_lo, zone_hi = plan.entry_zone
            r["fill"] = touch(p0, zone_hi if long else zone_lo, from_above=long)
            p1 = simulate_paths(candles, plan.entry, horizon, drift, seed=43)
            r["plan"] = race(p1, plan.targets[0]["price"], plan.stop_loss, long)
        results[name] = r
    return results


def outcome_rows(results: dict, res: dict, horizon: int, use_plan: bool) -> list:
    lv, plan = key_levels(res), res["plan"]
    up_p, up_l = lv["up"]
    dn_p, dn_l = lv["dn"]

    def row(label, fn):
        return {"Outcome": label, **{k: pct(fn(v)) for k, v in results.items()}}

    rows = [
        row(f"Touches {up_l} ({rnd(up_p)}) first", lambda v: v["race"]["target"]),
        row(f"Touches {dn_l} ({rnd(dn_p)}) first", lambda v: v["race"]["stop"]),
        row(f"Touches neither within {horizon} candles", lambda v: v["race"]["neither"]),
        row("Ends the window above the current price", lambda v: v["closes_up"]),
    ]
    if use_plan and plan.direction != "none":
        tp1 = plan.targets[0]["price"]
        rows += [
            row("Plan: pulls back into the entry zone", lambda v: v["fill"]["p"]),
            row(f"Plan, if entered: reaches TP1 ({rnd(tp1)}) first", lambda v: v["plan"]["target"]),
            row(f"Plan, if entered: hits the stop ({rnd(plan.stop_loss)}) first", lambda v: v["plan"]["stop"]),
            row("Plan, if entered: neither", lambda v: v["plan"]["neither"]),
        ]
    return rows


def scenario_lines(res_h: dict, res_l: dict, verdict: dict) -> list:
    last, atr, plan = res_l["last_close"], res_l["atr"], res_l["plan"]
    lv = key_levels(res_l)
    up_p, up_l = lv["up"]
    dn_p, dn_l = lv["dn"]
    lines = [
        f"Bullish: price holds above {dn_l} ({rnd(dn_p)}) and pushes through {up_l} ({rnd(up_p)}), "
        f"about {rnd((up_p - last) / atr, 1)} ATR away on the 1-minute chart.",
        f"Bearish: price loses {dn_l} ({rnd(dn_p)}) and heads lower, "
        f"about {rnd((last - dn_p) / atr, 1)} ATR away on the 1-minute chart.",
        f"1-hour chart: structure {res_h['structure_trend']}; trend bias {res_h['bias']['short']} / "
        f"{res_h['bias']['medium']} / {res_h['bias']['long']} (short / medium / long).",
        f"1-minute chart: structure {res_l['structure_trend']}; trend bias {res_l['bias']['short']} / "
        f"{res_l['bias']['medium']} / {res_l['bias']['long']}.",
    ]
    if verdict["status"] == "aligned":
        tp1 = plan.targets[0]
        lines.append(
            f"Plan ({plan.direction}): enter around {rnd(plan.entry)}, invalidated if price trades "
            f"through {rnd(plan.stop_loss)}, first target {rnd(tp1['price'])} ({tp1['rr']}R)."
        )
    else:
        lines.append("Plan: " + verdict["headline"])
    return lines


# ----------------------------------------------------------------------------
# UI sections
# ----------------------------------------------------------------------------
def chart_slot(spec: dict, demo_png: bytes | None):
    """Uploader plus per-chart options. Returns None (nothing yet), or a dict."""
    key, title = spec["key"], spec["title"]
    uploaded = st.file_uploader(f"Upload the {title}", type=["png", "jpg", "jpeg"], key=f"{key}_file")
    data = uploaded.getvalue() if uploaded else demo_png
    if data is None:
        return None
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        return {"error": f"Could not read the {title} as an image. Try a PNG or JPG."}
    height, width = bgr.shape[:2]

    price_range = roi = None
    with st.expander(f"Options for the {title} (price range, crop)"):
        if st.checkbox("I know the price range", key=f"{key}_use_range"):
            hi = st.number_input("Price at the highest wick", value=100.0, key=f"{key}_hi")
            lo = st.number_input("Price at the lowest wick", value=0.0, key=f"{key}_lo")
            if hi != lo:
                price_range = (hi, lo)
        if st.checkbox("Crop to candle area", key=f"{key}_use_roi"):
            st.caption(f"Image size: {width} x {height}")
            rx = st.number_input("X", 0, width - 1, 0, key=f"{key}_x")
            ry = st.number_input("Y", 0, height - 1, 0, key=f"{key}_y")
            rw = st.number_input("Width", 1, width, width, key=f"{key}_w")
            rh = st.number_input("Height", 1, height, height, key=f"{key}_h")
            roi = (int(rx), int(ry), int(min(rw, width - rx)), int(min(rh, height - ry)))
    return {"bgr": bgr, "roi": roi, "price_range": price_range}


def show_chart_detail(res: dict, spec: dict) -> None:
    tf_min, cfg = spec["minutes"], res["cfg"]
    units = "price" if res["calibrated"] else "normalized 0-100 units"
    st.image(res["png"], caption=f"Annotated {spec['title']}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Timeframe", spec["label"])
    c2.metric("Chart covers", fmt_minutes(res["n"] * tf_min))
    c3.metric("Candles detected", res["n"])
    c4.metric("Last close", rnd(res["last_close"]))
    st.caption(f"Levels are in {units}.")

    wins = bias_windows(res["n"])
    cols = st.columns(4)
    for col, name in zip(cols, ("short", "medium", "long")):
        k = wins[name]
        col.metric(f"{name.title()}: last {k} candles ({fmt_minutes(k * tf_min)})", res["bias"][name].title())
    cols[3].metric("Structure", res["structure_trend"].title())
    if res["events"]:
        e = res["events"][-1]
        st.caption(f"Last structure event: {e.direction} {e.kind} at {rnd(e.level)} (candle {e.idx})")

    active = [f for f in res["fvgs"] if f.filled_pct < cfg.max_fvg_fill]
    with st.expander(f"Fair Value Gaps ({len(res['fvgs'])} total, {len(active)} active)"):
        if res["fvgs"]:
            st.dataframe(
                [
                    {"Candle": f.idx, "Type": f.kind, "Bottom": rnd(f.bottom), "Top": rnd(f.top),
                     "Filled %": round(f.filled_pct * 100)}
                    for f in res["fvgs"]
                ]
            )
        else:
            st.write("None found.")
    with st.expander("Liquidity levels"):
        st.dataframe(
            [
                {"Price": rnd(item.price), "Side": item.side, "Label": item.label, "Swept": item.swept}
                for item in sorted(res["liquidity"], key=lambda x: -x.price)
            ]
        )


def show_verdict(res_h: dict, res_l: dict, verdict: dict) -> None:
    st.subheader("Outcome")
    status = verdict["status"]
    if status == "aligned":
        st.success(verdict["headline"])
    else:
        st.warning(verdict["headline"] + (" " + verdict["detail"] if verdict["detail"] else ""))

    st.table(
        [
            {"Chart": "1-hour", "Structure": res_h["structure_trend"].title(),
             "Short / Medium / Long": " / ".join(res_h["bias"][k].title() for k in ("short", "medium", "long")),
             "Bias score": verdict["h_score"],
             "Reads as": verdict["h_dir"].replace("neutral", "no clear bias")},
            {"Chart": "1-minute", "Structure": res_l["structure_trend"].title(),
             "Short / Medium / Long": " / ".join(res_l["bias"][k].title() for k in ("short", "medium", "long")),
             "Bias score": verdict["l_score"],
             "Reads as": verdict["l_dir"].replace("neutral", "no clear bias")},
        ]
    )

    if status != "aligned":
        return
    plan = res_l["plan"]
    units = "price" if res_l["calibrated"] else "normalized 0-100 units of the 1-minute chart"
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Direction", plan.direction.upper())
    d2.metric("Entry zone", f"{rnd(plan.entry_zone[0])} - {rnd(plan.entry_zone[1])}")
    d3.metric("Stop loss", rnd(plan.stop_loss))
    d4.metric("Confluence", f"{verdict['confluence']}/{verdict['confluence_max']}")
    st.caption(f"Limit entry at {rnd(plan.entry)} | risk per unit {rnd(plan.risk_per_unit)} | levels in {units}")
    if not res_l["calibrated"]:
        st.caption("Tip: enter the 1-minute chart's price range in its options to see real prices.")
    st.table(
        [
            {"Target": f"TP{i}", "Price": rnd(t["price"]), "R:R": t["rr"], "Source": t["source"]}
            for i, t in enumerate(plan.targets, 1)
        ]
    )
    st.write("**Checklist**")
    for name, ok in verdict["htf_factors"].items():
        st.write(f"- {'+' if ok else '-'} {name}")
    for note in plan.notes:
        st.write("- 1M: " + note if note[:1] in "+-" else "- " + note)


def show_outcomes(res_h: dict, res_l: dict, verdict: dict):
    st.subheader("Possible outcomes")
    horizon = st.slider(
        "Look-ahead window (1-minute candles)", 15, 240, 60, 5, key="horizon",
        help="How many 1-minute candles forward to look.",
    )
    st.caption(f"{horizon} candles is about {fmt_minutes(horizon * LTF['minutes'])}.")
    use_plan = verdict["status"] == "aligned"

    for line in scenario_lines(res_h, res_l, verdict):
        st.write("- " + line)

    results = outcomes(res_l, horizon, use_plan)
    rows = outcome_rows(results, res_l, horizon, use_plan)
    st.table(rows)

    nd = results["No drift"]["race"]
    lv = key_levels(res_l)

    def t(v):
        return "n/a" if v is None else f"{fmt_minutes(v * LTF['minutes'])} ({round(v)} candles)"

    st.caption(
        f"Typical time to first touch (no-drift run): {lv['up'][1]} {t(nd['target_median'])}, "
        f"{lv['dn'][1]} {t(nd['stop_median'])}."
    )
    st.caption(
        f"Odds come from {SIM_PATHS:,} simulated paths built from the 1-minute chart's own candle moves. "
        "'No drift' treats the recent trend as noise; 'Recent drift continues' assumes the trend "
        "keeps going exactly as before, which is the optimistic case and rarely holds for long. "
        "These are rough odds, not forecasts: they ignore news, spreads, and changes in market "
        "behavior, and a candle that touches both levels counts the stop first."
    )
    return horizon, rows


def main() -> None:
    st.set_page_config(page_title="MarketBot Chart Analyzer", layout="wide")
    st.title("MarketBot Chart Analyzer")
    st.caption(
        "Upload a 1-hour chart and a 1-minute chart of the same market (green = bullish, red = bearish). "
        "The outcome appears only when both are uploaded. The 1-hour chart sets the bias and the "
        "1-minute chart supplies the entry. Educational tool, not financial advice."
    )

    with st.sidebar:
        st.header("Detection settings")
        cfg = Config.from_env()
        cfg.min_candles = st.slider("Minimum candles", 5, 100, cfg.min_candles)
        cfg.swing_lookback = st.slider("Swing lookback", 1, 8, cfg.swing_lookback)
        cfg.min_saturation = st.slider("Color saturation cutoff", 20, 200, cfg.min_saturation)
        cfg.min_rr = st.slider("Minimum reward:risk", 1.0, 5.0, float(cfg.min_rr), 0.1)

    has_upload = any(st.session_state.get(f"{s['key']}_file") is not None for s in (HTF, LTF))
    if st.button("Try demo charts"):
        if has_upload:
            st.caption("Remove your uploaded files first to use the demo charts.")
        else:
            st.session_state["demo"] = {
                "htf": cv2.imencode(".png", demo_chart(seed=1, drift=2.0))[1].tobytes(),
                "ltf": cv2.imencode(".png", demo_chart(seed=11, drift=1.0))[1].tobytes(),
            }
    if has_upload:
        st.session_state.pop("demo", None)
    demo = st.session_state.get("demo", {})

    slots = {spec["key"]: chart_slot(spec, demo.get(spec["key"])) for spec in (HTF, LTF)}

    results = {}
    ready = True
    for spec in (HTF, LTF):
        slot = slots[spec["key"]]
        if slot is None:
            st.info(f"Waiting for the {spec['title']}.")
            ready = False
            continue
        if "error" in slot:
            st.error(slot["error"])
            ready = False
            continue
        try:
            results[spec["key"]] = analyze(slot["bgr"], cfg, slot["roi"], slot["price_range"])
            st.success(f"{spec['title'].capitalize()}: {results[spec['key']]['n']} candles detected.")
        except (ValueError, FileNotFoundError) as exc:
            st.error(f"{spec['title'].capitalize()}: {exc}")
            ready = False

    if not ready:
        st.info("The outcome appears once both the 1-hour and the 1-minute chart are uploaded and read.")
        return

    res_h, res_l = results["htf"], results["ltf"]
    verdict = combine(res_h, res_l)

    show_verdict(res_h, res_l, verdict)
    horizon, rows = show_outcomes(res_h, res_l, verdict)

    st.subheader("Chart details")
    tab_h, tab_l = st.tabs([HTF["title"].capitalize(), LTF["title"].capitalize()])
    with tab_h:
        show_chart_detail(res_h, HTF)
    with tab_l:
        show_chart_detail(res_l, LTF)

    report = {
        "verdict": {k: v for k, v in verdict.items()},
        "possible_outcomes": {"window_candles_1m": horizon, "table": rows},
        "chart_1h": res_h["report"],
        "chart_1m": res_l["report"],
    }
    d1, d2, d3 = st.columns(3)
    d1.download_button("1-hour annotated image", res_h["png"], "annotated_1h.png", "image/png")
    d2.download_button("1-minute annotated image", res_l["png"], "annotated_1m.png", "image/png")
    d3.download_button("JSON report", json.dumps(report, indent=2), "analysis.json", "application/json")


main()
