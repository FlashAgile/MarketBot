# Chart Analyzer

Analyzes candlestick chart screenshots (PNG/JPG) and produces:

- Trend bias (short / medium / long) and BOS / CHoCH market-structure events
- Fair Value Gaps (bullish / bearish, with fill percentage)
- Liquidity map: swing highs/lows, equal highs/lows, session extremes
- Trade plan: entry zone, stop loss, take-profit targets with R:R

> Educational tool. Not financial advice. Image-derived prices are approximations.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
# Normalized 0-100 price units
python main.py chart.png

# Map highest wick -> 5200, lowest wick -> 5050
python main.py chart.png --price-range 5200 5050

# Exact calibration: pixel_y1 price1 pixel_y2 price2 (read from axis labels)
python main.py chart.png --calib 80 5200 640 5050

# Restrict to the candle pane (x y width height) to exclude volume/axis labels
python main.py chart.png --roi 0 40 1500 700 --price-range 5200 5050
```

Outputs go to `output/`: `annotated.png` and `analysis.json`.

## Assumptions and limits

- Candles are filled green (bullish) and red (bearish) on any background.
  Tune hues in `chartai/config.py` or with env vars such as `CHARTAI_MIN_SATURATION=50`.
- Hollow/monochrome candles are not supported.
- Densely packed candles whose bodies touch merge into one blob and are skipped.
  Zoom in or crop to fewer candles.
- Use `--roi` to exclude volume bars, indicators, and price-label boxes in
  green/red.
- Charts carry no timestamps, so "session extremes" means the highest high
  and lowest low of the visible range.

## Config

All thresholds live in `chartai/config.py`. Any numeric field can be
overridden with an env var `CHARTAI_<FIELD_NAME>`, e.g. `CHARTAI_MIN_RR=2`.
