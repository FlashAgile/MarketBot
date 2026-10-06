from __future__ import annotations

from typing import List

import cv2
import numpy as np

from .models import FVG, LiquidityLevel, Swing, TradePlan
from .preprocess import PixelCandle, PriceScale

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _label(img, text, x, y, color):
    cv2.putText(img, text, (int(x), int(y)), _FONT, 0.4, color, 1, cv2.LINE_AA)


def annotate(
    bgr: np.ndarray,
    pcs: List[PixelCandle],
    scale: PriceScale,
    swings: List[Swing],
    fvgs: List[FVG],
    liquidity: List[LiquidityLevel],
    plan: TradePlan,
    path: str,
) -> None:
    img = bgr.copy()
    h, w = img.shape[:2]

    overlay = img.copy()
    for f in fvgs:
        if f.filled_pct >= 1.0:
            continue
        x0 = int(pcs[max(f.idx - 1, 0)].x)
        ya, yb = int(scale.y(f.top)), int(scale.y(f.bottom))
        color = (0, 200, 0) if f.kind == "bullish" else (0, 0, 220)
        cv2.rectangle(overlay, (x0, min(ya, yb)), (w - 1, max(ya, yb)), color, -1)
    img = cv2.addWeighted(overlay, 0.25, img, 0.75, 0)

    for s in swings:
        x, y = int(pcs[s.idx].x), int(scale.y(s.price))
        if s.kind == "high":
            _label(img, "SH", x - 8, y - 6, (0, 215, 255))
        else:
            _label(img, "SL", x - 8, y + 14, (255, 200, 0))

    for l in liquidity:
        if l.swept:
            continue
        y = int(scale.y(l.price))
        x0 = int(pcs[l.idx].x) if l.idx is not None else 0
        cv2.line(img, (x0, y), (w - 1, y), (200, 200, 200), 1, cv2.LINE_AA)
        _label(img, l.label, max(w - 110, x0), y - 3, (230, 230, 230))

    if plan.direction != "none" and plan.entry is not None:
        x0 = int(pcs[-1].x)
        lo, hi = plan.entry_zone
        cv2.rectangle(img, (x0, int(scale.y(hi))), (w - 1, int(scale.y(lo))), (255, 128, 0), 1)
        lines = [("ENTRY", plan.entry, (255, 128, 0)), ("SL", plan.stop_loss, (0, 0, 255))]
        lines += [(f"TP{i + 1} ({t['rr']}R)", t["price"], (0, 200, 0)) for i, t in enumerate(plan.targets)]
        for name, p, color in lines:
            y = int(scale.y(p))
            cv2.line(img, (x0, y), (w - 1, y), color, 2, cv2.LINE_AA)
            _label(img, name, x0 + 4, y - 4, color)

    cv2.imwrite(path, img)
