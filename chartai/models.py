from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class Candle:
    idx: int
    open: float
    high: float
    low: float
    close: float

    @property
    def bullish(self) -> bool:
        return self.close >= self.open


@dataclass
class Swing:
    idx: int
    price: float
    kind: str  # "high" | "low"


@dataclass
class FVG:
    idx: int  # index of the middle candle
    kind: str  # "bullish" | "bearish"
    top: float
    bottom: float
    filled_pct: float

    @property
    def mid(self) -> float:
        return (self.top + self.bottom) / 2.0


@dataclass
class StructureEvent:
    idx: int
    kind: str  # "BOS" | "CHoCH"
    direction: str  # "bullish" | "bearish"
    level: float


@dataclass
class LiquidityLevel:
    price: float
    side: str  # "buy-side" (above) | "sell-side" (below)
    label: str
    swept: bool
    idx: Optional[int] = None


@dataclass
class TradePlan:
    direction: str = "none"  # "long" | "short" | "none"
    bias_score: int = 0
    entry_zone: Optional[Tuple[float, float]] = None
    entry: Optional[float] = None
    stop_loss: Optional[float] = None
    risk_per_unit: Optional[float] = None
    targets: List[dict] = field(default_factory=list)
    confluence: int = 0
    notes: List[str] = field(default_factory=list)
