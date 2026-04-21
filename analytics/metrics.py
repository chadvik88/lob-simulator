import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from dataclasses import dataclass
from typing import Optional
from lob.book import Trade, LimitOrderBook
from lob.simulation import SimulationResult, SimulationSnapshot


@dataclass
class SpreadMetrics:
    """Time-series analysis of bid-ask spread."""
    timestamps: list[float]
    spreads: list[float]
    mean_spread: float
    median_spread: float
    std_spread: float
    pct_time_wide: float  # % of time spread > 2x mean


@dataclass
class OFIMetrics:
    """
    Order Flow Imbalance — Cont, Kukanov, Stoikov (2013).
    Measures buy vs sell pressure at the best quotes.
    Strongly predictive of short-term price moves.
    """
    timestamps: list[float]
    ofi_series: list[float]
    correlation_with_returns: float


@dataclass
class PriceImpactMetrics:
    """
    Price impact curve: how much does a trade of size X move the price?
    Follows the square root law: impact ~ sigma * sqrt(Q/V)
    """
    trade_sizes: list[float]
    price_impacts: list[float]
    fit_coefficient: float
    fit_exponent: float
    r_squared: float


@dataclass
class VWAPMetrics:
    """Volume-weighted average price of all trades."""
    vwap: float
    total_volume: int
    total_notional: float


@dataclass
class VolatilityMetrics:
    """Rolling realized volatility of mid-price returns."""
    timestamps: list[float]
    realized_vol: list[float]
    annualized_vol: float


@dataclass
class FullAnalytics:
    """Complete analytics output for a simulation run."""
    spread: SpreadMetrics
    ofi: OFIMetrics
    price_impact: PriceImpactMetrics
    vwap: VWAPMetrics
    volatility: VolatilityMetrics


def compute_spread_metrics(result: SimulationResult) -> SpreadMetrics:
    """
    Computes spread statistics over the full simulation.
    Identifies periods of spread widening (stress periods).
    """
    timestamps = result.timestamps
    spreads = result.spreads

    if not spreads:
        return SpreadMetrics([], [], 0, 0, 0, 0)

    arr = np.array(spreads)
    mean_spread = float(np.mean(arr))
    pct_wide = float(np.mean(arr > 2 * mean_spread))

    return SpreadMetrics(
        timestamps=timestamps,
        spreads=spreads,
        mean_spread=mean_spread,
        median_spread=float(np.median(arr)),
        std_spread=float(np.std(arr)),
        pct_time_wide=pct_wide,
    )


def compute_ofi(result: SimulationResult, window: int = 10) -> OFIMetrics:
    """
    Order Flow Imbalance per Cont, Kukanov & Stoikov (2013).

    OFI = (bid_depth_change - ask_depth_change) at the touch.
    Positive OFI = buying pressure. Negative = selling pressure.
    Computed as rolling difference in bid vs ask depth from snapshots.
    """
    snapshots = result.snapshots
    if len(snapshots) < window + 1:
        return OFIMetrics([], [], 0.0)

    df = pd.DataFrame([{
        "time": s.time,
        "bid_depth": s.bid_depth,
        "ask_depth": s.ask_depth,
        "mid_price": s.mid_price,
    } for s in snapshots if s.mid_price is not None])

    if df.empty:
        return OFIMetrics([], [], 0.0)

    df["bid_change"] = df["bid_depth"].diff()
    df["ask_change"] = df["ask_depth"].diff()
    df["ofi"] = df["bid_change"] - df["ask_change"]
    df["returns"] = df["mid_price"].pct_change()
    df = df.dropna()

    if len(df) < 2:
        return OFIMetrics([], [], 0.0)

    corr = float(df["ofi"].rolling(window).mean().corr(df["returns"]))
    corr = corr if not np.isnan(corr) else 0.0

    return OFIMetrics(
        timestamps=df["time"].tolist(),
        ofi_series=df["ofi"].tolist(),
        correlation_with_returns=corr,
    )


def _power_law(x: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Price impact model: impact = alpha * x^beta. Beta ~ 0.5 is the square root law."""
    return alpha * np.power(x, beta)


def compute_price_impact(result: SimulationResult) -> PriceImpactMetrics:
    """
    Fits the price impact curve to actual trades.
    Impact = price change caused by a trade of given size.
    Should follow a concave (square root) relationship.
    """
    trades = result.trades
    snapshots = result.snapshots

    if len(trades) < 10 or len(snapshots) < 2:
        return PriceImpactMetrics([], [], 0.0, 0.5, 0.0)

    snap_df = pd.DataFrame([{
        "time": s.time,
        "mid_price": s.mid_price,
    } for s in snapshots if s.mid_price is not None]).set_index("time")

    trade_df = pd.DataFrame([{
        "timestamp": t.timestamp,
        "price": t.price,
        "quantity": t.quantity,
    } for t in trades])

    trade_df = trade_df.sort_values("timestamp")
    trade_df["size_bucket"] = pd.qcut(trade_df["quantity"], q=20, duplicates="drop")
    grouped = trade_df.groupby("size_bucket", observed=True)

    sizes = []
    impacts = []
    for bucket, group in grouped:
        avg_size = float(group["quantity"].mean())
        avg_impact = float(group["price"].std()) if len(group) > 1 else 0.0
        if avg_size > 0 and avg_impact > 0:
            sizes.append(avg_size)
            impacts.append(avg_impact)

    if len(sizes) < 3:
        return PriceImpactMetrics(sizes, impacts, 0.0, 0.5, 0.0)

    try:
        popt, _ = curve_fit(_power_law, sizes, impacts, p0=[0.01, 0.5], maxfev=5000)
        alpha, beta = float(popt[0]), float(popt[1])
        predicted = _power_law(np.array(sizes), alpha, beta)
        ss_res = np.sum((np.array(impacts) - predicted) ** 2)
        ss_tot = np.sum((np.array(impacts) - np.mean(impacts)) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    except RuntimeError:
        alpha, beta, r2 = 0.0, 0.5, 0.0

    return PriceImpactMetrics(
        trade_sizes=sizes,
        price_impacts=impacts,
        fit_coefficient=alpha,
        fit_exponent=beta,
        r_squared=r2,
    )


def compute_vwap(result: SimulationResult) -> VWAPMetrics:
    """
    Volume-weighted average price across all trades.
    VWAP = sum(price * quantity) / sum(quantity)
    """
    trades = result.trades
    if not trades:
        return VWAPMetrics(0.0, 0, 0.0)

    total_notional = sum(t.price * t.quantity for t in trades)
    total_volume = sum(t.quantity for t in trades)
    vwap = total_notional / total_volume if total_volume > 0 else 0.0

    return VWAPMetrics(
        vwap=round(vwap, 4),
        total_volume=total_volume,
        total_notional=round(total_notional, 2),
    )


def compute_realized_volatility(
    result: SimulationResult,
    window: int = 20,
    trading_periods: int = 252,
) -> VolatilityMetrics:
    """
    Rolling realized volatility from mid-price log returns.
    Annualized using square root of time rule.
    Window is in snapshot intervals (default: 20 snapshots).
    """
    timestamps = result.timestamps
    mid_prices = result.mid_prices

    if len(mid_prices) < window + 1:
        return VolatilityMetrics([], [], 0.0)

    prices = np.array(mid_prices)
    log_returns = np.diff(np.log(prices))

    rolling_vol = []
    roll_times = []
    for i in range(window, len(log_returns)):
        window_returns = log_returns[i - window:i]
        vol = float(np.std(window_returns))
        rolling_vol.append(vol)
        roll_times.append(timestamps[i])

    annualized = float(np.mean(rolling_vol) * np.sqrt(trading_periods)) if rolling_vol else 0.0

    return VolatilityMetrics(
        timestamps=roll_times,
        realized_vol=rolling_vol,
        annualized_vol=round(annualized, 6),
    )


def run_full_analytics(result: SimulationResult) -> FullAnalytics:
    """Runs all metrics on a simulation result and returns a FullAnalytics object."""
    return FullAnalytics(
        spread=compute_spread_metrics(result),
        ofi=compute_ofi(result),
        price_impact=compute_price_impact(result),
        vwap=compute_vwap(result),
        volatility=compute_realized_volatility(result),
    )