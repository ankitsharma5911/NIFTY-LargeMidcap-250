"""
Compute Top 20 QUALITY stocks on each NSE LargeMidcap 250 rebalance date.

NOTE ON METHODOLOGY
-------------------
The real NSE "Quality" factor is built from FUNDAMENTALS (ROE, Debt/Equity,
5-year EPS-growth variability). This dataset only contains OHLCV price data, so
this script builds a PRICE-BASED PROXY that mirrors the *structure* of the NSE
quality methodology (three pillars -> cross-sectional Z-score -> normal-CDF
normalise -> equal-weight average -> rank), using price-derived analogues:

    NSE pillar (fundamental)          Price-based analogue        Better when
    --------------------------------  --------------------------  -----------
    ROE (profitability)               Annualised Sharpe ratio     higher
    Low financial leverage            Max drawdown (near 0)       higher
    Low EPS-growth variability        Return stability (std of    lower
                                      rolling monthly returns)

All three are measured over the trailing LOOKBACK trading days up to and
including the rebalance date. This is NOT the official NSE quality factor.

For each rebalance CSV in "NSE LargeMidcap 250 Historical data/":
  1. Read the 250 symbols
  2. Look up each symbol's parquet in "daywise stocks/"
  3. Compute the three raw price metrics at the rebalance date
  4. Z-score each metric across the universe, normalise with the normal CDF,
     average the three into a Quality Score
  5. Rank by highest Quality Score and pick top 20
  6. Save CSV to "Top 20 Quality stock/" folder
"""

import os
import math
import pandas as pd
import numpy as np
from pathlib import Path

# ---------- paths ----------
BASE = Path(os.getcwd())
REBALANCE_DIR = BASE / "NSE LargeMidcap 250 Historical data"
DAYWISE_DIR = BASE / "daywise stocks"
OUTPUT_DIR = BASE / "Top 10 Quality stock"

OUTPUT_DIR.mkdir(exist_ok=True)

# --- Tuned via parameter sweep (best risk-adjusted: highest CAGR/return with
# --- drawdown no worse than the 20-stock baseline). Concentrating to the
# --- best 10 names lifted CAGR ~1.5pts vs TOP_N=20 without deepening drawdown.
LOOKBACK = 252          # ~1 year of daily returns for every metric
MIN_OBS = 200           # minimum valid daily returns to trust the metrics
MONTHLY_WINDOW = 21     # ~1 month, for the return-stability metric
TRADING_DAYS = 252      # annualisation factor
Z_CLIP = 3.0            # winsorise Z-scores to [-3, 3] (NSE-style)
TOP_N = 10

# Pre-build a lookup: symbol -> parquet path (case-insensitive)
parquet_lookup = {}
for f in DAYWISE_DIR.iterdir():
    if f.suffix == ".parquet":
        parquet_lookup[f.stem.upper()] = f

# Cache loaded parquets
stock_cache = {}
def load_stock(symbol):
    """Load and cache a stock's parquet, sorted by date."""
    sym_upper = symbol.upper()
    if sym_upper in stock_cache:
        return stock_cache[sym_upper]

    parquet_path = parquet_lookup.get(sym_upper)
    if parquet_path is None:
        stock_cache[sym_upper] = None
        return None

    df = pd.read_parquet(parquet_path)
    df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
    df = (
        df.dropna(subset=["date_time"])
        .sort_values("date_time")
        .drop_duplicates("date_time")
        .reset_index(drop=True)
    )

    # Pre-compute daily returns
    df["daily_return"] = df["close"].pct_change()

    stock_cache[sym_upper] = df
    return df


def norm_cdf(z):
    """Standard-normal CDF (avoids a scipy dependency)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def zscore(series):
    """Cross-sectional Z-score, winsorised to [-Z_CLIP, Z_CLIP]."""
    std = series.std(ddof=1)
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=series.index)
    z = (series - series.mean()) / std
    return z.clip(-Z_CLIP, Z_CLIP)


# ---------- process each rebalance date ----------
rebalance_files = sorted(REBALANCE_DIR.glob("nifty_largemidcap_250_*.csv"))
print(f"Found {len(rebalance_files)} rebalance files\n")

for rb_file in rebalance_files:
    date_str = rb_file.stem.replace("nifty_largemidcap_250_", "")
    rebalance_date = pd.Timestamp(date_str).date()

    symbols_df = pd.read_csv(rb_file)
    symbols = symbols_df["Symbol"].astype(str).str.strip().tolist()

    print(f"--- Rebalance: {date_str} | Symbols: {len(symbols)} ---")

    rows = []
    missing_parquet = []
    missing_date = []
    insufficient_history = []

    for symbol in symbols:
        df = load_stock(symbol)

        if df is None:
            missing_parquet.append(symbol)
            continue

        # Find the exact rebalance date
        date_mask = df["date_time"].dt.date == rebalance_date
        if not date_mask.any():
            missing_date.append(symbol)
            continue

        pos = df.index.get_loc(df.index[date_mask][-1])
        if pos < LOOKBACK:
            insufficient_history.append(symbol)
            continue

        window = df.iloc[pos - LOOKBACK: pos + 1]
        close = window["close"]
        rets = window["daily_return"].iloc[1:]           # drop leading NaN

        if rets.notna().sum() < MIN_OBS:
            insufficient_history.append(symbol)
            continue

        # --- Metric 1: annualised Sharpe (profitability proxy) ---
        r_std = rets.std(ddof=1)
        if r_std == 0 or pd.isna(r_std):
            insufficient_history.append(symbol)
            continue
        sharpe = (rets.mean() / r_std) * math.sqrt(TRADING_DAYS)

        # --- Metric 2: max drawdown over window (resilience proxy) ---
        # Negative number; closer to 0 = shallower drawdown = better.
        running_max = close.cummax()
        max_drawdown = (close / running_max - 1.0).min()

        # --- Metric 3: return stability (low EPS-variability proxy) ---
        # Std of rolling monthly returns; lower = more consistent.
        monthly_rets = close.pct_change(MONTHLY_WINDOW).dropna()
        ret_stability = monthly_rets.std(ddof=1)
        if pd.isna(ret_stability):
            insufficient_history.append(symbol)
            continue

        rows.append({
            "Symbol": symbol,
            "Sharpe": sharpe,
            "MaxDrawdown": max_drawdown,
            "RetStability": ret_stability,
        })

    # Print missing info
    if missing_parquet:
        print(f"  Parquet file missing for: {', '.join(missing_parquet)}")
    if missing_date:
        print(f"  Data missing on {date_str} for: {', '.join(missing_date)}")
    if insufficient_history:
        print(f"  Insufficient history (<{LOOKBACK} days) for: {', '.join(insufficient_history)}")

    if not rows:
        print(f"  No data available for any stock on {date_str} — skipping.\n")
        continue

    metrics = pd.DataFrame(rows)

    # --- Cross-sectional Z-scores across the eligible universe ---
    # Higher Sharpe better, higher (closer-to-0) MaxDrawdown better,
    # LOWER RetStability better -> negate its Z-score.
    z_sharpe = zscore(metrics["Sharpe"])
    z_drawdown = zscore(metrics["MaxDrawdown"])
    z_stability = -zscore(metrics["RetStability"])

    # Normalise each Z-score to (0, 1) with the standard-normal CDF,
    # then equal-weight average into the Quality Score (NSE-style).
    metrics["QualityScore"] = (
        z_sharpe.apply(norm_cdf)
        + z_drawdown.apply(norm_cdf)
        + z_stability.apply(norm_cdf)
    ) / 3.0

    # Rank by HIGHEST Quality Score and pick top 20
    result_df = (
        metrics.sort_values("QualityScore", ascending=False)
        .head(TOP_N)
        .reset_index(drop=True)
    )
    result_df["QualityScore"] = result_df["QualityScore"].round(6)
    result_df["Sharpe"] = result_df["Sharpe"].round(4)
    result_df["MaxDrawdown"] = result_df["MaxDrawdown"].round(4)
    result_df["RetStability"] = result_df["RetStability"].round(6)
    result_df = result_df[["Symbol", "QualityScore", "Sharpe", "MaxDrawdown", "RetStability"]]
    result_df.index = result_df.index + 1
    result_df.index.name = "Rank"

    out_file = OUTPUT_DIR / f"top{TOP_N}_quality_{date_str}.csv"
    result_df.to_csv(out_file)
    print(f"  Saved: {out_file.name} | Best quality: "
          f"{result_df.iloc[0]['Symbol']} ({result_df.iloc[0]['QualityScore']})\n")

print(f"Done! All files saved to {OUTPUT_DIR.name}/ folder.")
