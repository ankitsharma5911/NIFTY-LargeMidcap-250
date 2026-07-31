"""
Compute Top 20 stocks by NSE-style Normalized Momentum Score
on each NSE LargeMidcap 250 rebalance date.

Methodology mirrors NSE's Nifty200 Momentum 30 index:

    1. For each stock, compute the risk-adjusted price return over
       6 months (~126 trading days) and 12 months (~252 trading days):

           ret_6m  = (Close_t / Close_{t-126}) - 1
           ret_12m = (Close_t / Close_{t-252}) - 1

           annual_vol = Std(daily returns over previous 252 days) * sqrt(252)

           ratio_6m  = ret_6m  / annual_vol
           ratio_12m = ret_12m / annual_vol

    2. Convert each risk-adjusted ratio to a z-score across the
       eligible universe:

           z = (ratio - mean(ratio)) / std(ratio)

    3. Normalized Momentum Score = average of the two z-scores:

           score = ( z_6m + z_12m ) / 2

    4. Rank by highest score and pick the top 20.
    5. Save CSV to "Top 20 NSE Momentum/" folder.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

# ======================================================
# CONFIGURATION
# ======================================================

BASE = Path(os.getcwd())

REBALANCE_DIR = BASE / "NSE LargeMidcap 250 Historical data"
DAYWISE_DIR = BASE / "daywise stocks"
OUTPUT_DIR = BASE / "Top 20 NSE Momentum"

OUTPUT_DIR.mkdir(exist_ok=True)

RETURN_LOOKBACK_6M = 126     # 6 months
RETURN_LOOKBACK_12M = 252    # 12 months
VOL_LOOKBACK = 252           # 1 year of daily returns for volatility
TRADING_DAYS = 252           # annualisation factor
TOP_N = 20

# Need enough history for the longest lookback / vol window
MIN_HISTORY = max(RETURN_LOOKBACK_12M, VOL_LOOKBACK)

# ======================================================
# Build parquet lookup
# ======================================================

parquet_lookup = {}

for f in DAYWISE_DIR.glob("*.parquet"):
    parquet_lookup[f.stem.upper()] = f

stock_cache = {}

# ======================================================
# Load stock
# ======================================================

def load_stock(symbol):

    symbol = symbol.upper()

    if symbol in stock_cache:
        return stock_cache[symbol]

    file = parquet_lookup.get(symbol)

    if file is None:
        stock_cache[symbol] = None
        return None

    df = pd.read_parquet(file)

    df["date_time"] = pd.to_datetime(df["date_time"])

    df = (
        df.sort_values("date_time")
        .drop_duplicates("date_time")
        .reset_index(drop=True)
    )

    stock_cache[symbol] = df

    return df

def zscore(series):
    """Standard z-score. Returns zeros if std is 0 to avoid divide-by-zero."""
    std = series.std(ddof=1)
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std

# ======================================================
# Process rebalance files
# ======================================================

rebalance_files = sorted(
    REBALANCE_DIR.glob("nifty_largemidcap_250_*.csv")
)

print(f"\nFound {len(rebalance_files)} rebalance files\n")

for rb_file in rebalance_files:

    date_str = rb_file.stem.replace(
        "nifty_largemidcap_250_",
        ""
    )

    rebalance_date = pd.Timestamp(date_str).date()

    symbols_df = pd.read_csv(rb_file)

    symbols = (
        symbols_df["Symbol"]
        .astype(str)
        .str.strip()
        .tolist()
    )

    print("=" * 70)
    print(date_str)
    print("=" * 70)

    rows = []

    missing_parquet = []
    missing_date = []
    insufficient_history = []

    for symbol in symbols:

        df = load_stock(symbol)

        if df is None:
            missing_parquet.append(symbol)
            continue

        mask = df["date_time"].dt.date == rebalance_date

        if not mask.any():
            missing_date.append(symbol)
            continue

        idx = df.index[mask][-1]

        pos = df.index.get_loc(idx)

        if pos < MIN_HISTORY:
            insufficient_history.append(symbol)
            continue

        close_today = df.iloc[pos]["close"]
        close_6m = df.iloc[pos - RETURN_LOOKBACK_6M]["close"]
        close_12m = df.iloc[pos - RETURN_LOOKBACK_12M]["close"]

        if (
            pd.isna(close_today)
            or pd.isna(close_6m)
            or pd.isna(close_12m)
            or close_6m <= 0
            or close_12m <= 0
        ):
            insufficient_history.append(symbol)
            continue

        # --------------------------------------
        # 6M and 12M returns
        # --------------------------------------

        return_6m = (close_today - close_6m) / close_6m
        return_12m = (close_today - close_12m) / close_12m

        # --------------------------------------
        # Annualised volatility (1yr daily returns)
        # --------------------------------------

        hist = df.iloc[pos - VOL_LOOKBACK: pos + 1]["close"]

        daily_returns = hist.pct_change().dropna()

        if len(daily_returns) < 200:
            insufficient_history.append(symbol)
            continue

        annual_volatility = (
            daily_returns.std(ddof=1)
            * np.sqrt(TRADING_DAYS)
        )

        if (
            np.isnan(annual_volatility)
            or annual_volatility <= 0
        ):
            insufficient_history.append(symbol)
            continue

        # --------------------------------------
        # Risk-adjusted returns (6M & 12M)
        # --------------------------------------

        ratio_6m = return_6m / annual_volatility
        ratio_12m = return_12m / annual_volatility

        rows.append(
            {
                "Symbol": symbol,
                "Return6M": return_6m,
                "Return12M": return_12m,
                "AnnualVolatility": annual_volatility,
                "Ratio6M": ratio_6m,
                "Ratio12M": ratio_12m,
            }
        )

    # ==================================================
    # Missing information
    # ==================================================

    if missing_parquet:
        print(
            "Missing parquet:",
            ", ".join(missing_parquet)
        )

    if missing_date:
        print(
            "Missing rebalance date:",
            ", ".join(missing_date)
        )

    if insufficient_history:
        print(
            "Insufficient history:",
            ", ".join(insufficient_history)
        )

    if len(rows) == 0:
        print("No valid stocks.\n")
        continue

    # ==================================================
    # DataFrame
    # ==================================================

    result = pd.DataFrame(rows)

    # --------------------------------------------------
    # Z-score of each risk-adjusted ratio (6M & 12M)
    # --------------------------------------------------

    result["Z_6M"] = zscore(result["Ratio6M"])
    result["Z_12M"] = zscore(result["Ratio12M"])

    # --------------------------------------------------
    # Final Normalized Momentum Score
    # --------------------------------------------------

    result["MomentumScore"] = (
        result["Z_6M"]
        + result["Z_12M"]
    ) / 2

    result = (
        result
        .sort_values(
            "MomentumScore",
            ascending=False,
        )
        .head(TOP_N)
        .reset_index(drop=True)
    )

    result.index += 1
    result.index.name = "Rank"

    # Round values

    result = result.round(
        {
            "Return6M": 6,
            "Return12M": 6,
            "AnnualVolatility": 6,
            "Ratio6M": 6,
            "Ratio12M": 6,
            "Z_6M": 4,
            "Z_12M": 4,
            "MomentumScore": 4,
        }
    )

    # Column order

    result = result[
        [
            "Symbol",
            "MomentumScore",
            "Return6M",
            "Return12M",
            "AnnualVolatility",
            "Ratio6M",
            "Ratio12M",
            "Z_6M",
            "Z_12M",
        ]
    ]

    # ==================================================
    # Save
    # ==================================================

    out_file = (
        OUTPUT_DIR
        / f"top{TOP_N}_momentum_{date_str}.csv"
    )

    result.to_csv(out_file)

    print(
        f"Saved : {out_file.name}"
    )

    print(
        f"Top Stock : {result.iloc[0]['Symbol']} "
        f"Score = {result.iloc[0]['MomentumScore']:.4f}"
    )

print("\nDone.")
