"""
Compute Top 20 stocks by NSE-style Lowest Annualised Volatility
on each NSE LargeMidcap 250 rebalance date.

Methodology mirrors NSE's Nifty Low Volatility indices:

    daily_return = ln(close_t / close_{t-1})          # natural-log return
    volatility   = Std(daily_return over previous 252 days)
    annual_vol   = volatility * sqrt(252)             # for readability only

NSE defines volatility as the standard deviation of daily LOG returns
over the trailing 1 year (~252 trading days). Annualising by sqrt(252)
is a constant multiplier, so it does not change the ranking.

Uses the 1 year (~252 trading days) of daily log returns up to and
including the rebalance date. Stocks with the LOWEST volatility
(most stable) are ranked first.

For each rebalance CSV in "NSE LargeMidcap 250 Historical data/":
  1. Read the 250 symbols
  2. Look up each symbol's parquet in "daywise stocks/"
  3. Compute annualised volatility at the rebalance date
  4. Rank by lowest volatility and pick top 20
  5. Save CSV to "Top 20 NSE Low Volatility/" folder
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

# ---------- paths ----------
BASE = Path(os.getcwd())
REBALANCE_DIR = BASE / "NSE LargeMidcap 250 Historical data"
DAYWISE_DIR = BASE / "daywise stocks"
OUTPUT_DIR = BASE / "Top 20 NSE Low Volatility"

OUTPUT_DIR.mkdir(exist_ok=True)

VOL_LOOKBACK = 252    # 1 year of daily returns for volatility
TRADING_DAYS = 252    # annualisation factor
MIN_RETURNS = 200     # require at least this many daily returns
TOP_N = 20

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
    df = df.dropna(subset=["date_time"]).sort_values("date_time").reset_index(drop=True)

    stock_cache[sym_upper] = df
    return df

# ---------- process each rebalance date ----------
rebalance_files = sorted(REBALANCE_DIR.glob("nifty_largemidcap_250_*.csv"))
print(f"Found {len(rebalance_files)} rebalance files\n")

for rb_file in rebalance_files:
    date_str = rb_file.stem.replace("nifty_largemidcap_250_", "")
    rebalance_date = pd.Timestamp(date_str).date()

    symbols_df = pd.read_csv(rb_file)
    symbols = symbols_df["Symbol"].str.strip().tolist()

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

        idx = df.index[date_mask][-1]
        pos = df.index.get_loc(idx)

        if pos < VOL_LOOKBACK:
            insufficient_history.append(symbol)
            continue

        # Annualised volatility (1yr daily LOG returns up to rebalance date)
        # NSE uses natural-log daily returns: ln(close_t / close_t-1)
        hist = df.iloc[pos - VOL_LOOKBACK: pos + 1]["close"]
        daily_returns = np.log(hist / hist.shift(1)).dropna()

        if len(daily_returns) < MIN_RETURNS:
            insufficient_history.append(symbol)
            continue

        annual_volatility = daily_returns.std(ddof=1) * np.sqrt(TRADING_DAYS)

        if np.isnan(annual_volatility) or annual_volatility <= 0:
            insufficient_history.append(symbol)
            continue

        rows.append({
            "Symbol": symbol,
            "AnnualVolatility": round(annual_volatility, 6),
        })

    # Print missing info
    if missing_parquet:
        print(f"  Parquet file missing for: {', '.join(missing_parquet)}")
    if missing_date:
        print(f"  Data missing on {date_str} for: {', '.join(missing_date)}")
    if insufficient_history:
        print(f"  Insufficient history (<{VOL_LOOKBACK} days) for: {', '.join(insufficient_history)}")

    # Rank by LOWEST volatility (most stable first) and pick top 20
    if rows:
        result_df = pd.DataFrame(rows)
        result_df = result_df.sort_values("AnnualVolatility", ascending=True).head(TOP_N).reset_index(drop=True)
        result_df.index = result_df.index + 1
        result_df.index.name = "Rank"

        out_file = OUTPUT_DIR / f"top{TOP_N}_low_vol_{date_str}.csv"
        result_df.to_csv(out_file)
        print(f"  Saved: {out_file.name} | Most stable: {result_df.iloc[0]['Symbol']} ({result_df.iloc[0]['AnnualVolatility']})")
    else:
        print(f"  No data available for any stock on {date_str} — skipping.")

print(f"Done! All files saved to {OUTPUT_DIR.name}/ folder.")
