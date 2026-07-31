"""
Compute Top 20 stocks by Lowest Volatility on each NSE 40 rebalance date.

Volatility = close.pct_change().rolling(ROLLING_WINDOW).std()

Uses the 20 trading days up to and including the rebalance date.
Stocks with LOWEST volatility (most stable) are ranked first.

For each rebalance CSV in "NSE LargeMidcap 250 Historical data/":
  1. Read the 250 symbols
  2. Look up each symbol's parquet in "daywise stocks/"
  3. Compute 20-day rolling volatility at the rebalance date
  4. Rank by lowest volatility and pick top 20
  5. Save CSV to "Top 20 low volatility stock/" folder
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

# ---------- paths ----------
BASE = Path(os.getcwd())
REBALANCE_DIR = BASE / "NSE LargeMidcap 250 Historical data"
DAYWISE_DIR = BASE / "daywise stocks"
OUTPUT_DIR = BASE / "Top 20 low volatility 40 days stock"

OUTPUT_DIR.mkdir(exist_ok=True)  # 40-day rolling std of daily returns
ROLLING_WINDOW = 40 

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

    # Pre-compute daily returns and rolling volatility
    df["daily_return"] = df["close"].pct_change()
    df["volatility_20d"] = df["daily_return"].rolling(ROLLING_WINDOW).std()

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
        vol_value = df.loc[idx, "volatility_20d"]

        if pd.isna(vol_value):
            insufficient_history.append(symbol)
            continue

        rows.append({
            "Symbol": symbol,
            "Volatility": round(vol_value, 6),
        })
    
    # Print missing info
    if missing_parquet:
        print(f"  Parquet file missing for: {', '.join(missing_parquet)}")
    if missing_date:
        print(f"  Data missing on {date_str} for: {', '.join(missing_date)}")
    if insufficient_history:
        print(f"  Insufficient history (<{ROLLING_WINDOW} days) for: {', '.join(insufficient_history)}")
        
    # Rank by LOWEST volatility (most stable first) and pick top 20
    if rows:
        result_df = pd.DataFrame(rows)
        result_df = result_df.sort_values("Volatility", ascending=True).head(20).reset_index(drop=True)
        result_df.index = result_df.index + 1
        result_df.index.name = "Rank"

        out_file = OUTPUT_DIR / f"top20_low_vol_{date_str}.csv"
        result_df.to_csv(out_file)
        print(f"  Saved: {out_file.name} | Most stable: {result_df.iloc[0]['Symbol']} ({result_df.iloc[0]['Volatility']})")
    else:
        print(f"  No data available for any stock on {date_str} — skipping.")

print(f"Done! All files saved to {OUTPUT_DIR.name}/ folder.")
