"""
Compute Top 20 stocks by Jensen's Alpha on each NSE LargeMidcap 250
rebalance date.

Jensen's Alpha (CAPM, risk-free rate = 0), estimated over the trailing
LOOKBACK trading days up to and including the rebalance date:

    market_return_t = mean( daily_return_t  over all eligible stocks )   (equal-weighted proxy)

    R_stock = alpha + beta * R_market + e
        beta        = Cov(R_stock, R_market) / Var(R_market)
        alpha_daily = mean(R_stock) - beta * mean(R_market)
        Alpha       = alpha_daily * 252   (annualised)

Stocks with the HIGHEST alpha (best risk-adjusted excess return) rank first.

For each rebalance CSV in "NSE LargeMidcap 250 Historical data/":
  1. Read the 250 symbols
  2. Look up each symbol's parquet in "daywise stocks/"
  3. Collect the trailing LOOKBACK daily returns at the rebalance date
  4. Build the equal-weighted market proxy and regress each stock on it
  5. Rank by highest alpha and pick top 20
  6. Save CSV to "Top 20 Alpha stock/" folder
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

# ---------- paths ----------
BASE = Path(os.getcwd())
REBALANCE_DIR = BASE / "NSE LargeMidcap 250 Historical data"
DAYWISE_DIR = BASE / "daywise stocks"
OUTPUT_DIR = BASE / "Top 20 Alpha stock"

OUTPUT_DIR.mkdir(exist_ok=True)

LOOKBACK = 252          # 1 year of daily returns for the regression
MIN_OBS = 200           # minimum overlapping observations to trust the fit
TRADING_DAYS = 252      # annualisation factor
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

# ---------- process each rebalance date ----------
rebalance_files = sorted(REBALANCE_DIR.glob("nifty_largemidcap_250_*.csv"))
print(f"Found {len(rebalance_files)} rebalance files\n")

for rb_file in rebalance_files:
    date_str = rb_file.stem.replace("nifty_largemidcap_250_", "")
    rebalance_date = pd.Timestamp(date_str).date()

    symbols_df = pd.read_csv(rb_file)
    symbols = symbols_df["Symbol"].astype(str).str.strip().tolist()

    print(f"--- Rebalance: {date_str} | Symbols: {len(symbols)} ---")

    missing_parquet = []
    missing_date = []
    insufficient_history = []

    # Collect the trailing daily-return window for every eligible stock,
    # indexed by date (needed to build the equal-weighted market proxy).
    returns_by_symbol = {}

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
        daily_returns = window["daily_return"].iloc[1:]          # drop leading NaN
        daily_returns.index = window["date_time"].iloc[1:].dt.date

        if daily_returns.notna().sum() < MIN_OBS:
            insufficient_history.append(symbol)
            continue

        returns_by_symbol[symbol] = daily_returns

    # Equal-weighted market proxy = mean daily return across the universe (per date)
    if returns_by_symbol:
        returns_matrix = pd.DataFrame(returns_by_symbol)
        market_return = returns_matrix.mean(axis=1, skipna=True)
    else:
        market_return = None

    # Jensen's alpha for each stock (risk-free = 0)
    rows = []
    if market_return is not None:
        for symbol, stock_return in returns_by_symbol.items():
            pair = pd.concat(
                [stock_return, market_return],
                axis=1,
                keys=["stock", "market"],
            ).dropna()

            if len(pair) < MIN_OBS:
                insufficient_history.append(symbol)
                continue

            market_var = pair["market"].var(ddof=1)
            if market_var == 0 or pd.isna(market_var):
                insufficient_history.append(symbol)
                continue

            beta = pair["stock"].cov(pair["market"]) / market_var
            alpha_daily = pair["stock"].mean() - beta * pair["market"].mean()
            alpha_annual = alpha_daily * TRADING_DAYS

            rows.append({
                "Symbol": symbol,
                "Alpha": round(alpha_annual, 6),
                "Beta": round(beta, 4),
                "AlphaDaily": round(alpha_daily, 8),
            })

    # Print missing info
    if missing_parquet:
        print(f"  Parquet file missing for: {', '.join(missing_parquet)}")
    if missing_date:
        print(f"  Data missing on {date_str} for: {', '.join(missing_date)}")
    if insufficient_history:
        print(f"  Insufficient history (<{LOOKBACK} days) for: {', '.join(insufficient_history)}")

    # Rank by HIGHEST alpha (best risk-adjusted excess return) and pick top 20
    if rows:
        result_df = pd.DataFrame(rows)
        result_df = result_df.sort_values("Alpha", ascending=False).head(TOP_N).reset_index(drop=True)
        result_df.index = result_df.index + 1
        result_df.index.name = "Rank"

        out_file = OUTPUT_DIR / f"top{TOP_N}_alpha_{date_str}.csv"
        result_df.to_csv(out_file)
        print(f"  Saved: {out_file.name} | Highest alpha: {result_df.iloc[0]['Symbol']} ({result_df.iloc[0]['Alpha']})")
    else:
        print(f"  No data available for any stock on {date_str} — skipping.")

print(f"Done! All files saved to {OUTPUT_DIR.name}/ folder.")
