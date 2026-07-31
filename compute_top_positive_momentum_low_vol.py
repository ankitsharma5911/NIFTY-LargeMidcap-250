import os
import pandas as pd
import numpy as np
from pathlib import Path

# ---------- paths ----------
BASE = Path(os.getcwd())
REBALANCE_DIR = BASE / "NSE LargeMidcap 250 Historical data"
DAYWISE_DIR = BASE / "daywise stocks"
OUTPUT_DIR = BASE / "Top 20 positive momentum 60 days low vol 40 days stock"

OUTPUT_DIR.mkdir(exist_ok=True)

LOOKBACK = 60      # momentum lookback (trading days)
ROLLING_WINDOW = 40  # volatility rolling window

# Pre-build a lookup: symbol -> parquet path (case-insensitive)
parquet_lookup = {}
for f in DAYWISE_DIR.iterdir():
    if f.suffix == ".parquet":
        parquet_lookup[f.stem.upper()] = f

# Cache loaded parquets
stock_cache = {}

def load_stock(symbol):
    """Load and cache a stock's parquet, sorted by date, with volatility pre-computed."""
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

    # Pre-compute volatility
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
    insufficient_data = []

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

        # Need at least 42 trading days before for momentum
        if pos < LOOKBACK:
            insufficient_data.append(symbol)
            continue

        close_t = df.iloc[pos]["close"]
        close_t_n = df.iloc[pos - LOOKBACK]["close"]
        vol_value = df.iloc[pos]["volatility_20d"]

        if (pd.isna(close_t) or pd.isna(close_t_n) or close_t_n == 0
                or pd.isna(vol_value) or vol_value == 0):
            insufficient_data.append(symbol)
            continue

        momentum = (close_t - close_t_n) / close_t_n
        score = momentum / vol_value

        rows.append({
            "Symbol": symbol,
            "Momentum": round(momentum, 6),
            "Volatility": round(vol_value, 6),
            "Score": round(score, 4),
        })

    # Print missing info
    if missing_parquet:
        print(f"  Parquet file missing for: {', '.join(missing_parquet)}")
    if missing_date:
        print(f"  Data missing on {date_str} for: {', '.join(missing_date)}")
    if insufficient_data:
        print(f"  Insufficient data for: {', '.join(insufficient_data)}")

    # Filter to POSITIVE momentum only (per docstring spec). During sharp drawdowns this may
    # yield fewer than 20 names — that's intentional: don't force-buy negative-momentum stocks.
    # The backtest handles variable-size lists (equal-weights across whatever count is provided).
    positive_rows = [r for r in rows if r["Momentum"] > 0]
    filtered_out = len(rows) - len(positive_rows)
    if filtered_out:
        print(f"  Filtered out {filtered_out} stock(s) with negative momentum")

    # Rank by HIGHEST score and pick top 20
    if positive_rows:
        result_df = pd.DataFrame(positive_rows)
        result_df = result_df.sort_values("Score", ascending=False).head(20).reset_index(drop=True)
        result_df.index = result_df.index + 1
        result_df.index.name = "Rank"

        out_file = OUTPUT_DIR / f"top20_mom_lowvol_{date_str}.csv"
        result_df.to_csv(out_file)
        print(f"  Saved: {out_file.name} | {len(result_df)} stocks | Top: {result_df.iloc[0]['Symbol']} (Score: {result_df.iloc[0]['Score']})")
    else:
        print(f"  No positive-momentum stocks on {date_str} — skipping (regime filter should have you in cash anyway).")



print(f"Done! All files saved to '{OUTPUT_DIR}' folder.")
