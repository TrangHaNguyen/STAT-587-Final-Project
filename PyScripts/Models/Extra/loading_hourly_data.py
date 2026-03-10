#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import yfinance as yf


cwd = Path.cwd()
for _ in range(5):
    if cwd.name != "STAT-587-Final-Project":
        cwd = cwd.parent
    else:
        break
else:
    raise FileNotFoundError("Could not find correct workspace folder.")

TICKERS_FILE = cwd / "PyScripts" / "Data" / "tickers.csv"
OUTPUT_FILE = cwd / "PyScripts" / "hourly_data.parquet"

INTERVAL = "1h"
PERIOD = "730d"
BATCH_SIZE = 40
MAX_POINTS = 4900
INDEX_TICKERS = ["^SPX"]

PRICE_FIELDS = ["Close", "High", "Low", "Open", "Volume"]


def _ensure_ticker_first_columns(dataframe: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if dataframe.empty:
        return dataframe

    if not isinstance(dataframe.columns, pd.MultiIndex):
        if len(tickers) == 1:
            dataframe.columns = pd.MultiIndex.from_product([tickers, dataframe.columns])
            return dataframe
        raise ValueError("Unexpected non-MultiIndex columns for multi-ticker download.")

    level0_values = set(dataframe.columns.get_level_values(0))
    level1_values = set(dataframe.columns.get_level_values(1))

    has_ticker_level0 = any(ticker in level0_values for ticker in tickers)
    has_ticker_level1 = any(ticker in level1_values for ticker in tickers)

    if has_ticker_level0:
        return dataframe
    if has_ticker_level1:
        return dataframe.swaplevel(0, 1, axis=1).sort_index(axis=1)

    raise ValueError("Could not detect ticker level in downloaded columns.")


def _download_type_block(tickers: list[str], asset_type: str) -> pd.DataFrame:
    pieces = []

    for start in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[start:start + BATCH_SIZE]
        print(f"Downloading {asset_type} batch {start // BATCH_SIZE + 1}: {len(batch)} tickers")

        raw = yf.download(
            tickers=batch,
            interval=INTERVAL,
            period=PERIOD,
            auto_adjust=False,
            actions=False,
            progress=False,
            threads=True,
            group_by="ticker",
        )

        if raw.empty:
            print("  - Empty batch response; skipping.")
            continue

        raw = _ensure_ticker_first_columns(raw, batch)

        batch_frames = []
        for ticker in batch:
            if ticker not in raw.columns.get_level_values(0):
                print(f"  - Missing ticker from response: {ticker}")
                continue

            ticker_df = raw[ticker].copy()
            available_fields = [field for field in PRICE_FIELDS if field in ticker_df.columns]
            if not available_fields:
                print(f"  - No required OHLCV fields for {ticker}")
                continue

            ticker_df = ticker_df[available_fields]
            ticker_df.columns = pd.MultiIndex.from_tuples(
                [(field, asset_type, ticker) for field in ticker_df.columns]
            )
            batch_frames.append(ticker_df)

        if batch_frames:
            batch_data = pd.concat(batch_frames, axis=1)
            pieces.append(batch_data)

    if not pieces:
        return pd.DataFrame()

    return pd.concat(pieces, axis=1)


def _load_asset_map() -> dict[str, list[str]]:
    tickers = pd.read_csv(TICKERS_FILE)["Ticker"].dropna().astype(str).str.strip()
    stock_tickers = sorted(ticker for ticker in tickers.unique() if ticker)

    asset_map = {
        "Index": INDEX_TICKERS,
        "Stocks": stock_tickers,
    }
    return {asset_type: symbols for asset_type, symbols in asset_map.items() if symbols}


def main() -> None:
    asset_map = _load_asset_map()
    print(f"Types configured for download: {sorted(asset_map)}")

    type_frames = []
    for asset_type, tickers in asset_map.items():
        print(f"\nBuilding hourly block for {asset_type}: {len(tickers)} tickers")
        block = _download_type_block(tickers=tickers, asset_type=asset_type)
        if not block.empty:
            type_frames.append(block)

    if not type_frames:
        raise RuntimeError("No hourly data was downloaded.")

    hourly = pd.concat(type_frames, axis=1).sort_index(axis=1)
    hourly = hourly[~hourly.index.duplicated(keep="last")].sort_index()

    if isinstance(hourly.index, pd.DatetimeIndex) and hourly.index.tz is not None:
        hourly.index = hourly.index.tz_convert(None)

    if MAX_POINTS is not None and len(hourly) > MAX_POINTS:
        hourly = hourly.tail(MAX_POINTS)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(OUTPUT_FILE, compression="gzip")

    unique_days = pd.to_datetime(hourly.index).normalize().nunique()
    print("\nSaved hourly parquet:", OUTPUT_FILE)
    print("Shape:", hourly.shape)
    print("Time range:", hourly.index.min(), "to", hourly.index.max())
    print("Unique days:", int(unique_days))


if __name__ == "__main__":
    main()
