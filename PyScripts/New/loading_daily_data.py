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
LOOKUP_FILE = cwd / "PyScripts" / "Data" / "stock_lookup_table.csv"
OUTPUT_DIR = cwd / "PyScripts" / "Data" / "New"
OUTPUT_FILE = OUTPUT_DIR / "raw_data_8_years.parquet"
# REFERENCE_FILE = cwd / "PyScripts" / "Data" / "raw_data_8_years.parquet"

START_DATE = "2018-01-01"
END_DATE = "2025-12-31"
INTERVAL = "1d"
BATCH_SIZE = 40
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


def _load_stock_tickers() -> list[str]:
    tickers = pd.read_csv(TICKERS_FILE)["Ticker"].dropna().astype(str).str.strip()
    lookup_tickers = pd.read_csv(LOOKUP_FILE)["Ticker"].dropna().astype(str).str.strip()

    tickers_list = [ticker for ticker in tickers.tolist() if ticker]
    lookup_list = [ticker for ticker in lookup_tickers.tolist() if ticker]

    if set(tickers_list) != set(lookup_list):
        raise ValueError("tickers.csv and stock_lookup_table.csv do not contain the same ticker set.")

    return tickers_list


def _download_type_block(tickers: list[str], asset_type: str) -> pd.DataFrame:
    pieces = []

    for start in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[start:start + BATCH_SIZE]
        print(f"Downloading {asset_type} batch {start // BATCH_SIZE + 1}: {len(batch)} tickers")

        raw = yf.download(
            tickers=batch,
            start=START_DATE,  # Inclusive lower bound for the download window, e.g. "2018-01-01".
            end=END_DATE,  # Exclusive upper bound; use the day after your desired final trading date.
            interval=INTERVAL,
            auto_adjust=True,
            repair=True,
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
            if len(available_fields) != len(PRICE_FIELDS):
                print(f"  - Missing required OHLCV fields for {ticker}")
                continue

            ticker_df = ticker_df[PRICE_FIELDS]
            ticker_df.columns = pd.MultiIndex.from_tuples(
                [(field, asset_type, ticker) for field in ticker_df.columns],
                names=["Metric", "Type", "Ticker"],
            )
            batch_frames.append(ticker_df)

        if batch_frames:
            pieces.append(pd.concat(batch_frames, axis=1))

    if not pieces:
        return pd.DataFrame()

    return pd.concat(pieces, axis=1)


# def _compare_with_reference(dataframe: pd.DataFrame) -> None:
#     if not REFERENCE_FILE.exists():
#         print("Reference parquet not found; skipping schema comparison.")
#         return
#
#     reference = pd.read_parquet(REFERENCE_FILE)
#     same_columns = dataframe.columns.equals(reference.columns)
#     same_index_name = dataframe.index.name == reference.index.name
#     same_start = dataframe.index.min() == reference.index.min()
#     same_end = dataframe.index.max() == reference.index.max()
#
#     print("\nReference comparison")
#     print("Same columns:", same_columns)
#     print("Same index name:", same_index_name)
#     print("Same start date:", same_start, dataframe.index.min(), reference.index.min())
#     print("Same end date:", same_end, dataframe.index.max(), reference.index.max())


def main() -> None:
    asset_map = {
        "Stocks": _load_stock_tickers(),
        "Index": INDEX_TICKERS,
    }

    print(f"Configured stock count: {len(asset_map['Stocks'])}")
    print(f"Configured index tickers: {asset_map['Index']}")

    type_frames = []
    for asset_type, tickers in asset_map.items():
        block = _download_type_block(tickers=tickers, asset_type=asset_type)
        if not block.empty:
            type_frames.append(block)

    if not type_frames:
        raise RuntimeError("No daily data was downloaded.")

    daily = pd.concat(type_frames, axis=1)
    daily = daily.sort_index(axis=1)
    daily = daily[~daily.index.duplicated(keep="last")].sort_index()

    if isinstance(daily.index, pd.DatetimeIndex) and daily.index.tz is not None:
        daily.index = daily.index.tz_convert(None)

    daily.columns = pd.MultiIndex.from_tuples(daily.columns, names=["Metric", "Type", "Ticker"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(OUTPUT_FILE, compression="gzip")

    print("\nSaved daily parquet:", OUTPUT_FILE)
    print("Shape:", daily.shape)
    print("Time range:", daily.index.min(), "to", daily.index.max())
    print("Column levels:", daily.columns.names)
    print("Metric values:", list(daily.columns.get_level_values(0).unique()))
    print("Type values:", list(daily.columns.get_level_values(1).unique()))
    # _compare_with_reference(daily)


if __name__ == "__main__":
    main()
