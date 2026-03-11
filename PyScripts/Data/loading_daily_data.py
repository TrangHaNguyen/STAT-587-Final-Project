#!/usr/bin/env python3
# Max-capacity run: YF_NETWORK_PROFILE=fast .venv/bin/python PyScripts/Data/loading_daily_data.py
from pathlib import Path
import time
import os

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
OUTPUT_DIR = cwd / "PyScripts" / "Data"
OUTPUT_FILE = OUTPUT_DIR / "raw_data_8_years.parquet"

START_DATE = "2018-01-01"
END_DATE = "2025-12-31"
INTERVAL = "1d"
INDEX_TICKERS = ["^SPX"]
PRICE_FIELDS = ["Close", "High", "Low", "Open", "Volume"]

# Capacity/network tuning (override with env var: YF_NETWORK_PROFILE=fast|stable|auto).
CPU_COUNT = os.cpu_count() or 4
NETWORK_PROFILE = os.getenv("YF_NETWORK_PROFILE", "auto").strip().lower()
if NETWORK_PROFILE not in {"auto", "fast", "stable"}:
    NETWORK_PROFILE = "auto"

BATCH_SIZE = min(80, max(20, CPU_COUNT * 4))
FALLBACK_BATCH_SIZE = max(5, min(15, CPU_COUNT))
MAX_SINGLE_TICKER_RECOVERY = max(20, min(60, CPU_COUNT * 3))
REPAIR_PRICE_DATA = False

if NETWORK_PROFILE == "fast":
    RETRY_ATTEMPTS = 1
    RETRY_WAIT_SECONDS = 0
    DOWNLOAD_THREADS = True
elif NETWORK_PROFILE == "stable":
    RETRY_ATTEMPTS = 3
    RETRY_WAIT_SECONDS = 2
    DOWNLOAD_THREADS = False
    BATCH_SIZE = min(BATCH_SIZE, 20)
else:  # auto
    RETRY_ATTEMPTS = 2
    RETRY_WAIT_SECONDS = 1
    DOWNLOAD_THREADS = True


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


def _download_with_retries(
    tickers: list[str],
    *,
    asset_type: str,
    batch_label: str,
    retry_attempts: int,
    retry_wait_seconds: int,
    threads: bool,
) -> pd.DataFrame:
    last_error = None
    for attempt in range(1, retry_attempts + 1):
        try:
            raw = yf.download(
                tickers=tickers,
                start=START_DATE,  # Inclusive lower bound for the download window, e.g. "2018-01-01".
                end=END_DATE,  # Exclusive upper bound; use the day after your desired final trading date.
                interval=INTERVAL,
                auto_adjust=True,
                repair=REPAIR_PRICE_DATA,
                actions=False,
                progress=False,
                threads=threads,
                group_by="ticker",
            )
            if not raw.empty:
                return raw
            last_error = "empty response"
        except Exception as exc:
            last_error = str(exc)

        if attempt < retry_attempts:
            print(f"  - Retry {attempt}/{retry_attempts - 1} for {asset_type} {batch_label} ({last_error})")
            time.sleep(retry_wait_seconds * attempt)

    print(f"  - Failed after retries for {asset_type} {batch_label}: {last_error}")
    return pd.DataFrame()


def _extract_ticker_frame(raw: pd.DataFrame, ticker: str, asset_type: str) -> pd.DataFrame | None:
    if ticker not in raw.columns.get_level_values(0):
        return None

    ticker_df = raw[ticker].copy()
    available_fields = [field for field in PRICE_FIELDS if field in ticker_df.columns]
    if len(available_fields) != len(PRICE_FIELDS):
        return None

    ticker_df = ticker_df[PRICE_FIELDS]
    ticker_df.columns = pd.MultiIndex.from_tuples(
        [(field, asset_type, ticker) for field in ticker_df.columns],
        names=["Metric", "Type", "Ticker"],
    )
    return ticker_df


def _extract_batch_frames(raw: pd.DataFrame, tickers: list[str], asset_type: str) -> tuple[list[pd.DataFrame], list[str]]:
    frames = []
    missing_tickers = []
    for ticker in tickers:
        ticker_frame = _extract_ticker_frame(raw, ticker, asset_type)
        if ticker_frame is None:
            missing_tickers.append(ticker)
        else:
            frames.append(ticker_frame)
    return frames, missing_tickers


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _download_type_block(tickers: list[str], asset_type: str) -> pd.DataFrame:
    pieces = []
    recovery_candidates = []
    batch_size = BATCH_SIZE
    retry_attempts = RETRY_ATTEMPTS
    retry_wait_seconds = RETRY_WAIT_SECONDS
    download_threads = DOWNLOAD_THREADS
    empty_batch_streak = 0

    for start in range(0, len(tickers), batch_size):
        batch = tickers[start:start + batch_size]
        batch_num = start // batch_size + 1
        print(f"Downloading {asset_type} batch {batch_num}: {len(batch)} tickers")
        raw = _download_with_retries(
            batch,
            asset_type=asset_type,
            batch_label=f"batch {batch_num}",
            retry_attempts=retry_attempts,
            retry_wait_seconds=retry_wait_seconds,
            threads=download_threads,
        )

        if raw.empty:
            empty_batch_streak += 1
            print("  - Empty batch response; scheduling these tickers for recovery pass.")
            recovery_candidates.extend(batch)
            if NETWORK_PROFILE == "auto" and empty_batch_streak >= 2 and download_threads:
                download_threads = False
                retry_attempts = max(retry_attempts, 3)
                retry_wait_seconds = max(retry_wait_seconds, 2)
                batch_size = max(10, batch_size // 2)
                print("  - Auto network fallback engaged: threads=False, higher retries, smaller batches.")
            continue

        empty_batch_streak = 0
        raw = _ensure_ticker_first_columns(raw, batch)
        batch_frames, missing_tickers = _extract_batch_frames(raw, batch, asset_type)
        recovery_candidates.extend(missing_tickers)

        if batch_frames:
            pieces.append(pd.concat(batch_frames, axis=1))

    # Recovery pass: retry missing symbols in smaller batches (faster than one-by-one retries).
    recovery_candidates = _dedupe_preserve_order(recovery_candidates)
    unresolved = []
    if recovery_candidates:
        print(f"\nRecovery pass for {asset_type}: {len(recovery_candidates)} unresolved tickers")
    for start in range(0, len(recovery_candidates), FALLBACK_BATCH_SIZE):
        batch = recovery_candidates[start:start + FALLBACK_BATCH_SIZE]
        batch_num = start // FALLBACK_BATCH_SIZE + 1
        raw = _download_with_retries(
            batch,
            asset_type=asset_type,
            batch_label=f"recovery batch {batch_num}",
            retry_attempts=retry_attempts,
            retry_wait_seconds=retry_wait_seconds,
            threads=download_threads,
        )
        if raw.empty:
            unresolved.extend(batch)
            continue

        raw = _ensure_ticker_first_columns(raw, batch)
        batch_frames, missing_tickers = _extract_batch_frames(raw, batch, asset_type)
        unresolved.extend(missing_tickers)
        if batch_frames:
            pieces.append(pd.concat(batch_frames, axis=1))

    # Final pass: recover only a limited number individually to control total runtime.
    final_unresolved = unresolved
    if unresolved and len(unresolved) <= MAX_SINGLE_TICKER_RECOVERY:
        final_unresolved = []
        print(f"Final single-ticker recovery for {asset_type}: {len(unresolved)} tickers")
        for ticker in unresolved:
            raw_single = _download_with_retries(
                [ticker],
                asset_type=asset_type,
                batch_label=f"ticker {ticker}",
                retry_attempts=retry_attempts,
                retry_wait_seconds=retry_wait_seconds,
                threads=download_threads,
            )
            if raw_single.empty:
                final_unresolved.append(ticker)
                continue
            raw_single = _ensure_ticker_first_columns(raw_single, [ticker])
            ticker_frame = _extract_ticker_frame(raw_single, ticker, asset_type)
            if ticker_frame is None:
                final_unresolved.append(ticker)
                continue
            pieces.append(ticker_frame)

    if final_unresolved:
        print(f"Unresolved {asset_type} tickers ({len(final_unresolved)}): {sorted(set(final_unresolved))}")

    if not pieces:
        return pd.DataFrame()

    return pd.concat(pieces, axis=1)




def main() -> None:
    asset_map = {
        "Stocks": _load_stock_tickers(),
        "Index": INDEX_TICKERS,
    }

    print(f"Configured stock count: {len(asset_map['Stocks'])}")
    print(f"Configured index tickers: {asset_map['Index']}")
    print(
        "Runtime tuning:",
        f"cpu={CPU_COUNT}, profile={NETWORK_PROFILE}, batch={BATCH_SIZE},",
        f"fallback_batch={FALLBACK_BATCH_SIZE}, retries={RETRY_ATTEMPTS}, threads={DOWNLOAD_THREADS}",
    )

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
