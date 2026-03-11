#!/usr/bin/env python3
# Max-capacity run: YF_NETWORK_PROFILE=fast .venv/bin/python PyScripts/Data/loading_hourly_data.py
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
OUTPUT_FILE = cwd / "PyScripts" / "Data" / "hourly_data.parquet"

INTERVAL = "1h"
PERIOD = "730d"
MAX_POINTS = 4900
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
                interval=INTERVAL,
                period=PERIOD,
                auto_adjust=False,
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
    if not available_fields:
        return None

    ticker_df = ticker_df[available_fields]
    ticker_df.columns = pd.MultiIndex.from_tuples(
        [(field, asset_type, ticker) for field in ticker_df.columns]
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
            batch_data = pd.concat(batch_frames, axis=1)
            pieces.append(batch_data)

    # Recovery pass: retry unresolved symbols in smaller batches.
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

    # Final pass: recover a limited number individually to avoid excessive runtime.
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
    print(
        "Runtime tuning:",
        f"cpu={CPU_COUNT}, profile={NETWORK_PROFILE}, batch={BATCH_SIZE},",
        f"fallback_batch={FALLBACK_BATCH_SIZE}, retries={RETRY_ATTEMPTS}, threads={DOWNLOAD_THREADS}",
    )

    type_frames = []
    for asset_type, tickers in asset_map.items():
        print(f"\nBuilding hourly block for {asset_type}: {len(tickers)} tickers")
        block = _download_type_block(tickers=tickers, asset_type=asset_type)
        if not block.empty:
            type_frames.append(block)

    if not type_frames:
        raise RuntimeError("No hourly data was downloaded.")

    hourly = pd.concat(type_frames, axis=1, sort=False).sort_index(axis=1)
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
