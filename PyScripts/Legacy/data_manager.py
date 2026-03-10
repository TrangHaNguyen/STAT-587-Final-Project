#!/usr/bin/env python3
import yfinance as yf
import pandas as pd
import sys
import time
from pathlib import Path

# ------------------------------------------------------------------------------------------------------------------
# Retrieving the following data will utilize
# - S&P 500 stocks
# - Indexes:
#   - S&P 500 (^SPX),
#   - Russel 2000 (^RUT),
#   - NASDAQ (^IXIC),
#   - Dow Jones Industrial (^DJI),
#   - CBOE Volatility Index (^VIX),
#   - Nikkei 225 (^N225) [Japan],
#   - DAX (^GDAXI) [Germany].
# - Commodities
#   - Crude Oil (CL=F),
#   - Natural Gas (NG=F),
#   - Gold (GC=F),
#   - Silver (SI=F),
#   - Corn (ZC=F)
#
# This data will be formatted as such:
# | Metric       | Close                                                                               | High                                                                                | Low                                                                                 | Open                                                                                | Volume                                                                              |
# | Type         | Stocks                  | Commodities                     | Indexes                 | Stocks                  | Commodities                     | Indexes                 | Stocks                  | Commodities                     | Indexes                 | Stocks                  | Commodities                     | Indexes                 | Stocks                  | Commodities                     | Indexes                 |
# | Ticker       | Stock 1 | ... | Stock m | Commodity 1 | ... | Commodity 5 | Index 1 | ... | Index 7 | Stock 1 | ... | Stock m | Commodity 1 | ... | Commodity 5 | Index 1 | ... | Index 7 | Stock 1 | ... | Stock m | Commodity 1 | ... | Commodity 5 | Index 1 | ... | Index 7 | Stock 1 | ... | Stock m | Commodity 1 | ... | Commodity 5 | Index 1 | ... | Index 7 | Stock 1 | ... | Stock m | Commodity 1 | ... | Commodity 5 | Index 1 | ... | Index 7 |
# | Index (Date) |         |     |         |             |     |             |         |     |         |         |     |         |             |     |             |         |     |         |         |     |         |             |     |             |         |     |         |         |     |         |             |     |             |         |     |         |         |     |         |             |     |             |         |     |         |
# | Date 0       |         |     |         |             |     |             |         |     |         |         |     |         |             |     |             |         |     |         |         |     |         |             |     |             |         |     |         |         |     |         |             |     |             |         |     |         |         |     |         |             |     |             |         |     |         |
# |  ...         |         |     |         |             |     |             |         |     |         |         |     |         |             |     |             |         |     |         |         |     |         |             |     |             |         |     |         |         |     |         |             |     |             |         |     |         |         |     |         |             |     |             |         |     |         |
# | Date n       |         |     |         |             |     |             |         |     |         |         |     |         |             |     |             |         |     |         |         |     |         |             |     |             |         |     |         |         |     |         |             |     |             |         |     |         |         |     |         |             |     |             |         |     |         |
#
# With a stock lookup table as such:
# | Stock (Name) | Sectors | Market Capitalization (in tens of millions) |
# | Stock 1      |         |                                             |
# |   .          |         |                                             |
# | Stock m      |         |                                             |
# ------------------------------------------------------------------------------------------------------------------

start_date='2024-01-01'
end_date='2025-12-31'

print(f"Running: {sys.argv[0]}")

CLEANED_DATA=pd.DataFrame()
if (not (Path("bin") / "total_data.csv").exists()):
    print("Setting up necessary variables...")
    TICKERS=sys.argv[1:]
    print("Retrieved TICKERS.")
    INDEXES=["^SPX", "^RUT", "^IXIC", "^DJI", "^VIX", "^N225", "^GDAXI"]
    COMMODS=["CL=F", "NG=F", "GC=F", "SI=F", "ZC=F"]
    print("Finished setting up necessary variables.")

    print("Downloading necessary data.")
    INDEX_DATA=yf.download(INDEXES, start=start_date, end=end_date, threads=1)
    COMMO_DATA=yf.download(COMMODS, start=start_date, end=end_date, threads=1)
    print("TICKERS size is: ",  len(TICKERS))
    cycles = int(len(TICKERS) / 50)
    print("Downloading over ", cycles + 1, " cycles.")
    print("Cycle 1")
    STOCK_DATA=yf.download(TICKERS[0:50], start=start_date, end=end_date, threads=2)
    time.sleep(7)
    for i in range(cycles):
        print("Cycle ", i+2)
        STOCK_DATA=pd.concat([STOCK_DATA, yf.download(TICKERS[50*(i+1):50*(i+2)], start=start_date, end=end_date, threads=2)], axis=1)
        time.sleep(7)
    print("Finished downloading data.")

    print("Formatting data...")
    temp_df={
        'Stocks': STOCK_DATA,
        'Commodities': COMMO_DATA,
        'Indexes': INDEX_DATA
    }
    TOTAL_DATA=pd.concat(temp_df, axis=1)
    TOTAL_DATA.columns.names=["Type", "Metric", "Ticker"]
    TOTAL_DATA=TOTAL_DATA.reorder_levels([1, 0, 2], axis=1)
    print(TOTAL_DATA["Close"].head())
    print("Finished formatting data.")

    print("Cleaning up data utilizing linear interpolation, forward and backward fill...")
    CLEANED_DATA=TOTAL_DATA.interpolate(method="linear", limit_direction="both", limit=1).ffill().bfill()
    print(CLEANED_DATA["Close"].head())
    print("Finished cleaning up the data.")
    print("Final NA count: ", CLEANED_DATA.isnull().sum().sum())

    print("Generating .csv file to store data at bin folder named: total_data.csv")
    CLEANED_DATA.to_csv(Path("bin") / 'total_data.csv', index=True)
    print("Finished creating file.")
else:
    confirmation=input("Data file was found, would you like to download and overwrite existing file (true/yes)? ")
    if confirmation.lower() in ["true", "t", "yes", "y"]:
        print("Setting up necessary variables...")
        TICKERS=sys.argv[1:]
        print("Retrieved TICKERS.")
        INDEXES=["^SPX", "^RUT", "^IXIC", "^DJI", "^VIX", "^N225", "^GDAXI"]
        COMMODS=["CL=F", "NG=F", "GC=F", "SI=F", "ZC=F"]
        print("Finished setting up necessary variables.")

        print("Downloading necessary data.")
        INDEX_DATA=yf.download(INDEXES, start=start_date, end=end_date, threads=1)
        COMMO_DATA=yf.download(COMMODS, start=start_date, end=end_date, threads=1)
        print("TICKERS size is: ",  len(TICKERS))
        cycles = int(len(TICKERS) / 50)
        print("Downloading over ", cycles + 1, " cycles.")
        print("Cycle 1")
        STOCK_DATA=yf.download(TICKERS[0:50], start=start_date, end=end_date, threads=2)
        time.sleep(7)
        for i in range(cycles):
            print("Cycle ", i+2)
            STOCK_DATA=pd.concat([STOCK_DATA, yf.download(TICKERS[50*(i+1):50*(i+2)], start=start_date, end=end_date, threads=2)], axis=1)
            time.sleep(7)
        print("Finished downloading data.")

        print("Formatting data...")
        temp_df={
            'Stocks': STOCK_DATA,
            'Commodities': COMMO_DATA,
            'Indexes': INDEX_DATA
        }
        TOTAL_DATA=pd.concat(temp_df, axis=1)
        TOTAL_DATA.columns.names=["Type", "Metric", "Ticker"]
        TOTAL_DATA=TOTAL_DATA.reorder_levels([1, 0, 2], axis=1)
        print(TOTAL_DATA["Close"].head())
        print("Finished formatting data.")

        print("Cleaning up data utilizing linear interpolation, forward and backward fill...")
        CLEANED_DATA=TOTAL_DATA.interpolate(method="linear", limit_direction="both", limit=1).ffill().bfill()
        print(CLEANED_DATA["Close"].head())
        print("Finished cleaning up the data.")
        print("Final NA count: ", CLEANED_DATA.isnull().sum().sum())

        print("Generating .csv file to store data at bin folder named: total_data.csv")
        CLEANED_DATA.to_csv(Path("bin") / 'total_data.csv', index=True)
        print("Finished creating file.")
    else:
        print("Did not redownload.")

# pd.read_csv('bin/input_data.csv', header=[0, 1, 2], index_col=0, parse_dates=True) for reading csv with multiple headers.
