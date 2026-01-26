#!/usr/bin/env python3
import yfinance
import sys

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
# | Index (Date) | Stock 1 | ... | Stock 500 | Index 1 | ... | Index 7 | Commodity 1 | ... | Commodity 5 | 
# | Date 0       |         |     |           |         |     |         |             |     |             |
# | Date 1       |         |     |           |         |     |         |             |     |             |
# |   .          |         |     |           |         |     |         |             |     |             |
# |   .          |         |     |           |         |     |         |             |     |             |
# | Date n       |         |     |           |         |     |         |             |     |             |
#
# With a stock lookup table as such:
# | Stock (Name) | Sectors | Market Capitalization (in Mil.) |
# | Stock 1      |         |                                 |
# |   .          |         |                                 |
# |   .          |         |                                 |
# | Stock 500    |         |                                 |
# ------------------------------------------------------------------------------------------------------------------

print(f"Running: {sys.argv[0]}")
TICKERS=sys.argv[1:]
print("Retrieved TICKERS.")

print("Setting up necessary variables...")
INDEXES=["^SPX", "^RUT", "^IXIC", "^DJI", "^VIX", "^N225", "^GDAXI"]
COMMODS=["CL=F", "NG=F", "GC=F", "SI=F", "ZC=F"]
print("Finished setting up necessary variables.")
