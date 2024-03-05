import os
import sys
import time
import logging
import yfinance as yf


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    start = "2022-01-01"
    end = "2024-01-01"
    interval = "1d"

    tickers = [
        "^SPX",
        "^NDX",
        "AMZN",
        "AAPL",
        "NFLX",
        "GOOG",
        "TSLA",
        "META",
        "NET",
        "NVDA",
        "TSM",
        "BABA",
        "SQ",
        "PLTR",
        "SHOP",
    ]

    for ticker in tickers:
        if os.path.exists(f"datasets/{ticker}_{interval}.csv"):
            continue

        logging.info(f"fetching ohlcv for {ticker}")
        df = (
            yf.Ticker(ticker)
            .history(interval=interval, start=start, end=end)
            .rename(
                columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            )
            .drop(["Dividends", "Stock Splits"], axis=1)
        )
        df.index.names = ["timestamp"]
        df.to_csv(f"datasets/{ticker}_{interval}.csv")
        time.sleep(1)
