import time
import logging
import yfinance as yf
import polars as pl
from datetime import datetime
from typing import Optional


def download_yfinance(
    tickers: list[str],
    start: datetime,
    end: datetime,
    interval: str,
    dir: Optional[str],
) -> list[pl.DataFrame]:
    """
    Download historical data from Yahoo Finance.
    """
    date_format = "%Y-%m-%d"

    dfs = []

    for ticker in tickers:
        logging.info(
            f"fetching ohlcv for {ticker} from {start} to {end} on {interval} interval"
        )
        df = (
            yf.Ticker(ticker)
            .history(
                interval=interval,
                start=start.strftime(date_format),
                end=end.strftime(date_format),
            )
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

        if dir is not None:
            df.to_csv(f"{dir}/{ticker}_{interval}.csv")

        dfs.append(pl.from_pandas(df, include_index=True))
        time.sleep(0.2)

    return dfs
