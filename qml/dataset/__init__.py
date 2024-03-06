import torch
import polars as pl
from datetime import datetime
from torch.utils.data import Dataset
from qml.dataset.downloader import download_yfinance
from typing import Optional


class YahooFinanceDataset(Dataset):
    r"""A custom implemention of `torch.utils.data.Dataset` for loading data from Yahoo Finance."""

    def __init__(
        self,
        tickers: list[str],
        start: datetime,
        end: datetime,
        interval: str,
        root: Optional[str],
    ):
        """
        Instantiate an instance of `YahooFinanceDataset`.

        Parameters
        ----------
        tickers
            Ticker(s) to load. For example, `^SPX` for Standard & Poor 500 or `AAPL` for Apple.
        start
            The datetime that marks the beginning of the historical data.
        end
            The datetime that marks the end of the historical data.
        interval
            The interval for the historical data, can be `1d`, `1h`, etc.
        root
            If specified, the dataset will be downloaded to the provided path.

        Returns
        -------
        YahooFinanceDataset
            A newly instantiated instance.
        """
        self.ticker_names = tickers
        self.tickers = download_yfinance(
            tickers=tickers, start=start, end=end, interval=interval, dir=root
        )

        # Drop unused columns and rename the 'close' column to the ticker name
        dfs = [
            df.drop(["open", "high", "low", "volume"]).rename({"close": ticker})
            for df, ticker in zip(self.tickers, tickers)
        ]

        # Join all the tickers to form one single dataframe
        self.data = dfs[0]
        for i in range(1, len(dfs)):
            self.data = self.data.join(dfs[i], on="timestamp")

    def __len__(self):
        return self.data.select(pl.len()).item()

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.from_numpy(self.data[index].to_numpy())
