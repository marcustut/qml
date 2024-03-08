import torch
import numpy as np
from datetime import datetime
from torch import Tensor
from torch.utils.data import Dataset
from qml.dataset.downloader import download_yfinance
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Optional


class GANData:
    def __init__(self, g_in: Tensor, g_out: Tensor, d_in: Tensor):
        self.g_in = g_in
        self.g_out = g_out
        self.d_in = d_in


class StockGANDataset(Dataset):
    r"""A custom implemention of `torch.utils.data.Dataset` for loading data from Yahoo Finance."""

    def __init__(
        self,
        tickers: list[str],
        target_ticker: str,
        start: datetime,
        end: datetime,
        interval: str,
        sliding_window: int,
        train_test_pct: tuple[float, float],
        root: Optional[str],
    ):
        """
        Instantiate an instance of `StockGANDataset`.

        Parameters
        ----------
        tickers
            Ticker(s) to load. For example, `^SPX` for Standard & Poor 500 or `AAPL` for Apple.
        target_ticker
            Ticker to predict. For example, put `^SPX` if you want to predict the price of `^SPX`.
            Note that the target_ticker must be a ticker present in `tickers` otherwise it will
            raise an Exception.
        start
            The datetime that marks the beginning of the historical data.
        end
            The datetime that marks the end of the historical data.
        interval
            The interval for the historical data, can be `1d`, `1h`, etc.
        sliding_window
            The window size for constructing the dataset. For example, if 3 is given and the
            interval is `1d` then the data would look like:

            ```
            [
                [day1, day2, day3],
                [day2, day3, day4],
                [day3, day4, day5],
            ]
            ```
        train_test_pct
            The split percentage of train and test data. Both of the number must add up to 1.0,
            otherwise an Exception is thrown.
        root
            If specified, the dataset will be downloaded to the provided path.

        Returns
        -------
        StockGANDataset
            A newly instantiated instance.
        """
        if target_ticker not in tickers:
            raise Exception(
                f"target_ticker ({target_ticker}) must be one of the ticker in tickers ({tickers})"
            )

        if train_test_pct[0] + train_test_pct[1] != 1.0:
            raise Exception(
                f"percentage in train_test_pct must add up to 1.0, received {train_test_pct}"
            )

        target_index = tickers.index(target_ticker)

        self.ticker_names = tickers
        self.tickers = download_yfinance(
            tickers=tickers, start=start, end=end, interval=interval, dir=root
        )

        # Check if all the ticker has the same start and end date
        ticker = self.tickers[0]
        for i in range(1, len(self.tickers)):
            if self.tickers[i]["timestamp"][0] != ticker["timestamp"][0]:
                raise Exception(
                    f"not all tickers has the same start date, {self.tickers[i]["timestamp"][0]} for {self.ticker_names[i]} and {ticker["timestamp"][0]} for {self.ticker_names[0]}"
                )

            if self.tickers[i]["timestamp"][-1] != ticker["timestamp"][-1]:
                raise Exception(
                    f"not all tickers has the same end date, {self.tickers[i]["timestamp"][0]} for {self.ticker_names[i]} and {ticker["timestamp"][0]} for {self.ticker_names[0]}"
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

        # Construct the tensors from dataframe
        data = self.data.drop(
            "timestamp"
        ).to_numpy()  # timestamp field is not needed for training

        # Split the dataset
        train_data, test_data = train_test_split(
            data,
            train_size=train_test_pct[0],
            test_size=train_test_pct[1],
            shuffle=False,  # must not shuffle time series data
        )
        train_in, train_out = (
            train_data,
            train_data.T[target_index].reshape(-1, 1),
        )
        test_in, test_out = (
            test_data,
            test_data.T[target_index].reshape(-1, 1),
        )

        self.in_scaler = MinMaxScaler(feature_range=(0, 1))
        self.out_scaler = MinMaxScaler(feature_range=(0, 1))

        # Construct the tensors using the sliding window
        g_in, g_out, d_in = StockGANDataset.extract_gan_features(
            in_data=self.in_scaler.fit_transform(train_in),
            out_data=self.out_scaler.fit_transform(train_out),
            window_size=sliding_window,
        )
        self.train = GANData(g_in=g_in, g_out=g_out, d_in=d_in)

        g_in, g_out, d_in = StockGANDataset.extract_gan_features(
            in_data=self.in_scaler.transform(test_in),
            out_data=self.out_scaler.transform(test_out),
            window_size=sliding_window,
        )
        self.test = GANData(g_in=g_in, g_out=g_out, d_in=d_in)

    def __len__(self):
        return self.train.g_in.shape[0]

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.train.g_in[index], self.train.d_in[index]

    @staticmethod
    def extract_gan_features(
        in_data: np.ndarray, out_data: np.ndarray, window_size: int
    ) -> tuple[Tensor, Tensor, Tensor]:
        assert in_data.shape[0] == out_data.shape[0]

        g_tensor_in = []
        g_tensor_out = []
        d_tensor_in = []

        for i in range(window_size, in_data.shape[0]):
            g_tensor_in.append(in_data[i - window_size : i, :])
            g_tensor_out.append(out_data[i])
            d_tensor_in.append(out_data[i - window_size : i + 1])

        # Make sure the length is correct (number of data)
        assert len(g_tensor_in) == len(g_tensor_out) == len(d_tensor_in)

        return (
            torch.from_numpy(np.array(g_tensor_in)).float(),
            torch.from_numpy(np.array(g_tensor_out)).float(),
            torch.from_numpy(np.array(d_tensor_in)).float(),
        )
