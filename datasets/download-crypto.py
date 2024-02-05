import sys
import time
import ccxt
import polars as pl
import logging
from datetime import datetime, timezone
from dataclasses import dataclass


@dataclass
class Datasource:
    exchange: ccxt.Exchange
    symbol: str
    timeframe: tuple[str, int]


def normalise_ohlcv(ohlcv: list[any]):
    return {
        "timestamp": ohlcv[0],
        "open": ohlcv[1],
        "high": ohlcv[2],
        "low": ohlcv[3],
        "close": ohlcv[4],
        "volume": ohlcv[5],
    }


def fetch_ohlcv(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: tuple[str, int],
    since: datetime,
    until: datetime,
) -> pl.DataFrame:
    since = int(datetime.timestamp(since) * 1000)
    until = int(datetime.timestamp(until) * 1000)
    limit = 1000

    logging.info(
        f"[fetch_ohlcv] {datasource.exchange.name} ({datasource.timeframe[0]}) {datasource.symbol}"
    )
    df = pl.DataFrame()

    while since < until:
        logging.info(
            f"[fetch_ohlcv] {datasource.exchange.name} ({datasource.timeframe[0]}) {datasource.symbol}"
        )
        time.sleep(exchange.rateLimit / 1000)
        ohlcvs = map(
            normalise_ohlcv,
            exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe[0],
                limit=limit,
                since=since,
                params={"exclude_current_candle": True},
            ),
        )
        df = pl.concat([df, pl.from_dicts(ohlcvs)])
        since += timeframe[1] * limit

    return df


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    bitstamp = ccxt.bitstamp()
    bybit = ccxt.bybit()

    since = datetime(year=2022, month=1, day=1, tzinfo=timezone.utc)
    until = datetime(year=2024, month=1, day=1, tzinfo=timezone.utc)
    timeframe = ("4h", 60 * 60 * 1000 * 4)
    datasources = [
        # Datasource(
        #     exchange=bitstamp,
        #     symbol="BTC/USD",
        #     timeframe=("1d", 60 * 60 * 1000 * 24),
        # )
        Datasource(exchange=bybit, symbol="BTC/USDT", timeframe=timeframe),
        Datasource(exchange=bybit, symbol="ETH/USDT", timeframe=timeframe),
        Datasource(exchange=bybit, symbol="ADA/USDT", timeframe=timeframe),
        Datasource(exchange=bybit, symbol="MATIC/USDT", timeframe=timeframe),
    ]

    for datasource in datasources:
        if not datasource.exchange.has["fetchOHLCV"]:
            logging.error("The exchange does not support fetching OHLCV")
            exit(1)

        ohlcv = fetch_ohlcv(
            exchange=datasource.exchange,
            symbol=datasource.symbol,
            since=since,
            until=until,
            timeframe=datasource.timeframe,
        )
        ohlcv.write_csv(
            f'{datasource.exchange.name.lower()}_{datasource.symbol.replace('/','')}_{datasource.timeframe[0]}.csv'
        )
