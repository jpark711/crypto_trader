#!/usr/bin/env python3
"""
Async Binance Spot Market OHLCV Downloader (Single CSV, Inline Config, Native REST + Feature Engineering)

Downloads specified-interval OHLCV bars for selected or all spot symbols on Binance
into one CSV (with a symbol column), using Binance's native REST API in parallel via asyncio + aiohttp.
Automatically computes returns, EWMA volatility, and multi-day returns so the output can feed directly into GenerateStockData.
Outputs a `Date` column (YYYY-MM-DD) instead of raw timestamp.

Configure the parameters below and run:
    python async_binance_downloader.py
"""

import os
import asyncio
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import aiohttp
from tqdm.asyncio import tqdm_asyncio  # pip install tqdm

#
# ── CONFIGURE HERE ─────────────────────────────────────────────────────────────
#
OUTPUT_FILE = "data/0_raw/binance_features_all.csv"  # path to output CSV file
INTERVAL = "1d"  # Binance interval: '1m','5m','1h','1d', etc.
START_DATE = None  # 'YYYY-MM-DD' or None for full history
TICKERS = None  # list of symbols e.g. ['BTCUSDT','ETHUSDT'], or None for all
CONCURRENCY = 5  # max number of symbols to fetch in parallel
#
# ────────────────────────────────────────────────────────────────────────────────
#

BASE = "https://api.binance.com"
KLINES_ENDPOINT = "/api/v3/klines"
MAX_LIMIT = 1000
HORIZONS = [5, 20, 60, 65, 180, 250, 260]
EWMA_SPAN = 20  # span for EWMA volatility


def interval_to_ms(interval: str) -> int:
    unit = interval[-1]
    num = int(interval[:-1])
    if unit == "m":
        return num * 60_000
    if unit == "h":
        return num * 3_600_000
    if unit == "d":
        return num * 86_400_000
    raise ValueError(f"Unsupported interval unit: {unit}")


class AsyncDownloader:
    def __init__(self, output_file, interval, start_date, tickers, concurrency):
        self.output_file = output_file
        self.interval = interval
        self.tickers = tickers
        self.concurrency = concurrency
        self.start_ms = 0
        if start_date:
            dt = datetime.fromisoformat(start_date)
            self.start_ms = int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)
            logging.info(f"Starting from {start_date} (ms: {self.start_ms})")
        else:
            logging.info("No START_DATE; pulling full history from epoch.")
        os.makedirs(os.path.dirname(self.output_file) or ".", exist_ok=True)

    async def get_all_symbols(self, session):
        url = BASE + "/api/v3/exchangeInfo"
        async with session.get(url) as resp:
            data = await resp.json()
        return [
            s["symbol"]
            for s in data["symbols"]
            if s["status"] == "TRADING" and s["isSpotTradingAllowed"]
        ]

    async def fetch_symbol(self, session, symbol):
        since = self.start_ms
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        all_bars = []
        while since < now_ms:
            params = {
                "symbol": symbol,
                "interval": self.interval,
                "startTime": since,
                "limit": MAX_LIMIT,
            }
            async with session.get(BASE + KLINES_ENDPOINT, params=params) as resp:
                bars = await resp.json()
            if not bars:
                break
            all_bars.extend(bars)
            since = bars[-1][0] + interval_to_ms(self.interval)

        if not all_bars:
            return None

        df = pd.DataFrame(
            all_bars,
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_asset_volume",
                "trades",
                "taker_buy_base",
                "taker_buy_quote",
                "ignore",
            ],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        # shift timestamp by 4 hours so daily bars align at 20:00:00 UTC reference time
        df["Date"] = (df["timestamp"] - pd.Timedelta(hours=4)).dt.strftime("%Y-%m-%d")

        df = df[["Date", "open", "high", "low", "close", "volume"]]
        df[["open", "high", "low", "close", "volume"]] = df[
            ["open", "high", "low", "close", "volume"]
        ].astype(float)

        df.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Vol",
            },
            inplace=True,
        )
        df.insert(0, "StockID", symbol)

        # compute features
        df["Ret"] = df["Close"].pct_change()
        df["EWMA_vol"] = (
            df["Ret"].ewm(span=EWMA_SPAN, adjust=False, min_periods=1).std().fillna(0)
        )
        for h in HORIZONS:
            df[f"Ret_{h}d"] = (1 + df["Ret"]).rolling(h).apply(np.prod, raw=True) - 1

        df.dropna(subset=["Ret"], inplace=True)
        return df.reset_index(drop=True)

    async def download_all(self):
        async with aiohttp.ClientSession() as session:
            symbols = self.tickers or await self.get_all_symbols(session)
            logging.info(
                f"Downloading {len(symbols)} symbols at interval {self.interval} (parallel {self.concurrency})"
            )
            sem = asyncio.Semaphore(self.concurrency)
            header_written = False

            async def sem_task(sym):
                async with sem:
                    return sym, await self.fetch_symbol(session, sym)

            tasks = [sem_task(s) for s in symbols]
            for fut in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
                sym, df = await fut
                if df is None:
                    logging.warning(f"{sym}: no data, skipping")
                    continue
                rows = len(df)
                first_date = df["Date"].iloc[0]
                last_date = df["Date"].iloc[-1]

                df.to_csv(
                    self.output_file,
                    mode="w" if not header_written else "a",
                    header=not header_written,
                    index=False,
                )
                header_written = True
                logging.info(f"✔ {sym}: {rows} rows ({first_date} → {last_date})")
        logging.info(f"Download complete: {self.output_file}")


async def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    dl = AsyncDownloader(OUTPUT_FILE, INTERVAL, START_DATE, TICKERS, CONCURRENCY)
    await dl.download_all()


if __name__ == "__main__":
    asyncio.run(main())
