# -*- coding: utf-8 -*-

import asyncio
import ccxt.async_support as ccxta  # noqa: E402
import time
import pandas as pd
import streamlit as st
import itertools
import numpy as np


class CryptoTrader:
    def __init__(self, exchanges):
        self.exchanges = exchanges
        self.fiats = ["USD", "KRW"]
        self.price_fields = ["last", "low", "high"]
        self.other_fields = ["volume"]
        self.all_fields = self.price_fields + self.other_fields
        self.reference_asset = "USDT"
        self.volume_threshold = 20000

    # Define the asynchronous function to fetch tickers
    async def fetch_tickers(self, exchange):
        client = getattr(ccxta, exchange)()
        tickers = await client.fetch_tickers()
        await client.close()
        return tickers

    # Define the function to fetch tickers from multiple exchanges
    async def multi_tickers(self):
        input_coroutines = [self.fetch_tickers(exchange) for exchange in self.exchanges]
        tickers = await asyncio.gather(*input_coroutines, return_exceptions=True)
        return tickers

    # Define the function to format and combine tickers
    def format_tickers(self, tickers):
        # Instatiate an empty dictionary of all fields as keys
        combined_tickers = {}
        for field in self.all_fields:
            combined_tickers[field] = {}

        # Gather the USDT prices
        for exchange, ticker_data in zip(self.exchanges, tickers):
            if isinstance(ticker_data, Exception):
                continue
            # Extract the quote currency list
            quote_list = []
            usdt_prices = {}
            avail_tickers = ticker_data.keys()
            for symbol, data in ticker_data.items():
                parts = symbol.split("/")
                if len(parts) == 2:
                    base_currency, quote_currency = parts
                    quote_list.append(quote_currency)
                    # Retrieve USDT prices for quote currencies
                    if base_currency == self.reference_asset:
                        usdt_prices[(exchange, quote_currency)] = 1 / data["last"]

            # Extract non-fiat quote currency prices using USDT as a base currency
            quote_set = set(quote_list)
            for quote_ticker in quote_set:
                usdt_quote = quote_ticker + "/" + self.reference_asset
                if quote_ticker not in self.fiats and usdt_quote not in avail_tickers:
                    for fiat in self.fiats:
                        quote_fiat_ticker = quote_ticker + "/" + fiat
                        if quote_fiat_ticker in avail_tickers:
                            quote_fiat_price = ticker_data[quote_fiat_ticker]["last"]
                            usdt_prices[(exchange, quote_ticker)] = (
                                quote_fiat_price * usdt_prices[(exchange, fiat)]
                            )

            # Convert all prices to USDT using the gathered USDT prices
            for symbol, data in ticker_data.items():
                if "/USDT" in symbol:
                    base_currency = symbol.replace("/" + self.reference_asset, "")
                    ticker_key = (base_currency, "USDT", exchange)
                    combined_tickers["last"][ticker_key] = data["last"]
                    for fld in self.price_fields:
                        combined_tickers[fld][ticker_key] = data[fld]
                else:
                    parts = symbol.split("/")
                    if len(parts) == 2:
                        base_currency, quote_currency = parts
                        if (exchange, quote_currency) in usdt_prices:
                            ticker_key = (base_currency, quote_currency, exchange)
                            for fld in self.price_fields:
                                if data[fld] is not None:
                                    combined_tickers[fld][ticker_key] = (
                                        data[fld]
                                        * usdt_prices[(exchange, quote_currency)]
                                    )
                # Populate trading volumes
                combined_tickers["volume"][ticker_key] = data["baseVolume"]

        # Convert the combined tickers dictionary to a pandas DataFrame
        formatted_tickers = {}
        for k, v in combined_tickers.items():
            formatted_tickers[k] = pd.DataFrame.from_dict(
                v, orient="index", columns=["Price"]
            )
            formatted_tickers[k].index = pd.MultiIndex.from_tuples(
                formatted_tickers[k].index, names=["Symbol", "Quote", "Exchange"]
            )
            formatted_tickers[k] = formatted_tickers[k].unstack(level=[2, 1])
            formatted_tickers[k].columns = formatted_tickers[k].columns.droplevel()
        return formatted_tickers

    # Define a function to calculate the arbitrage across the exchanges
    def calculate_arbitrage(self, ticker_data):
        # Generate all combinations of exchanges
        prices = ticker_data["last"]
        volumes = ticker_data["volume"]
        exchanges = prices.columns
        combinations = list(itertools.combinations(exchanges, 2))

        # Filter out tickers with low trading volumes
        for col in prices.columns:
            if col in volumes.columns:
                volumes[col] = prices[col] * volumes[col]
                small_vol_ind = volumes[col] < self.volume_threshold
                prices.loc[small_vol_ind, col] = np.nan

        # Compute price ratio
        price_ratio_dict = {}
        for combo in combinations:
            (ex1, ex2) = combo
            price_ratio_val = prices[ex1] / prices[ex2]
            price_ratio_dict[(ex1, ex2)] = price_ratio_val
            price_ratio_dict[(ex2, ex1)] = 1 / price_ratio_val
        price_ratio = pd.DataFrame(price_ratio_dict)
        price_ratio.dropna(how="all", inplace=True)

        # Store the maximum price ratio and the corresponding exchange pair
        max_price_ratio = pd.DataFrame(index=price_ratio.index)
        exchange_pair = price_ratio.idxmax(axis=1)
        max_price_ratio[("max_price_ratio", None)] = round(
            price_ratio.max(axis=1) - 1, 2
        )

        # Instatiate columns of long and short position fields
        col_names = ["exchange", "quote"] + self.all_fields
        positions = ["long", "short"]
        col_tuples = list(itertools.product(positions, col_names))
        max_price_ratio.loc[:, col_tuples] = None
        max_price_ratio.columns = pd.MultiIndex.from_tuples(
            max_price_ratio.columns, names=["Position", "Field"]
        )

        # For each price ratio, append prices of the coins, volumes, 24-hour low and highs
        exchange_flds = ["exchange", "quote"]
        low_high_vol = ["low", "high", "volume"]
        for coin_name in max_price_ratio.index:
            (ex1, ex2) = exchange_pair[coin_name]
            ex_dict = {"long": ex2, "short": ex1}
            for pos in positions:
                ex_key = ex_dict[pos]
                max_price_ratio.loc[coin_name, (pos, exchange_flds)] = ex_key
                max_price_ratio.loc[coin_name, (pos, "last")] = prices.loc[
                    coin_name, ex_key
                ]
                for key in low_high_vol:
                    field = ticker_data.get(key)
                    if (ex_key in field.columns) and (coin_name in field.index):
                        max_price_ratio.loc[coin_name, (pos, key)] = field.loc[
                            coin_name, ex_key
                        ]
        round_cols = low_high_vol + ["last"]
        for pos in positions:
            # Round dollar columns
            max_price_ratio.loc[:, (pos, round_cols)] = (
                max_price_ratio.loc[:, (pos, round_cols)].apply(pd.to_numeric).round(2)
            )

        # Sort values by maximum price ratio
        max_price_ratio.sort_values(
            by=("max_price_ratio", None), ascending=False, inplace=True
        )

        # Flatten columns for Streamlit Visulaization
        max_price_ratio.columns = max_price_ratio.columns.to_flat_index()
        return max_price_ratio

    # Define the asynchronous function to update streamlit objects
    async def update_streamlit(self):
        table_title = st.empty()
        table_title.write("### Prices in " + self.reference_asset)
        quote_table = st.empty()
        last_updated = st.empty()
        timer = st.empty()
        error_log = st.empty()
        while True:
            try:
                tic = time.time()
                tickers = await self.multi_tickers()
                ticker_data = self.format_tickers(tickers)
                price_ratios = self.calculate_arbitrage(ticker_data)
                async_time = time.time() - tic
                last_updated.write(
                    "Last updated: " + time.strftime("%H:%M:%S %B %d, %Y")
                )
                timer.write(f"Async call took {async_time:.2f} seconds")
                quote_table.dataframe(
                    price_ratios, use_container_width=True, height=900
                )
            except Exception as e:
                error_log.write(f"An error occurred: {e}")

    # Define the Streamlit app
    async def main(self):
        st.set_page_config(layout="wide")
        st.title("Cryptocurrency Market Data from Multiple Exchanges")
        await self.update_streamlit()


# Run the Streamlit app
if __name__ == "__main__":
    exchanges = ["binance", "kucoin", "bithumb", "gateio", "coinone"]
    crypto_trader = CryptoTrader(exchanges)
    asyncio.run(crypto_trader.main())
