# -*- coding: utf-8 -*-

import asyncio
import time
import pandas as pd
import streamlit as st
import itertools
import numpy as np
import exchange as ex
from config.config import Config
from datetime import datetime


class CryptoTrader:
    def __init__(self):
        """
        Initializes a new instance of the CryptoTrader class.

        The constructor loads the configuration data and parameters and
        instantiates the exchange clients and loads their trading fees.

        Attributes:
            params (dict): The configuration parameters.
            cred (dict): The configuration credentials.
            exchange_names (list): The names of the exchanges.
            exchanges (dict): The exchange client instances.
            trading_fees (dict): The trading fees for each exchange.
        """
        pass
        # Load config data and parameters
        config = Config()
        self.params = config.params
        self.cred = config.credentials
        self.exchange_names = self.params["exchanges"]

        # Instantiate the exchange clients and load trading fees
        self.exchanges = {}
        self.trading_fees = {}
        for exc in self.exchange_names:
            exc_client = getattr(ex, exc)
            self.exchanges[exc] = exc_client()
            exc_info = self.exchanges[exc].describe()
            self.trading_fees[exc] = exc_info["fees"]["trading"]

    # Define the asynchronous function to fetch tickers
    async def fetch_data(
        self, exchange_name, exchange_instance, method, symbol=None, **kwargs
    ):
        """
        Fetches data from a given exchange using a given method and
        (optionally) a given symbol.

        If the symbol is None, the method is called with the given keyword
        arguments and the result is stored in the session state under the
        given method name with the given exchange name as the key.

        If the symbol is not None, the method is called with the given keyword
        arguments and the symbol and the result is stored in the session state
        under the given method name with the given exchange name as the key
        and the given symbol as a sub-key.

        If an exception occurs, the error is printed and the function sleeps
        for 5 seconds before retrying.

        Args:
            exchange_name (str): The name of the exchange to fetch data from.
            exchange_instance (ccxt.pro.Exchange): The instance of the exchange to
                fetch data from.
            method (str): The method to call to fetch the data.
            symbol (str, optional): The symbol to fetch data for. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the method.

        Returns:
            None
        """
        if symbol is None:
            while True:
                try:
                    data = await getattr(exchange_instance, method)(**kwargs)
                    st.session_state[method][exchange_name] = data
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    st.session_state["timestamp"][method][exchange_name] = timestamp
                except Exception as e:
                    print(f"[Error] fetch_data - {method} from {exchange_name}: {e}")
                    await asyncio.sleep(5)
        else:
            while True:
                try:
                    data = await getattr(exchange_instance, method)(
                        symbol=symbol, **kwargs
                    )
                    st.session_state[method][exchange_name][symbol] = data
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    st.session_state["timestamp"][method][exchange_name] = timestamp
                except Exception as e:
                    print(f"[Error] fetch_data - {method} from {exchange_name}: {e}")
                    await asyncio.sleep(5)

    # Define the asynchronous function to fetch all symbols data
    async def fetch_all_symbols_data(
        self, exchange_name, exchange_instance, method, **kwargs
    ):
        """
        Fetches the data for all symbols from an exchange.

        Args:
            exchange_name (str): The name of the exchange to fetch data from.
            exchange_instance (ccxt.pro.Exchange): The instance of the exchange to
                fetch data from.
            method (str): The method to call to fetch the data.
            **kwargs: Additional keyword arguments to pass to the method.

        Returns:
            None
        """
        try:
            # Instatiate a session state by exchange
            if exchange_name not in st.session_state[method]:
                st.session_state[method][exchange_name] = {}
                st.session_state["timestamp"][method][exchange_name] = {}

            # Extract all symbols
            all_symbols = []
            for market_data in self.markets[exchange_name].values():
                all_symbols.append(market_data["symbol"])

            # Fetch data for all symbols
            all_symbols_coroutines = []
            for symbol in all_symbols:
                coroutine = self.fetch_data(
                    exchange_name, exchange_instance, method, symbol=symbol, **kwargs
                )
                all_symbols_coroutines.append(coroutine)
            await asyncio.gather(*all_symbols_coroutines)
        except Exception as e:
            print(
                f"[Error] fetch_all_symbols_data - {method} from {exchange_name}: {e}"
            )

    # Define the function to fetch bid ask data of multiple exchanges
    async def fetch_bid_ask(self, tickers):
        """
        Asynchronously fetches the bid and ask prices and quantities for multiple
        exchanges.

        This function iterates over the provided tickers, sets the tickers for each
        exchange object, and retrieves the bid and ask data using the cached order
        books.

        Args:
            tickers (dict): A dictionary where keys are exchange names and values
                are the ticker data for the respective exchanges.

        Returns:
            dict: A dictionary where keys are exchange names and values are
                dictionaries containing bid and ask prices, quantities, and values
                for each ticker.
        """

        bid_ask_data = {}
        for exc, ticker_dat in tickers.items():
            # Instantiate an exchange object
            self.exchanges[exc].tickers = ticker_dat
            # Fetch bid ask data
            bid_ask_dict = self.exchanges[exc].fetch_bid_ask(use_cached=True)
            bid_ask_data[exc] = bid_ask_dict
        return bid_ask_data

    #  Define a function to fetch an order book
    async def fetch_order_book(self, position, exchange, bid_ask_type, bid_ask_data):
        # Revert the ticker exceptions to get correct order book data
        """
        Asynchronously fetches and processes order book data for a given market position.

        This function adjusts the ticker information based on exceptions, retrieves
        order book data from an exchange, converts it into a pandas DataFrame, and
        adjusts prices and values relative to a reference currency. It also appends
        the current timestamp to the output.

        Args:
            position (str): The market position identifier used to determine the
                correct ticker.
            exchange (ccxt.pro.Exchange): The exchange instance used to fetch the
                order book data.
            bid_ask_type (str): The type of order book data to fetch ("bid" or "ask").
            bid_ask_data (pd.DataFrame): DataFrame containing exchange rate and other
                relevant data for adjusting the order book values.

        Returns:
            dict: A dictionary with the processed order book data as a pandas DataFrame
                under the "data" key and the timestamp of retrieval under the
                "timestamp" key.
        """

        ticker_map = self.params["tickers"]["reverse_map"]
        if position in ticker_map.keys():
            query_position = ticker_map[position]
        else:
            query_position = position

        # Fetch order book data
        ticker = query_position[2] + "/" + query_position[1]
        length_dict = self.params["order_books"]["length"]
        order_book_len = length_dict[position[0]]
        order_book = await exchange.fetch_order_book(
            symbol=ticker, limit=order_book_len
        )

        # Convert the data into DataFrame
        order_df_obj = pd.DataFrame(
            order_book[bid_ask_type + "s"], columns=["price", "quantity"]
        )
        order_df = order_df_obj.apply(pd.to_numeric)

        # Adjust price and value relative to the reference currency
        exc_rate = bid_ask_data.loc[position, bid_ask_type + "_exchange_rate"]
        order_df["price"] = exc_rate * order_df["price"]
        order_df["value"] = order_df["price"] * order_df["quantity"]

        # Retrieve timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        output_dict = {"data": order_df, "timestamp": timestamp}
        return output_dict

    # Define a function to format order book data
    async def format_order_book(self, arb_table, bid_ask_data):
        # Filter out the top positions based on the arbitrage gain filter
        """
        Asynchronously fetches and processes order book data for the top positions in
        the arbitrage table.

        This function filters out the top positions based on the arbitrage gain filter,
        and then fetches order book data for each of the bid and ask positions of the
        top positions. It then converts the order book data into DataFrames, adjusts
        the prices and values relative to a reference currency, and appends the current
        timestamp to the output.

        Args:
            arb_table (pd.DataFrame): The arbitrage table containing the positions to
                consider.
            bid_ask_data (pd.DataFrame): DataFrame containing exchange rate and other
                relevant data for adjusting the order book values.

        Returns:
            tuple: A tuple containing the processed order book data as a dictionary and
                the number of positions considered.
        """
        if self.params["arbitrage"]["gain_filter"] is None:
            arb_filter = len(arb_table)
        else:
            arb_filter = min(self.params["arbitrage"]["gain_filter"], len(arb_table))
        order_books = {}
        timestamps = {}
        for row_ind in range(arb_filter):
            # Fetch order books for bid and ask positions
            for hdr in self.params["order_books"]["header"]:
                pos = hdr + "_position"
                order_book_pos = arb_table.loc[row_ind, pos]
                # Fetch order book data
                this_exc = self.exchanges[order_book_pos[0]]
                order_book_key = str(row_ind) + "_" + hdr
                order_books[order_book_key] = self.fetch_order_book(
                    position=order_book_pos,
                    exchange=this_exc,
                    bid_ask_type=hdr,
                    bid_ask_data=bid_ask_data,
                )
        # Gather the tasks
        order_book_values = await asyncio.gather(*order_books.values())
        order_book_dict = dict(zip(order_books.keys(), order_book_values))
        return order_book_dict, arb_filter

    # Define the function to fetch data from multiple exchanges
    async def fetch_multi_exchanges(self, method, with_symbols: bool, **kwargs):
        """
        Fetches data from multiple exchanges using the provided method.

        Args:
            method (str): The method to use for fetching data from each exchange.
            with_symbols (bool): Whether to fetch data for all symbols available on
                each exchange using `fetch_all_symbols_data` or just the default
                data for each exchange using `fetch_data`.
            **kwargs: Additional keyword arguments to pass to the chosen method.

        Returns:
            None
        """
        try:
            if method not in st.session_state:
                st.session_state[method] = {}
                st.session_state["timestamp"][method] = {}
            if with_symbols:
                input_coroutines = [
                    self.fetch_all_symbols_data(
                        exchange_name=exc_name,
                        exchange_instance=exc_instnace,
                        method=method,
                        **kwargs,
                    )
                    for exc_name, exc_instnace in self.exchanges.items()
                ]
            else:
                input_coroutines = [
                    self.fetch_data(
                        exchange_name=exc_name,
                        exchange_instance=exc_instnace,
                        method=method,
                        **kwargs,
                    )
                    for exc_name, exc_instnace in self.exchanges.items()
                ]
            await asyncio.gather(*input_coroutines, return_exceptions=True)
        except Exception as e:
            print(
                f"[Error] fetch_multi_exchanges - {method} for multiple exchanges: {e}"
            )

    # Define the function to format and combine tickers
    def format_tickers(self, tickers):
        # Instatiate an empty dictionary of all fields as keys
        """
        Formats and combines tickers data from multiple exchanges.

        Args:
            tickers (dict): A dictionary of tickers data from multiple exchanges.
                The keys are the exchange names, and the values are dictionaries of
                tickers data, where the keys are the ticker symbols and the values
                are dictionaries of ticker data.

        Returns:
            pandas.DataFrame: A DataFrame containing the formatted and combined
                tickers data. The index is a MultiIndex with the exchange name,
                quote currency, and ticker symbol. The columns are the fields of
                the tickers data, and the values are the corresponding data for
                each ticker.

        Raises:
            None
        """
        combined_tickers = {}
        for field in self.params["tickers"]["fields"]:
            combined_tickers[field] = {}

        # Gather the USDT prices
        for exchange, ticker_data in tickers.items():
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
                    if base_currency == self.params["tickers"]["ref"]:
                        usdt_prices[(exchange, quote_currency)] = 1 / data["last"]

            # Extract non-fiat quote currency prices using USDT as a base currency
            quote_set = set(quote_list)
            for quote_ticker in quote_set:
                usdt_quote = quote_ticker + "/" + self.params["tickers"]["ref"]
                if (
                    quote_ticker not in self.params["tickers"]["fiats"]
                    and usdt_quote not in avail_tickers
                ):
                    for fiat in self.params["tickers"]["fiats"]:
                        quote_fiat_ticker = quote_ticker + "/" + fiat
                        if quote_fiat_ticker in avail_tickers:
                            quote_fiat_price = ticker_data[quote_fiat_ticker]["last"]
                            usdt_prices[(exchange, quote_ticker)] = (
                                quote_fiat_price * usdt_prices[(exchange, fiat)]
                            )

            # Convert all prices to USDT using the gathered USDT prices
            for symbol, data in ticker_data.items():
                if "/USDT" in symbol:
                    base_currency = symbol.replace(
                        "/" + self.params["tickers"]["ref"], ""
                    )
                    ticker_key = (exchange, "USDT", base_currency)
                    combined_tickers["last"][ticker_key] = data["last"]
                    for fld in self.params["tickers"]["fields"]:
                        combined_tickers[fld][ticker_key] = data[fld]
                else:
                    parts = symbol.split("/")
                    if len(parts) == 2:
                        base_currency, quote_currency = parts
                        if (exchange, quote_currency) in usdt_prices:
                            ticker_key = (exchange, quote_currency, base_currency)
                            for fld in self.params["tickers"]["fields"]:
                                if data[fld] is not None:
                                    combined_tickers[fld][ticker_key] = (
                                        data[fld]
                                        * usdt_prices[(exchange, quote_currency)]
                                    )

        # Convert the combined tickers dictionary to a pandas DataFrame
        ticker_df_list = []
        for k, v in combined_tickers.items():
            ticker_tbl = pd.DataFrame.from_dict(
                combined_tickers[k], orient="index", columns=[k]
            )

            ticker_df_list.append(ticker_tbl)
        ticker_df = pd.concat(ticker_df_list, axis=1)
        ticker_df.index = pd.MultiIndex.from_tuples(
            ticker_df.index, names=["exchange", "quote", "ticker"]
        )
        # Rename tickers with exceptions
        ticker_data = self.rename_exceptions(df=ticker_df)
        ticker_data.rename(columns=self.params["tickers"]["column_map"], inplace=True)
        return ticker_data

    # Define a function to format currency data
    def format_currencies(self, currencies):
        """
        Aggregate currency data into a pandas DataFrame.

        Args:
            currencies (dict): The currency data to be formatted.

        Returns:
            pandas.DataFrame: The formatted currency data.

        Notes:
            The currency data is aggregated from the various exchanges and formatted into a single DataFrame.
            The DataFrame has the following columns:

                - ticker (str): The ticker symbol of the currency.
                - exchange (str): The exchange on which the currency is traded.
                - networks (list of str): The list of networks supported by the currency.
                - active (bool): Whether the currency is active or not.
                - deposit (bool): Whether the currency supports deposits.
                - withdraw (bool): Whether the currency supports withdrawals.
                - fee (float): The fee for the currency.
                - trading_fee (float): The trading fee for the currency.

            The DataFrame is indexed by the exchange and ticker symbol.

        """
        currency_list = []
        for exchange, currency_values in currencies.items():
            for ticker, ticker_data in currency_values.items():
                network_dict = ticker_data["networks"]
                if len(network_dict) > 0:
                    withdraw_network = []
                    deposit_network = []
                    for network, network_data in network_dict.items():
                        if network_data["active"] and network_data["withdraw"]:
                            withdraw_network.append(network)
                        if network_data["active"] and network_data["deposit"]:
                            deposit_network.append(network)
                    currency_values[ticker]["withdraw_networks"] = withdraw_network
                    currency_values[ticker]["deposit_networks"] = deposit_network
            currency_df = pd.DataFrame.from_dict(currency_values, orient="index")
            currency_df.loc[:, "exchange"] = exchange
            currency_df.dropna(axis=1, how="all", inplace=True)
            currency_list.append(currency_df)
        currency_table = pd.concat(currency_list)
        currency_table.rename({"id": "ticker"}, axis=1, inplace=True)

        # Filter the selected columns
        curr_data = currency_table[self.params["currencies"]["columns"]]
        curr_data.loc[:, "ticker"] = curr_data.loc[:, "ticker"].str.upper()
        curr_data.set_index(["exchange", "ticker"], inplace=True)

        # Rename exception tickers
        ticker_exceptions = {}
        for old_ind, new_ind in self.params["tickers"]["exceptions"].items():
            curr_old_ind = old_ind[:1] + old_ind[1 + 1 :]
            curr_new_ind = new_ind[:1] + new_ind[1 + 1 :]
            ticker_exceptions[curr_old_ind] = curr_new_ind
        currency_data = self.rename_exceptions(
            df=curr_data, ticker_exceptions=ticker_exceptions
        )
        return currency_data

    # Define the function to format bid ask data
    def format_bid_ask(self, bid_ask_data):
        """
        Aggregate bid ask data into a pandas DataFrame.

        Args:
            bid_ask_data (dict): The bid ask data to be formatted.

        Returns:
            pandas.DataFrame: The formatted bid ask data.

        Notes:
            The bid ask data is aggregated from the various exchanges and formatted into a single DataFrame.
            The DataFrame has the following columns:

                - ticker (str): The ticker symbol of the currency.
                - exchange (str): The exchange on which the currency is traded.
                - networks (list of str): The list of networks supported by the currency.
                - active (bool): Whether the currency is active or not.
                - deposit (bool): Whether the currency supports deposits.
                - withdraw (bool): Whether the currency supports withdrawals.
                - fee (float): The fee for the currency.
                - trading_fee (float): The trading fee for the currency.
                - bid_price (float): The bid price of the currency.
                - bid_quantity (float): The bid quantity of the currency.
                - bid_value (float): The bid value of the currency.
                - ask_price (float): The ask price of the currency.
                - ask_quantity (float): The ask quantity of the currency.
                - ask_value (float): The ask value of the currency.

            The DataFrame is indexed by the exchange and ticker symbol.

        """
        idx = pd.IndexSlice
        bid_ask_df_list = []
        for exc, bid_ask_dict in bid_ask_data.items():
            # Convert the data into DataFrame
            bid_ask_df = pd.DataFrame.from_dict(bid_ask_dict, orient="index")

            # Split the tickers by "/" and create a MultiIndex
            split_index = [ind.split("/")[::-1] for ind in bid_ask_df.index]
            multi_index = [[exc] + lst for lst in split_index]
            bid_ask_df.index = pd.MultiIndex.from_tuples(
                multi_index, names=self.params["order_books"]["levels"]
            )
            bid_ask_df.dropna(how="all", axis=1, inplace=True)
            quote_cols = [
                x + "_quote" for x in self.params["order_books"]["price_cols"]
            ]
            bid_ask_df.loc[:, quote_cols] = bid_ask_df.loc[
                :, self.params["order_books"]["price_cols"]
            ].values

            # Convert prices quoted in other currencies to the reference currency
            quote_currencies = list(bid_ask_df.index.get_level_values("quote").unique())
            if self.params["tickers"]["ref"] in quote_currencies:
                quote_currencies.remove(self.params["tickers"]["ref"])
                ref_curr_rows = idx[:, self.params["tickers"]["ref"], :]
                exc_cols = self.params["order_books"]["exc_cols"]
                bid_ask_df.loc[ref_curr_rows, exc_cols] = 1

            # Find conversion rate for each quoted currency
            for quote_curr in quote_currencies:
                conv_index = (exc, quote_curr, self.params["tickers"]["ref"])
                rev_conv_index = (exc, self.params["tickers"]["ref"], quote_curr)
                if conv_index in bid_ask_df.index:
                    conv_rate = bid_ask_df.loc[conv_index]
                elif rev_conv_index in bid_ask_df.index:
                    conv_rate = 1 / bid_ask_df.loc[rev_conv_index]
                else:
                    for fiat_quote in self.params["tickers"]["fiats"]:
                        fiat_quote_index = (exc, fiat_quote, quote_curr)
                        if fiat_quote_index in bid_ask_df.index:
                            fiat_quote_conv = bid_ask_df.loc[fiat_quote_index]
                            fiat_ref_index = (
                                exc,
                                fiat_quote,
                                self.params["tickers"]["ref"],
                            )
                            fiat_ref_conv = bid_ask_df.loc[fiat_ref_index]
                            conv_rate = fiat_ref_conv / fiat_quote_conv

                # Find bid ask index quoted in the target currency
                quote_curr_index = idx[:, quote_curr, :]

                # Convert the unit of the bid ask prices to the reference currency
                for hdr in self.params["order_books"]["header"]:
                    exchange_rate = 1 / conv_rate[hdr + "_price"]
                    nominal_col = self.params["order_books"][hdr + "_nominal"]
                    bid_ask_df.loc[quote_curr_index, nominal_col] = (
                        bid_ask_df * exchange_rate
                    )
                    exc_col = hdr + "_exchange_rate"
                    bid_ask_df.loc[quote_curr_index, exc_col] = exchange_rate
            bid_ask_df_list.append(bid_ask_df)

        # Concatenate the DataFrames into one
        bid_ask_tbl = pd.concat(bid_ask_df_list, axis=0)
        bid_ask_table = self.rename_exceptions(df=bid_ask_tbl)
        return bid_ask_table

    # Define a function to rename tickers with exceptions
    def rename_exceptions(self, df, ticker_exceptions=None):
        """
        Renames the tickers in the index of the DataFrame based on the exceptions defined in the "tickers" parameter.

        Args:
            df (pd.DataFrame): The DataFrame to be processed.
            ticker_exceptions (dict, optional): The dictionary of old ticker names to new ticker names. Defaults to None.

        Returns:
            pd.DataFrame: The DataFrame with the renamed tickers in the index.
        """
        if ticker_exceptions is None:
            ticker_exceptions = self.params["tickers"]["exceptions"]
        df_index = list(df.index)
        df_index_names = df.index.names
        for old_ind, new_ind in ticker_exceptions.items():
            df_index[df_index.index(old_ind)] = new_ind
        df.index = pd.MultiIndex.from_tuples(df_index, names=df_index_names)
        return df

    # Define a function to calculate the arbitrage across the exchanges
    def calculate_arbitrage(self, ticker_data, currency_data, bid_ask_data):
        """
        Calculate the arbitrage opportunities across the exchanges.

        Args:
            ticker_data (pd.DataFrame): The DataFrame containing the ticker data.
            currency_data (pd.DataFrame): The DataFrame containing the currency data.
            bid_ask_data (pd.DataFrame): The DataFrame containing the bid ask data.

        Returns:
            pd.DataFrame: The DataFrame containing the arbitrage opportunities.

        Notes:
            The function filters out inactive tickers and those with low trading volume.
            It then calculates the best arbitrage opportunity for each ticker and
            appends additional data to the result.

        """
        # Filter out inactive tickers from the bid ask data
        merged_bid_ask = bid_ask_data.join(currency_data)
        avail_bid_ask = merged_bid_ask[~merged_bid_ask.active.isna()]

        # Filter out tickers below 24-hour volume threshold
        merged_bid_ask_vol = avail_bid_ask.join(ticker_data)
        master_data = merged_bid_ask_vol[
            merged_bid_ask_vol.volume > self.params["tickers"]["volume_threshold"]
        ]

        # For each unique ticker, find the best arbitrage opportunity
        unique_tickers = set(master_data.index.get_level_values("ticker"))
        bid_ask_ratio_list = []
        idx = pd.IndexSlice

        for ticker in unique_tickers:
            ticker_dat = master_data.loc[idx[:, :, ticker]].sort_index()
            trade_pairs = list(itertools.permutations(ticker_dat.index, 2))
            max_ratio = -np.inf
            bid_ask_ratio_dict = {}
            bid_ask_ratio_dict["ticker"] = ticker
            bid_ask_ratio_dict["arb_gain"] = None
            bid_ask_ratio_dict["order_size"] = None
            bid_ask_ratio_dict["bid_ask_spread"] = None
            bid_ask_ratio_dict["bid_timestamp"] = None
            bid_ask_ratio_dict["ask_timestamp"] = None
            for trade_pair in trade_pairs:
                (bid_exc, ask_exc) = trade_pair
                bid_data = ticker_dat.loc[bid_exc]
                ask_data = ticker_dat.loc[ask_exc]
                tradable = ask_data.withdraw and bid_data.deposit
                if not tradable:
                    continue
                bid_ask_ratio = bid_data["bid_price"] / ask_data["ask_price"]
                if bid_ask_ratio > max_ratio:
                    bid_trading_fee = self.trading_fees[bid_exc[0]]["taker"]
                    ask_trading_fee = self.trading_fees[ask_exc[0]]["taker"]
                    bid_ask_ratio_dict["bid_ask_spread"] = (
                        bid_ask_ratio - bid_trading_fee - ask_trading_fee
                    )
                    bid_ask_ratio_dict["bid_position"] = bid_exc + (ticker,)
                    bid_ask_ratio_dict["ask_position"] = ask_exc + (ticker,)
                    bid_ask_ratio_dict["bid_price"] = bid_data["bid_price"]
                    bid_ask_ratio_dict["ask_price"] = ask_data["ask_price"]
                    bid_ask_ratio_dict["bid_price_quote"] = bid_data["bid_price_quote"]
                    bid_ask_ratio_dict["ask_price_quote"] = ask_data["ask_price_quote"]
                    bid_ask_ratio_dict["bid_value"] = ticker_dat.loc[
                        bid_exc, "bid_value"
                    ]
                    bid_ask_ratio_dict["ask_value"] = ticker_dat.loc[
                        ask_exc, "ask_value"
                    ]
                    bid_ask_ratio_dict["bid_trading_fee"] = bid_trading_fee
                    bid_ask_ratio_dict["ask_trading_fee"] = ask_trading_fee
                    max_ratio = bid_ask_ratio
            if bid_ask_ratio_dict["bid_ask_spread"] is not None:
                bid_ask_ratio_list.append(bid_ask_ratio_dict)
        bid_ask_ratio = pd.DataFrame(bid_ask_ratio_list)
        bid_ask_ratio["bid_ask_spread"] = (
            (bid_ask_ratio["bid_ask_spread"] - 1) * 100
        ).round(2)
        arb_table = bid_ask_ratio.sort_values(
            by="bid_ask_spread", ascending=False
        ).set_index("ticker", drop=True)

        # Define columns of long and short positions
        ticker_cols = list(ticker_data.columns)
        additional_cols = (
            self.params["currencies"]["networks"]
            + ticker_cols
            + self.params["currencies"]["fields"]
        )
        positions = ["bid", "ask"]
        col_names = [pos + "_" + col for col in additional_cols for pos in positions]
        arb_table.loc[:, col_names] = None

        # For each price ratio, append additional data
        for coin_name in arb_table.index:
            coin_data = arb_table.loc[coin_name]
            bid_ask_dat = {
                "bid": master_data.loc[coin_data.bid_position, additional_cols],
                "ask": master_data.loc[coin_data.ask_position, additional_cols],
            }
            for k, v in bid_ask_dat.items():
                col_names = [k + "_" + col for col in additional_cols]
                arb_table.loc[coin_name, col_names] = v.values

        # Round columns in the reference currency
        for pos in positions:
            round_cols = [pos + "_" + col for col in ticker_cols]
            arb_table.loc[:, round_cols] = (
                arb_table.loc[:, round_cols].apply(pd.to_numeric).round(4)
            )

        # Drop unnecessary columns
        drop_cols = [
            pos + "_" + curr
            for pos in positions
            for curr in self.params["currencies"]["fields"]
        ] + self.params["currencies"]["drop_columns"]
        arb_table.drop(drop_cols, axis=1, inplace=True)
        arb_table.reset_index(inplace=True)
        pos_arb_table = arb_table.loc[arb_table.bid_ask_spread > 0]
        return pos_arb_table

    # Define a function to compute the nominal arbitrage gain
    async def calculate_arbitrage_gain(self, arb_table, order_books):
        """
        Computes the arbitrage gain for each row in the `arb_table` using the order book data in `order_books`.

        The function iterates over each row in `arb_table`, and for each row, it iterates over the matching
        order books in `order_books`. It accumulates the quantity of each order book that could be used to
        make a trade and the value of that trade. The value is the difference between the bid and ask prices
        multiplied by the quantity of the trade.

        The function also loads the timestamps of the order books into the `arb_table`.

        The function then rounds the arbitrage gain and sorts the table by the values.

        Args:
            arb_table (pd.DataFrame): A DataFrame containing the arbitrage opportunities.
            order_books (dict): A dictionary containing the order book data.

        Returns:
            pd.DataFrame: The DataFrame with the calculated arbitrage gains and order quantities.

        """
        # Compute arbitrage gains and order quantities
        for row_ind in range(self.arb_filter):
            bid_hdr = str(row_ind) + "_" + "bid"
            ask_hdr = str(row_ind) + "_" + "ask"
            quantity = 0
            value = 0
            ask_len = len(order_books[ask_hdr]["data"])
            bid_len = len(order_books[bid_hdr]["data"])
            ask_ind = 0
            bid_ind = 0
            ask_price = -np.inf
            bid_price = np.inf
            ask_quantity = 0
            bid_quantity = 0
            while (ask_ind < ask_len) and (bid_ind < bid_len):
                ask_data = order_books[ask_hdr]["data"].loc[ask_ind]
                bid_data = order_books[bid_hdr]["data"].loc[bid_ind]
                ask_price = ask_data["price"]
                bid_price = bid_data["price"]
                if ask_price >= bid_price:
                    break
                ask_quantity = ask_data["quantity"]
                bid_quantity = bid_data["quantity"]

                if ask_quantity <= bid_quantity:
                    quantity += ask_quantity
                    value += ask_quantity * (bid_price - ask_price)
                    bid_quantity -= ask_quantity
                    ask_ind += 1
                else:
                    quantity += bid_quantity
                    value += bid_quantity * (bid_price - ask_price)
                    ask_quantity -= bid_quantity
                    bid_ind += 1
                    if ask_ind == ask_len:
                        print("ask orderbook overload")
                        print(arb_table.loc[row_ind]["ask_position"])
                    if bid_ind == bid_len:
                        print("bid order book overload")
                        print(arb_table.loc[row_ind]["bid_position"])
            arb_table.loc[row_ind, "arb_gain"] = value
            arb_table.loc[row_ind, "order_size"] = quantity

            # Load timestamps
            arb_table.loc[row_ind, "bid_timestamp"] = order_books[bid_hdr]["timestamp"]
            arb_table.loc[row_ind, "ask_timestamp"] = order_books[ask_hdr]["timestamp"]

        # Round the arbitrage gain and sort the table by the values
        arb_table["arb_gain"] = pd.to_numeric(arb_table["arb_gain"]).round(0)
        arb_table.sort_values("arb_gain", ascending=False, inplace=True)

        return arb_table

    # Define the asynchronous function to update streamlit objects
    async def update_streamlit(self):
        """
        Asynchronously updates the Streamlit objects with the arbitrage data.

        This function runs an infinite loop that waits for the ticker and currency
        data to be loaded, fetches and formats the bid ask data, finds arbitrage
        trade opportunities, fetches and formats order books data, and computes the
        nominal arbitrage gain. It then updates the Streamlit objects with the
        computed data.

        Args:
            None

        Returns:
            None
        """
        # Instantiate the Streamlit objects
        table_title = st.empty()
        table_title.write("### Prices in " + self.params["tickers"]["ref"])
        quote_table = st.empty()
        timestamp_table = st.empty()
        last_updated = st.empty()
        timer = st.empty()
        error_log = st.empty()

        while True:
            try:
                error_log.write("")
                tic = time.time()
                # Wait for the ticker and currency data to be loaded
                for method in self.params["fetch_methods"].keys():
                    assert method in st.session_state, method + " is not loaded"
                    assert len(st.session_state[method]) == len(self.exchange_names), (
                        method + " is loading for all exchanges"
                    )
                # Load ticker and currency data
                tickers = st.session_state["fetch_tickers"]
                currencies = st.session_state["fetch_currencies"]
                # Format ticker and currency data
                ticker_data = self.format_tickers(tickers)
                currency_data = self.format_currencies(currencies)
                # Fetch and format bid ask data
                bid_ask_prices = await self.fetch_bid_ask(tickers)
                bid_ask_data = self.format_bid_ask(bid_ask_prices)
                # Find arbitrage trade opportunities
                arb_table = self.calculate_arbitrage(
                    ticker_data, currency_data, bid_ask_data
                )
                # Fetch and format order books data
                order_books, self.arb_filter = await self.format_order_book(
                    arb_table, bid_ask_data
                )
                # Compute nominal arbitrage gain
                arb_gain = await self.calculate_arbitrage_gain(arb_table, order_books)
                # Print the computation time
                async_time = time.time() - tic
                last_updated.write(
                    "Last updated: " + time.strftime("%H:%M:%S %B %d, %Y")
                )
                timer.write(f"Async call took {async_time:.2f} seconds")
                quote_table.dataframe(arb_gain, use_container_width=True, height=900)
                timestamp_df = pd.DataFrame.from_dict(st.session_state["timestamp"])
                timestamp_table.dataframe(timestamp_df)
            except AssertionError as e:
                # Print the error
                error_log.write(f"[Loading] {e}")
                # Pause for 1 second and try again
                await asyncio.sleep(1)

            except Exception as e:
                # Print the error
                error_log.write(f"[Error] {e}")
                # Pause for 1 second and try again
                await asyncio.sleep(1)

    # Define a function to close all exchange clients
    async def close_all_exchanges(self):
        """
        Close all exchange clients.

        This method is used to close all exchange clients when the Streamlit app is stopped.

        Args:
            None

        Returns:
            None
        """
        client_list = []
        for exc in self.exchange_names:
            client_list.append(self.exchanges[exc].close())
        await asyncio.gather(*client_list)

    # Define the Streamlit app
    async def main(self):
        """
        Build the Streamlit app and create async tasks for updating exchange data.

        This method is the entry point for the Streamlit app and is responsible for
        loading the markets for each exchange, creating the session states, and
        creating async tasks for updating exchange data.

        Args:
            None

        Returns:
            None
        """
        # Load markets of each exchange
        load_markets_coroutines = []
        for exchange_instance in self.exchanges.values():
            load_markets_coroutines.append(exchange_instance.load_markets())
        loaded_markets = await asyncio.gather(*load_markets_coroutines)
        self.markets = dict(zip(self.exchanges.keys(), loaded_markets))

        # Innstatiate the session states
        st.session_state["timestamp"] = {}

        # Create async tasks for updating exchange data
        for method, method_param in self.params["fetch_methods"].items():
            asyncio.create_task(
                self.fetch_multi_exchanges(
                    method=method, with_symbols=method_param["with_symbols"]
                )
            )

        # Build the Streamlit app
        st.set_page_config(layout="wide")
        st.title("Crypto Spacial Arbitrage Tracker")

        # Update the Streamlit application
        await self.update_streamlit()


# Run the Streamlit app
if __name__ == "__main__":
    # Instantiate the CryptoTrader object
    crypto_trader = CryptoTrader()
    asyncio.run(crypto_trader.main())
