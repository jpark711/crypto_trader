import requests
import ccxt.pro
import pandas as pd
import numpy as np
from ccxt.base.types import Currencies, OrderBook
from ccxt.abstract.bithumb import ImplicitAPI


class bithumb(ccxt.pro.bithumb, ImplicitAPI):
    def __init__(self):
        """
        Initializes a new instance of the class.

        This method initializes the `tickers`, `order_books`, and `quote_currencies` attributes of the class.

        Parameters:
            None

        Returns:
            None
        """
        super().__init__()
        self.tickers = None
        self.order_books = None
        self.info = self.describe()
        self.quote_currencies = list(self.info["options"]["quoteCurrencies"].keys())
        self.api_urls = {
            "order_books": "https://api.bithumb.com/public/orderbook/ALL_",
            "currencies": "https://api.bithumb.com/public/assetsstatus/multichain/ALL",
        }
        self.headers = {"accept": "application/json"}
        self.currency_cols = {
            "currency": "id",
            "deposit_status": "deposit",
            "withdrawal_status": "withdraw",
            "net_type": "network",
        }
        self.bool_cols = ["withdraw", "deposit"]
        self.replace_dict = {"1": True, "0": False}

    def fetch_bid_ask(self, use_cached=False):
        """
        Fetches the bid and ask prices and quantities for each ticker in the `order_books` dictionary.

        Args:
            use_cached (bool, optional): Whether to reuse the cached order books data. Currently not used for Bithumb

        Returns:
            dict: A dictionary containing the bid and ask prices, quantities, and values for each ticker.
                The keys are the ticker symbols in the format "base_currency/quote_currency".
                The values are dictionaries with the following keys:
                    - "bid_price": The bid price.
                    - "bid_quantity": The bid quantity.
                    - "bid_value": The bid value (price * quantity).
                    - "ask_price": The ask price.
                    - "ask_quantity": The ask quantity.
                    - "ask_value": The ask value (price * quantity).
        """
        order_books = self.fetch_order_books()
        bid_ask = {}
        for ticker, order_book in order_books.items():
            bid_ask_list = [
                order_book["bids"][0]["price"],
                order_book["bids"][0]["quantity"],
                order_book["asks"][0]["price"],
                order_book["asks"][0]["quantity"],
            ]
            bid_ask_floats = [np.nan if x is None else float(x) for x in bid_ask_list]
            bid_ask[ticker] = {
                "bid_price": bid_ask_floats[0],
                "bid_quantity": bid_ask_floats[1],
                "bid_value": bid_ask_floats[0] * bid_ask_floats[1],
                "ask_price": bid_ask_floats[2],
                "ask_quantity": bid_ask_floats[3],
                "ask_value": bid_ask_floats[2] * bid_ask_floats[3],
            }
        return bid_ask

    def fetch_order_books(self, quote_currencies=None):
        """
        Fetches the order books for the specified quote currencies.

        Args:
            quote_currencies (Optional[List[str] | str]): The quote currencies for which to fetch the order books.
                If not provided, the default quote currencies will be used.

        Raises:
            AssertionError: If `quote_currencies` is an empty list or contains invalid values.
            AssertionError: If `quote_currencies` is a string that is not supported.

        Returns:
            Dict[str, Dict[str, List[List[float]]]]: A dictionary containing the order books for each ticker.
                The keys are the tickers in the format "base_currency/quote_currency".
                The values are dictionaries with the keys "bids" and "asks", which are lists of lists representing
                the bid and ask prices and quantities respectively.
        """
        # Validate `quote_currencies` input
        if quote_currencies is None:
            quote_currencies = self.quote_currencies
        elif type(quote_currencies) is list:
            assert (
                len(quote_currencies) > 0
            ), "`quote_currencies` cannot be an empty list"
            assert all(
                x in self.quote_currencies for x in quote_currencies
            ), "`quote_currencies` contains invalid values"
        elif type(quote_currencies) is str:
            assert (
                quote_currencies in self.quote_currencies
            ), "`quote_currencies` is not supported"

        # Download order books for `quote_currencies`
        order_books = {}
        for quote_currency in quote_currencies:
            url = self.api_urls["order_books"] + quote_currency
            response = requests.get(url, headers=self.headers)
            response_dict = response.json()
            response_data = response_dict["data"]
            quote = response_data.pop("payment_currency")
            del response_data["timestamp"]
            for order_book in response_data.values():
                ticker = order_book["order_currency"] + "/" + quote
                order_books[ticker] = {
                    "bids": order_book["bids"],
                    "asks": order_book["asks"],
                }
        return order_books

    async def fetch_order_book(
        self, symbol: str, limit: int = None, params: dict = {}, use_cached: bool = True
    ) -> OrderBook:
        """
        Fetches information on open orders with bid(buy) and ask(sell) prices, volumes, and other data.

        This function sends a GET request to the Bithumb API to retrieve the order book for a specified market.
        The API endpoint is: https://api.bithumb.com/public/orderbook/{baseId}_{quoteId}

        Parameters:
        - symbol (str): Unified symbol of the market to fetch the order book for.
        - limit (int, optional): The maximum amount of order book entries to return. Default is None, which means the API's default limit will be used.
        - params (dict, optional): Extra parameters specific to the exchange API endpoint. Default is an empty dictionary.
        - use_cached (bool, optional): Whether to use cached order book data. Default is True.

        Returns:
        - dict: A dictionary of `order book structures <https://docs.ccxt.com/#/?id=order-book-structure>` indexed by market symbols.
              The dictionary contains the following keys:
              - "bids": A list of bid prices and quantities.
              - "asks": A list of ask prices and quantities.
              - "timestamp": The timestamp of the order book data.
        """
        if use_cached and (self.order_books is not None):
            return self.order_books[symbol]
        else:
            await self.load_markets()
            market = self.market(symbol)
            request: dict = {
                "baseId": market["baseId"],
                "quoteId": market["quoteId"],
            }
            if limit is not None:
                request["count"] = limit  # default 30, max 30
            response = await self.publicGetOrderbookBaseIdQuoteId(
                self.extend(request, params)
            )
            #
            #     {
            #         "status":"0000",
            #         "data":{
            #             "timestamp":"1587621553942",
            #             "payment_currency":"KRW",
            #             "order_currency":"BTC",
            #             "bids":[
            #                 {"price":"8652000","quantity":"0.0043"},
            #                 {"price":"8651000","quantity":"0.0049"},
            #                 {"price":"8650000","quantity":"8.4791"},
            #             ],
            #             "asks":[
            #                 {"price":"8654000","quantity":"0.119"},
            #                 {"price":"8655000","quantity":"0.254"},
            #                 {"price":"8658000","quantity":"0.119"},
            #             ]
            #         }
            #     }
            #
            data = self.safe_value(response, "data", {})
            timestamp = self.safe_integer(data, "timestamp")
            return self.parse_order_book(
                data, symbol, timestamp, "bids", "asks", "price", "quantity"
            )

    async def fetch_currencies(self, params={}) -> Currencies:
        """
        Fetches the currency data from the Bithumb API.

        This function sends a GET request to the Bithumb API to retrieve the currency data.
        The API endpoint is: https://api.bithumb.com/public/assetsstatus/multichain/ALL

        Returns:
            dict: A dictionary containing the currency data. The keys are the currency IDs,
                  and the values are dictionaries with the following keys:
                  - "id" (str): The currency ID.
                  - "deposit" (bool): Indicates if the currency is available for deposit.
                  - "withdraw" (bool): Indicates if the currency is available for withdrawal.
        """
        # Fetch Bithumb currency data
        currency = "ALL"
        request: dict = {"currency": currency}
        response_dict = await self.public_get_assetsstatus_multichain_currency(
            self.extend(request, params)
        )
        currency_list = response_dict["data"]

        # Convert the data into a DataFrame and extract required fields
        currency_table = pd.DataFrame(currency_list)
        currency_table.rename(self.currency_cols, axis=1, inplace=True)
        currency_table[self.bool_cols] = (
            currency_table[self.bool_cols].apply(pd.to_numeric).astype(bool)
        )
        currency_table["active"] = (
            currency_table["deposit"] & currency_table["withdraw"]
        )
        bool_columns = self.bool_cols + ["active"]
        agg_currency_tbl = currency_table.groupby("id")[bool_columns].any()

        # Convert it back to dict
        agg_currency_tbl["id"] = agg_currency_tbl.index
        currencies = agg_currency_tbl.to_dict(orient="index")
        for ticker in currencies.keys():
            network_tbl = currency_table[currency_table.id == ticker]
            network_tbl.loc[:, "id"] = None
            network_tbl.set_index("network", inplace=True, drop=False)
            currencies[ticker]["networks"] = network_tbl.to_dict(orient="index")
        return currencies
