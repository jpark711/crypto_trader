import ccxt.pro
import numpy as np


class kucoin(ccxt.pro.kucoin):
    def __init__(self):
        """
        Initializes a new instance of the class.

        This method initializes the `tickers` attribute of the class.
        The `tickers` attribute is set to None to indicate that the tickers have not been fetched yet.

        Parameters:
            None

        Returns:
            None
        """
        super().__init__()
        self.tickers = None

    def fetch_bid_ask(self, use_cached=False):
        """
        Fetches the bid and ask prices and quantities for each ticker in the `order_books` dictionary.

        Args:
            use_cached (bool, optional): Whether to reuse the cached order books data. Defaults to False.

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
        #  Reuse the cached order books data to obtain the bid ask data if instructed
        if use_cached and (self.tickers is not None):
            ticker_data = self.tickers
        else:
            ticker_data = self.fetch_tickers()
        bid_ask = {}
        for ticker, ticker_info in ticker_data.items():
            bid_ask_list = [
                ticker_info["bid"],
                ticker_info["bidVolume"],
                ticker_info["ask"],
                ticker_info["askVolume"],
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
