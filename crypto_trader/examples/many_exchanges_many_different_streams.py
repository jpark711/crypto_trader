# -*- coding: utf-8 -*-

import ccxt.pro
from asyncio import gather, run


async def symbol_loop(exchange, method, symbol):
    print("Starting", exchange.id, method, symbol)
    while True:
        try:
            response = await getattr(exchange, method)(symbol)
            now = exchange.milliseconds()
            iso8601 = exchange.iso8601(now)
            if method == "watchOrderBook":
                print(
                    iso8601,
                    exchange.id,
                    method,
                    symbol,
                    response["asks"][0],
                    response["bids"][0],
                )
            elif method == "watchTicker":
                print(
                    iso8601,
                    exchange.id,
                    method,
                    symbol,
                    response["high"],
                    response["low"],
                    response["bid"],
                    response["ask"],
                )
            elif method == "watchTrades":
                print(iso8601, exchange.id, method, symbol, len(response), "trades")

        except Exception as e:
            print(str(e))
            # raise e  # uncomment to break all loops in case of an error in any one of them
            break  # you can break just this one loop if it fails


async def method_loop(exchange, method, symbols):
    print("Starting", exchange.id, method, symbols)
    loops = [symbol_loop(exchange, method, symbol) for symbol in symbols]
    await gather(*loops)


async def exchange_loop(exchange_id, methods):
    print("Starting", exchange_id, methods)
    exchange = getattr(ccxt.pro, exchange_id)()
    loops = [
        method_loop(exchange, method, symbols) for method, symbols in methods.items()
    ]
    await gather(*loops)
    await exchange.close()


async def main():
    exchanges = {
        "gateio": {
            "watchOrderBook": ["BTC/USDT", "ETH/BTC", "ETH/USDT"],
            "watchTicker": ["BTC/USDT"],
        },
        "kucoin": {
            "watchOrderBook": ["BTC/USDT", "ETH/USDT"],
            "watchTrades": ["ETH/USDT"],
        },
    }
    loops = [
        exchange_loop(exchange_id, methods)
        for exchange_id, methods in exchanges.items()
    ]
    await gather(*loops)


run(main())
