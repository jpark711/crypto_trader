# -*- coding: utf-8 -*-

import asyncio
import ccxt.pro


async def loop(exchange, symbol, n, i):
    try:
        orderbook = await exchange.watch_order_book(symbol)
        now = exchange.milliseconds()
        print(symbol)
        print(orderbook)
        print(n)
        print(i)
        print(exchange.iso8601(now))
        # i = how many updates there were in total
        # n = the number of the pair to count subscriptions
        # print(
        #     exchange.iso8601(now),
        #     n,
        #     symbol,
        #     i,
        #     orderbook["asks"][0],
        #     orderbook["bids"][0],
        # )
    except Exception as e:
        print(str(e))
        # raise e  # uncomment to break all loops in case of an error in any one of them
        # break  # you can also break just this one loop if it fails


async def main():
    i = 0
    while True:
        try:
            exchange = ccxt.pro.bithumb()
            await exchange.load_markets()
            markets = list(exchange.markets.values())
            # symbols = [market["symbol"] for market in markets if not market["darkpool"]]
            symbols = [market["symbol"] for market in markets]
            await asyncio.gather(
                *[loop(exchange, symbol, n, i) for n, symbol in enumerate(symbols)]
            )
            i += 1
            await asyncio.sleep(1)
        except Exception as e:
            print(str(e))
            await exchange.close()


asyncio.run(main())
