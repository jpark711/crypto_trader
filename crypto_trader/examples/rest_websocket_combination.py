import streamlit as st
import ccxt.pro as ccxtpro
import asyncio

# Initialize exchanges
exchanges = {
    "gateio": ccxtpro.gateio(
        {
            "enableRateLimit": True,
        }
    ),
    "kucoin": ccxtpro.kucoin(
        {
            "enableRateLimit": True,
        }
    ),
}


# Define a generic function to get real-time order book data via WebSocket
async def fetch_order_book_ws(exchange, exchange_name, symbol):
    while True:
        try:
            order_book = await exchange.watch_order_book(symbol)
            st.session_state[f"{exchange_name}_order_book_{symbol}"] = order_book
        except Exception as e:
            print(f"WebSocket Error ({exchange_name} - {symbol}): {e}")
            await asyncio.sleep(5)  # Retry after a short delay


# Define a function to get currency data via async REST API
async def fetch_currency_rest(exchange):
    try:
        currency_data = await exchange.fetch_currencies()
        st.session_state["rest_currency"] = currency_data
        exchange.close()
        return currency_data
    except Exception as e:
        print(f"REST API Error: {e}")


# Start Streamlit app
async def main():
    # Symbols to watch
    symbols = {
        "gateio": ["BTC/USDT", "ETH/USDT"],
        "kucoin": ["BTC/USDT", "ETH/USDT"],
    }
    currency = "BTC"

    # Initialize session state
    containers = {}
    for exchange_name, exchange_symbols in symbols.items():
        containers[exchange_name] = {}
        for symbol in exchange_symbols:
            if f"{exchange_name}_order_book_{symbol}" not in st.session_state:
                st.session_state[f"{exchange_name}_order_book_{symbol}"] = {}
                containers[exchange_name][symbol] = st.empty()

    # Start WebSocket order book data fetching in the background for all exchanges and symbols
    for exchange_name, exchange_instance in exchanges.items():
        for symbol in symbols[exchange_name]:
            asyncio.create_task(
                fetch_order_book_ws(exchange_instance, exchange_name, symbol)
            )
            asyncio.create_task(fetch_currency_rest(exchange_instance))

    # wb_header = st.empty()
    # wb_header.write(f"### Real-time Ticker Data for {symbol}")
    # wb_text = st.empty()
    rest_header = st.empty()
    rest_header.write(f"### Currency Data for {currency}")
    # rest_text = st.empty()

    while True:
        # Fetch currency data via async REST API periodically
        # curr_dat = await fetch_currency_rest(currency)

        # Display real-time data in Streamlit
        # Update the data in the Streamlit containers
        for exchange_name, exchange_symbols in symbols.items():
            for symbol in exchange_symbols:
                order_book = st.session_state[f"{exchange_name}_order_book_{symbol}"]
                containers[exchange_name][symbol].subheader(
                    f"Real-time Order Book for {symbol} ({exchange_name})"
                )
                containers[exchange_name][symbol].write(order_book)

        # Refresh the Streamlit app periodically
        await asyncio.sleep(1)
        # st.rerun()


# Run the main function with asyncio
if __name__ == "__main__":
    asyncio.run(main())
