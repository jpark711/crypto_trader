import websockets
import asyncio


async def hello():
    uri = "wss://pubwss.bithumb.com/pub/ws"

    async with websockets.connect(uri, ping_interval=None) as websocket:
        greeting = await websocket.recv()
        print(greeting)

        # 구독 요청
        data = '{"type":"ticker", "symbols": ["BTC_KRW"], "tickTypes": ["30M"]}'
        await websocket.send(data)

        print("after send data")

        while True:
            recv_data = await websocket.recv()
            print(recv_data)


asyncio.get_event_loop().run_until_complete(hello())
