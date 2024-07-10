import asyncio
import websockets

HOST = "127.0.0.1"
PORT = 13394

async def listen_for_messages(uri):
    try:
        async with websockets.connect(uri) as websocket:
            print(f"[ Successfully connected to server {HOST} {PORT} ]")
            while True:
                try:
                    message = await websocket.recv()
                    print("MESSAGE RECEIVED:", message)
                except websockets.ConnectionClosed as e:
                    print(f"[ Connection closed while receiving: {e} ]")
                    break
                except Exception as e:
                    print(f"[ Error receiving message: {e} ]")
                    break
    except Exception as e:
        print(f"[ Unable to connect to host {HOST} and port {PORT} ]: {e}")

if __name__ == "__main__":
    uri = f"ws://{HOST}:{PORT}/ws/chat"
    asyncio.run(listen_for_messages(uri))
