import asyncio
import threading
import websockets

HOST = "127.0.0.1"
PORT = 13349

def listen_for_messages_from_server(websocket):
    async def listen():
        async for message in websocket:
            if message:
                username, content = message.split("~", 1)
                print(f"[{username}] {content}")
            else:
                print("[ message from server is empty ]")

    asyncio.run(listen())

def send_message_to_server(websocket):
    async def send():
        while True:
            message = input("Message: ")
            if message:
                await websocket.send(message)
            else:
                print("[ empty message ]")
                await websocket.close()
                break

    asyncio.run(send())

def communicate_to_server(websocket):
    username = input("Enter username: ")
    if username:
        asyncio.run(websocket.send(username))
    else:
        print("[ username cannot be empty ]")
        asyncio.run(websocket.close())
        return

    listen_thread = threading.Thread(target=listen_for_messages_from_server, args=(websocket,))
    send_thread = threading.Thread(target=send_message_to_server, args=(websocket,))

    listen_thread.start()
    send_thread.start()

    listen_thread.join()
    send_thread.join()

async def main():
    uri = f"ws://{HOST}:{PORT}/ws/chat"
    try:
        async with websockets.connect(uri) as websocket:
            print(f"[ Successfully connected to server {HOST} {PORT} ]")
            communicate_to_server(websocket)
    except Exception as e:
        print(f"[ Unable to connect to host {HOST} and port {PORT} ]: {e}")

if __name__ == "__main__":
    asyncio.run(main())
