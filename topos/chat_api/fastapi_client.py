# This code almost works. It conencts to the socket, and sends, but does not receive messages.

import asyncio
import threading
import websockets

HOST = "127.0.0.1"
PORT = 13349

async def listen_for_messages_from_server(websocket):
    try:
        while True:
            try:
                message = await websocket.recv()
                print("MESSAGE RECEIVED:", message)
                if message:
                    try:
                        username, content = message.split("~", 1)
                        print(f"[{username}] {content}")
                    except ValueError as e:
                        print(f"[ Error parsing message: {e} ]")
                else:
                    print("[ message from server is empty ]")
            except websockets.ConnectionClosed as e:
                print(f"[ Connection closed while receiving: {e} ]")
                break
            except Exception as e:
                print(f"[ Error receiving message: {e} ]")
                break
    except Exception as e:
        print(f"[ Error in listening for messages: {e} ]")

async def send_message_to_server(websocket):
    try:
        while True:
            message = input("Message: ")
            if message:
                try:
                    await websocket.send(message)
                    print("MESSAGE SENT:", message)
                except Exception as e:
                    print(f"[ Error sending message: {e} ]")
            else:
                print("[ empty message ]")
                await websocket.close()
                break
    except websockets.ConnectionClosed as e:
        print(f"[ Connection closed while sending: {e} ]")
    except Exception as e:
        print(f"[ Error in sending messages: {e} ]")

async def communicate_to_server(websocket):
    try:
        username = input("Enter username: ")
        if username:
            try:
                await websocket.send(username)
                print(f"USERNAME SENT: {username}")
            except Exception as e:
                print(f"[ Error sending username: {e} ]")
        else:
            print("[ username cannot be empty ]")
            await websocket.close()
            return

        listen_task = asyncio.create_task(listen_for_messages_from_server(websocket))
        send_task = asyncio.create_task(send_message_to_server(websocket))

        await asyncio.gather(listen_task, send_task)
    except Exception as e:
        print(f"[ Error in communication: {e} ]")

def start_event_loop(loop):
    try:
        asyncio.set_event_loop(loop)
        loop.run_forever()
    except Exception as e:
        print(f"[ Error in event loop: {e} ]")

async def main():
    uri = f"ws://{HOST}:{PORT}/ws/chat"
    try:
        async with websockets.connect(uri) as websocket:
            print(f"[ Successfully connected to server {HOST} {PORT} ]")
            await communicate_to_server(websocket)
    except websockets.ConnectionClosed as e:
        print(f"[ Connection closed: {e} ]")
    except Exception as e:
        print(f"[ Unable to connect to host {HOST} and port {PORT} ]: {e}")

if __name__ == "__main__":
    try:
        new_loop = asyncio.new_event_loop()
        t = threading.Thread(target=start_event_loop, args=(new_loop,))
        t.start()

        asyncio.run_coroutine_threadsafe(main(), new_loop)
    except Exception as e:
        print(f"[ Error starting client: {e} ]")
