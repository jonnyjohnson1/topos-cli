import asyncio
import threading
import websockets
import json
import random
import string
import datetime

HOST = "https://6kppfbg5ti9u.share.zrok.io"
PORT = 13394

global username
global session_id

# Generate a random 8-digit session ID
def generate_id(length=8):
    return ''.join(random.choices(string.digits, k=length))

user_id = generate_id()

async def listen_for_messages_from_server(websocket):
    try:
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
        print(f"[ Error in listening for messages: {e} ]")

async def send_message_to_server(websocket):
    try:
        while True:
            message = input("Message: ")
            if message:
                try:
                    msg = {
                        'message_id': generate_id(),
                        'message_type': 'user',
                        'num_participants': 1,
                        'content': {
                            'user_id': user_id,
                            'session_id': session_id,
                            'username': username,
                            'text': message
                        },
                        'timestamp': datetime.datetime.now().isoformat(),
                        'metadata': {
                            'priority': '',
                            'tags': []
                        },
                        'attachments': [{'file_name': None, 'file_type': None, 'url': None}]
                    }
                    await websocket.send(json.dumps(msg))
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
        global username, session_id
        username = input("Enter username: ")
        session_id = input("Enter sessionId: ")
        if username:
            try:
                msg = {
                    'message_type': 'join_server',
                    'user_id': user_id,
                    'session_id': session_id,
                    'created_at': datetime.datetime.now().isoformat(),
                    'username': username
                }
                await websocket.send(json.dumps(msg))
                print(f"USERNAME SENT: {username}")
            except Exception as e:
                print(f"[ Error sending username: {e} ]")
        else:
            print("[ username cannot be empty ]")
            await websocket.close()
            return

        # listen_task = asyncio.create_task(listen_for_messages_from_server(websocket))
        send_task = asyncio.create_task(send_message_to_server(websocket))

        await asyncio.gather(send_task) #listen_task, 
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
        async with websockets.connect(uri, ping_interval=20, ping_timeout=200) as websocket:
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
