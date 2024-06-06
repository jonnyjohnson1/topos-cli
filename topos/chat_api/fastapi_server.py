from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from concurrent.futures import ThreadPoolExecutor
from typing import List
import uvicorn

app = FastAPI()

active_clients: List[WebSocket] = []

executor = ThreadPoolExecutor(max_workers=10)

# function to send a message to a single client
async def send_message_to_client(client: WebSocket, message: str):
    await client.send_text(message)

# function to send any new message to all the clients
# currently connected to the server.
async def send_message_to_all(message: str):
    print(f"[ broadcasting message to {len(active_clients)} members]")
    for client in active_clients:
        await send_message_to_client(client, message)

# function to handle client connection and communication
async def handle_client(websocket: WebSocket):
    await websocket.accept()
    print("client joined")
    active_clients.append(websocket)
    try:
        while True:
            username = await websocket.receive_text()
            print(username)
            if username:
                # actions to take when user joins server
                prompt_message = f"SERVER~{username} joined the chat"
                await send_message_to_all(prompt_message)
                break

        while True:
            message = await websocket.receive_text()
            if message:
                final_msg = f"{username}~{message}"
                await send_message_to_all(final_msg)
            else:
                print(f"[ Message from client {username} is empty ]")
    except WebSocketDisconnect:
        active_clients.remove(websocket)
        disconnect_message = f"SERVER~{username} left the chat"
        await send_message_to_all(disconnect_message)
    except Exception as e:
        print(f"[ Error: {e} ]")

@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await handle_client(websocket)

# Run the app with multiple threads using uvicorn
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=13349, workers=4)


# print("\033[92mINFO:\033[0m     API docs available at: \033[1mhttp://127.0.0.1:13349/docs\033[0m")
# uvicorn.run(app, host="127.0.0.1", port=13349, workers=4)