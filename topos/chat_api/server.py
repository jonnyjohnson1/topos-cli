from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketState
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple
import uvicorn
import json
import asyncio
import datetime
from ..utilities.utils import generate_deci_code

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to the specific origins you want to allow
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SessionManager:
    def __init__(self):
        self.active_sessions: Dict[str, List[Tuple[str, WebSocket]]] = {}
        self.user_sessions: Dict[str, str] = {}
        self.usernames: Dict[str, str] = {}

    def add_session(self, session_id: str, user_id: str, websocket: WebSocket):
        print(f"[ adding {session_id} to active_sessions ]")
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = []
        self.active_sessions[session_id].append((user_id, websocket))

    def get_active_sessions(self):
        return self.active_sessions

    def get_user_sessions(self):
        return self.user_sessions

    def add_user_session(self, user_id: str, session_id: str):
        print(f"[ adding {user_id} to user_sessions ]")
        self.user_sessions[user_id] = session_id

    def add_username(self, user_id: str, username: str):
        print(f"[ adding {username} for {user_id} ]")
        self.usernames[user_id] = username

    def get_username(self, user_id: str) -> str:
        return self.usernames.get(user_id, "Unknown user")

session_manager = SessionManager()

async def send_message_to_client(client: WebSocket, message: dict):
    if not isinstance(message, dict):
        print("Message is not a dictionary")
        return

    if not client.application_state == WebSocketState.CONNECTED:
        print("Client is not connected")
        return

    try:
        await client.send_json(message)
    except Exception as e:
        print(e)

async def send_message_to_all(session_id: str, sender_user_id: str, message: dict, session_manager: SessionManager):
    active_sessions = session_manager.get_active_sessions()
    print("send_message_to_all")
    print(session_id in active_sessions)
    if message['message_type'] != 'server':
        print(f"[ message to user :: {message['content']['text']}]")
    if session_id in active_sessions:
        for user_id, client in active_sessions[session_id]:
            if message['message_type'] == 'server':
                await send_message_to_client(client, message)
            elif user_id != sender_user_id:
                await send_message_to_client(client, message)

async def send_to_all_clients_on_all_sessions(sender_user_id: str, message: dict, session_manager: SessionManager):
    active_sessions = session_manager.get_active_sessions()
    print("send_message_to_all")
    if message['message_type'] != 'server':
        print(f"[ message to user :: {message['content']['text']}]")
    for session_id in active_sessions:
        message["session_id"] = session_id
        for user_id, client in active_sessions[session_id]:
            if message['message_type'] == 'server':
                await send_message_to_client(client, message)
            elif user_id != sender_user_id:
                await send_message_to_client(client, message)

async def handle_client(websocket: WebSocket, session_manager: SessionManager, inactivity_event: asyncio.Event):
    await websocket.accept()
    print("client joined")
    try:
        while True:
            data = await asyncio.wait_for(websocket.receive_text(), timeout=600.0) # removes user if they haven't spoken in 10 minutes
            if data:
                payload = json.loads(data)
                inactivity_event.set()  # Reset the inactivity event
                print(payload)
                message_type = payload['message_type']
                print(message_type)
                active_sessions = session_manager.get_active_sessions()
                user_sessions = session_manager.get_user_sessions()

                if message_type == "create_server":
                    session_id = generate_deci_code(6)
                    print(f"[ client created chat :: session_id {session_id} ]")
                    user_id = payload['user_id']
                    host_name = payload['host_name']
                    username = payload['username']
                    session_manager.add_session(session_id, user_id, websocket)
                    session_manager.add_user_session(user_id, session_id)
                    session_manager.add_username(user_id, username)
                    print(session_manager.get_active_sessions()) # shows value
                    active_sessions = session_manager.get_active_sessions()

                    prompt_message = f"{host_name} created the chat"
                    data = {
                        "message_type": "server",
                        "session_id": session_id,
                        "message": prompt_message,
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                    await send_message_to_all(session_id, user_id, data, session_manager)

                elif message_type == "join_server":
                    session_id = payload['session_id']
                    user_id = payload['user_id']
                    username = payload['username']
                    active_sessions = session_manager.get_active_sessions()
                    print(session_id)
                    print("ACTIVE SESSIONS: ", session_manager.get_active_sessions())
                    print("ACTIVE SESSIONS: ", active_sessions) # shows empty when client connects
                    print(session_id in active_sessions)
                    if session_id in active_sessions:
                        print(f"[ {username} joined chat :: session_id {session_id} ]")
                        session_manager.add_session(session_id, user_id, websocket)
                        session_manager.add_user_session(user_id, session_id)
                        session_manager.add_username(user_id, username)
                        join_message = f"{username} joined the chat"
                        data = {
                            "message_type": "server",
                            "session_id": session_id,
                            "message": join_message,
                            "timestamp": datetime.datetime.now().isoformat()
                        }
                        await send_message_to_all(session_id, user_id, data, session_manager)
                    else:
                        await websocket.send_json({"error": "Invalid session ID"})
                break
        while True:
            data = await websocket.receive_text()
            if data:
                payload = json.loads(data)
                inactivity_event.set()  # Reset the inactivity event
                print("RECEIVED: ", payload)
                session_id = payload['content']['session_id']
                user_id = payload['content']['user_id']
                if session_id:
                    print(f"sending {session_id}")
                    await send_message_to_all(session_id, user_id, payload, session_manager)
            else:
                print(f"[ Message from client is empty ]")
    except WebSocketDisconnect:
        print("client disconnected")
        await handle_disconnect(websocket, session_manager)
    except asyncio.TimeoutError:
        print("client disconnected due to timeout")
        await handle_disconnect(websocket, session_manager)
    except Exception as e:
        print(f"client disconnected due to error: {e}")
        await handle_disconnect(websocket, session_manager)

async def handle_disconnect(websocket, session_manager):
    active_sessions = session_manager.get_active_sessions()
    user_sessions = session_manager.get_user_sessions()
    for session_id, clients in active_sessions.items():
        for user_id, client in clients:
            if client == websocket:
                clients.remove((user_id, client))
                if not clients:
                    del active_sessions[session_id]
                username = session_manager.get_username(user_id)
                disconnect_message = f"{username} left the chat"
                await asyncio.shield(send_message_to_all(session_id, user_id, {
                    "message_type": "server",
                    "session_id": session_id,
                    "message": disconnect_message,
                    "timestamp": datetime.datetime.now().isoformat()
                }, session_manager))
                break
    user_sessions.pop(user_id, None)
    session_manager.usernames.pop(user_id, None)

@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    print("[ client connected :: preparing setup ]")
    print(f" current connected sessions :: {session_manager.get_active_sessions()}")
    inactivity_event = asyncio.Event()
    # inactivity_task = asyncio.create_task(check_inactivity(inactivity_event)) # not applicable for local builds
    await handle_client(websocket, session_manager, inactivity_event)
    # inactivity_task.cancel()

async def check_inactivity(inactivity_event: asyncio.Event):
    while True:
        try:
            await asyncio.wait_for(inactivity_event.wait(), timeout=600.0)
            inactivity_event.clear()
        except asyncio.TimeoutError:
            print("No activity detected for 10 minutes, shutting down...")
            disconnect_message = f"Conserving power...shutting down..."
            await asyncio.shield(send_to_all_clients_on_all_sessions("senderUSErID#45",
                {
                "message_type": "server",
                "message": disconnect_message,
                "timestamp": datetime.datetime.now().isoformat()
            }, session_manager))
            asyncio.get_event_loop().stop()

# perform healthcheck for GCP requirement 
@app.get("/healthcheck/")
async def root():
    return {"message": "Status: OK"}

@app.post("/test")
async def test():
    return {"response": True}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)