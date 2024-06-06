from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from concurrent.futures import ThreadPoolExecutor
from typing import List
import uvicorn
import json

from ..utilities.utils import generate_hex_code

import datetime

app = FastAPI()

active_clients: List[WebSocket] = []

executor = ThreadPoolExecutor(max_workers=10)

""" Message JSON Schema

{ 
"message_id": "<unique_id>", 
"message_type": "<type>", // OPTIONS: user, ai, server 
“num_participants”: <int>,
"content": 
	{ 
	  "sender_id": "<id>",
	  "conversation_id": "<id>", 
	  "username": "<name>", 
	  "text": "<message_content>" 
	}, 
"timestamp": "<datetime>", 
"metadata": { 
	"priority": "<level>", // e.g., normal, high 
	"tags": ["<tag1>", "<tag2>"] 
	}, 
"attachments": [ 
	{ 
	  "file_name": "<file_name>", 
	  "file_type": "<type>", 
	  "url": "<file_url>" 
	} 
] }

"""

# function to send a message to a single client
async def send_message_to_client(client: WebSocket, message: dict):
    await client.send_json(message)

# function to send any new message to all the clients
# currently connected to the server.
async def send_message_to_all(message: dict):
    print(f"[ broadcasting message to {len(active_clients)} members]")
    for client in active_clients:
        # TODO send message to only the clients in the matching joinIDs
        # Do not broadcast back to the original sender
        await send_message_to_client(client, message)

# function to handle client connection and communication
async def handle_client(websocket: WebSocket):
    await websocket.accept()
    print("client joined")
    active_clients.append(websocket) # MOVE THIS TO THE START AND SET IN LIST OF session_ids : list<active clients>
    try:
        while True:
            # initial client contact
            data = await websocket.receive_text()
            if data:
                # actions to take when user joins server
                payload = json.loads(data)
                message_type = payload['message_type']
                if message_type == "create_server":
                    session_id = generate_hex_code(10)
                    print(f"[ client created chat :: session_id {session_id} ]")
                    print("DATA: ", data)
                    print(payload)
                    conversation_id = payload['conversation_id']
                    host_name = payload['host_name']
                    
                    prompt_message = f"{host_name} created the chat"
                    data = {
                        "message_type" : "server",
                        "session_id": session_id,
                        "message": prompt_message,
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                    #
                    await send_message_to_all(data)
                
                elif message_type == "join_server":
                    print(f"[ client joined chat :: session_id {session_id} ]")

                elif message_type == "start_server_from_chat":
                    print(f"[ client joined chat :: session_id {session_id} ]")
                
                break
        while True:
            data = await websocket.receive_text()
            if data:
                payload = json.loads(data)
                await send_message_to_all(payload)
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