# chat_server.py
# this version works locally, but doesn't work with zrok

import socket
import threading
import json
import datetime
from typing import Dict, List, Tuple
from http.server import BaseHTTPRequestHandler, HTTPServer

from topos.utilities.utils import generate_deci_code

HOST = "127.0.0.1"
PORT = 13394
HTTP_PORT = 8080
LISTENER_LIMIT = 5
active_clients: Dict[str, List[Tuple[str, socket.socket]]] = {}
user_sessions: Dict[str, str] = {}

class SessionManager:
    def __init__(self):
        self.active_sessions: Dict[str, List[Tuple[str, socket.socket]]] = active_clients
        self.user_sessions: Dict[str, str] = user_sessions

    def add_session(self, session_id: str, user_id: str, client: socket.socket):
        print(f"[ adding {session_id} to active_sessions ]")
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = []
        self.active_sessions[session_id].append((user_id, client))

    def get_active_sessions(self):
        return self.active_sessions

    def get_user_sessions(self):
        return self.user_sessions

    def add_user_session(self, user_id: str, session_id: str):
        print(f"[ adding {user_id} to user_sessions ]")
        self.user_sessions[user_id] = session_id

session_manager = SessionManager()

# Function to listen for incoming messages from client
def listen_for_message(client, session_id, user_id):
    while True:
        max_msg_size = 2048
        try:
            message = client.recv(max_msg_size).decode('utf-8')
            if message:
                payload = json.loads(message)
                print(f"RECEIVED: {payload}")
                if 'content' in payload and 'session_id' in payload['content']:
                    session_id = payload['content']['session_id']
                    user_id = payload['content']['user_id']
                    if session_id:
                        print(f"sending {session_id}")
                        send_message_to_all(session_id, user_id, payload)
            else:
                print(f"[ Message from client {user_id} is empty ]")
        except:
            handle_disconnect(client, session_id, user_id)
            break

# Function to send message to a single client
def send_message_to_client(client, message):
    try:
        client.sendall(json.dumps(message).encode())
    except:
        print("Error sending message to client")

# Function to send any new message to all the clients currently connected to the server
def send_message_to_all(session_id, sender_user_id, message):
    active_sessions = session_manager.get_active_sessions()
    if message['message_type'] != 'server':
        print(f"[ message to user :: {message['content']['text']}]")
    if session_id in active_sessions:
        for user_id, client in active_sessions[session_id]:
            if message['message_type'] == 'server':
                send_message_to_client(client, message)
            elif user_id != sender_user_id:
                send_message_to_client(client, message)

# Function to handle client
def client_handler(client):
    max_msg_size = 2048
    try:
        data = client.recv(max_msg_size).decode('utf-8')
        if data:
            payload = json.loads(data)
            message_type = payload['message_type']
            user_id = payload['user_id']
            username = payload['username']
            
            print(f"[ {username} joined the chat")
            if message_type == "create_server":
                session_id = generate_deci_code(6)
                print(f"[ client created chat :: session_id {session_id} ]")
                session_manager.add_session(session_id, user_id, client)
                session_manager.add_user_session(user_id, session_id)
                print(session_manager.get_active_sessions())
                prompt_message = f"{payload['host_name']} created the chat"
                data = {
                    "message_type": "server",
                    "session_id": session_id,
                    "message": prompt_message,
                    "timestamp": datetime.datetime.now().isoformat()
                }
                send_message_to_all(session_id, user_id, data)
                threading.Thread(target=listen_for_message, args=(client, session_id, user_id)).start()

            elif message_type == "join_server":
                session_id = payload['session_id']
                username = payload['username']
                if session_id in session_manager.get_active_sessions():
                    print(f"[ {username} joined chat :: session_id {session_id} ]")
                    session_manager.add_session(session_id, user_id, client)
                    session_manager.add_user_session(user_id, session_id)
                    join_message = f"{username} joined the chat"
                    data = {
                        "message_type": "server",
                        "session_id": session_id,
                        "message": join_message,
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                    send_message_to_all(session_id, user_id, data)
                    threading.Thread(target=listen_for_message, args=(client, session_id, user_id)).start()
                else:
                    send_message_to_client(client, json.dumps({"error": "Invalid session ID"}))
        else:
            print("[ Client username is empty ]")
    except Exception as e:
        print(f"Error: {e}")
        client.close()

# Handle disconnect
def handle_disconnect(client, session_id, user_id):
    active_sessions = session_manager.get_active_sessions()
    if session_id in active_sessions:
        clients = active_sessions[session_id]
        clients = [c for c in clients if c[1] != client]
        if not clients:
            del active_sessions[session_id]
        disconnect_message = f"{user_id} left the chat"
        send_message_to_all(session_id, user_id, {
            "message_type": "server",
            "session_id": session_id,
            "message": disconnect_message,
            "timestamp": datetime.datetime.now().isoformat()
        })
        user_sessions = session_manager.get_user_sessions()
        user_sessions.pop(user_id, None)
    client.close()

# HTTP request handler for health check
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/test":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            response = {"response": True}
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()

# Function to start the HTTP server for health check
def start_http_server():
    http_server = HTTPServer((HOST, HTTP_PORT), HealthCheckHandler)
    print(f"[ HTTP server running on port {HTTP_PORT} ]")
    http_server.serve_forever()


# Main function to start the chat server and HTTP server
def main():
    # Start the HTTP server in a separate thread
    threading.Thread(target=start_http_server, daemon=True).start()

    # Start the TCP chat server
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        server.bind((HOST, PORT))
        print(f"[ Running the server on {HOST} {PORT} ]")
    except:
        print(f"[ Unable to bind to host {HOST} and port {PORT} ]")
        return

    server.listen(LISTENER_LIMIT)
    print("[ Server is listening for connections ]")

    while True:
        client, address = server.accept()
        print(f"New connected client {address[0]} {address[1]}")
        threading.Thread(target=client_handler, args=(client,)).start()

if __name__ == "__main__":
    main()
