import random
import string
import json
import asyncio

# Mock WebSocket class for testing purposes
class WebSocket:
    async def receive_text(self):
        await asyncio.sleep(1)
        if not hasattr(self, 'message_index'):
            self.message_index = 0
        
        if self.message_index == 0:
            self.message_index += 1
            return json.dumps({"message_type": "create_server", "user_id": self.user_id, "host_name": "host1"})
        else:
            return json.dumps({"message_type": "join_server", "session_id": self.session_id, "user_id": self.user_id})

# Generate a random 8-digit session ID
def generate_session_id(length=8):
    return ''.join(random.choices(string.digits, k=length))

class SessionManager:
    def __init__(self):
        self.active_sessions = {}
        self.user_sessions = {}

    def add_session(self, session_id, user_id, websocket):
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = []
        self.active_sessions[session_id].append((user_id, websocket))

    def get_active_sessions(self):
        return self.active_sessions

    def get_user_sessions(self):
        return self.user_sessions

    def add_user_session(self, user_id, session_id):
        self.user_sessions[user_id] = session_id

async def test_session_manager():
    session_manager = SessionManager()
    websocket = WebSocket()
    websocket.user_id = "user1"

    while True:
        data = await websocket.receive_text()
        if data:
            payload = json.loads(data)
            print("Received payload:", payload)
            message_type = payload['message_type']
            active_sessions = session_manager.get_active_sessions()
            user_sessions = session_manager.get_user_sessions()

            if message_type == "create_server":
                session_id = generate_session_id()
                websocket.session_id = session_id
                print(f"[ client created chat :: session_id {session_id} ]")
                user_id = payload['user_id']
                host_name = payload['host_name']
                session_manager.add_session(session_id, user_id, websocket)
                session_manager.add_user_session(user_id, session_id)
                print("Active Sessions:", active_sessions)
                print("User Sessions:", user_sessions)
                session_created = True

            elif message_type == "join_server":
                session_id = payload.get('session_id')
                user_id = payload['user_id']
                session_manager.add_session(session_id, user_id, websocket)
                session_manager.add_user_session(user_id, session_id)
                print(f"User {user_id} joined session {session_id}")
                print("Active Sessions:", active_sessions)
                print("User Sessions:", user_sessions)
        
        # Break the loop after one iteration for testing purposes
        # if session_created:
        #     break

# Run the test case
asyncio.run(test_session_manager())
