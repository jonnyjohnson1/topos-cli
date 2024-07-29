import asyncio
import json
from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, List

class LobbyServer:
    def __init__(self):
        self.general_lobby = []
        self.game_lobbies = {}  # game_id -> list of websockets
        self.websocket_to_user = {}  # websocket -> user_id

    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        self.websocket_to_user[websocket] = user_id
        print(f"\t[ User {user_id} connected ]")

    async def disconnect(self, websocket: WebSocket):
        user_id = self.websocket_to_user[websocket]
        del self.websocket_to_user[websocket]
        print(f"\t[ User {user_id} disconnected ]")
        if websocket in self.general_lobby:
            self.general_lobby.remove(websocket)
        for game_id in self.game_lobbies:
            if websocket in self.game_lobbies[game_id]:
                self.game_lobbies[game_id].remove(websocket)

    async def join_general_lobby(self, websocket: WebSocket):
        self.general_lobby.append(websocket)
        await self.send_message(websocket, "GeneralLobbyEntered", {"status": "ok"})
        print(f"\t\t[ User joined general lobby ]")
        await self.send_available_games(websocket)

    async def join_game_lobby(self, websocket: WebSocket, game_id: str):
        if game_id not in self.game_lobbies:
            self.game_lobbies[game_id] = []
        self.game_lobbies[game_id].append(websocket)
        await self.send_message(websocket, "InGameLobbyJoined", {"status": "ok", "game_id": game_id})
        print(f"\t\t[ User joined game lobby {game_id} ]")

    async def send_message(self, websocket: WebSocket, message_type: str, data: dict):
        message = {"type": message_type, "data": data}
        await websocket.send_text(json.dumps(message))

    async def send_available_games(self, websocket: WebSocket):
        # This is a placeholder. Implement your logic to get the available games.
        available_games = [{"gameId": "1", "name": "Debate Game 1"}, {"gameId": "2", "name": "Debate Game 2"}]
        await self.send_message(websocket, "AvailableGames", {"games": available_games})

    async def handle_message(self, websocket: WebSocket, message: str):
        data = json.loads(message)
        message_type = data["type"]
        user_id = self.websocket_to_user[websocket]

        if message_type == "JoinGeneralLobby":
            await self.join_general_lobby(websocket)
        elif message_type == "JoinInGameLobby":
            game_id = data["game_id"]
            await self.join_game_lobby(websocket, game_id)
        elif message_type == "ReadyGame":
            game_id = data["game_id"]
            await self.send_message(websocket, "GameReadied", {"status": "ready", "game_id": game_id})
        elif message_type == "UnreadyGame":
            game_id = data["game_id"]
            await self.send_message(websocket, "GameUnreadied", {"status": "unready", "game_id": game_id})

    async def handle_connection(self, websocket: WebSocket, user_id: str):
        await self.connect(websocket, user_id)
        try:
            while True:
                data = await websocket.receive_text()
                await self.handle_message(websocket, data)
        except WebSocketDisconnect:
            await self.disconnect(websocket)
        except Exception as e:
            print(f"Error: {e}")
            await self.disconnect(websocket)

lobby_server = LobbyServer()
