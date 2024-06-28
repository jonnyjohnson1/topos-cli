# test_debate_jwt_flow.py

import unittest
from uuid import uuid4
from datetime import datetime, timedelta
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.testclient import TestClient
from topos.services.database.app_state import AppState
from topos.channel.debatesim import DebateSimulator
import json
import jwt
from jwt.exceptions import InvalidTokenError
import asyncio

class TestDebateJWTFlow(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        AppState._instance = None
        load_dotenv()  # Load environment variables

        # Initialize app state
        self.app_state = AppState(use_neo4j=False)

        # Initialize DebateSimulator without Neo4j
        self.debate_simulator = DebateSimulator(use_neo4j=False)

    async def asyncTearDown(self):
        # Reset the singleton instance
        AppState._instance = None

    async def test_jwt_generation_and_validation(self):
        user_id = f"user_{str(uuid4())}"
        session_id = f"session_{str(uuid4())}"

        # Generate JWT token
        token = self.debate_simulator.generate_jwt_token(user_id, session_id)
        self.assertIsNotNone(token)

        # Validate JWT token
        try:
            decoded_token = jwt.decode(token, self.debate_simulator.jwt_secret, algorithms=["HS256"])
            self.assertEqual(decoded_token["user_id"], user_id)
            self.assertEqual(decoded_token["session_id"], session_id)
        except InvalidTokenError:
            self.fail("JWT token validation failed")

    async def test_debate_flow_with_jwt(self):
        app = FastAPI()

        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()

            app_state = AppState.get_instance()

            user_id = f"user_{str(uuid4())}"
            session_id = f"session_{str(uuid4())}"
            token = self.debate_simulator.generate_jwt_token(user_id, session_id)

            app_state.set_value("user_id", user_id)
            app_state.set_value("session_id", session_id)
            app_state.set_value("prior_ontology", [])
            app_state.set_value("message_history", [])

            message_data = [
                {"role": "user", "data": {"user_id": "userA",
                                          "content": "Human activity, particularly the burning of fossil fuels, has significantly increased the concentration of greenhouse gases in the atmosphere."}},
                {"role": "user", "data": {"user_id": "userB",
                                          "content": "Climate change is a natural phenomenon that has occurred throughout Earth's history, long before human activity."}}
            ]

            for message in message_data:
                user_id = message["data"]["user_id"]
                content = message["data"]["content"]
                data = json.dumps({
                    "message": content,
                    "user_id": user_id,
                    "generation_nonce": str(uuid4())
                })
                await self.debate_simulator.integrate(token, data, app_state)

                # Ensure message history is updated
                message_history = app_state.get_value(f"message_history_{session_id}", [])
                self.assertTrue(any(m["data"]["content"] == content for m in message_history if isinstance(m, dict)))

                # Ensure prior ontology is updated
                prior_ontology = app_state.get_value(f"prior_ontology_{session_id}", [])
                self.assertTrue(any(content in ontology for ontology in prior_ontology))

        client = TestClient(app)

        with client.websocket_connect("/ws") as websocket:
            completed = False
            full_response = ""
            while not completed:
                response = websocket.receive_json()
                if response["status"] == "initial_clusters":
                    full_response += json.dumps(response)
                elif response["status"] == "updated_clusters":
                    full_response += json.dumps(response)
                elif response["status"] == "wepcc_result":
                    full_response += json.dumps(response)
                elif response["status"] == "completed":
                    full_response += json.dumps(response)
                    completed = True

        # Verify overall response
        self.assertIn("initial_clusters", full_response)
        self.assertIn("updated_clusters", full_response)
        self.assertIn("wepcc_result", full_response)

if __name__ == "__main__":
    unittest.main()
