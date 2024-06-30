# test_debate_flow.py

import unittest
from uuid import uuid4
from datetime import datetime, timedelta, UTC
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, status
from fastapi.testclient import TestClient
from fastapi.security import OAuth2PasswordBearer
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import json
import jwt
from jwt.exceptions import InvalidTokenError
import asyncio
from topos.services.database.app_state import AppState
from topos.channel.debatesim import DebateSimulator
from topos.api.debate_routes import router, SECRET_KEY, ALGORITHM

app = FastAPI()
app.include_router(router)

class TestDebateJWTFlow(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        AppState._instance = None
        load_dotenv()  # Load environment variables

        # Initialize app state
        self.app_state = AppState(use_neo4j=False)

        # Initialize DebateSimulator without Neo4j
        self.debate_simulator = await DebateSimulator.get_instance()

    async def asyncTearDown(self):
        # Make sure to cancel the processing task when tearing down
        if self.debate_simulator.processing_task:
            self.debate_simulator.processing_task.cancel()
            try:
                await self.debate_simulator.processing_task
            except asyncio.CancelledError:
                pass

        # Reset the singleton instance
        AppState._instance = None

    def test_jwt_generation_and_validation(self):
        user_id = f"user_{str(uuid4())}"
        session_id = f"session_{str(uuid4())}"

        # Generate JWT token
        token_data = {
            "user_id": user_id,
            "exp": datetime.now(UTC) + timedelta(hours=1)
        }
        token = jwt.encode(token_data, SECRET_KEY, algorithm=ALGORITHM)
        self.assertIsNotNone(token)

        # Validate JWT token
        try:
            decoded_token = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            self.assertEqual(decoded_token["user_id"], user_id)
        except InvalidTokenError:
            self.fail("JWT token validation failed")

    async def test_debate_flow_with_jwt(self):
        client = TestClient(app)

        # Request JWT token
        response = client.post("/token", data={"username": "user", "password": "pass"})
        self.assertEqual(response.status_code, 200)
        token = response.json().get("access_token")
        self.assertIsNotNone(token)

        # Get list of sessions
        response = client.get("/sessions", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(response.status_code, 200)
        sessions = response.json().get("sessions", [])

        if not sessions:
            # Create a new session if none exists
            response = client.post("/create_session", headers={"Authorization": f"Bearer {token}"})
            self.assertEqual(response.status_code, 200)
            session_id = response.json().get("session_id")
            self.assertIsNotNone(session_id)
        else:
            session_id = sessions[0]

        message_data = [
            {"role": "user", "data": {"user_id": "userA", "content": "Human activity impacts climate change."}},
            {"role": "user", "data": {"user_id": "userB", "content": "Natural cycles cause climate change."}}
        ]

        with client.websocket_connect(f"/ws?token={token}&session_id={session_id}") as websocket:
            for message in message_data:
                websocket.send_json({
                    "message": message["data"]["content"],
                    "user_id": message["data"]["user_id"],
                    "generation_nonce": str(uuid4())
                })
                print(f"Sent message: {message['data']['content']}")
                await asyncio.sleep(1)  # Increase wait time to allow for task processing
                print("Finished waiting after sending message")

                initial_response_received = False
                clusters_received = False
                updated_clusters_received = False
                wepcc_result_received = False

                # Wait for and process multiple responses
                while not (initial_response_received and clusters_received and updated_clusters_received and wepcc_result_received):
                    try:
                        response = websocket.receive_json()
                        print(f"Received response: {response}")

                        if response["status"] == "message_processed":
                            self.assertIn("initial_analysis", response)
                            initial_response_received = True

                        if response["status"] == "initial_clusters":
                            self.assertIn("clusters", response)
                            clusters_received = True

                        if response["status"] == "updated_clusters":
                            self.assertIn("clusters", response)
                            updated_clusters_received = True

                        if response["status"] == "wepcc_result":
                            self.assertIn("wepcc_result", response)
                            wepcc_result_received = True

                    except asyncio.TimeoutError:
                        print("Timeout waiting for WebSocket response")
                        self.fail("Test timed out waiting for response")

            print(f"Messaged processed: {message['data']['content']}")

        print("Test completed")

        # Verify that all expected responses were received
        self.assertTrue(initial_response_received, "Did not receive initial response.")
        self.assertTrue(clusters_received, "Did not receive initial clusters.")
        self.assertTrue(updated_clusters_received, "Did not receive updated clusters.")
        self.assertTrue(wepcc_result_received, "Did not receive WEPCC result.")

if __name__ == "__main__":
    unittest.main()

