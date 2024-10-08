# test_debate_flow_4_evolution.py

import os
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
import threading

from topos.services.database.app_state import AppState
from topos.channel.debatesim import DebateSimulator
from topos.api.debate_routes import router, SECRET_KEY, ALGORITHM

app = FastAPI()
app.include_router(router)


class WebSocketThread(threading.Thread):
    def __init__(self, url, messages, responses):
        threading.Thread.__init__(self)
        self.url = url
        self.messages = messages
        self.responses = responses
        self.client = TestClient(app)

    def run(self):
        with self.client.websocket_connect(self.url) as websocket:
            for message in self.messages:
                websocket.send_json(message)
                response = websocket.receive_json()
                self.responses.append(response)

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
        if self.debate_simulator.channel_engine.processing_task:
            self.debate_simulator.channel_engine.processing_task.cancel()
            try:
                await self.debate_simulator.channel_engine.processing_task
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

    def test_debate_flow_with_jwt(self):
        client = TestClient(app)

        response = client.post("/admin_set_accounts", data={"Evolutionist": "pass", "Creationist": "pass"})
        self.assertEqual(response.status_code, 200)

        # Create tokens for two users
        response = client.post("/token", data={"username": "Evolutionist", "password": "pass"})
        self.assertEqual(response.status_code, 200)
        token_user_a = response.json().get("access_token")
        self.assertIsNotNone(token_user_a)

        response = client.post("/token", data={"username": "Creationist", "password": "pass"})
        self.assertEqual(response.status_code, 200)
        token_user_b = response.json().get("access_token")
        self.assertIsNotNone(token_user_b)

        # Get or create session for userA
        response = client.get("/sessions", headers={"Authorization": f"Bearer {token_user_a}"})
        self.assertEqual(response.status_code, 200)
        sessions = response.json().get("sessions", [])

        if not sessions:
            response = client.post("/create_session", headers={"Authorization": f"Bearer {token_user_a}"})
            self.assertEqual(response.status_code, 200)
            session_id = response.json().get("session_id")
            self.assertIsNotNone(session_id)
        else:
            session_id = sessions[0]

        message_data = [
                        {"role": "user", "data": {"user_id": "Evolutionist", "content": "The theory of evolution is supported by extensive scientific evidence. Fossil records provide clear evidence of species changing over time."}},
                        {"role": "user", "data": {"user_id": "Creationist", "content": "Evolution is just a theory, not a proven fact. The fossil record is incomplete and doesn't show transitional forms."}},
                        {"role": "user", "data": {"user_id": "Evolutionist", "content": "Genetic studies confirm common ancestry between species. Natural selection has been observed in real-time, such as with antibiotic resistance in bacteria."}},
                        {"role": "user", "data": {"user_id": "Creationist", "content": "Genetic similarities could be due to a common designer, not common ancestry. Microevolution within species occurs, but not macroevolution between species."}},
                        {"role": "user", "data": {"user_id": "Evolutionist", "content": "The age of the Earth, approximately 4.5 billion years, provides ample time for evolution to occur. Comparative anatomy shows homologous structures across species, indicating shared ancestry."}},
                        {"role": "user", "data": {"user_id": "Creationist", "content": "The Earth is much younger, only thousands of years old, not billions. Similar structures could be evidence of a common design plan, not evolution."}},
                        {"role": "user", "data": {"user_id": "Evolutionist", "content": "Biogeography supports evolution, with species distribution matching continental drift patterns. Vestigial organs in various species point to evolutionary history."}},
                        {"role": "user", "data": {"user_id": "Creationist", "content": "Species distribution could be explained by migration after a global flood event. So-called vestigial organs often have functions we haven't discovered yet."}},
                        {"role": "user", "data": {"user_id": "Evolutionist", "content": "The scientific community overwhelmingly accepts evolution as the best explanation for biodiversity. Evolutionary theory has predictive power, such as in discovering Tiktaalik, a transitional fossil."}},
                        {"role": "user", "data": {"user_id": "Creationist", "content": "Many scientists disagree with evolution, showing it's not a settled matter. Creation science also makes predictions, like the discovery of soft tissue in dinosaur fossils."}},
                        ]

        unique_users = set(message["data"]["user_id"] for message in message_data)
        user_a_name, user_b_name = list(unique_users)

        message_data = self.debate_simulator.break_into_sentences(message_data)

        # Open WebSocket connections for both users
        with client.websocket_connect(f"/ws?token={token_user_a}&session_id={session_id}") as websocket_a, \
             client.websocket_connect(f"/ws?token={token_user_b}&session_id={session_id}") as websocket_b:

            for message in message_data:
                if message["data"]["user_id"] == user_a_name:
                    websocket_a.send_json({
                        "message": message["data"]["content"],
                        "user_id": message["data"]["user_id"],
                        "generation_nonce": str(uuid4())
                    })
                else:
                    websocket_b.send_json({
                        "message": message["data"]["content"],
                        "user_id": message["data"]["user_id"],
                        "generation_nonce": str(uuid4())
                    })

                print(f"\t[ Sent message: {message['data']['content']} ]")

                initial_response_received = False
                clusters_received = False
                updated_clusters_received = False
                wepcc_result_received = False
                final_results_received = False

                # Wait for and process multiple responses
                while not (initial_response_received and clusters_received and updated_clusters_received
                           and wepcc_result_received and final_results_received):
                    if message["data"]["user_id"] == user_a_name:
                        response = websocket_a.receive_json()
                    else:
                        response = websocket_b.receive_json()

                    # print(f"\t\t[ Received response: {response} ]")

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
                        print("RESPONSE:\n", response)

                    if response["status"] == "final_results":
                        self.assertIn("results", response)
                        final_results_received = True
                        print("RESPONSE:\n", response)

                print(f"\t[ Messaged processed: {message['data']['content']} ]")
                
        print("Test completed")

        # Verify that all expected responses were received
        self.assertTrue(initial_response_received, "Did not receive initial response.")
        self.assertTrue(clusters_received, "Did not receive initial clusters.")
        self.assertTrue(updated_clusters_received, "Did not receive updated clusters.")
        self.assertTrue(wepcc_result_received, "Did not receive WEPCC result.")
        self.assertTrue(final_results_received, "Did not receive final results.")


if __name__ == "__main__":

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    unittest.main()

