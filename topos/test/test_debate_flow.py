# test_debate_flow.py

import os
import unittest
import warnings
from datetime import datetime
from topos.services.database.app_state import AppState, Neo4jConnection
from topos.FC.ontological_feature_detection import OntologicalFeatureDetection
from topos.channel.debatesim import DebateSimulator
from dotenv import load_dotenv
from uuid import uuid4
import json
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient
import asyncio


class TestDebateFlow(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        load_dotenv()  # Load environment variables

        self.neo4j_uri = os.getenv("NEO4J_URI")
        self.neo4j_user = os.getenv("NEO4J_USER")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD")
        self.neo4j_test_database = os.getenv("NEO4J_TEST_DATABASE")

        # Initialize the Neo4j connection
        self.neo4j_conn = Neo4jConnection(self.neo4j_uri, self.neo4j_user, self.neo4j_password)

        # Initialize the ontological feature detection with the test database
        self.ofd = OntologicalFeatureDetection(self.neo4j_uri, self.neo4j_user, self.neo4j_password,
                                               self.neo4j_test_database)

        # Initialize DebateSimulator
        self.debate_simulator = DebateSimulator()

        # Clean the database before each test
        self.clean_database()

    async def asyncTearDown(self):
        # Close the connection properly

        # Get the existing instance of AppState
        app_state = AppState.get_instance()
        app_state.close()
        # Reset the singleton instance
        Neo4jConnection._instance = None

    def clean_database(self):
        app_state = AppState.get_instance()
        with app_state.driver.session(database=self.neo4j_test_database) as session:
            session.run("MATCH (n) DETACH DELETE n")

    async def test_debate_flow(self):
        app = FastAPI()

        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()

            app_state = {
                "user_id": f"user_{str(uuid4())}",
                "session_id": f"session_{str(uuid4())}",
                "prior_ontology": []
            }

            message_data = [
                {"role": "user",
                 "data": {"user_id": "userABC", "content": "Checkers is better than chess because it is simpler."}},
                {"role": "user",
                 "data": {"user_id": "userFED", "content": "Chess is better than checkers because it has more depth."}},
                {"role": "user", "data": {"user_id": "userABC",
                                          "content": "Checkers is better than chess because there are fewer rules."}},
                {"role": "user", "data": {"user_id": "userFED",
                                          "content": "Chess is better than checkers because it requires more strategy."}},
                {"role": "user",
                 "data": {"user_id": "userABC", "content": "Checkers is better than chess because it is faster."}}
            ]

            expected_messages_count = {
                "userABC": 0,
                "userFED": 0
            }

            # send messages one by one to ensure the graph database gets filled out step by step
            for message in message_data:
                data = json.dumps({
                    "user_id": message["data"]["user_id"],
                    "message": message["data"]["content"],
                    "message_history": [],
                    "model": "dolphin-llama3",
                    "temperature": 0.3,
                    "topic": "Chess vs Checkers"
                })
                await self.debate_simulator.debate_step(websocket, data, app_state)

                # Update expected messages count
                user_id = message["data"]["user_id"]
                expected_messages_count[user_id] += 1

                # Print messages for each user after each step
                user_messages = self.ofd.get_messages_by_user(user_id, "SENT")
                print("\n\n\n")
                print(f"Messages for user {user_id}:")
                for msg in user_messages:
                    print(f"Message ID: {msg['message_id']}, Content: {msg['message']}, Timestamp: {msg['timestamp']}")

                # Assert the number of messages retrieved
                self.assertEqual(len(user_messages), expected_messages_count[user_id])
                print("\n\n\n")

            # Final message to get response
            data = json.dumps({
                "user_id": "userABC",
                "message": "Chess actually can win in far fewer moves than checkers, if the player is skilled enough.",
                "message_history": message_data,
                "model": "dolphin-llama3",
                "temperature": 0.3,
                "topic": "Chess vs Checkers"
            })
            await self.debate_simulator.debate_step(websocket, data, app_state)

        client = TestClient(app)

        with client.websocket_connect("/ws") as websocket:
            completed = False
            full_response = ""
            while not completed:
                response = websocket.receive_json()
                if response["status"] == "generating":
                    full_response += response["response"]
                elif response["status"] == "completed":
                    full_response += response["response"]
                    completed = True
                    self.assertIn("semantic_category", response)
                    self.assertTrue(response["completed"])


if __name__ == "__main__":
    unittest.main()
