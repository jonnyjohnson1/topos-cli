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

        response = client.post("/admin_set_accounts", data={"Speaker_MS": "pass", "Speaker_WV": "pass"})
        self.assertEqual(response.status_code, 200)

        # Create tokens for two users
        response = client.post("/token", data={"username": "Speaker_MS", "password": "pass"})
        self.assertEqual(response.status_code, 200)
        token_user_a = response.json().get("access_token")
        self.assertIsNotNone(token_user_a)

        response = client.post("/token", data={"username": "Speaker_WV", "password": "pass"})
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
            {"role": "user", "data": {"user_id": "Speaker_MS",
                                      "content": "Ladies and gentlemen, we are here to discuss why Microsoft GraphRAG's knowledge graph-based method is the best approach for building a RAG workflow for complex queries. First, Microsoft GraphRAG leverages the power of structured knowledge graphs, which provide a robust and well-defined schema for data relationships. This ensures high accuracy and reliability in query results, as the graph inherently understands the semantic connections between different data points. Furthermore, the integration with Microsoft's suite of tools offers seamless interoperability, making it easier for enterprises to adopt and scale their RAG workflows efficiently. These advantages position Microsoft GraphRAG as the superior choice for handling complex queries."}},
            {"role": "user", "data": {"user_id": "Speaker_WV",
                                      "content": "While Microsoft GraphRAG's structured approach has its merits, Weaviate's Verba vector-based method offers unparalleled flexibility and scalability, which are critical for modern applications. Verba's use of vector embeddings allows it to handle unstructured data more effectively, adapting to a wide range of query types without the need for predefined schemas. This adaptability is particularly important in today's fast-evolving data landscape, where the types and sources of data are constantly changing. Moreover, Verba's performance in real-time query processing and its ability to provide contextually relevant results make it an invaluable tool for dynamic and complex query environments."}},
            {"role": "user", "data": {"user_id": "Speaker_MS",
                                      "content": "One of the key strengths of Microsoft GraphRAG's knowledge graph-based method is its inherent ability to maintain data integrity and consistency. Knowledge graphs are built on a foundation of ontologies and taxonomies, which ensure that the data relationships are both logical and meaningful. This structure is essential for complex queries, as it eliminates ambiguities and reduces the likelihood of errors in the query results. Additionally, the visual nature of knowledge graphs makes it easier for users to understand and interact with the data, leading to better decision-making and more insightful analyses."}},
            {"role": "user", "data": {"user_id": "Speaker_WV",
                                      "content": "While data integrity is important, the real world often requires handling diverse and unstructured data. Verba's vector-based method excels in this regard by using machine learning models to create rich, multidimensional representations of data points. This allows Verba to capture nuanced relationships and patterns that may not be immediately apparent in a structured knowledge graph. Furthermore, the ability to continuously update and refine vector embeddings based on new data ensures that Verba remains highly relevant and accurate over time, providing a more dynamic and responsive solution for complex queries."}},
            {"role": "user", "data": {"user_id": "Speaker_MS",
                                      "content": "The scalability of Microsoft GraphRAG's method is another significant advantage. Knowledge graphs can be scaled horizontally by distributing data across multiple nodes, ensuring that the system can handle large volumes of data and high query loads. This scalability is complemented by Microsoft's cloud infrastructure, which provides robust support for large-scale deployments. Additionally, the graph-based approach supports efficient query optimization techniques, enabling fast and reliable query execution even as the dataset grows. These capabilities make Microsoft GraphRAG a highly scalable and performant solution for enterprise-level RAG workflows."}},
            {"role": "user", "data": {"user_id": "Speaker_WV",
                                      "content": "In contrast, Verba's vector-based method offers superior scalability in terms of both data size and complexity. By leveraging distributed computing and advanced indexing techniques, Verba can process massive datasets with ease. Its ability to parallelize computations and distribute workload across multiple servers ensures that performance remains high even under heavy query loads. Moreover, the vector-based approach allows for more complex and sophisticated query capabilities, such as similarity searches and contextual understanding, which are crucial for modern applications. This makes Verba not only scalable but also highly adaptable to the needs of diverse and evolving data environments."}},
            {"role": "user", "data": {"user_id": "Speaker_MS",
                                      "content": "Another important consideration is the ease of integration and adoption. Microsoft GraphRAG benefits from seamless integration with other Microsoft products, such as Azure, Office 365, and Dynamics 365. This integration simplifies the deployment and management of RAG workflows, allowing organizations to leverage their existing investments in Microsoft technology. Furthermore, the comprehensive documentation and support provided by Microsoft ensure that users can quickly get up to speed with the GraphRAG platform, reducing the learning curve and accelerating time-to-value."}},
            {"role": "user", "data": {"user_id": "Speaker_WV",
                                      "content": "Ease of integration is also a strength of Weaviate's Verba. Verba is designed to be highly interoperable, with support for a wide range of data sources and formats. Its open architecture allows for easy integration with existing systems and workflows, minimizing disruption and maximizing compatibility. Additionally, Verba's extensive API support and modular design enable organizations to customize and extend the platform to meet their specific needs. This flexibility, combined with robust community support and extensive documentation, makes Verba a highly accessible and adaptable solution for building RAG workflows."}},
            {"role": "user", "data": {"user_id": "Speaker_MS",
                                      "content": "In conclusion, Microsoft GraphRAG's knowledge graph-based method offers a structured, scalable, and highly integrative approach to building RAG workflows for complex queries. Its reliance on well-defined data relationships ensures accuracy and consistency, while its seamless integration with Microsoft's ecosystem facilitates ease of adoption and deployment. These advantages make Microsoft GraphRAG the best choice for organizations seeking a reliable and robust solution for handling complex queries."}},
            {"role": "user", "data": {"user_id": "Speaker_WV",
                                      "content": "To summarize, Weaviate's Verba vector-based method provides unmatched flexibility and scalability, making it ideally suited for the dynamic and diverse data environments of today. Its ability to handle unstructured data and deliver contextually relevant results ensures that it can adapt to a wide range of query types and requirements. This versatility, combined with its performance and ease of integration, makes Verba the superior choice for building RAG workflows that can meet the evolving demands of modern applications."}}
        ]

        unique_users = set(message["data"]["user_id"] for message in message_data)
        user_a_name, user_b_name = list(unique_users)

        # message_data = self.debate_simulator.break_into_sentences(message_data)

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

                    if response["status"] == "final_results":
                        self.assertIn("results", response)
                        final_results_received = True

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

