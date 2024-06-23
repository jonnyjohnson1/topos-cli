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
        AppState._instance = None
        load_dotenv()  # Load environment variables

        self.neo4j_uri = os.getenv("NEO4J_URI")
        self.neo4j_user = os.getenv("NEO4J_USER")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD")
        self.neo4j_test_database = os.getenv("NEO4J_TEST_DATABASE")

        # Initialize app state with Neo4j connection details
        self.app_state = AppState(self.neo4j_uri, self.neo4j_user, self.neo4j_password, self.neo4j_test_database)

        # Initialize the Neo4j connection
        self.neo4j_conn = Neo4jConnection(self.neo4j_uri, self.neo4j_user, self.neo4j_password)

        # Initialize the ontological feature detection with the test database
        self.ofd = OntologicalFeatureDetection(self.neo4j_uri, self.neo4j_user, self.neo4j_password,
                                               self.neo4j_test_database, True)

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

            app_state = AppState.get_instance()

            app_state.set_state("user_id", f"user_{str(uuid4())}")
            app_state.set_state("session_id", f"session_{str(uuid4())}")
            app_state.set_state("prior_ontology", [])

            message_data = [
                {"role": "user", "data": {"user_id": "userA",
                                          "content": "Human activity, particularly the burning of fossil fuels, has significantly increased the concentration of greenhouse gases in the atmosphere."}},
                {"role": "user", "data": {"user_id": "userB",
                                          "content": "Climate change is a natural phenomenon that has occurred throughout Earth's history, long before human activity."}},
                {"role": "user", "data": {"user_id": "userA",
                                          "content": "Scientific studies show a direct correlation between industrialization and the rise in global temperatures."}},
                {"role": "user", "data": {"user_id": "userB",
                                          "content": "There are many natural factors, such as volcanic eruptions and solar radiation, that contribute to global temperature changes."}},
                {"role": "user", "data": {"user_id": "userA",
                                          "content": "The Intergovernmental Panel on Climate Change (IPCC) reports that human influence is the dominant cause of global warming since the mid-20th century."}},
                {"role": "user", "data": {"user_id": "userB",
                                          "content": "The climate models used to predict human impact are based on assumptions that might not accurately reflect complex climate systems."}},
                {"role": "user", "data": {"user_id": "userA",
                                          "content": "Deforestation for agriculture and urban development reduces the Earth's capacity to absorb CO2, exacerbating climate change."}},
                {"role": "user", "data": {"user_id": "userB",
                                          "content": "The Earth's climate has always fluctuated; there were periods of warming and cooling even in the pre-industrial era."}},
                {"role": "user", "data": {"user_id": "userA",
                                          "content": "Human-driven emissions of CO2 and methane are accelerating the melting of polar ice caps and glaciers."}},
                {"role": "user", "data": {"user_id": "userB",
                                          "content": "Some studies suggest that current climate changes could be part of a natural cycle, not necessarily driven by human activities."}},
                {"role": "user", "data": {"user_id": "userA",
                                          "content": "While natural factors do play a role, the rapid increase in CO2 levels coincides with the industrial revolution and increased fossil fuel use."}},
                {"role": "user", "data": {"user_id": "userB",
                                          "content": "Historical data shows that the Earth has experienced higher CO2 levels and warmer temperatures in the past without human influence."}},
                {"role": "user", "data": {"user_id": "userA",
                                          "content": "The majority of climate scientists agree that human activity is the primary driver of recent climate change."}},
                {"role": "user", "data": {"user_id": "userB",
                                          "content": "Some scientists argue that the impact of human activities on climate change is overstated."}},
                {"role": "user", "data": {"user_id": "userA",
                                          "content": "The acidification of oceans due to increased CO2 levels is a direct result of human emissions, affecting marine ecosystems."}},
                {"role": "user", "data": {"user_id": "userB",
                                          "content": "The costs of transitioning to renewable energy are high and could negatively impact economies."}},
                {"role": "user", "data": {"user_id": "userA",
                                          "content": "Extreme weather events, such as hurricanes and wildfires, have become more frequent and severe due to human-induced climate change."}},
                {"role": "user", "data": {"user_id": "userB",
                                          "content": "Efforts to combat climate change should focus on adaptation rather than trying to control the climate."}},
                {"role": "user", "data": {"user_id": "userA",
                                          "content": "Renewable energy sources like wind and solar power can reduce our dependence on fossil fuels and mitigate climate change."}},
                {"role": "user", "data": {"user_id": "userB",
                                          "content": "Developing countries need affordable energy to grow their economies, and restricting fossil fuel use could hinder their development."}},
                {"role": "user", "data": {"user_id": "userA",
                                          "content": "Ignoring the human impact on climate change could lead to catastrophic consequences for future generations."}},
                {"role": "user", "data": {"user_id": "userB",
                                          "content": "It's important to consider the economic implications of drastic measures to reduce emissions."}},
                {"role": "user", "data": {"user_id": "userA",
                                          "content": "Investing in green technology creates jobs and boosts the economy while protecting the environment."}},
                {"role": "user", "data": {"user_id": "userB",
                                          "content": "Developing new technologies for adaptation can also create jobs and economic growth."}},
                {"role": "user", "data": {"user_id": "userA",
                                          "content": "International agreements like the Paris Accord aim to reduce global emissions and limit temperature rise, benefiting everyone."}},
                {"role": "user", "data": {"user_id": "userB",
                                          "content": "Policies should balance environmental concerns with economic needs to avoid undue hardship."}},
                {"role": "user", "data": {"user_id": "userA",
                                          "content": "The health impacts of air pollution from fossil fuels are significant, causing respiratory problems and premature deaths."}},
                {"role": "user", "data": {"user_id": "userB",
                                          "content": "Natural climate variability must be accounted for in any climate policy."}},
                {"role": "user", "data": {"user_id": "userA",
                                          "content": "Preserving natural habitats and biodiversity is crucial for a sustainable future and mitigating climate change."}},
                {"role": "user", "data": {"user_id": "userB",
                                          "content": "Focusing on resilience and adaptation strategies can help communities better cope with climate changes."}},
                {"role": "user", "data": {"user_id": "userA",
                                          "content": "Proactive measures to reduce emissions will ultimately be more cost-effective than dealing with the aftermath of climate disasters."}},
                {"role": "user", "data": {"user_id": "userB",
                                          "content": "There's still uncertainty in the extent of human impact on climate change."}},
                {"role": "user", "data": {"user_id": "userA",
                                          "content": "Global cooperation is essential to address a problem that affects all nations."}},
                {"role": "user", "data": {"user_id": "userB",
                                          "content": "Climate policies should be flexible and adaptable to new scientific findings."}},
                {"role": "user", "data": {"user_id": "userA",
                                          "content": "The transition to a green economy is an opportunity to innovate and lead in sustainable practices."}},
                {"role": "user", "data": {"user_id": "userB",
                                          "content": "Economic growth and environmental protection can go hand in hand with smart policies."}},
                {"role": "user", "data": {"user_id": "userA",
                                          "content": "Public awareness and education about climate change can drive more sustainable behaviors."}},
                {"role": "user", "data": {"user_id": "userB",
                                          "content": "Innovation in carbon capture and storage technologies can reduce emissions without drastic lifestyle changes."}},
                {"role": "user", "data": {"user_id": "userA",
                                          "content": "Protecting the environment is a moral responsibility to ensure a livable planet for future generations."}},
                {"role": "user", "data": {"user_id": "userB",
                                          "content": "Addressing climate change requires a balanced approach that considers both environmental and economic factors."}}
            ]

            expected_messages_count = {
                "userA": 0,
                "userB": 0
            }

            built_message_history = []

            # send messages one by one to ensure the graph database gets filled out step by step
            for message in message_data:
                data = json.dumps({
                    "user_id": message["data"]["user_id"],
                    "message": message["data"]["content"],
                    "message_history": built_message_history,
                    "model": "dolphin-llama3",
                    "temperature": 0.3,
                    "topic": "Climate Change: Is human activity the primary cause?"
                })
                await self.debate_simulator.debate_step(websocket, data, app_state)

                built_message_history.append(message)

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
                "user_id": "userA",
                "message": "Mitigating climate change requires a global effort to reduce emissions and transition to renewable energy sources.",
                "message_history": message_data,
                "model": "dolphin-llama3",
                "temperature": 0.3,
                "topic": "Climate Change: Is human activity the primary cause?"
            })
            await self.debate_simulator.debate_step(websocket, data, app_state)

            kl_divergences = app_state.get_value("kl_divergences", [])
            print(f"KL-Divergences: {kl_divergences}")

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
