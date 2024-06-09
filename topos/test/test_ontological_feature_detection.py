# test_ontological_feature_detection.py

import os
import unittest
from datetime import datetime
from topos.services.database.app_state import AppState, Neo4jConnection
from topos.FC.ontological_feature_detection import OntologicalFeatureDetection
from dotenv import load_dotenv


class TestOntologicalFeatureDetection(unittest.TestCase):
    def setUp(self):
        load_dotenv()  # Load environment variables

        self.neo4j_uri = os.getenv("NEO4J_URI")
        self.neo4j_user = os.getenv("NEO4J_USER")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD")
        self.neo4j_test_database = os.getenv("NEO4J_TEST_DATABASE")

        # Initialize the Neo4j connection
        self.neo4j_conn = Neo4jConnection(self.neo4j_uri, self.neo4j_user, self.neo4j_password)

        # # Create the test database if it doesn't exist
        #@note neo4j only allows one database per instance
        # self.create_test_database()

        # Initialize the ontological feature detection with the test database
        self.ofd = OntologicalFeatureDetection(self.neo4j_uri, self.neo4j_user, self.neo4j_password,
                                               self.neo4j_test_database)

        # Clean the database before each test
        self.clean_database()

    def tearDown(self):
        # Close the connection properly

        # Get the existing instance of AppState
        app_state = AppState.get_instance()
        app_state.close()
        # Reset the singleton instance
        Neo4jConnection._instance = None

    # def create_test_database(self):
    #     self.neo4j_conn.create_database(self.neo4j_test_database)

    def clean_database(self):
        app_state = AppState.get_instance()
        with app_state.driver.session(database=self.neo4j_test_database) as session:
            session.run("MATCH (n) DETACH DELETE n")

    def test_ontological_detection(self):
        load_dotenv()  # Load environment variables

        neo4j_uri = os.getenv("NEO4J_URI")
        neo4j_user = os.getenv("NEO4J_USER")
        neo4j_password = os.getenv("NEO4J_PASSWORD")
        neo4j_test_database = os.getenv("NEO4J_TEST_DATABASE")

        ofd = OntologicalFeatureDetection(neo4j_uri, neo4j_user, neo4j_password, neo4j_test_database)

        # Example with paragraph input
        paragraph = (
            "John, a software engineer from New York, bought a new laptop from Amazon on Saturday. "
            "He later met with his friend Alice, who is a data scientist at Google, for coffee at Starbucks. "
            "They discussed a variety of topics including the recent advancements in artificial intelligence, "
            "machine learning, and the future of technology. Alice suggested attending the AI conference in San Francisco next month."
        )
        mermaid_syntax_paragraph = ofd.extract_mermaid_syntax(paragraph, input_type="paragraph")
        print("Mermaid Syntax for Paragraph Input:")
        print(mermaid_syntax_paragraph)

        # Example with semantically compressed data input
        compressed_data = "Theoretical Computer Science::1=field within theoretical computer science;2=inherent difficulty;3=solve computational problems;4=achievable with algorithms and computation"
        mermaid_syntax_compressed = ofd.extract_mermaid_syntax(compressed_data, input_type="compressed_data")
        print("Mermaid Syntax for Compressed Data Input:")
        print(mermaid_syntax_compressed)

    def test_timestamp_and_accessors(self):
        user_id = "userRAM"
        session_id = "sessionZIP"
        message = "Hello, this is a test message!"

        # Extract the current timestamp
        timestamp = datetime.now().isoformat()

        # Create and test the ontology extraction with timestamp
        composable_string = f"for user {user_id}, of {session_id}, the message is: {message}"
        mermaid_syntax = self.ofd.extract_mermaid_syntax(composable_string, input_type="paragraph",
                                                         timestamp=timestamp)
        print("Mermaid Syntax with Timestamp:")
        print(mermaid_syntax)

        # Insert test data into Neo4j
        entities, pos_tags, dependencies, relations, srl_results, timestamp = self.ofd.build_ontology_from_paragraph(composable_string)

        message = composable_string

        self.ofd.store_ontology(user_id, session_id, message, timestamp)

        # Test search functions
        messages_by_user = self.ofd.get_messages_by_user(user_id, "SENT")
        print("Messages by User:")
        for msg in messages_by_user:
            print(msg)
        assert messages_by_user, "No messages found for the specified user."

        messages_by_session = self.ofd.get_messages_by_session(session_id, "CONTAINS")
        print("Messages by Session:")
        for msg in messages_by_session:
            print(msg)
        assert messages_by_session, "No messages found for the specified session."

        users_by_session = self.ofd.get_users_by_session(session_id, "PARTICIPATED_IN")
        print("Users by Session:")
        for user in users_by_session:
            print(user)
        assert users_by_session, "No users found for the specified session."

        sessions_by_user = self.ofd.get_sessions_by_user(user_id, "PARTICIPATED_IN")
        print("Sessions by User:")
        for session in sessions_by_user:
            print(session)
        assert sessions_by_user, "No sessions found for the specified user."


if __name__ == "__main__":
    unittest.main()
