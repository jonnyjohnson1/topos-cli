# test_ontological_feature_detection.py

import os
import unittest
from topos.FC.ontological_feature_detection import OntologicalFeatureDetection
from dotenv import load_dotenv


class TestOntologicalFeatureDetection(unittest.TestCase):
    def setUp(self):
        neo4j_uri = "bolt://localhost:7687"
        neo4j_user = "neo4j"
        neo4j_password = "pass"

        # Initialize the ontological feature detection
        self.ofd = OntologicalFeatureDetection(neo4j_uri,
                                               neo4j_user,
                                               neo4j_password)

    def tearDown(self):
        # Close the connection (if needed)
        self.ofd.close()

    def test_ontological_detection(self):
        load_dotenv()  # Load environment variables

        neo4j_uri = os.getenv("NEO4J_URI")
        neo4j_user = os.getenv("NEO4J_USER")
        neo4j_password = os.getenv("NEO4J_PASSWORD")

        ofd = OntologicalFeatureDetection(neo4j_uri, neo4j_user, neo4j_password)

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

        ofd.close()


if __name__ == "__main__":
    unittest.main()
