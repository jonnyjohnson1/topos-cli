# test_ontological_feature_detection.py

import unittest
import json
from ..FC.ontological_feature_detection import OntologicalFeatureDetection


class TestOntologicalFeatureDetection(unittest.TestCase):
    def setUp(self):
        neo4j_uri = "bolt://localhost:7687"
        neo4j_user = "neo4j"
        neo4j_password = ""

        # Initialize the ontological feature detection
        self.ofd = OntologicalFeatureDetection(neo4j_uri,
                                               neo4j_user,
                                               neo4j_password)

    def tearDown(self):
        # Close the connection (if needed)
        self.ofd.close()

    def test_ontological_detection(self):
        # Sample JSON string
        json_string = '{"role":"moderator", "content":"I believe the topic of discussion is Chess versus Checkers, a fascinating comparison between these two classic board games.", "certainty_score": 9}'

        # Parse the JSON string
        data = json.loads(json_string)

        # Extract the relevant fields
        role = data.get('role', '')
        content = data.get('content', '')
        certainty_score = data.get('certainty_score', 0)

        # Print the extracted fields for verification
        print(f"Role: {role}")
        print(f"Content: {content}")
        print(f"Certainty Score: {certainty_score}")

        # Process the extracted content
        self.ofd.build_ontology(content)


if __name__ == "__main__":
    unittest.main()
