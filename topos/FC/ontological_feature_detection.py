# ontological_feature_detection.py

import subprocess
import nltk
import spacy
import warnings
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from datetime import datetime
from topos.services.database.neo4j_connector import Neo4jConnection

# Suppress specific warnings related to model initialization
warnings.filterwarnings("ignore", message="Some weights of the model checkpoint at")

# Check and download NLTK data only if not already downloaded
nltk_packages = [
    ('tokenizers/punkt', 'punkt'),
    ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
    ('corpora/wordnet', 'wordnet')
]

for resource, package in nltk_packages:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(package)


class OntologicalFeatureDetection:
    def __init__(self, neo4j_uri, neo4j_user, neo4j_password, neo4j_database_name):
        # Initialize the tokenizer and model for SRL
        self.tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
        self.model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

        spacy_model_name = 'en_core_web_lg'

        # Load SpaCy models
        try:
            self.nlp = spacy.load(spacy_model_name)
        except OSError:
            print(f"SpaCy model '{spacy_model_name}' not found. Downloading...")
            subprocess.run(["python", "-m", "spacy", "download", spacy_model_name])
            self.nlp = spacy.load(spacy_model_name)

        # Add custom entities using EntityRuler
        ruler = self.nlp.add_pipe("entity_ruler")
        patterns = [{"label": "USER", "pattern": "userABC"}, {"label": "SESSION", "pattern": "sessionXYZ"}]
        ruler.add_patterns(patterns)

        #@note: neo4j only allows one database per instance
        self.database = "neo4j"#neo4j_database_name

        # Initialize Neo4j connection using singleton
        self.neo4j_conn = Neo4jConnection(neo4j_uri, neo4j_user, neo4j_password)
        self.driver = self.neo4j_conn.get_driver()

    def close(self):
        # Only close the connection if it is not shared by other instances
        if self.neo4j_conn:
            self.neo4j_conn.close()

    def perform_ner(self, text):
        doc = self.nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        print(f"NER results: {entities}")
        return entities

    def perform_pos_tagging(self, text):
        doc = self.nlp(text)
        pos_tags = [(token.text, token.pos_) for token in doc]
        print(f"POS tagging results: {pos_tags}")
        return pos_tags

    def perform_dependency_parsing(self, text):
        doc = self.nlp(text)
        dependencies = [(token.text, token.dep_, token.head.text) for token in doc]
        print(f"Dependency parsing results: {dependencies}")
        return dependencies

    def perform_srl(self, text):
        nlp_pipeline = pipeline("ner", model=self.model, tokenizer=self.tokenizer)
        results = nlp_pipeline(text)
        print(f"SRL results: {results}")
        return results

    def perform_relation_extraction(self, text):
        relations = []
        print(f"Relation extraction results: {relations}")
        return relations

    def add_entity(self, tx, entity, label, timestamp):
        print(f"Adding entity: {entity}, label: {label} at {timestamp}")
        tx.run("MERGE (e:Entity {name: $entity, label: $label, created_at: $timestamp})",
               entity=entity, label=label, timestamp=timestamp)

    def add_relation(self, tx, entity1, relation, entity2, timestamp):
        print(f"Attempting to create relationship: ({entity1})-[:{relation}]->({entity2}) at {timestamp}")
        result = tx.run("MATCH (a:Entity {name: $entity1}), (b:Entity {name: $entity2}) "
                        "MERGE (a)-[r:RELATION {type: $relation, created_at: $timestamp}]->(b) "
                        "RETURN r",
                        entity1=entity1, relation=relation, entity2=entity2, timestamp=timestamp)
        relationships = result.values()
        # print(f"Relationships created: {relationships}")
        assert len(relationships) > 0, f"Failed to create relationship ({entity1})-[:{relation}]->({entity2})"

        # print(f"Relationship created: ({entity1})-[:{relation}]->({entity2}) at {timestamp}")

    def build_ontology_from_paragraph(self, text):
        print(f"Processing text for ontology: {text}")
        entities = self.perform_ner(text)
        pos_tags = self.perform_pos_tagging(text)
        dependencies = self.perform_dependency_parsing(text)
        srl_results = self.perform_srl(text)
        relations = self.perform_relation_extraction(text)
        timestamp = datetime.now().isoformat()

        user = "userABC"
        session_entity = "sessionXYZ"
        message_entity = text.replace(" ", "_")

        with self.driver.session(database=self.database) as neo4j_session:
            # Insert user and session entities
            neo4j_session.execute_write(self.add_entity, user, "USER", timestamp)
            neo4j_session.execute_write(self.add_entity, session_entity, "SESSION", timestamp)

            # Insert message entity
            neo4j_session.execute_write(self.add_entity, message_entity, "MESSAGE", timestamp)

            # Create relationships between user, session, and message
            print(f"Creating relationship: ({user})-[:SENT]->({message_entity})")
            neo4j_session.execute_write(self.add_relation, user, "SENT", message_entity, timestamp)
            print(f"Creating relationship: ({session_entity})-[:CONTAINS]->({message_entity})")
            neo4j_session.execute_write(self.add_relation, session_entity, "CONTAINS", message_entity, timestamp)
            print(f"Creating relationship: ({user})-[:PARTICIPATED_IN]->({session_entity})")
            neo4j_session.execute_write(self.add_relation, user, "PARTICIPATED_IN", session_entity, timestamp)

            # Verify data insertion
            self.verify_data_insertion(neo4j_session, user, session_entity, message_entity)

        return entities, dependencies, srl_results

    def verify_data_insertion(self, neo4j_session, user, session_entity, message_entity):
        # Verify user entity
        result = neo4j_session.run("MATCH (e:Entity {name: $name, label: 'USER'}) RETURN e", name=user)
        user_node = result.single()
        print(f"Verification - User Node: {user_node}")
        assert user_node is not None, f"User {user} not found in database."

        # Verify session entity
        result = neo4j_session.run("MATCH (e:Entity {name: $name, label: 'SESSION'}) RETURN e", name=session_entity)
        session_node = result.single()
        print(f"Verification - Session Node: {session_node}")
        assert session_node is not None, f"Session {session_entity} not found in database."

        # Verify message entity
        result = neo4j_session.run("MATCH (e:Entity {name: $name, label: 'MESSAGE'}) RETURN e", name=message_entity)
        message_node = result.single()
        print(f"Verification - Message Node: {message_node}")
        assert message_node is not None, f"Message {message_entity} not found in database."

        # Verify relationships
        self.verify_relationship(neo4j_session, user, message_entity, "SENT")
        self.verify_relationship(neo4j_session, session_entity, message_entity, "CONTAINS")
        self.verify_relationship(neo4j_session, user, session_entity, "PARTICIPATED_IN")

    def verify_relationship(self, neo4j_session, start_entity, end_entity, relation_type):
        result = neo4j_session.run(
            "MATCH (a:Entity {name: $start_entity})-[r:RELATION {type: $relation_type}]->(b:Entity {name: $end_entity}) "
            "RETURN r",
            start_entity=start_entity, relation_type=relation_type, end_entity=end_entity)
        relationships = result.values()
        print(f"Verification - Relationships ({start_entity})-[:{relation_type}]->({end_entity}): {relationships}")
        assert relationships, f"No relationships ({start_entity})-[:{relation_type}]->({end_entity}) found in database."
        return relationships[0]

    def parse_input(self, input_str):
        topic, concepts = input_str.split("::")
        concepts = concepts.split(";")
        parsed_data = {
            "topic": topic,
            "concepts": [concept.split("=") for concept in concepts]
        }
        return parsed_data

    def build_ontology_from_compressed_data(self, input_str):
        parsed_data = self.parse_input(input_str)
        timestamp = datetime.now().isoformat()

        topic = parsed_data["topic"]
        entities = [(topic, "Topic")]
        relations = []

        for concept in parsed_data["concepts"]:
            if len(concept) == 2:
                index, description = concept
                entities.append((description, "Concept"))
                relations.append((topic, f"related_to_{index}", description))

        with self.driver.session(database=self.database) as session:
            for entity, label in entities:
                session.execute_write(self.add_entity, entity, label, timestamp)
            for relation in relations:
                session.execute_write(self.add_relation, relation[0], relation[1], relation[2], timestamp)

        return entities, relations

    def extract_mermaid_syntax(self, input_data, input_type="paragraph", timestamp=None):
        if input_type == "paragraph":
            entities, dependencies, srl_results = self.build_ontology_from_paragraph(input_data)
        else:
            entities, relations = self.build_ontology_from_compressed_data(input_data)

        entity_set = set()
        edges = set()

        for entity, label in entities:
            node_id = entity.replace(" ", "_")
            entity_set.add(f'{node_id}["{entity} ({label})"]')

        if input_type == "paragraph":
            for token, dep, head in dependencies:
                if dep in ["nsubj", "dobj", "pobj"]:  # Simplified dependency types
                    token_id = token.replace(" ", "_")
                    head_id = head.replace(" ", "_")
                    edges.add(f'{head_id} --> {token_id}')

            for result in srl_results:
                if 'entity' in result and 'word' in result:
                    entity = result['word'].replace(" ", "_")
                    srl_entity = result["entity"].replace(" ", "_")
                    edges.add(f'{entity} --> {srl_entity}')
        else:
            for relation in relations:
                head_id = relation[0].replace(" ", "_")
                token_id = relation[2].replace(" ", "_")
                edges.add(f'{head_id} --> {token_id}')

        if timestamp is None:
            timestamp = datetime.now().isoformat()

        entity_set.add(f'timestamp["Timestamp: {timestamp}"]')

        mermaid_syntax = "graph LR\n"
        for node in entity_set:
            mermaid_syntax += f"    {node}\n"
        for edge in edges:
            mermaid_syntax += f"    {edge}\n"

        if entities:
            first_entity_id = entities[0][0].replace(" ", "_")
            mermaid_syntax += f"    timestamp --> {first_entity_id}\n"

        return mermaid_syntax

    def get_messages_by_user(self, user_id):
        query = """
        MATCH (u:Entity {name: $user_id, label: 'USER'})-[:SENT]->(m:Entity {label: 'MESSAGE'})
        RETURN m.name AS message, m.created_at AS timestamp
        """
        with self.driver.session(database=self.database) as neo4j_session:
            result = neo4j_session.run(query, user_id=user_id)
            data = [record.data() for record in result]
            print(f"Messages by user {user_id}: {data}")
            return data

    def get_messages_by_session(self, session_id):
        query = """
        MATCH (s:Entity {name: $session_id, label: 'SESSION'})-[:CONTAINS]->(m:Entity {label: 'MESSAGE'})
        RETURN m.name AS message, m.created_at AS timestamp
        """
        with self.driver.session(database=self.database) as neo4j_session:
            result = neo4j_session.run(query, session_id=session_id)
            data = [record.data() for record in result]
            print(f"Messages by session {session_id}: {data}")
            return data

    def get_users_by_session(self, session_id):
        query = """
        MATCH (s:Entity {name: $session_id, label: 'SESSION'})<-[:PARTICIPATED_IN]-(u:Entity {label: 'USER'})
        RETURN u.name AS user_id, u.created_at AS created_at
        """
        with self.driver.session(database=self.database) as neo4j_session:
            result = neo4j_session.run(query, session_id=session_id)
            data = [record.data() for record in result]
            print(f"Users by session {session_id}: {data}")
            return data

    def get_sessions_by_user(self, user_id):
        query = """
        MATCH (u:Entity {name: $user_id, label: 'USER'})-[:PARTICIPATED_IN]->(s:Entity {label: 'SESSION'})
        RETURN s.name AS session_id, s.created_at AS created_at
        """
        with self.driver.session(database=self.database) as neo4j_session:
            result = neo4j_session.run(query, user_id=user_id)
            data = [record.data() for record in result]
            print(f"Sessions by user {user_id}: {data}")
            return data

    def print_ascii(self, hierarchy, nodes, node_id, indent=0, is_last=True, prefix=""):
        node_label = nodes[node_id]
        if indent == 0:
            output = f"{prefix}{node_label}\n"
        else:
            connector = "`- " if is_last else "|- "
            output = f"{prefix}{connector}{node_label}\n"
            prefix += "   " if is_last else "|  "

        children = hierarchy.get(node_id, [])
        for i, child in enumerate(children):
            is_last_child = i == (len(children) - 1)
            output += self.print_ascii(hierarchy, nodes, child, indent + 1, is_last_child, prefix)

        return output

    @staticmethod
    def build_hierarchy(nodes, edges):
        hierarchy = {}
        for parent, child in edges:
            if parent not in hierarchy:
                hierarchy[parent] = []
            hierarchy[parent].append(child)
        return hierarchy

    @staticmethod
    def parse_mermaid(input_text):
        lines = input_text.strip().split("\n")
        nodes = {}
        edges = []

        for line in lines:
            if "-->" in line:
                parts = line.split("-->")
                parent = parts[0].strip()
                child = parts[1].strip()
                edges.append((parent, child))
                if parent not in nodes:
                    nodes[parent] = parent
                if child not in nodes:
                    nodes[child] = child
            elif "graph" not in line and "[" in line:
                node_id, node_label = line.split("[")
                node_label = node_label.rstrip("]").strip().strip('"')
                nodes[node_id.strip()] = node_label

        return nodes, edges

    @staticmethod
    def find_root_nodes(nodes, edges):
        all_nodes = set(nodes.keys())
        child_nodes = set(child for _, child in edges)
        root_nodes = list(all_nodes - child_nodes)
        return root_nodes

    def mermaid_to_ascii(self, mermaid_input):
        nodes, edges = self.parse_mermaid(mermaid_input)
        hierarchy = self.build_hierarchy(nodes, edges)
        root_nodes = self.find_root_nodes(nodes, edges)
        ascii_output = ""
        for root_node in root_nodes:
            ascii_output += self.print_ascii(hierarchy, nodes, root_node)
        return ascii_output


# Example usage
# if __name__ == "__main__":
#     load_dotenv()  # Load environment variables
#
#     neo4j_uri = os.getenv("NEO4J_URI")
#     neo4j_user = os.getenv("NEO4J_USER")
#     neo4j_password = os.getenv("NEO4J_PASSWORD")
#
#     ofd = OntologicalFeatureDetection(neo4j_uri, neo4j_user, neo4j_password)
#
#     # Example with paragraph input
#     paragraph = (
#         "John, a software engineer from New York, bought a new laptop from Amazon on Saturday. "
#         "He later met with his friend Alice, who is a data scientist at Google, for coffee at Starbucks. "
#         "They discussed a variety of topics including the recent advancements in arti