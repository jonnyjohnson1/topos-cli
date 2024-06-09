# ontological_feature_detection.py

import subprocess
from collections import deque

import nltk
import spacy
import warnings
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from datetime import datetime
from topos.services.database.app_state import AppState

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

        # Add custom entities using EntityRuler with regex patterns
        ruler = self.nlp.add_pipe("entity_ruler")
        patterns = [
            {"label": "USER", "pattern": [{"TEXT": {"REGEX": "^user.*$"}}]},
            {"label": "SESSION", "pattern": [{"TEXT": {"REGEX": "^session.*$"}}]}
        ]
        ruler.add_patterns(patterns)

        self.showroom_db_name = "neo4j"

        # Initialize app state with Neo4j connection details
        self.app_state = AppState(neo4j_uri, neo4j_user, neo4j_password, self.showroom_db_name)

    def perform_ner(self, text):
        doc = self.nlp(text)

        for token in doc:
            print(f"{token.text}: {token.pos_}")

        entities = [(ent.text, ent.label_) for ent in doc.ents]
        print(f"\t[ NER results: {entities} ]")
        return entities

    def perform_pos_tagging(self, text):
        doc = self.nlp(text)
        pos_tags = [(token.text, token.pos_) for token in doc]
        print(f"\t[ POS tagging results: {pos_tags} ]")
        return pos_tags

    def perform_dependency_parsing(self, text):
        doc = self.nlp(text)
        dependencies = [(token.text, token.dep_, token.head.text) for token in doc]
        print(f"\t[ Dependency parsing results: {dependencies} ]")
        return dependencies

    def perform_srl(self, text):
        doc = self.nlp(text)
        srl_results = []

        subject = None
        verb = None
        comparative = None
        compared_entities = []

        # Identify comparative structures and their scopes
        comparative_scopes = []
        scope_start = None
        for i, token in enumerate(doc):
            if token.pos_ == "ADJ" and doc[i + 1].text == "than":
                comparative = token.text
                scope_start = i
            elif scope_start is not None and (token.pos_ == "CCONJ" or i == len(doc) - 1):
                comparative_scopes.append((scope_start, i))
                scope_start = None

        # Adjust POS tags for entities within comparative scopes
        for start, end in comparative_scopes:
            for i in range(start, end + 1):
                if doc[i].pos_ == "VERB":
                    doc[i].pos_ = "PROPN"

        # Identify subject, verb, comparative, and compared entities
        for token in doc:
            if token.dep_ == "nsubj":
                subject = token.text
                srl_results.append({"entity": token.text, "role": "SUBJECT", "word": token.text})
            elif token.dep_ == "ROOT" and token.pos_ in ["VERB", "AUX"]:
                verb = token.text
                srl_results.append({"entity": token.text, "role": "VERB", "word": token.text})
            elif token.dep_ == "acomp" and token.head.pos_ in ["VERB", "AUX"]:
                comparative = token.text
                srl_results.append({"entity": token.text, "role": "COMPARATIVE", "word": token.text})
            elif token.pos_ in ["NOUN", "PROPN"] and any(start <= token.i <= end for start, end in comparative_scopes):
                compared_entities.append(token.text)

        # Generate comparative relationships
        if subject and comparative and compared_entities:
            for i in range(len(compared_entities)):
                if i == 0:
                    srl_results.append({"entity": subject, "role": "MORE_COMPLICATED", "word": comparative})
                    srl_results.append(
                        {"entity": compared_entities[i], "role": "LESS_COMPLICATED", "word": comparative})
                else:
                    srl_results.append({"entity": subject, "role": "LESS_COMPLICATED", "word": comparative})
                    srl_results.append(
                        {"entity": compared_entities[i], "role": "MORE_COMPLICATED", "word": comparative})

        print(f"\t[ SRL results: {srl_results} ]")

        return srl_results

    def perform_relation_extraction(self, text, srl_results):
        relations = []
        entities = set()

        comparative_adjective = None
        subject = None
        more_complicated = []
        less_complicated = []

        for result in srl_results:
            if result['role'] == 'COMPARATIVE':
                comparative_adjective = result['word']
            elif result['role'] == 'SUBJECT' and result['entity'] != 'message':
                subject = result['entity']
                entities.add(subject)
            elif result['role'] == 'MORE_COMPLICATED':
                more_complicated.append(result['entity'])
                entities.add(result['entity'])
            elif result['role'] == 'LESS_COMPLICATED':
                less_complicated.append(result['entity'])
                entities.add(result['entity'])

        if comparative_adjective and subject:
            relation_type = f"{comparative_adjective}_than"
            for entity in less_complicated:
                if entity != subject:
                    relations.append((subject, relation_type, entity))
            for entity in more_complicated:
                if entity != subject:
                    relations.append((entity, relation_type, subject))

        print(f"\t[ Relation extraction results: {relations} :: {entities} ]")

        return relations, list(entities)



    # def perform_relation_extraction(self, text):
    #     doc = self.nlp(text)
    #     relations = []
    #
    #     for token in doc:
    #         # Handle subject-verb-object relationships
    #         if token.dep_ in ("nsubj", "dobj") and token.head.pos_ == "VERB":
    #             subject = None
    #             object_ = None
    #             verb = token.head.text
    #
    #             if token.dep_ == "nsubj":
    #                 subject = token.text
    #                 for child in token.head.children:
    #                     if child.dep_ in ("dobj", "pobj"):
    #                         object_ = child.text
    #                         relations.append((subject, verb, object_))
    #
    #             elif token.dep_ == "dobj":
    #                 object_ = token.text
    #                 for child in token.head.children:
    #                     if child.dep_ == "nsubj":
    #                         subject = child.text
    #                         relations.append((subject, verb, object_))
    #
    #         # Handle comparative structures (e.g., "better than", "greater than", "more complicated than")
    #         if token.dep_ == "acomp" and token.head.pos_ in ("VERB", "AUX"):
    #             comparative_adjective = token.text
    #             verb = token.head.text
    #             subject = None
    #             comparative_object = None
    #
    #             for child in token.head.children:
    #                 if child.dep_ == "nsubj":
    #                     subject = child.text
    #
    #             for child in token.children:
    #                 if child.dep_ == "prep" and child.text == "than":
    #                     for obj in child.children:
    #                         if obj.dep_ == "pobj":
    #                             comparative_object = f"{comparative_adjective} than {obj.text}"
    #                             relations.append((subject, verb, comparative_object))
    #
    #         # Handle multiple comparatives in the same sentence
    #         if token.dep_ == "cc" and token.text in ("but", "and"):
    #             for next_comp in token.head.conjuncts:
    #                 if next_comp.dep_ == "acomp":
    #                     next_adjective = next_comp.text
    #                     next_subject = None
    #                     next_object = None
    #
    #                     for child in next_comp.children:
    #                         if child.dep_ == "nsubj":
    #                             next_subject = child.text
    #
    #                         if child.dep_ == "prep" and child.text == "than":
    #                             for obj in child.children:
    #                                 if obj.dep_ == "pobj":
    #                                     next_object = f"{next_adjective} than {obj.text}"
    #                                     relations.append((next_subject, verb, next_object))
    #
    #     print(f"Relation extraction results: {relations}")
    #     return relations

    def add_entity(self, tx, entity_id: str, entity_label: str, properties: dict):
        query = "MERGE (e:{label} {{id: $entity_id}})\n".format(label=entity_label)
        for key, value in properties.items():
            query += "SET e.{key} = ${key}\n".format(key=key)
        tx.run(query, entity_id=entity_id, **properties)

    def add_relation(self, tx, source_id: str, relation_type: str, target_id: str, properties: dict):
        query = (
            "MATCH (source {{id: $source_id}}), (target {{id: $target_id}})\n"
            "MERGE (source)-[r:{relation_type}]->(target)\n".format(relation_type=relation_type)
        )
        for key, value in properties.items():
            query += "SET r.{key} = ${key}\n".format(key=key)
        tx.run(query, source_id=source_id, target_id=target_id, **properties)

        # print(f"Relationship created: ({entity1})-[:{relation}]->({entity2}) at {timestamp}")

    def build_ontology_from_paragraph(self, user_id, session_id, message_id, text):
        print(f"Processing text for ontology: {text}")
        entities = self.perform_ner(text)
        pos_tags = self.perform_pos_tagging(text)
        dependencies = self.perform_dependency_parsing(text)
        srl_results = self.perform_srl(text)
        relations, relational_entities = self.perform_relation_extraction(text, srl_results)
        timestamp = datetime.now().isoformat()

        context_entities = [(user_id, "USER"), (session_id, "SESSION"), (message_id, "MESSAGE")]

        return entities, pos_tags, dependencies, relations, srl_results, timestamp, context_entities

    def store_ontology(self, user_id, session_id, message_id, message, timestamp, context_entities, relations):
        with self.app_state.get_driver_session() as neo4j_session:
            def _store_ontology(tx):
                # Insert message entity with properties
                print(f"[ user_id :: {user_id} :: session_id :: {session_id} :: message :: {message} ]")
                print(f"Inserting message entity: {message_id} with content: {message} and timestamp: {timestamp}")
                self.add_entity(tx, message_id, "MESSAGE", {"content": message, "timestamp": timestamp})

                # Insert context entities and create relationships with the message
                for entity, label in context_entities:
                    print(f"Inserting context entity: ({entity}) with label: {label}")
                    self.add_entity(tx, entity, label, {"timestamp": timestamp})
                    print(f"Creating relationship: ({message_id})-[:HAS_ENTITY]->({entity})")
                    self.add_relation(tx, message_id, "HAS_ENTITY", entity, {"timestamp": timestamp})

                # Create relationships based on relation extraction results
                for subject, relation, object_ in relations:
                    print(f"Creating relationship: ({subject})-[:{relation}]->({object_})")
                    self.add_relation(tx, subject, relation, object_, {"timestamp": timestamp})

                # Create relationships between user, session, and message
                print(f"Creating relationship: ({user_id})-[:SENT]->({message_id})")
                self.add_relation(tx, user_id, "SENT", message_id, {"timestamp": timestamp})
                print(f"Creating relationship: ({session_id})-[:CONTAINS]->({message_id})")
                self.add_relation(tx, session_id, "CONTAINS", message_id, {"timestamp": timestamp})
                print(f"Creating relationship: ({user_id})-[:PARTICIPATED_IN]->({session_id})")
                self.add_relation(tx, user_id, "PARTICIPATED_IN", session_id, {"timestamp": timestamp})

            neo4j_session.execute_write(_store_ontology)

            # Verify data insertion
            # self.verify_data_insertion(user_id, session_id, message_id, message, timestamp, context_entities)

    def verify_data_insertion(self, user_id, session_id, message_id, message, timestamp, context_entities):
        with self.app_state.get_driver_session() as neo4j_session:
            # Retrieve message content and timestamp
            result = neo4j_session.run(
                "MATCH (m:MESSAGE {id: $message_id}) "
                "RETURN m.content AS message, m.timestamp AS timestamp",
                message_id=message_id
            )
            message_data = result.single()
            assert message_data, f"Message with ID {message_id} not found."
            assert message_data["message"] == message, f"Expected message: {message}, got: {message_data['message']}"
            assert message_data[
                       "timestamp"] == timestamp, f"Expected timestamp: {timestamp}, got: {message_data['timestamp']}"
            print("Message content and timestamp verified.")

            # Retrieve user_id
            print(f"Verifying user for message_id: {message_id}")
            result = neo4j_session.run(
                "MATCH (u:USER)-[:SENT]->(m:MESSAGE {id: $message_id}) "
                "RETURN u.id AS user_id",
                message_id=message_id
            )
            record = result.single()
            assert record, f"No user found for message_id: {message_id}"
            retrieved_user_id = record["user_id"]
            assert retrieved_user_id == user_id, f"Expected user_id: {user_id}, got: {retrieved_user_id}"
            print("User ID verified.")

            # Retrieve session_id
            print(f"Verifying session for message_id: {message_id}")
            result = neo4j_session.run(
                "MATCH (s:SESSION)-[:CONTAINS]->(m:MESSAGE {id: $message_id}) "
                "RETURN s.id AS session_id",
                message_id=message_id
            )
            record = result.single()
            assert record, f"No session found for message_id: {message_id}"
            retrieved_session_id = record["session_id"]
            assert retrieved_session_id == session_id, f"Expected session_id: {session_id}, got: {retrieved_session_id}"
            print("Session ID verified.")

            # Retrieve context entities
            print(f"Verifying context entities for message_id: {message_id}")
            result = neo4j_session.run(
                "MATCH (m:MESSAGE {id: $message_id})-[:HAS_ENTITY]->(e) "
                "RETURN e.id AS entity, labels(e) AS labels",
                message_id=message_id
            )
            retrieved_context_entities = [(record["entity"], record["labels"][0]) for record in result]
            assert set(retrieved_context_entities) == set(
                context_entities), f"Expected context entities: {context_entities}, got: {retrieved_context_entities}"
            print("Context entities verified.")

            print("Data insertion verified successfully.")

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

        with self.app_state.get_driver_session() as session:
            for entity, label in entities:
                session.execute_write(self.add_entity, entity, label, {"timestamp": timestamp})
            for relation in relations:
                session.execute_write(self.add_relation, relation[0], relation[1], relation[2],
                                      {"timestamp": timestamp})

        return entities, relations

    def extract_mermaid_syntax(self, input_data, input_type="paragraph", timestamp=None):
        message = input_data
        user_id = "userPRIME"
        session_id = "sessionTEMP"
        message_id = "message"

        if input_type == "paragraph":
            entities, pos_tags, dependencies, relations, srl_results, timestamp, context_entities = self.build_ontology_from_paragraph(
                user_id, session_id, message_id, input_data)
        elif input_type == "components":
            message, entities, dependencies, relations, srl_results, timestamp, context_entities = input_data
        else:  # input_type == "compressed_data"
            entities, relations = self.build_ontology_from_compressed_data(input_data)
            context_entities = [(user_id, "USER"), (session_id, "SESSION"), (message_id, "MESSAGE")]
            srl_results = []
            dependencies = []
            pos_tags = []
            message = input_data

        if timestamp is None:
            timestamp = datetime.now().isoformat()

        mermaid_syntax = "flowchart LR\n"

        # Add user, session, and timestamp nodes
        mermaid_syntax += f'    {user_id}["{user_id} (USER)"] --> {session_id}["{session_id} (SESSION)"]\n'
        mermaid_syntax += f'    timestamp["{timestamp}"] --> {user_id}\n'

        # Add message node and its connections
        mermaid_syntax += f'    {message_id}["{message} (MESSAGE)"] --> {user_id}\n'
        mermaid_syntax += f'    {message_id} --> {session_id}\n'
        mermaid_syntax += f'    {message_id} --> timestamp\n'

        # Add subgraph for entities and relations
        mermaid_syntax += f'    subgraph subgraph_{message_id}["Message Subgraph"]\n'
        added_entities = set()
        for relation in relations:
            subject, relation_type, object_ = relation
            if subject not in added_entities:
                mermaid_syntax += f'        {subject}["{subject}"]\n'
                added_entities.add(subject)
            if object_ not in added_entities:
                mermaid_syntax += f'        {object_}["{object_}"]\n'
                added_entities.add(object_)
            relation_label = "more complicated" if "more" in relation_type else "less complicated"
            mermaid_syntax += f'        {subject} -->|"{relation_label}"| {object_}\n'
        mermaid_syntax += f'    end\n'

        mermaid_syntax += f'    {message_id} --> subgraph_{message_id}\n'

        return mermaid_syntax

    def get_messages_by_user(self, user_id, relation_type):
        query = """
        MATCH (u:USER {id: $user_id})-[r:{relation_type}]->(m:MESSAGE)
        RETURN m.id AS message_id, m.content AS message, m.timestamp AS timestamp
        """
        query = query.replace("{relation_type}", relation_type)
        with self.app_state.get_driver_session() as neo4j_session:
            result = neo4j_session.run(query, user_id=user_id)
            data = [record.data() for record in result]
            print(f"Messages by user {user_id}: {data}")
            return data

    def get_messages_by_session(self, session_id, relation_type):
        query = """
        MATCH (s:SESSION {id: $session_id})-[r:CONTAINS]->(m:MESSAGE)
        RETURN m.id AS message_id, m.content AS message, m.timestamp AS timestamp
        """
        with self.app_state.get_driver_session() as neo4j_session:
            result = neo4j_session.run(query, session_id=session_id)
            data = [record.data() for record in result]
            print(f"Messages by session {session_id}: {data}")
            return data

    def get_users_by_session(self, session_id, relation_type):
        query = """
        MATCH (s:SESSION {id: $session_id})<-[r:PARTICIPATED_IN]-(u:USER)
        RETURN u.id AS user_id
        """
        with self.app_state.get_driver_session() as neo4j_session:
            result = neo4j_session.run(query, session_id=session_id)
            data = [record.data() for record in result]
            print(f"Users by session {session_id}: {data}")
            return data

    def get_sessions_by_user(self, user_id, relation_type):
        query = """
        MATCH (u:USER {id: $user_id})-[r:PARTICIPATED_IN]->(s:SESSION)
        RETURN s.id AS session_id
        """
        with self.app_state.get_driver_session() as neo4j_session:
            result = neo4j_session.run(query, user_id=user_id)
            data = [record.data() for record in result]
            print(f"Sessions by user {user_id}: {data}")
            return data

    def get_message_by_id(self, message_id):
        query = """
        MATCH (m:MESSAGE {id: $message_id})
        RETURN m.content AS message, m.timestamp AS timestamp
        """
        with self.app_state.get_driver_session() as neo4j_session:
            result = neo4j_session.run(query, message_id=message_id)
            data = [record.data() for record in result]
            print(f"Query executed: {query}")
            print(f"Parameters: message_id={message_id}")
            print(f"Query result: {data}")
            return data

    @staticmethod
    def check_message_exists(message_id):
        app_state = AppState.get_instance()
        exists = app_state.value_exists(label="MESSAGE", key="name", value=message_id)
        print(f"Message with ID {message_id} exists: {exists}")
        return exists

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

    from collections import deque

    def mermaid_to_ascii(self, mermaid_input):
        nodes, edges = self.parse_mermaid(mermaid_input)
        root_nodes = self.find_root_nodes(nodes, edges)

        ascii_output = ""
        visited_nodes = set()

        for root_node in root_nodes:
            if root_node.startswith("subgraph_"):
                subgraph_label = nodes[root_node]
                ascii_output += f"subgraph {subgraph_label}\n"
                ascii_output += self.traverse_hypergraph(root_node, nodes, edges, visited_nodes, indent=1)
            else:
                ascii_output += self.traverse_hypergraph(root_node, nodes, edges, visited_nodes)
            ascii_output += "\n"  # Add a newline between root nodes

        return ascii_output.strip()  # Remove any trailing newline

    def traverse_hypergraph(self, node_id, nodes, edges, visited_nodes, indent=0, max_depth=None):
        output = ""
        queue = deque([(node_id, indent, False)])

        while queue:
            current_node, current_indent, is_last = queue.popleft()

            if current_node in visited_nodes:
                continue

            visited_nodes.add(current_node)

            node_label = nodes[current_node]
            prefix = self.get_prefix(current_indent, is_last)
            output += f"{prefix}{node_label}\n"

            if max_depth is not None and current_indent >= max_depth:
                continue

            connected_nodes = self.get_connected_nodes(current_node, edges)
            for i, child_node in enumerate(connected_nodes):
                is_last_child = i == len(connected_nodes) - 1
                queue.append((child_node, current_indent + 1, is_last_child))

        return output

    def get_prefix(self, indent, is_last):
        if indent == 0:
            return ""
        elif is_last:
            return "    " * (indent - 1) + "└── "
        else:
            return "    " * (indent - 1) + "├── "

    def get_connected_nodes(self, node_id, edges):
        connected_nodes = set()
        for edge in edges:
            source, target = edge
            if source == node_id:
                connected_nodes.add(target)
        return list(connected_nodes)


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