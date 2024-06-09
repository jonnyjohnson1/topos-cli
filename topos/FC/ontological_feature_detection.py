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

        return srl_results

    def perform_relation_extraction(self, text):
        doc = self.nlp(text)
        relations = []

        for token in doc:
            if token.dep_ in ("nsubj", "dobj") and token.head.pos_ == "VERB":
                subject = None
                object_ = None
                verb = token.head.text

                if token.dep_ == "nsubj":
                    subject = token.text
                    for child in token.head.children:
                        if child.dep_ in ("dobj", "pobj"):
                            object_ = child.text
                            relations.append((subject, verb, object_))

                elif token.dep_ == "dobj":
                    object_ = token.text
                    for child in token.head.children:
                        if child.dep_ == "nsubj":
                            subject = child.text
                            relations.append((subject, verb, object_))

            if token.dep_ == "acomp" and token.head.pos_ in ("VERB", "AUX"):
                comparative_adjective = token.text
                verb = token.head.text
                subject = None
                comparative_object = None

                for child in token.head.children:
                    if child.dep_ == "nsubj":
                        subject = child.text

                for child in token.children:
                    if child.dep_ == "prep" and child.text == "than":
                        for obj in child.children:
                            if obj.dep_ == "pobj":
                                comparative_object = f"{comparative_adjective} than {obj.text}"
                                relations.append((subject, verb, comparative_object))

            if token.dep_ == "cc" and token.text in ("but", "and"):
                for next_comp in token.head.conjuncts:
                    if next_comp.dep_ == "acomp":
                        next_adjective = next_comp.text
                        next_subject = None
                        next_object = None

                        for child in next_comp.children:
                            if child.dep_ == "nsubj":
                                next_subject = child.text

                            if child.dep_ == "prep" and child.text == "than":
                                for obj in child.children:
                                    if obj.dep_ == "pobj":
                                        next_object = f"{next_adjective} than {obj.text}"
                                        relations.append((next_subject, verb, next_object))

        print(f"\t[ SRL parsing results: {relations} ]")
        return relations


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

    def build_ontology_from_paragraph(self, user_id, session_id, message_id, text):
        print(f"Processing text for ontology: {text}")
        entities = self.perform_ner(text)
        pos_tags = self.perform_pos_tagging(text)
        dependencies = self.perform_dependency_parsing(text)
        srl_results = self.perform_srl(text)
        relations = self.perform_relation_extraction(text)
        timestamp = datetime.now().isoformat()

        context_entities = [(user_id, "USER"), (session_id, "SESSION"), (message_id, "MESSAGE")]

        return entities, pos_tags, dependencies, relations, srl_results, timestamp, context_entities

    def store_ontology(self, user_id, session_id, message_id, message, timestamp, context_entities):
        message_entity = f"message_{message_id}"
        with self.app_state.get_driver_session() as neo4j_session:
            # Insert context entities
            for entity, label in context_entities:
                neo4j_session.execute_write(self.add_entity, entity, label, timestamp)

            # Insert message entity
            neo4j_session.execute_write(self.add_entity, message_entity, "MESSAGE", timestamp, message)

            # Create relationships between user, session, and message
            print(f"Creating relationship: ({user_id})-[:SENT]->({message_entity})")
            neo4j_session.execute_write(self.add_relation, user_id, "SENT", message_entity, timestamp)
            print(f"Creating relationship: ({session_id})-[:CONTAINS]->({message_entity})")
            neo4j_session.execute_write(self.add_relation, session_id, "CONTAINS", message_entity, timestamp)
            print(f"Creating relationship: ({user_id})-[:PARTICIPATED_IN]->({session_id})")
            neo4j_session.execute_write(self.add_relation, user_id, "PARTICIPATED_IN", session_id, timestamp)

            # Verify data insertion
            # self.verify_data_insertion(neo4j_session, user, session_entity, message_entity)

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

        with self.app_state.get_driver_session() as session:
            for entity, label in entities:
                session.execute_write(self.add_entity, entity, label, timestamp)
            for relation in relations:
                session.execute_write(self.add_relation, relation[0], relation[1], relation[2], timestamp)

        return entities, relations

    def extract_mermaid_syntax(self, input_data, input_type="paragraph", timestamp=None):
        message = input_data
        temp_user_id = "user_unknown"
        session_user_id = "session_unknown"
        message_id = "message_unknown"

        if input_type == "paragraph":
            entities, pos_tags, dependencies, relations, srl_results, timestamp, context_entities = self.build_ontology_from_paragraph(
                temp_user_id, session_user_id, message_id, input_data)
        elif input_type == "components":
            message, entities, dependencies, relations, srl_results, timestamp, context_entities = input_data
        else:  # input_type == "compressed_data"
            entities, relations = self.build_ontology_from_compressed_data(input_data)
            context_entities = [(temp_user_id, "USER"), (session_user_id, "SESSION"), (message_id, "MESSAGE")]
            srl_results = []
            dependencies = []
            pos_tags = []
            message = input_data

        subgraph_entities = set()
        edges = set()

        if input_type == "paragraph" or input_type == "components":
            for result in srl_results:
                if 'entity' in result and 'word' in result:
                    entity = result['word'].replace(" ", "_")
                    srl_entity = result["entity"].replace(" ", "_")
                    role = result["role"]

                    if role in ["MORE_COMPLICATED", "LESS_COMPLICATED"]:
                        subgraph_entities.add(entity)
                        subgraph_entities.add(srl_entity)
                        subgraph_entities.add(role.lower())
                        if role == "MORE_COMPLICATED":
                            edges.add(f'{entity} --> {srl_entity}')
                        else:
                            edges.add(f'{srl_entity} --> {entity}')

        if timestamp is None:
            timestamp = datetime.now().isoformat()

        mermaid_syntax = "flowchart LR\n"

        if context_entities:
            user_entity = next((entity for entity, label in context_entities if label == "USER"), None)
            session_entity = next((entity for entity, label in context_entities if label == "SESSION"), None)
            if user_entity and session_entity:
                user_entity_id = user_entity.replace(" ", "_")
                session_entity_id = session_entity.replace(" ", "_")
                mermaid_syntax += f"    {user_entity_id}[\"{user_entity} (USER)\"] --> {session_entity_id}[\"{session_entity} (SESSION)\"]\n"
                mermaid_syntax += f"    timestamp[\"{timestamp}\"] --> {user_entity_id}\n"

        message_entity_id = "message"
        message_subgraph_id = f"subgraph_{message_entity_id}"
        mermaid_syntax += f" subgraph {message_subgraph_id}[\"Message Subgraph\"]\n"

        for entity in subgraph_entities:
            if entity not in ["less_complicated", "more_complicated"]:
                mermaid_syntax += f"        {entity}[\"{entity}\"]\n"
            else:
                mermaid_syntax += f"        {entity}[\"{entity.replace('_', ' ')}\"]\n"

        mermaid_syntax += "  end\n"

        for edge in sorted(edges):
            mermaid_syntax += f"    {edge}\n"

        mermaid_syntax += f"    {message_entity_id}[\"{message} (MESSAGE)\"] --> {user_entity_id}\n"
        mermaid_syntax += f"    {message_entity_id} --> {session_entity_id}\n"
        mermaid_syntax += f"    {message_entity_id} --> timestamp\n"
        mermaid_syntax += f"    {message_entity_id} --> {message_subgraph_id}\n"

        return mermaid_syntax

    def get_messages_by_user(self, user_id, relation_type):
        query = """
        MATCH (u:Entity {name: $user_id, label: 'USER'})-[:RELATION {type: $relation_type}]->(m:Entity {label: 'MESSAGE'})
        RETURN m.name AS message, m.created_at AS timestamp
        """
        with self.app_state.get_driver_session() as neo4j_session:
            result = neo4j_session.run(query, user_id=user_id, relation_type=relation_type)
            data = [record.data() for record in result]
            print(f"Messages by user {user_id}: {data}")
            return data

    def get_messages_by_session(self, session_id, relation_type):
        query = """
        MATCH (s:Entity {name: $session_id, label: 'SESSION'})-[:RELATION {type: $relation_type}]->(m:Entity {label: 'MESSAGE'})
        RETURN m.name AS message, m.created_at AS timestamp
        """
        with self.app_state.get_driver_session() as neo4j_session:
            result = neo4j_session.run(query, session_id=session_id, relation_type=relation_type)
            data = [record.data() for record in result]
            print(f"Messages by session {session_id}: {data}")
            return data

    def get_users_by_session(self, session_id, relation_type):
        query = """
        MATCH (s:Entity {name: $session_id, label: 'SESSION'})<-[:RELATION {type: $relation_type}]-(u:Entity {label: 'USER'})
        RETURN u.name AS user_id, u.created_at AS created_at
        """
        with self.app_state.get_driver_session() as neo4j_session:
            result = neo4j_session.run(query, session_id=session_id, relation_type=relation_type)
            data = [record.data() for record in result]
            print(f"Users by session {session_id}: {data}")
            return data

    def get_sessions_by_user(self, user_id, relation_type):
        query = """
        MATCH (u:Entity {name: $user_id, label: 'USER'})-[:RELATION {type: $relation_type}]->(s:Entity {label: 'SESSION'})
        RETURN s.name AS session_id, s.created_at AS created_at
        """
        with self.app_state.get_driver_session() as neo4j_session:
            result = neo4j_session.run(query, user_id=user_id, relation_type=relation_type)
            data = [record.data() for record in result]
            print(f"Sessions by user {user_id}: {data}")
            return data

    def get_message_by_id(self, message_id):
        query = """
        MATCH (m:Entity {name: $message_id, label: 'MESSAGE'})
        RETURN m.name AS message_id, m.created_at AS timestamp
        """
        with self.app_state.get_driver_session() as neo4j_session:
            result = neo4j_session.run(query, message_id=message_id)
            data = [record.data() for record in result]
            print(f"Message {message_id}: {data}")
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