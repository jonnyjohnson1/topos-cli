# ontological_feature_detection.py

import nltk
import spacy
import warnings
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from neo4j import GraphDatabase

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
    def __init__(self, neo4j_uri, neo4j_user, neo4j_password):
        # Initialize the tokenizer and model for SRL
        self.tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
        self.model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

        # Load SpaCy models
        self.nlp = spacy.load('en_core_web_sm')

        # Initialize Neo4j connection
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    def close(self):
        self.driver.close()

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
        # Placeholder for relation extraction
        relations = []
        print(f"Relation extraction results: {relations}")
        return relations

    def add_entity(self, tx, entity, label):
        print(f"Adding entity: {entity}, label: {label}")
        tx.run("MERGE (e:Entity {name: $entity, label: $label})", entity=entity, label=label)

    def add_relation(self, tx, entity1, relation, entity2):
        print(f"Adding relation: ({entity1})-[:{relation}]->({entity2})")
        tx.run("MATCH (a:Entity {name: $entity1}), (b:Entity {name: $entity2}) "
               "MERGE (a)-[r:RELATION {type: $relation}]->(b)",
               entity1=entity1, relation=relation, entity2=entity2)

    def build_ontology_from_paragraph(self, text):
        entities = self.perform_ner(text)
        pos_tags = self.perform_pos_tagging(text)
        dependencies = self.perform_dependency_parsing(text)
        srl_results = self.perform_srl(text)
        relations = self.perform_relation_extraction(text)

        with self.driver.session() as session:
            for entity, label in entities:
                session.execute_write(self.add_entity, entity, label)

            for relation in relations:
                session.execute_write(self.add_relation, relation['entity1'], relation['relation'], relation['entity2'])

        return entities, dependencies, srl_results

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

        topic = parsed_data["topic"]
        entities = [(topic, "Topic")]
        relations = []

        for concept in parsed_data["concepts"]:
            if len(concept) == 2:
                index, description = concept
                entities.append((description, "Concept"))
                relations.append((topic, f"related_to_{index}", description))

        with self.driver.session() as session:
            for entity, label in entities:
                session.execute_write(self.add_entity, entity, label)

            for relation in relations:
                session.execute_write(self.add_relation, relation[0], relation[1], relation[2])

        return entities, relations

    def extract_mermaid_syntax(self, input_data, input_type="paragraph"):
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

        mermaid_syntax = "graph LR\n"
        for node in entity_set:
            mermaid_syntax += f"    {node}\n"
        for edge in edges:
            mermaid_syntax += f"    {edge}\n"

        return mermaid_syntax


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
#         "They discussed a variety of topics including the recent advancements in artificial intelligence, "
#         "machine learning, and the future of technology. Alice suggested attending the AI conference in San Francisco next month."
#     )
#     mermaid_syntax_paragraph = ofd.extract_mermaid_syntax(paragraph, input_type="paragraph")
#     print("Mermaid Syntax for Paragraph Input:")
#     print(mermaid_syntax_paragraph)
#
#     # Example with semantically compressed data input
#     compressed_data = "Theoretical Computer Science::1=field within theoretical computer science;2=inherent difficulty;3=solve computational problems;4=achievable with algorithms and computation"
#     mermaid_syntax_compressed = ofd.extract_mermaid_syntax(compressed_data, input_type="compressed_data")
#     print("Mermaid Syntax for Compressed Data Input:")
#     print(mermaid_syntax_compressed)
#
#     ofd.close()
