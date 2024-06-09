# app_state.py
from datetime import datetime
from threading import Lock
from neo4j import GraphDatabase
from topos.services.database.neo4j_connector import Neo4jConnection


class AppState:
    driver = None
    _instance = None
    _lock = Lock()

    def __new__(cls, neo4j_uri=None, neo4j_user=None, neo4j_password=None, neo4j_db_name=None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(AppState, cls).__new__(cls)
                    cls._instance._init_state(neo4j_uri, neo4j_user, neo4j_password, neo4j_db_name)
                    print("\t[ app_state :: new instance created ]")
        else:
            print("\t[ app_state :: returning singleton instance ]")
        return cls._instance

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            raise Exception("AppState has not been initialized, call AppState with parameters first.")
        return cls._instance

    def _init_state(self, neo4j_uri, neo4j_user, neo4j_password, neo4j_db_name):
        print("\t\t[ app_state :: init ]")
        self.state = {}
        self.neo4j_db_name = neo4j_db_name

        # Initialize Neo4j connection using singleton
        self.neo4j_conn = Neo4jConnection(neo4j_uri, neo4j_user, neo4j_password)
        self.driver = self.neo4j_conn.get_driver()
        self.closed = False

    def get_driver(self):
        return self.driver

    def get_driver_session(self):
        return self.driver.session(database=self.neo4j_db_name)

    def get_state(self, key):
        return self.state

    def set_state(self, key, value):
        self.state[key] = value

    def get_value(self, key, default=None):
        return self.state.get(key, default)

    def write_ontology(self, ontology):
        if 'ontology' not in self.state:
            self.state['ontology'] = []
        self.state['ontology'].append(ontology)

    def read_ontology(self):
        return self.state.get('ontology', [])

    def close(self):
        print("\t\t[ app_state :: try close ]")

        # Only close the connection if it is not shared by other instances
        if self.neo4j_conn:
            self.neo4j_conn.close()

        self.state = {}
        print("\t\t\t[ app_state :: close successful ]")

    # @staticmethod
    # def add_entity(tx, entity_id: str, entity_label: str, properties: dict):
    #     query = "MERGE (e:{label} {{id: $entity_id}})\n".format(label=entity_label)
    #     for key, value in properties.items():
    #         query += "SET e.{key} = ${key}\n".format(key=key)
    #     tx.run(query, entity_id=entity_id, **properties)
    #
    # @staticmethod
    # def add_relation(self, tx, source_id: str, relation_type: str, target_id: str, properties: dict):
    #     query = (
    #         "MATCH (source), (target)\n"
    #         "WHERE source.id = $source_id AND target.id = $target_id\n"
    #         "MERGE (source)-[r:{relation_type}]->(target)\n".format(relation_type=relation_type)
    #     )
    #     for key, value in properties.items():
    #         query += "SET r.{key} = ${key}\n".format(key=key)
    #     tx.run(query, source_id=source_id, target_id=target_id, **properties)

    def value_exists(self, label, key, value):
        with self.get_driver_session() as session:
            result = session.run(
                f"MATCH (n:{label} {{{key}: $value}}) RETURN COUNT(n) > 0 AS exists",
                value=value
            )
            return result.single()["exists"]

