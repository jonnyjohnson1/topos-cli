# app_state.py
from datetime import datetime
from threading import Lock
from neo4j import GraphDatabase
from topos.services.database.neo4j_connector import Neo4jConnection


class AppState:
    _instance = None
    _lock = Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(AppState, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, neo4j_uri=None, neo4j_user=None, neo4j_password=None, neo4j_db_name=None, use_neo4j=True):
        if self._initialized:
            return
        self._init_state(neo4j_uri, neo4j_user, neo4j_password, neo4j_db_name, use_neo4j)
        self._initialized = True
        print("\t[ app_state :: instance initialized ]")

    @classmethod
    def get_instance(cls):
        if cls._instance is None or not cls._instance._initialized:
            raise Exception("AppState has not been initialized, call AppState with parameters first.")
        return cls._instance

    def _init_state(self, neo4j_uri, neo4j_user, neo4j_password, neo4j_db_name, use_neo4j=True):
        print("\t\t[ app_state :: init ]")
        self.state = {}
        self.neo4j_db_name = neo4j_db_name
        self.driver = None
        self.neo4j_conn = None

        if use_neo4j:
            # Initialize Neo4j connection using singleton
            self.neo4j_conn = Neo4jConnection(neo4j_uri, neo4j_user, neo4j_password)
            self.driver = self.neo4j_conn.get_driver()

    def get_driver(self):
        return self.driver

    def get_driver_session(self):
        if not self.driver:
            raise Exception("Neo4j driver is not initialized.")
        return self.driver.session(database=self.neo4j_db_name)

    def get_state(self):
        return self.state

    def set_state(self, key, value):
        self.state[key] = value

    def set_value(self, key, value):
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
        if self.neo4j_conn:
            self.neo4j_conn.close()
        self.state = {}
        self._initialized = False
        print("\t\t\t[ app_state :: close successful ]")

    def value_exists(self, label, key, value):
        if not self.driver:
            raise Exception("Neo4j driver is not initialized.")
        with self.get_driver_session() as session:
            result = session.run(
                f"MATCH (n:{label} {{{key}: $value}}) RETURN COUNT(n) > 0 AS exists",
                value=value
            )
            return result.single()["exists"]