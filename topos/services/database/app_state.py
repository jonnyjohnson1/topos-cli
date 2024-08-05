# app_state.py

from datetime import datetime
from threading import Lock
from typing import Union, Dict, Any
from topos.services.database.database_interface import DatabaseInterface
from topos.services.database.neo4j_database import Neo4jDatabase
from topos.services.database.supabase_database import SupabaseDatabase

class AppState:
    _instance = None
    _lock = Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(AppState, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, db_type: str = None, **kwargs):
        if self._initialized:
            return
        self._init_state(db_type, **kwargs)
        self._initialized = True
        print("\t[ AppState :: instance initialized ]")

    @classmethod
    def get_instance(cls):
        if cls._instance is None or not cls._instance._initialized:
            raise Exception("AppState has not been initialized, call AppState with parameters first.")
        return cls._instance

    def _init_state(self, db_type: str = None, **kwargs):
        print("\t\t[ AppState :: init ]")
        self.state: Dict[str, Any] = {}
        self.db: Union[DatabaseInterface, None] = None
        self.db_type: Union[str, None] = None

        if db_type:
            self.set_database(db_type, **kwargs)

    def set_database(self, db_type: str, **kwargs):
        print(f"\t\t[ AppState :: setting database to {db_type} ]")
        if db_type == "neo4j":
            self.db = Neo4jDatabase(
                kwargs.get('neo4j_uri'),
                kwargs.get('neo4j_user'),
                kwargs.get('neo4j_password'),
                kwargs.get('neo4j_db_name')
            )
            self.db_type = "neo4j"
        elif db_type == "supabase":
            self.db = SupabaseDatabase(
                kwargs.get('supabase_url'),
                kwargs.get('supabase_key')
            )
            self.db_type = "supabase"
        else:
            raise ValueError(f"Unsupported database type: {db_type}")

    def get_db(self) -> DatabaseInterface:
        if not self.db:
            raise Exception("Database has not been initialized.")
        return self.db

    def get_state(self) -> Dict[str, Any]:
        return self.state

    def set_state(self, key: str, value: Any):
        self.state[key] = value

    def get_value(self, key: str, default: Any = None) -> Any:
        return self.state.get(key, default)

    def write_ontology(self, ontology: Dict[str, Any]):
        if 'ontology' not in self.state:
            self.state['ontology'] = []
        self.state['ontology'].append(ontology)

    def read_ontology(self) -> list:
        return self.state.get('ontology', [])

    def close(self):
        print("\t\t[ AppState :: try close ]")
        if self.db:
            self.db.close()
        self.state = {}
        self._initialized = False
        print("\t\t\t[ AppState :: close successful ]")

    def value_exists(self, label: str, key: str, value: str) -> bool:
        return self.get_db().value_exists(label, key, value)