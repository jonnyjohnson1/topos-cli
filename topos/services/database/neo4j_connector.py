# neo4j_connector.py

from neo4j import GraphDatabase
from threading import Lock

class Neo4jConnection:
    _instance = None
    _lock = Lock()

    def __new__(cls, uri, user, password):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(Neo4jConnection, cls).__new__(cls)
                    cls._instance._init_connection(uri, user, password)
                    print("\t[ neo4j :: new instance created ]")
        else:
            print("\t[ neo4j :: returning singleton instance ]")
        return cls._instance

    def _init_connection(self, uri, user, password):
        print("\t\t[ neo4j :: init ]")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.closed = False

    def close(self):
        print("\t\t[ neo4j :: try close ]")
        if self.driver is not None and not self.closed:
            print("\t\t\t[ neo4j :: close successful ]")
            self.driver.close()
            self.closed = True

    def get_driver(self):
        if self.driver is None or self.closed:
            raise Exception("Driver has been closed.")
        return self.driver

    def create_database(self, database_name):
        with self.driver.session() as session:
            session.run(f"CREATE DATABASE {database_name} IF NOT EXISTS")


