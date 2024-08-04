# neo4j_database.py

from topos.services.database.database_interface import DatabaseInterface
from neo4j import GraphDatabase
from typing import List, Dict, Any

class Neo4jDatabase(DatabaseInterface):
    """
    Neo4j's implementation of the DatabaseInterface.
    """

    def __init__(self, uri: str, user: str, password: str, database: str):
        """
        Initialize the Neo4j database connection.

        :param uri: URI of the Neo4j database
        :param user: Username for authentication
        :param password: Password for authentication
        :param database: Name of the database to use
        """
        super().__init__()
        print(f"\t[ Neo4jDatabase init ]")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        print(f"\t\t[ Neo4jDatabase uri :: {uri} ]")
        print(f"\t\t[ Neo4jDatabase database :: {database} ]")

    def add_entity(self, entity_id: str, entity_label: str, properties: Dict[str, Any]) -> None:
        with self.driver.session(database=self.database) as session:
            session.run(
                f"MERGE (e:{entity_label} {{id: $entity_id}}) SET e += $properties",
                entity_id=entity_id, properties=properties
            )
        print(f"\t[ Neo4jDatabase add_entity :: {entity_id}, {entity_label} ]")

    def add_relation(self, source_id: str, relation_type: str, target_id: str, properties: Dict[str, Any]) -> None:
        with self.driver.session(database=self.database) as session:
            session.run(
                f"MATCH (source {{id: $source_id}}), (target {{id: $target_id}}) "
                f"MERGE (source)-[r:{relation_type}]->(target) SET r += $properties",
                source_id=source_id, target_id=target_id, properties=properties
            )
        print(f"\t[ Neo4jDatabase add_relation :: {source_id} -[{relation_type}]-> {target_id} ]")

    def get_messages_by_user(self, user_id: str, relation_type: str) -> List[Dict[str, Any]]:
        with self.driver.session(database=self.database) as session:
            result = session.run(
                f"MATCH (u:USER {{id: $user_id}})-[r:{relation_type}]->(m:MESSAGE) "
                "RETURN m.id AS message_id, m.content AS message, m.timestamp AS timestamp",
                user_id=user_id
            )
            return [dict(record) for record in result]

    def get_messages_by_session(self, session_id: str, relation_type: str) -> List[Dict[str, Any]]:
        with self.driver.session(database=self.database) as session:
            result = session.run(
                f"MATCH (s:SESSION {{id: $session_id}})-[r:{relation_type}]->(m:MESSAGE) "
                "RETURN m.id AS message_id, m.content AS message, m.timestamp AS timestamp",
                session_id=session_id
            )
            return [dict(record) for record in result]

    def get_users_by_session(self, session_id: str, relation_type: str) -> List[Dict[str, Any]]:
        with self.driver.session(database=self.database) as session:
            result = session.run(
                f"MATCH (s:SESSION {{id: $session_id}})<-[r:{relation_type}]-(u:USER) "
                "RETURN u.id AS user_id",
                session_id=session_id
            )
            return [dict(record) for record in result]

    def get_sessions_by_user(self, user_id: str, relation_type: str) -> List[Dict[str, Any]]:
        with self.driver.session(database=self.database) as session:
            result = session.run(
                f"MATCH (u:USER {{id: $user_id}})-[r:{relation_type}]->(s:SESSION) "
                "RETURN s.id AS session_id",
                user_id=user_id
            )
            return [dict(record) for record in result]

    def get_message_by_id(self, message_id: str) -> Dict[str, Any]:
        with self.driver.session(database=self.database) as session:
            result = session.run(
                "MATCH (m:MESSAGE {id: $message_id}) "
                "RETURN m.content AS message, m.timestamp AS timestamp",
                message_id=message_id
            )
            return dict(result.single())

    def value_exists(self, label: str, key: str, value: str) -> bool:
        with self.driver.session(database=self.database) as session:
            result = session.run(
                f"MATCH (n:{label} {{{key}: $value}}) RETURN COUNT(n) > 0 AS exists",
                value=value
            )
            return result.single()["exists"]

    def close(self):
        """
        Close the Neo4j driver connection.
        """
        self.driver.close()
        print(f"\t[ Neo4jDatabase connection closed ]")