import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor, Json
from typing import List, Dict, Any
from topos.services.database.database_interface import DatabaseInterface

class PostgresDatabase(DatabaseInterface):
    def __init__(self, dbname: str, user: str, password: str, host: str = 'localhost', port: str = '5432'):
        print("[ In PostgresDatabase init ]")
        super().__init__()
        self.pool = psycopg2.pool.SimpleConnectionPool(1, 20,
            dbname=dbname, user=user, password=password, host=host, port=port
        )
        print(f"\t[ PostgresDatabase init ]")

    def __del__(self):
        if hasattr(self, 'pool'):
            self.pool.closeall()

    def _get_conn(self):
        return self.pool.getconn()

    def _put_conn(self, conn):
        self.pool.putconn(conn)

    def add_entity(self, entity_id: str, entity_label: str, properties: Dict[str, Any]) -> None:
        query = """
        INSERT INTO entities (id, label, properties)
        VALUES (%s, %s, %s)
        ON CONFLICT (id) DO UPDATE
        SET label = EXCLUDED.label, properties = EXCLUDED.properties
        """
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(query, (entity_id, entity_label, Json(properties)))
            conn.commit()
        finally:
            self._put_conn(conn)

    def add_relation(self, source_id: str, relation_type: str, target_id: str, properties: Dict[str, Any]) -> None:
        query = """
        INSERT INTO relations (source_id, relation_type, target_id, properties)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (source_id, relation_type, target_id) DO UPDATE
        SET properties = EXCLUDED.properties
        """
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(query, (source_id, relation_type, target_id, Json(properties)))
            conn.commit()
        finally:
            self._put_conn(conn)

    def get_messages_by_user(self, user_id: str, relation_type: str) -> List[Dict[str, Any]]:
        query = """
        SELECT e.id as message_id, e.properties->>'content' as message, e.properties->>'timestamp' as timestamp
        FROM relations r
        JOIN entities e ON r.target_id = e.id
        WHERE r.source_id = %s AND r.relation_type = %s AND e.label = 'MESSAGE'
        """
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (user_id, relation_type))
                return cur.fetchall()
        finally:
            self._put_conn(conn)

    def get_messages_by_session(self, session_id: str, relation_type: str) -> List[Dict[str, Any]]:
        query = """
        SELECT e.id as message_id, e.properties->>'content' as message, e.properties->>'timestamp' as timestamp
        FROM relations r
        JOIN entities e ON r.target_id = e.id
        WHERE r.source_id = %s AND r.relation_type = %s AND e.label = 'MESSAGE'
        """
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (session_id, relation_type))
                return cur.fetchall()
        finally:
            self._put_conn(conn)

    def get_users_by_session(self, session_id: str, relation_type: str) -> List[Dict[str, Any]]:
        query = """
        SELECT r.source_id as user_id
        FROM relations r
        WHERE r.target_id = %s AND r.relation_type = %s
        """
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (session_id, relation_type))
                return cur.fetchall()
        finally:
            self._put_conn(conn)

    def get_sessions_by_user(self, user_id: str, relation_type: str) -> List[Dict[str, Any]]:
        query = """
        SELECT r.target_id as session_id
        FROM relations r
        JOIN entities e ON r.target_id = e.id
        WHERE r.source_id = %s AND r.relation_type = %s AND e.label = 'SESSION'
        """
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (user_id, relation_type))
                return cur.fetchall()
        finally:
            self._put_conn(conn)

    def get_message_by_id(self, message_id: str) -> Dict[str, Any]:
        query = """
        SELECT properties->>'content' as message, properties->>'timestamp' as timestamp
        FROM entities
        WHERE id = %s AND label = 'MESSAGE'
        """
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (message_id,))
                result = cur.fetchone()
                return result if result else {}
        finally:
            self._put_conn(conn)

    def value_exists(self, label: str, key: str, value: str) -> bool:
        query = """
        SELECT 1
        FROM entities
        WHERE label = %s AND properties->>%s = %s
        LIMIT 1
        """
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(query, (label, key, value))
                return bool(cur.fetchone())
        finally:
            self._put_conn(conn)

    def override_conversational_cache(self, session_id: str, new_messages: List[Dict[str, Any]]) -> None:
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                # Remove existing messages for the session
                cur.execute("DELETE FROM relations WHERE source_id = %s AND relation_type = 'CONTAINS'", (session_id,))

                # Add new messages and relations
                for msg in new_messages:
                    cur.execute("""
                        INSERT INTO entities (id, label, properties)
                        VALUES (%s, 'MESSAGE', %s)
                        ON CONFLICT (id) DO UPDATE
                        SET properties = EXCLUDED.properties
                    """, (msg['message_id'], Json({'content': msg['content'], 'timestamp': msg['timestamp']})))

                    cur.execute("""
                        INSERT INTO relations (source_id, relation_type, target_id)
                        VALUES (%s, 'CONTAINS', %s)
                    """, (session_id, msg['message_id']))

            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            self._put_conn(conn)
