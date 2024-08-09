import unittest
import os
from dotenv import load_dotenv
from topos.services.database.postgres_database import PostgresDatabase
import psycopg2
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()


class TestPostgresDatabase(unittest.TestCase):
    """
    Integration tests for PostgresDatabase class.

    SETUP INSTRUCTIONS:
    1. Install PostgreSQL:
       brew install postgresql

    2. Start PostgreSQL service:
       brew services start postgresql

    3. Create a test database:
       createdb test_topos_db

    4. Install required Python packages:
       poetry add psycopg2-binary python-dotenv

    5. Create a .env file in the same directory as this test file with the following content:
       POSTGRES_DB=test_topos_db
       POSTGRES_USER=your_username
       POSTGRES_PASSWORD=your_password
       POSTGRES_HOST=localhost
       POSTGRES_PORT=5432

    6. Create necessary tables in your test database:
       psql test_topos_db

       Then run the following SQL commands:

       CREATE TABLE entities (
           id TEXT PRIMARY KEY,
           label TEXT NOT NULL,
           properties JSONB
       );

       CREATE TABLE relations (
           source_id TEXT,
           relation_type TEXT,
           target_id TEXT,
           properties JSONB,
           PRIMARY KEY (source_id, relation_type, target_id)
       );

       CREATE ROLE your_username WITH LOGIN PASSWORD 'your_password';
       GRANT ALL PRIVILEGES ON DATABASE test_topos_db TO your_username

       -- and maybe...

       psql -d test_topos_db
       GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO test_username;
       GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO test_username;
       GRANT ALL PRIVILEGES ON SCHEMA public TO test_username;

       ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO test_username;
       ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO test_username;

       -- If you're using PostgreSQL 14 or later, you can also add these:
       GRANT pg_read_all_data TO test_username;
       GRANT pg_write_all_data TO test_username;

    7. Run the tests:
       python -m unittest test_postgres_database.py
    """

    db = None

    @classmethod
    def setUpClass(cls):
        logging.info("Setting up TestPostgresDatabase class")
        cls.db = PostgresDatabase(
            dbname=os.getenv("POSTGRES_DB"),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD"),
            host=os.getenv("POSTGRES_HOST"),
            port=os.getenv("POSTGRES_PORT")
        )
        logging.info(f"Database connection established: {cls.db}")

    def setUp(self):
        logging.info("Setting up test case")
        conn = self.db._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE TABLE relations, entities RESTART IDENTITY")
            conn.commit()
            logging.info("Test tables cleared")
        except psycopg2.Error as e:
            logging.error(f"Error clearing test tables: {e}")
            conn.rollback()
        finally:
            self.db._put_conn(conn)

    def test_add_entity(self):
        logging.info("Running test_add_entity")
        entity_id = "test_entity_1"
        entity_label = "TEST_ENTITY"
        properties = {"name": "Test Entity", "value": 42}

        self.db.add_entity(entity_id, entity_label, properties)
        logging.info(f"Entity added: {entity_id}")

        conn = self.db._get_conn()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT * FROM entities WHERE id = %s", (entity_id,))
                result = cur.fetchone()

            self.assertIsNotNone(result)
            self.assertEqual(result['id'], entity_id)
            self.assertEqual(result['label'], entity_label)
            self.assertEqual(result['properties']['name'], "Test Entity")
            self.assertEqual(result['properties']['value'], 42)
            logging.info("Entity verification successful")
        finally:
            self.db._put_conn(conn)

    def test_add_relation(self):
        logging.info("Running test_add_relation")
        source_id = "source_entity"
        target_id = "target_entity"
        relation_type = "TEST_RELATION"
        properties = {"strength": 0.8}

        self.db.add_entity(source_id, "SOURCE", {})
        self.db.add_entity(target_id, "TARGET", {})
        logging.info("Source and target entities added")

        self.db.add_relation(source_id, relation_type, target_id, properties)
        logging.info("Relation added")

        conn = self.db._get_conn()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT * FROM relations WHERE source_id = %s AND target_id = %s", (source_id, target_id))
                result = cur.fetchone()

            self.assertIsNotNone(result)
            self.assertEqual(result['source_id'], source_id)
            self.assertEqual(result['target_id'], target_id)
            self.assertEqual(result['relation_type'], relation_type)
            self.assertEqual(result['properties']['strength'], 0.8)
            logging.info("Relation verification successful")
        finally:
            self.db._put_conn(conn)

    def test_get_messages_by_user(self):
        logging.info("Running test_get_messages_by_user")
        user_id = "test_user"
        message_ids = ["message1", "message2", "message3"]
        relation_type = "SENT"

        self.db.add_entity(user_id, "USER", {})
        for msg_id in message_ids:
            self.db.add_entity(msg_id, "MESSAGE",
                               {"content": f"Test message {msg_id}", "timestamp": "2023-01-01T00:00:00"})
            self.db.add_relation(user_id, relation_type, msg_id, {})
        logging.info("User and messages added")

        messages = self.db.get_messages_by_user(user_id, relation_type)
        logging.info(f"Retrieved {len(messages)} messages")

        self.assertEqual(len(messages), 3)
        for msg in messages:
            self.assertIn(msg['message_id'], message_ids)
            self.assertTrue(msg['message'].startswith("Test message"))
            self.assertEqual(msg['timestamp'], "2023-01-01T00:00:00")
        logging.info("Messages verification successful")

    def test_get_messages_by_session(self):
        logging.info("Running test_get_messages_by_session")
        session_id = "test_session"
        message_ids = ["message1", "message2", "message3"]
        relation_type = "CONTAINS"

        self.db.add_entity(session_id, "SESSION", {})
        for msg_id in message_ids:
            self.db.add_entity(msg_id, "MESSAGE",
                               {"content": f"Test message {msg_id}", "timestamp": "2023-01-01T00:00:00"})
            self.db.add_relation(session_id, relation_type, msg_id, {})
        logging.info("Session and messages added")

        messages = self.db.get_messages_by_session(session_id, relation_type)
        logging.info(f"Retrieved {len(messages)} messages")

        self.assertEqual(len(messages), 3)
        for msg in messages:
            self.assertIn(msg['message_id'], message_ids)
            self.assertTrue(msg['message'].startswith("Test message"))
            self.assertEqual(msg['timestamp'], "2023-01-01T00:00:00")
        logging.info("Messages verification successful")

    def test_get_users_by_session(self):
        logging.info("Running test_get_users_by_session")
        session_id = "test_session"
        user_ids = ["user1", "user2", "user3"]
        relation_type = "PARTICIPATED"

        self.db.add_entity(session_id, "SESSION", {})
        for user_id in user_ids:
            self.db.add_entity(user_id, "USER", {})
            self.db.add_relation(user_id, relation_type, session_id, {})
        logging.info("Session and users added")

        users = self.db.get_users_by_session(session_id, relation_type)
        logging.info(f"Retrieved {len(users)} users")

        self.assertEqual(len(users), 3)
        for user in users:
            self.assertIn(user['user_id'], user_ids)
        logging.info("Users verification successful")

    def test_get_sessions_by_user(self):
        logging.info("Running test_get_sessions_by_user")
        user_id = "test_user"
        session_ids = ["session1", "session2", "session3"]
        relation_type = "PARTICIPATED"

        self.db.add_entity(user_id, "USER", {})
        for session_id in session_ids:
            self.db.add_entity(session_id, "SESSION", {})
            self.db.add_relation(user_id, relation_type, session_id, {})
        logging.info("User and sessions added")

        sessions = self.db.get_sessions_by_user(user_id, relation_type)
        logging.info(f"Retrieved {len(sessions)} sessions")

        self.assertEqual(len(sessions), 3)
        for session in sessions:
            self.assertIn(session['session_id'], session_ids)
        logging.info("Sessions verification successful")

    def test_get_message_by_id(self):
        logging.info("Running test_get_message_by_id")
        message_id = "test_message"
        content = "This is a test message"
        timestamp = "2023-01-01T00:00:00"

        self.db.add_entity(message_id, "MESSAGE", {"content": content, "timestamp": timestamp})
        logging.info("Test message added")

        message = self.db.get_message_by_id(message_id)
        logging.info(f"Retrieved message: {message}")

        self.assertEqual(message['message'], content)
        self.assertEqual(message['timestamp'], timestamp)
        logging.info("Message verification successful")

    def test_value_exists(self):
        logging.info("Running test_value_exists")
        entity_id = "test_entity"
        entity_label = "TEST_ENTITY"
        properties = {"name": "Test Entity", "value": 42}

        self.db.add_entity(entity_id, entity_label, properties)
        logging.info("Test entity added")

        self.assertTrue(self.db.value_exists(entity_label, "name", "Test Entity"))
        self.assertFalse(self.db.value_exists(entity_label, "name", "Non-existent Entity"))
        logging.info("Value existence checks completed")


if __name__ == '__main__':
    unittest.main()