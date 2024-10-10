import psycopg2.extras
import threading

from topos.FC.conversation_cache_manager import ConversationCacheManager
import json

import shutil
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

       CREATE ROLE test_username WITH LOGIN PASSWORD 'test_password';
       GRANT ALL PRIVILEGES ON DATABASE test_topos_db TO test_username

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
        cls._ensure_table_exists()

    def setUp(self):
        logging.info("Setting up test case")
        conn = self.db._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE TABLE relations, entities, conversation_cache RESTART IDENTITY")
            conn.commit()
            logging.info("Test tables cleared")
        except psycopg2.Error as e:
            logging.error(f"Error clearing test tables: {e}")
            conn.rollback()
        finally:
            self.db._put_conn(conn)

    @classmethod
    def _ensure_table_exists(cls):
        conn = cls.db._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS conversation_cache (
                        conv_id TEXT PRIMARY KEY,
                        message_data JSONB
                    )
                """)
            conn.commit()
        except psycopg2.Error as e:
            logging.error(f"Error creating conversation_cache table: {e}")
            conn.rollback()
        finally:
            cls.db._put_conn(conn)

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

    def test_override_conversational_cache(self):
        logging.info("Running test_override_conversational_cache")
        session_id = "test_session_cache"
        user_id = "test_user_cache"
        message_ids = ["message1", "message2", "message3"]
        relation_type = "CONTAINS"

        # Set up initial session and messages
        self.db.add_entity(session_id, "SESSION", {})
        self.db.add_entity(user_id, "USER", {})
        for msg_id in message_ids:
            self.db.add_entity(msg_id, "MESSAGE",
                            {"content": f"Test message {msg_id}", "timestamp": "2023-01-01T00:00:00"})
            self.db.add_relation(session_id, relation_type, msg_id, {})
        self.db.add_relation(user_id, "PARTICIPATED", session_id, {})
        logging.info("Initial session, user, and messages added")

        # Test initial cache
        initial_messages = self.db.get_messages_by_session(session_id, relation_type)
        self.assertEqual(len(initial_messages), 3)

        # Override cache with new messages
        new_message_ids = ["new_message1", "new_message2"]
        new_messages = [
            {"message_id": "new_message1", "content": "New test message 1", "timestamp": "2023-01-02T00:00:00"},
            {"message_id": "new_message2", "content": "New test message 2", "timestamp": "2023-01-02T00:00:01"}
        ]
        self.db.override_conversational_cache(session_id, new_messages)
        logging.info("Conversational cache overridden")

        # Verify overridden cache
        updated_messages = self.db.get_messages_by_session(session_id, relation_type)
        self.assertEqual(len(updated_messages), 2)
        for msg in updated_messages:
            self.assertIn(msg['message_id'], new_message_ids)
            self.assertTrue(msg['message'].startswith("New test message"))
            self.assertTrue(msg['timestamp'].startswith("2023-01-02"))
        logging.info("Overridden cache verification successful")

    def test_conversation_cache_manager_file(self):
        logging.info("Running test_conversation_cache_manager_file")
        cache_dir = "./_test_conv_cache"
        cache_manager = ConversationCacheManager(cache_dir=cache_dir, use_postgres=False)

        conv_id = "test_conversation"
        message_data = {
            "message1": {"content": "Test message 1", "timestamp": "2023-01-01T00:00:00"},
            "message2": {"content": "Test message 2", "timestamp": "2023-01-01T00:00:01"}
        }

        # Save to cache
        cache_manager.save_to_cache(conv_id, message_data)

        # Load from cache
        loaded_data = cache_manager.load_from_cache(conv_id)

        self.assertIsNotNone(loaded_data)
        self.assertIn(conv_id, loaded_data)

        loaded_conv_data = loaded_data[conv_id]

        # Check if all original messages are in the loaded data
        for msg_id, msg_content in message_data.items():
            self.assertIn(msg_id, loaded_conv_data)
            self.assertEqual(loaded_conv_data[msg_id], msg_content)

        # Check if the number of messages is the same
        self.assertEqual(len(loaded_conv_data), len(message_data))

        # Clean up
        cache_manager.clear_cache()
        shutil.rmtree(cache_dir, ignore_errors=True)

    def test_conversation_cache_manager_postgres(self):
        logging.info("Running test_conversation_cache_manager_postgres")
        db_config = {
            "dbname": os.getenv("POSTGRES_DB"),
            "user": os.getenv("POSTGRES_USER"),
            "password": os.getenv("POSTGRES_PASSWORD"),
            "host": os.getenv("POSTGRES_HOST"),
            "port": os.getenv("POSTGRES_PORT")
        }
        logging.info(f"Database configuration: {db_config}")
        try:
            cache_manager = ConversationCacheManager(use_postgres=True, db_config=db_config)
        except Exception as e:
            logging.error(f"Failed to create ConversationCacheManager: {e}")
            raise

        # Clear the cache before starting the test
        try:
            cache_manager.clear_cache()
            logging.info("Cache cleared successfully")
        except Exception as e:
            logging.error(f"Failed to clear cache: {e}")
            raise

        conv_id = "test_conversation_pg"
        message_data = {
            "message1": {"content": "Test message 1", "timestamp": "2023-01-01T00:00:00"},
            "message2": {"content": "Test message 2", "timestamp": "2023-01-01T00:00:01"}
        }

        # Save to cache
        try:
            cache_manager.save_to_cache(conv_id, message_data)
            logging.info(f"Data saved to cache for conversation: {conv_id}")
        except Exception as e:
            logging.error(f"Failed to save data to cache: {e}")
            raise

        # Verify data is in the database
        try:
            with psycopg2.connect(**db_config) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT message_data FROM conversation_cache WHERE conv_id = %s", (conv_id,))
                    result = cur.fetchone()
                    logging.info(f"Query result: {result}")
                    self.assertIsNotNone(result, "No data found in database")
                    self.assertEqual(result[0], message_data, "Data in database does not match expected data")
            logging.info("Data verification in database successful")
        except Exception as e:
            logging.error(f"Failed to verify data in database: {e}", exc_info=True)
            raise

        # Load from cache
        try:
            loaded_data = cache_manager.load_from_cache(conv_id)
            logging.info(f"Data loaded from cache for conversation: {conv_id}")
        except Exception as e:
            logging.error(f"Failed to load data from cache: {e}")
            raise

        self.assertIsNotNone(loaded_data, "Loaded data is None")
        self.assertIn(conv_id, loaded_data, f"Conversation ID {conv_id} not found in loaded data")
        loaded_conv_data = loaded_data[conv_id]

        # Check if all original messages are in the loaded data
        for msg_id, msg_content in message_data.items():
            self.assertIn(msg_id, loaded_conv_data, f"Message ID {msg_id} not found in loaded data")
            self.assertEqual(loaded_conv_data[msg_id], msg_content, f"Message content for {msg_id} does not match")

        # Check if the number of messages is the same
        self.assertEqual(len(loaded_conv_data), len(message_data), "Number of messages does not match")

        # Test updating existing conversation
        updated_message_data = {
            "message1": {"content": "Updated message 1", "timestamp": "2023-01-02T00:00:00"},
            "message3": {"content": "New message 3", "timestamp": "2023-01-02T00:00:01"}
        }
        try:
            cache_manager.save_to_cache(conv_id, updated_message_data)
            logging.info(f"Updated data saved to cache for conversation: {conv_id}")
        except Exception as e:
            logging.error(f"Failed to save updated data to cache: {e}")
            raise

        try:
            updated_loaded_data = cache_manager.load_from_cache(conv_id)
            logging.info(f"Updated data loaded from cache for conversation: {conv_id}")
        except Exception as e:
            logging.error(f"Failed to load updated data from cache: {e}")
            raise

        self.assertIsNotNone(updated_loaded_data, "Updated loaded data is None")
        self.assertIn(conv_id, updated_loaded_data, f"Conversation ID {conv_id} not found in updated loaded data")
        self.assertEqual(updated_loaded_data[conv_id], updated_message_data, "Updated data does not match expected data")

        # Test clearing cache
        try:
            cache_manager.clear_cache()
            logging.info("Cache cleared successfully")
        except Exception as e:
            logging.error(f"Failed to clear cache: {e}")
            raise

        try:
            cleared_data = cache_manager.load_from_cache(conv_id)
            self.assertIsNone(cleared_data, "Cleared data is not None")
            logging.info("Cache clearing verified successfully")
        except Exception as e:
            logging.error(f"Failed to verify cache clearing: {e}")
            raise

        # Verify database state after clearing
        try:
            with psycopg2.connect(**db_config) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM conversation_cache")
                    count = cur.fetchone()[0]
                    self.assertEqual(count, 0, "Database should be empty after clearing cache")
            logging.info("Database state after clearing verified successfully")
        except (psycopg2.Error, AssertionError) as e:
            logging.error(f"Failed to verify database state after clearing: {e}")
            raise

        # Test concurrent access
        threads = []
        for i in range(5):
            thread = threading.Thread(target=self._concurrent_save_load, args=(cache_manager, f"conv_{i}"))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Verify all conversations were saved
        try:
            with psycopg2.connect(**db_config) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM conversation_cache")
                    count = cur.fetchone()[0]
                    self.assertEqual(count, 5, "Database should contain 5 conversations after concurrent access")
            logging.info("Concurrent access test completed successfully")
        except (psycopg2.Error, AssertionError) as e:
            logging.error(f"Failed to verify database state after concurrent access: {e}")
            raise

    def _concurrent_save_load(self, cache_manager, conv_id):
        message_data = {
            "message": {"content": f"Concurrent message for {conv_id}", "timestamp": "2023-01-03T00:00:00"}
        }
        cache_manager.save_to_cache(conv_id, message_data)
        loaded_data = cache_manager.load_from_cache(conv_id)
        self.assertIsNotNone(loaded_data)
        self.assertIn(conv_id, loaded_data)
        self.assertEqual(loaded_data[conv_id], message_data, f"Concurrent operation failed for {conv_id}")


if __name__ == '__main__':
    unittest.main()
