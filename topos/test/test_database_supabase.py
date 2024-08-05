import unittest
import os

from dotenv import load_dotenv
from topos.services.database.supabase_database import SupabaseDatabase
from postgrest.exceptions import APIError

# Load environment variables
load_dotenv()

class TestSupabaseDatabase(unittest.TestCase):
    db = None
    supabase_key = None
    supabase_url = None
    entities_table = "fixed_entities"
    relations_table = "fixed_relations"

    @classmethod
    def setUpClass(cls):
        # Initialize the SupabaseDatabase with environment variables
        cls.supabase_url = os.getenv("SUPABASE_URL")
        cls.supabase_key = os.getenv("SUPABASE_KEY")
        cls.db = SupabaseDatabase(cls.supabase_url, cls.supabase_key)

    def setUp(self):
        # Clear the test tables before each test
        try:
            # Ensure using the 'id' column with a text condition
            self.db.client.table(self.entities_table).delete().neq('id', '0').execute()
            self.db.client.table(self.relations_table).delete().neq('id', '0').execute()
        except APIError as e:
            print(f"Error clearing test tables: {e}")
    def test_add_entity(self):
        entity_id = "test_entity_1"
        entity_label = "TEST_ENTITY"
        properties = {"name": "Test Entity", "value": 42}

        self.db.add_entity(entity_id, entity_label, properties, table_name=self.entities_table)

        # Verify the entity was added
        result = self.db.client.table(self.entities_table).select('*').eq('id', entity_id).execute()
        self.assertEqual(len(result.data), 1)
        self.assertEqual(result.data[0]['id'], entity_id)
        self.assertEqual(result.data[0]['label'], entity_label)
        self.assertEqual(result.data[0]['name'], "Test Entity")
        self.assertEqual(result.data[0]['value'], 42)

    def test_add_relation(self):
        source_id = "source_entity"
        target_id = "target_entity"
        relation_type = "TEST_RELATION"
        properties = {"strength": 0.8}

        # Add source and target entities first
        self.db.add_entity(source_id, "SOURCE", {}, table_name=self.entities_table)
        self.db.add_entity(target_id, "TARGET", {}, table_name=self.entities_table)

        self.db.add_relation(source_id, relation_type, target_id, properties, table_name=self.relations_table)

        # Verify the relation was added
        result = self.db.client.table(self.relations_table).select('*').eq('source_id', source_id).eq('target_id', target_id).execute()
        self.assertEqual(len(result.data), 1)
        self.assertEqual(result.data[0]['source_id'], source_id)
        self.assertEqual(result.data[0]['target_id'], target_id)
        self.assertEqual(result.data[0]['relation_type'], relation_type)
        self.assertEqual(result.data[0]['strength'], 0.8)

    def test_get_messages_by_user(self):
        user_id = "test_user"
        message_ids = ["message1", "message2", "message3"]
        relation_type = "SENT"

        # Add user and messages
        self.db.add_entity(user_id, "USER", {}, table_name=self.entities_table)
        for msg_id in message_ids:
            self.db.add_entity(msg_id, "MESSAGE", {"content": f"Test message {msg_id}", "timestamp": "2023-01-01T00:00:00"}, table_name=self.entities_table)
            self.db.add_relation(user_id, relation_type, msg_id, {}, table_name=self.relations_table)

        messages = self.db.get_messages_by_user(user_id, relation_type)

        self.assertEqual(len(messages), 3)
        for msg in messages:
            self.assertIn(msg['message_id'], message_ids)
            self.assertTrue(msg['message'].startswith("Test message"))
            self.assertEqual(msg['timestamp'], "2023-01-01T00:00:00")

    def test_get_messages_by_session(self):
        session_id = "test_session"
        message_ids = ["message1", "message2", "message3"]
        relation_type = "CONTAINS"

        # Add session and messages
        self.db.add_entity(session_id, "SESSION", {}, table_name=self.entities_table)
        for msg_id in message_ids:
            self.db.add_entity(msg_id, "MESSAGE", {"content": f"Test message {msg_id}", "timestamp": "2023-01-01T00:00:00"}, table_name=self.entities_table)
            self.db.add_relation(session_id, relation_type, msg_id, {}, table_name=self.relations_table)

        messages = self.db.get_messages_by_session(session_id, relation_type)

        self.assertEqual(len(messages), 3)
        for msg in messages:
            self.assertIn(msg['message_id'], message_ids)
            self.assertTrue(msg['message'].startswith("Test message"))
            self.assertEqual(msg['timestamp'], "2023-01-01T00:00:00")

    def test_get_users_by_session(self):
        session_id = "test_session"
        user_ids = ["user1", "user2", "user3"]
        relation_type = "PARTICIPATED"

        # Add session and users
        self.db.add_entity(session_id, "SESSION", {}, table_name=self.entities_table)
        for user_id in user_ids:
            self.db.add_entity(user_id, "USER", {}, table_name=self.entities_table)
            self.db.add_relation(user_id, relation_type, session_id, {}, table_name=self.relations_table)

        users = self.db.get_users_by_session(session_id, relation_type)

        self.assertEqual(len(users), 3)
        for user in users:
            self.assertIn(user['user_id'], user_ids)

    def test_get_sessions_by_user(self):
        user_id = "test_user"
        session_ids = ["session1", "session2", "session3"]
        relation_type = "PARTICIPATED"

        # Add user and sessions
        self.db.add_entity(user_id, "USER", {}, table_name=self.entities_table)
        for session_id in session_ids:
            self.db.add_entity(session_id, "SESSION", {}, table_name=self.entities_table)
            self.db.add_relation(user_id, relation_type, session_id, {}, table_name=self.relations_table)

        sessions = self.db.get_sessions_by_user(user_id, relation_type)

        self.assertEqual(len(sessions), 3)
        for session in sessions:
            self.assertIn(session['session_id'], session_ids)

    def test_get_message_by_id(self):
        message_id = "test_message"
        content = "This is a test message"
        timestamp = "2023-01-01T00:00:00"

        self.db.add_entity(message_id, "MESSAGE", {"content": content, "timestamp": timestamp}, table_name=self.entities_table)

        message = self.db.get_message_by_id(message_id)

        self.assertEqual(message['message'], content)
        self.assertEqual(message['timestamp'], timestamp)

    def test_value_exists(self):
        entity_id = "test_entity"
        entity_label = "TEST_ENTITY"
        properties = {"name": "Test Entity", "value": 42}

        self.db.add_entity(entity_id, entity_label, properties, table_name=self.entities_table)

        self.assertTrue(self.db.value_exists(entity_label, "name", "Test Entity"))
        self.assertFalse(self.db.value_exists(entity_label, "name", "Non-existent Entity"))

if __name__ == '__main__':
    unittest.main()
