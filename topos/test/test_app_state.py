# test_app_state.py

import unittest
from unittest.mock import MagicMock, patch
from topos.services.database.app_state import AppState
from topos.services.database.database_interface import DatabaseInterface

class TestAppState(unittest.TestCase):

    def setUp(self):
        # Reset the AppState singleton before each test
        AppState._instance = None
        AppState._initialized = False

    def test_singleton_instance(self):
        print("\t[ Test: Singleton Instance ]")
        instance1 = AppState()
        instance2 = AppState()
        self.assertIs(instance1, instance2)

    @patch('topos.services.database.app_state.Neo4jDatabase')
    def test_neo4j_initialization(self, mock_neo4j):
        print("\t[ Test: Neo4j Initialization ]")
        mock_neo4j.return_value = MagicMock(spec=DatabaseInterface)
        app_state = AppState(db_type="neo4j", neo4j_uri="uri", neo4j_user="user", neo4j_password="pass", neo4j_db_name="db")
        self.assertIsNotNone(app_state.db)
        self.assertEqual(app_state.db_type, "neo4j")
        mock_neo4j.assert_called_once_with("uri", "user", "pass", "db")

    @patch('topos.services.database.app_state.SupabaseDatabase')
    def test_supabase_initialization(self, mock_supabase):
        print("\t[ Test: Supabase Initialization ]")
        mock_supabase.return_value = MagicMock(spec=DatabaseInterface)
        app_state = AppState(db_type="supabase", supabase_url="url", supabase_key="key")
        self.assertIsNotNone(app_state.db)
        self.assertEqual(app_state.db_type, "supabase")
        mock_supabase.assert_called_once_with("url", "key")

    def test_invalid_db_type(self):
        print("\t[ Test: Invalid DB Type ]")
        with self.assertRaises(ValueError):
            AppState(db_type="invalid")

    def test_get_uninitialized_db(self):
        print("\t[ Test: Get Uninitialized DB ]")
        app_state = AppState()
        with self.assertRaises(Exception):
            app_state.get_db()

    def test_state_operations(self):
        print("\t[ Test: State Operations ]")
        app_state = AppState()
        app_state.set_state("key", "value")
        self.assertEqual(app_state.get_value("key"), "value")
        self.assertEqual(app_state.get_value("nonexistent", "default"), "default")

    def test_ontology_operations(self):
        print("\t[ Test: Ontology Operations ]")
        app_state = AppState()
        ontology1 = {"entity": "test1"}
        ontology2 = {"entity": "test2"}
        app_state.write_ontology(ontology1)
        app_state.write_ontology(ontology2)
        self.assertEqual(app_state.read_ontology(), [ontology1, ontology2])

    @patch('topos.services.database.app_state.Neo4jDatabase')
    def test_value_exists(self, mock_neo4j):
        print("\t[ Test: Value Exists ]")
        mock_db = MagicMock(spec=DatabaseInterface)
        mock_db.value_exists.return_value = True
        mock_neo4j.return_value = mock_db
        app_state = AppState(db_type="neo4j", neo4j_uri="uri", neo4j_user="user", neo4j_password="pass", neo4j_db_name="db")
        self.assertTrue(app_state.value_exists("Label", "key", "value"))
        mock_db.value_exists.assert_called_once_with("Label", "key", "value")

    @patch('topos.services.database.app_state.Neo4jDatabase')
    def test_close(self, mock_neo4j):
        print("\t[ Test: Close ]")
        mock_db = MagicMock(spec=DatabaseInterface)
        mock_db.close = MagicMock()  # Explicitly add close method to the mock
        mock_neo4j.return_value = mock_db
        app_state = AppState(db_type="neo4j", neo4j_uri="uri", neo4j_user="user", neo4j_password="pass",
                             neo4j_db_name="db")
        app_state.close()
        mock_db.close.assert_called_once()
        self.assertEqual(app_state.state, {})
        self.assertFalse(app_state._initialized)

    @patch('topos.services.database.app_state.Neo4jDatabase')
    @patch('topos.services.database.app_state.SupabaseDatabase')
    def test_switch_database(self, mock_supabase, mock_neo4j):
        print("\t[ Test: Switch Database ]")
        mock_neo4j.return_value = MagicMock(spec=DatabaseInterface)
        mock_supabase.return_value = MagicMock(spec=DatabaseInterface)
        app_state = AppState(db_type="neo4j", neo4j_uri="uri", neo4j_user="user", neo4j_password="pass", neo4j_db_name="db")
        self.assertEqual(app_state.db_type, "neo4j")
        app_state.set_database("supabase", supabase_url="url", supabase_key="key")
        self.assertEqual(app_state.db_type, "supabase")
        mock_supabase.assert_called_once_with("url", "key")

if __name__ == '__main__':
    unittest.main()