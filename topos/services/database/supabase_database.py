from supabase import create_client, Client
from typing import List, Dict, Any
from topos.services.database.database_interface import DatabaseInterface

class SupabaseDatabase(DatabaseInterface):
    """
    Supabase implementation of the DatabaseInterface.
    """

    def __init__(self, url: str, key: str):
        """
        Initialize the Supabase client.

        :param url: Supabase project URL
        :param key: Supabase project API key
        """
        super().__init__()
        print(f"\t[ SupabaseDatabase init ]")
        self.client: Client = create_client(url, key)
        print(f"\t\t[ SupabaseDatabase url :: {url} ]")

    def add_entity(self, entity_id: str, entity_label: str, properties: Dict[str, Any], table_name: str = 'fixed_entities') -> None:
        data = {"id": entity_id, "label": entity_label, **properties}
        self.client.table(table_name).upsert(data).execute()
        print(f"\t[ SupabaseDatabase add_entity :: {entity_id}, {entity_label} to {table_name} ]")

    def add_relation(self, source_id: str, relation_type: str, target_id: str, properties: Dict[str, Any], table_name: str = 'fixed_relations') -> None:
        data = {
            "source_id": source_id,
            "relation_type": relation_type,
            "target_id": target_id,
            **properties
        }
        self.client.table(table_name).upsert(data).execute()
        print(f"\t[ SupabaseDatabase add_relation :: {source_id} -[{relation_type}]-> {target_id} ]")

    def get_messages_by_user(self, user_id: str, relation_type: str) -> List[Dict[str, Any]]:
        result = self.client.table('fixed_relations').select(
            'target_id'
        ).eq('source_id', user_id).eq('relation_type', relation_type).execute()

        message_ids = [r['target_id'] for r in result.data]
        messages = self.client.table('fixed_entities').select(
            'id', 'content', 'timestamp'
        ).in_('id', message_ids).eq('label', 'MESSAGE').execute()

        return [{"message_id": r['id'], "message": r['content'], "timestamp": r['timestamp']} for r in messages.data]

    def get_messages_by_session(self, session_id: str, relation_type: str) -> List[Dict[str, Any]]:
        result = self.client.table('fixed_relations').select(
            'target_id'
        ).eq('source_id', session_id).eq('relation_type', relation_type).execute()

        message_ids = [r['target_id'] for r in result.data]
        messages = self.client.table('fixed_entities').select(
            'id', 'content', 'timestamp'
        ).in_('id', message_ids).eq('label', 'MESSAGE').execute()

        return [{"message_id": r['id'], "message": r['content'], "timestamp": r['timestamp']} for r in messages.data]

    def get_users_by_session(self, session_id: str, relation_type: str) -> List[Dict[str, Any]]:
        result = self.client.table('fixed_relations').select(
            'source_id'
        ).eq('target_id', session_id).eq('relation_type', relation_type).execute()
        return [{"user_id": r['source_id']} for r in result.data]

    def get_sessions_by_user(self, user_id: str, relation_type: str) -> List[Dict[str, Any]]:
        result = self.client.table('fixed_relations').select(
            'target_id'
        ).eq('source_id', user_id).eq('relation_type', relation_type).execute()

        session_ids = [r['target_id'] for r in result.data]
        sessions = self.client.table('fixed_entities').select(
            'id'
        ).in_('id', session_ids).eq('label', 'SESSION').execute()

        return [{"session_id": r['id']} for r in sessions.data]

    def get_message_by_id(self, message_id: str) -> Dict[str, Any]:
        result = self.client.table("fixed_entities").select(
            'content',
            'timestamp'
        ).eq('id', message_id).eq('label', 'MESSAGE').limit(1).execute()
        if result.data:
            return {"message": result.data[0]['content'], "timestamp": result.data[0]['timestamp']}
        return {}

    def value_exists(self, label: str, key: str, value: str) -> bool:
        result = self.client.table('fixed_entities').select('id').eq('label', label).eq(key, value).limit(1).execute()
        return len(result.data) > 0
