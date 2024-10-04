from chat_server.managers.missed_message_manager import MissedMessageManager
from chat_server.services.group_management_service import GroupManagementService
from chat_server.utils.utils import sqlite_timestamp_to_ms

KAFKA_TOPIC = 'chat_topic'

class MissedMessageService:
    def __init__(self) -> None:
        self.missed_message_manager = MissedMessageManager()
        pass 
    # houskeeping if required 
    # if you need to inject the group management service here it could be an option ??
    
    async def get_missed_messages(self,user_id :str ,group_management_service :GroupManagementService):
        last_seen = group_management_service.get_user_last_seen_online(user_id=user_id)
        if(last_seen):
            users_groups = group_management_service.get_user_groups(user_id=user_id)
            group_ids = [group["group_id"] for group in users_groups]
            # get the last timestamp msg processed by the user
            return await self.missed_message_manager.get_filtered_missed_messages(key_filter=group_ids,timestamp_ms=sqlite_timestamp_to_ms(last_seen))
        else:
            return []