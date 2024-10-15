from topos.services.messages.missed_message_manager import MissedMessageManager
from topos.services.messages.group_management_service import GroupManagementService
from datetime import datetime
import pytz

class MissedMessageService:
    def __init__(self, db_params: dict) -> None:
        self.missed_message_manager = MissedMessageManager()
        self.group_management_service = GroupManagementService(db_params)

    async def get_missed_messages(self, user_id: str):
        last_seen = self.group_management_service.get_user_last_seen_online(user_id=user_id)
        if last_seen:
            users_groups = self.group_management_service.get_user_groups(user_id=user_id)
            group_ids = set(group["group_id"] for group in users_groups)
            timestamp_ms = int(datetime.fromisoformat(last_seen).replace(tzinfo=pytz.UTC).timestamp() * 1000)
            return await self.missed_message_manager.get_filtered_missed_messages(key_filter=group_ids, timestamp_ms=timestamp_ms)
        else:
            return []
