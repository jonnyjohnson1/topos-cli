# database_interface.py

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class DatabaseInterface(ABC):
    """
    Abstract base class defining the interface for database operations.
    All concrete database implementations should inherit from this class.
    """

    @abstractmethod
    def add_entity(self, entity_id: str, entity_label: str, properties: Dict[str, Any]) -> None:
        """
        Add an entity to the database.

        :param entity_id: Unique identifier for the entity
        :param entity_label: Label or type of the entity
        :param properties: Dictionary of properties associated with the entity
        """
        pass

    @abstractmethod
    def add_relation(self, source_id: str, relation_type: str, target_id: str, properties: Dict[str, Any]) -> None:
        """
        Add a relation between two entities in the database.

        :param source_id: ID of the source entity
        :param relation_type: Type of the relation
        :param target_id: ID of the target entity
        :param properties: Dictionary of properties associated with the relation
        """
        pass

    @abstractmethod
    def get_messages_by_user(self, user_id: str, relation_type: str) -> List[Dict[str, Any]]:
        """
        Retrieve messages associated with a user.

        :param user_id: ID of the user
        :param relation_type: Type of relation to consider
        :return: List of dictionaries containing message details
        """
        pass

    @abstractmethod
    def get_messages_by_session(self, session_id: str, relation_type: str) -> List[Dict[str, Any]]:
        """
        Retrieve messages associated with a session.

        :param session_id: ID of the session
        :param relation_type: Type of relation to consider
        :return: List of dictionaries containing message details
        """
        pass

    @abstractmethod
    def get_users_by_session(self, session_id: str, relation_type: str) -> List[Dict[str, Any]]:
        """
        Retrieve users associated with a session.

        :param session_id: ID of the session
        :param relation_type: Type of relation to consider
        :return: List of dictionaries containing user details
        """
        pass

    @abstractmethod
    def get_sessions_by_user(self, user_id: str, relation_type: str) -> List[Dict[str, Any]]:
        """
        Retrieve sessions associated with a user.

        :param user_id: ID of the user
        :param relation_type: Type of relation to consider
        :return: List of dictionaries containing session details
        """
        pass

    @abstractmethod
    def get_message_by_id(self, message_id: str) -> Dict[str, Any]:
        """
        Retrieve a specific message by its ID.

        :param message_id: ID of the message
        :return: Dictionary containing message details
        """
        pass

    @abstractmethod
    def value_exists(self, label: str, key: str, value: str) -> bool:
        """
        Check if a value exists in the database for a given label and key.

        :param label: Label or type of the entity
        :param key: Key to check
        :param value: Value to look for
        :return: Boolean indicating whether the value exists
        """
        pass

    def __init__(self):
        print(f"\t[ DatabaseInterface init ]")