# chat_model.py
# Abstract Base Class for Chat Models
from abc import ABC, abstractmethod
from typing import List, Dict, Generator


class ChatModel(ABC):
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.api_key = api_key

    @abstractmethod
    def stream_chat(self, message_history: List[Dict[str, str]], temperature: float = 0) -> Generator[str, None, None]:
        pass

    @abstractmethod
    def generate_response(self, context: str, prompt: str, temperature: float = 0) -> str:
        pass

    @abstractmethod
    def generate_response_messages(self, message_history: List[Dict[str, str]], temperature: float = 0) -> str:
        pass