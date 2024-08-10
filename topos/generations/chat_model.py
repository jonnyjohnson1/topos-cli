from abc import ABC, abstractmethod
import openai


# Abstract Base Class for Chat Models
class ChatModel(ABC):

    def __init__(self, model_name, api_key):
        self.model_name = model_name
        self.api_key = api_key

    @abstractmethod
    def stream_chat(self, message_history, temperature=0):
        pass

    @abstractmethod
    def generate_response(self, context, prompt, temperature=0):
        pass
