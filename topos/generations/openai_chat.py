# openai_chat.py

from openai import OpenAI
from typing import List, Dict, Generator
from topos.generations.chat_model import ChatModel


class OpenAIChatModel(ChatModel):

    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name, api_key)
        self.client = OpenAI(
            api_key=self.api_key
        )

    def stream_chat(self, message_history: List[Dict[str, str]], temperature: float = 0) -> Generator[str, None, None]:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=message_history,
                temperature=temperature,
                stream=True
            )
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            print(f"Error in stream_chat: {e}")
            yield f"Error: {str(e)}"

    def generate_response(self, context: str, prompt: str, temperature: float = 0) -> str:
        try:
            messages = [
                {"role": "system", "content": context},
                {"role": "user", "content": prompt}
            ]
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in generate_response: {e}")
            return f"Error: {str(e)}"

    def generate_response_messages(self, message_history: List[Dict[str, str]], temperature: float = 0) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=message_history,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in generate_response_messages: {e}")
            return f"Error: {str(e)}"
