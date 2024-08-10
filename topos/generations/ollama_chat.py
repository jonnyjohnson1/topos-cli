# ollama_chat.py

import ollama
from typing import List, Dict, Generator
from topos.generations.chat_model import ChatModel


class OllamaChatModel(ChatModel):

    def __init__(self, model_name: str):
        super().__init__(model_name, "unused")

    def stream_chat(self, message_history: List[Dict[str, str]], temperature: float = 0) -> Generator[str, None, None]:
        try:
            stream = ollama.chat(
                model=self.model_name,
                messages=message_history,
                stream=True,
                temperature=temperature
            )
            for chunk in stream:
                yield chunk['message']['content']
        except Exception as e:
            print(f"Error in stream_chat: {e}")
            yield f"Error: {str(e)}"

    def generate_response(self, context: str, prompt: str, temperature: float = 0) -> str:
        messages = [
            {"role": "system", "content": context},
            {"role": "user", "content": prompt}
        ]
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                temperature=temperature
            )
            return response['message']['content']
        except Exception as e:
            print(f"Error in generate_response: {e}")
            return f"Error: {str(e)}"