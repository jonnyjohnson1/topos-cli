from typing import List, Dict, Generator

from .llm_client import LLMClient

# Assuming OpenAI is a pre-defined client for API interactions

default_models = {
    "groq": "llama-3.1-70b-versatile",
    "openai": "gpt-4o",
    "ollama": "dolphin-llama3"
    }

class LLMController:
    def __init__(self, model_name: str, provider: str, api_key: str):
        self.provier = provider
        self.api_key = api_key
        self.client = LLMClient(provider, api_key).get_client()
        self.model_name = self._init_model(model_name, provider)
        
    def _init_model(self, model_name: str, provider: str):
        if len(model_name) > 0:
            return model_name
        else:
            if provider == 'ollama':
                return model_name
            else:
                return default_models[provider]

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
                temperature=temperature,
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"

    def generate_response_messages(self, message_history: List[Dict[str, str]], temperature: float = 0) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=message_history,
                temperature=temperature,
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"