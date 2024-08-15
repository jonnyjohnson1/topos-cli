from openai import OpenAI

api_url_dict = {
    'ollama': 'http://localhost:11434/v1',
    'openai': None,
    'groq': 'https://api.groq.com/openai/v1'
}

class LLMClient:
    def __init__(self, provider: str, api_key: str):
        if provider not in api_url_dict:
            print(f"Unsupported provider: {self.provider}")
        self.provider = provider.lower()
        self.api_key = api_key
        self.client = self._init_client()
        print(f"Init client:: {self.provider}")
    
    def _init_client(self):
        if self.provider == "openai":
            return OpenAI(api_key=self.api_key)
        else:
            url = api_url_dict[self.provider]
            return OpenAI(api_key=self.api_key, base_url=url)

    def get_client(self):
        return self.client

    def get_provider(self):
        return self.provider