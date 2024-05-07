import json
import logging
import math

# from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.llms import Ollama
from topos.FC.cache_manager import CacheManager


class SemanticCompression:
    def __init__(self, api_key, model="ollama:solar", max_tokens_category=128, max_tokens_contextualize=128,
                 max_tokens_recompose=256, max_tokens_decode=1024, cache_enabled=True):
        self.api_key = api_key
        self.model_provider, self.model_type = self.parse_model(model)
        self.max_tokens_semantic_category = max_tokens_category
        self.max_tokens_contextualize = max_tokens_contextualize
        self.max_tokens_recompose = max_tokens_recompose
        self.max_tokens_decode = max_tokens_decode
        self.cache_enabled = cache_enabled

        self.cache_manager = CacheManager()

    @staticmethod
    def parse_model(model):
        if ":" in model:
            return model.split(":", 1)
        else:
            return "ollama", model

    def get_content_key(self, key, token_limit_for_task):
        content_key = f"{key}.{self.model_provider}.{self.model_type}.{token_limit_for_task}"
        return content_key

    def fetch_semantic_category(self, input_text, modifiers_limit=3):
        content_key = self.get_content_key(input_text, self.max_tokens_semantic_category)

        cached_response = self.cache_manager.load_from_cache(content_key)
        if cached_response:
            return cached_response

        if self.model_provider == "ollama":
            prompt = f"Summarize the following into one or more words with up to {modifiers_limit} modifiers: {input_text}"
            ollama = Ollama(model=self.model_type)
            response_text = ""
            for chunk in ollama.stream(prompt):
                response_text += chunk
            response = {"content": response_text}
        else:
            try:
                openai_client = OpenAI(api_key=self.api_key)
                response = openai_client.chat.completions.create(
                    model=self.model_type,
                    messages=[{"role": "system",
                               "content": f"Summarize the following into one or more words with up to {modifiers_limit} modifiers: {input_text}"}],
                    max_tokens=self.max_tokens_semantic_category,
                    n=1,
                    stop=None,
                    temperature=0.3)
                response = response.choices[0].message
            except Exception as e:
                logging.error(f"Error in fetch_semantic_compression: {e}")
                return None

        self.cache_manager.save_to_cache(content_key, response)
        return response