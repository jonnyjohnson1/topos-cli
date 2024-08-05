# semantic_compression.py

import json
import logging
import math

# from dotenv import load_dotenv
from openai import OpenAI
from topos.FC.cache_manager import CacheManager
from topos.FC.similitude_module import load_model, util


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

        self.embedding_model_smallest_80_14200 = 'all-MiniLM-L6-v2'
        self.embedding_model_small_120_7500 = 'all-MiniLM-L12-v2'
        self.embedding_model_medium_420_2800 = 'all-mpnet-base-v2'

        self.model = self.load_model()

        self.cache_manager = CacheManager()

    def load_model(self):
        return load_model(self.embedding_model_medium_420_2800)

    @staticmethod
    def parse_model(model):
        if ":" in model:
            return model.split(":", 1)
        else:
            return "ollama", model

    def get_content_key(self, key, token_limit_for_task):
        content_key = f"{key}.{self.model_provider}.{self.model_type}.{token_limit_for_task}"
        return content_key

    def fetch_semantic_category(self, input_text, extra_fingerprint=""):
        content_string = ""

        if self.model_provider == "openai":
            content_string = f"""Summarize the following into six or less words: {input_text}"""
            # content_string = f"""Summarize the following into one or more words with up to {modifiers_limit} modifiers: {input_text}"""
        elif self.model_provider == "ollama" and self.model_type == "phi3":
            content_string = f"""Summarize the following into one or more words: {input_text}
                   in the format of: 
                   ___Summarized Hypernym/Category___"""
        elif self.model_provider == "ollama" and self.model_type == "dolphin-llama3":
            content_string = f"""You are an expert at finding the simplest hypernyms & synopsis possible."""
        elif self.model_provider == "claude":
            content_string = f"""Summarize the following into six or less words:"""

        # Construct the JSON object using a Python dictionary and convert it to a JSON string
        messages = [
            {
                "role": "system",
                "content": content_string
            }
        ]

        if self.model_provider == "ollama" and self.model_type == "dolphin-llama3":
            messages.append({"role": "user",
                             "content": f"in the format of ___Summarized Hypernym/Category___, give me the (six or less words) hypernym for the following text: {input_text}\nRemember - 6 words or less!!"})

        if self.model_provider == "claude":
            messages.append({"role": "user", "content": f"{input_text}"})

        # default temp is 0.3
        temperature = 0.3

        # gpt-4o is a bit more conservative, so we need to use a higher temperature otherwise it'll just degenerate
        # into a single hypernym.
        if self.model_provider == "openai" and self.model_type == "gpt-4o":
            temperature = 0.5

        # Use json.dumps to safely create a JSON string
        # Attempt to parse the template as JSON
        formatted_json = json.dumps(messages, indent=4)

        content_key = self.get_content_key(formatted_json + extra_fingerprint, self.max_tokens_semantic_category)

        cached_response = self.cache_manager.load_from_cache(content_key)
        if cached_response:
            return cached_response

        try:
            ollama_base = "http://localhost:11434/v1"
            client = OpenAI(
                base_url=ollama_base,
                api_key="ollama",
            )

            response = client.chat.completions.create(
                model=self.model_type,
                messages=json.loads(formatted_json),
                max_tokens=self.max_tokens_semantic_category,
                n=1,
                stop=None,
                temperature=temperature)
            self.cache_manager.save_to_cache(content_key, response.
                                             choices[0].message)
            return response.choices[0].message
        except Exception as e:
            logging.error(f"Error in fetch_semantic_compression: {e}")
            return None

    def get_semantic_distance(self, detail_dict, modified_text):
        original_embeddings = self.model.encode(detail_dict)
        modified_embeddings = self.model.encode(modified_text)
        return util.pytorch_cos_sim(original_embeddings, modified_embeddings)[0][0]
