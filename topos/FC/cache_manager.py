#cachemanager.py

#(c)2024 chris forrester - free for all license, no warranty or liability

import os
import pickle
import hashlib
import logging
from dotenv import load_dotenv

class CacheManager:
    def __init__(self, cache_dir="../_cache"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)  # Ensure the cache directory exists

    @staticmethod
    def _get_input_hash(input_text):
        # Create an SHA-256 hash of the input text
        hash_object = hashlib.sha256(input_text.encode('utf-8'))
        return hash_object.hexdigest()

    def _get_cache_path(self, input_text, prefix=""):
        # Create a valid filename for the cache based on the input text and an optional prefix
        filename = f"cache_{prefix}_{self._get_input_hash(input_text)}.pkl"
        return os.path.join(self.cache_dir, filename)

    def load_from_cache(self, input_text, prefix=""):
        """Load data from the cache using a specific prefix."""
        cache_path = self._get_cache_path(input_text, prefix)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as file:
                    return pickle.load(file)
            except Exception as e:
                logging.error(f"Failed to load from cache {cache_path}: {e}")
                return None
        return None

    def save_to_cache(self, input_text, data, prefix=""):
        """Save data to the cache using a specific prefix."""
        cache_path = self._get_cache_path(input_text, prefix)
        try:
            with open(cache_path, "wb") as file:
                pickle.dump(data, file)
        except Exception as e:
            logging.error(f"Failed to save to cache {cache_path}: {e}")

    def clear_cache(self):
        """Clear the cache directory."""
        try:
            for filename in os.listdir(self.cache_dir):
                file_path = os.path.join(self.cache_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        except Exception as e:
            logging.error(f"Failed to clear cache directory: {e}")
