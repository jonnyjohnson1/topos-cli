# cache_manager.py

import os
import pickle
import hashlib
import logging
from dotenv import load_dotenv

class ConversationCacheManager:
    def __init__(self, cache_dir="./_conv_cache"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)  # Ensure the cache directory exists

    def _get_cache_path(self, conv_id, prefix=""):
        # Create a valid filename for the cache based on the input text and an optional prefix
        filename = f"cache_{prefix}_{conv_id}.pkl"
        try:
            print(self.cache_dir)
        except Exception as e:
            print("FAILED TO GET CACHE self.cache_dir")
            logging.error(f"Failed to load from cache {cache_path}: {e}")
        try:
            os.path.join(self.cache_dir, filename)
        except Exception as e:
            print("FAILED TO GET CACHE PATH")
            logging.error(f"Failed to load from cache {cache_path}: {e}")
        return os.path.join(self.cache_dir, filename)

    def load_from_cache(self, conv_id, prefix=""):
        """Load data from the cache using a specific prefix."""
        cache_path = self._get_cache_path(conv_id, prefix)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as file:
                    return pickle.load(file)
            except Exception as e:
                logging.error(f"Failed to load from cache {cache_path}: {e}")
                return None
        return None

    def save_to_cache(self, conv_id, data, prefix=""):
        """Save data to the cache using a specific prefix."""
        print("save_to_cache_func")
        cache_path = self._get_cache_path(conv_id, prefix)
        print("CACHE PATH", cache_path)
        print("CACHE DATA", cache_path)
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
