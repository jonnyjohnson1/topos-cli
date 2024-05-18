# cache_manager.py

import os
import pickle
import hashlib
import logging
from dotenv import load_dotenv
from collections import OrderedDict

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
        """Load data from the cache using a specific prefix and order messages by timestamp."""
        cache_path = self._get_cache_path(conv_id, prefix)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as file:
                    data = pickle.load(file)
                    conversation_dict = data.get(conv_id, {})
                    
                    # Order the messages by timestamp
                    ordered_conversation = OrderedDict(
                        sorted(conversation_dict.items(), key=lambda item: item[1]['timestamp'])
                    )
                    data[conv_id] = ordered_conversation
                    return data
            except Exception as e:
                logging.error(f"Failed to load from cache {cache_path}: {e}")
                return None
        return None

    # def save_to_cache(self, conv_id, data, prefix=""):
    #     """Save data to the cache using a specific prefix."""
    #     cache_path = self._get_cache_path(conv_id, prefix)
    #     try:
    #         with open(cache_path, "wb") as file:
    #             pickle.dump(data, file)
    #     except Exception as e:
    #         logging.error(f"Failed to save to cache {cache_path}: {e}")

    def save_to_cache(self, conv_id, new_data, prefix=""):
        """Save data to the cache using a specific prefix and update existing dictionary."""
        cache_path = self._get_cache_path(conv_id, prefix)
        
        # Load existing data from the cache if it exists
        try:
            with open(cache_path, "rb") as file:
                existing_data = pickle.load(file)
        except (FileNotFoundError, EOFError):
            existing_data = {conv_id: {}}
        except Exception as e:
            logging.error(f"Failed to load from cache {cache_path}: {e}")
            existing_data = {conv_id: {}}
        
        # Extract the conversation dictionary from the existing data
        conversation_dict = existing_data.get(conv_id, {})
        # Extract the new message_id and data
        message_id, message_data = next(iter(new_data.items()))
        
        # Update the conversation dictionary with the new message data
        conversation_dict[message_id] = message_data
        
        # Update the existing data with the updated conversation dictionary
        existing_data[conv_id] = conversation_dict
        
        # Save the updated data back to the cache
        try:
            with open(cache_path, "wb") as file:
                pickle.dump(existing_data, file)
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
