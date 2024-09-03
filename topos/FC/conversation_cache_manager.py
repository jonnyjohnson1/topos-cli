# cache_manager.py
import psycopg2
from psycopg2.extras import Json

import os
import pickle
import hashlib
import logging
from dotenv import load_dotenv
from collections import OrderedDict

class ConversationCacheManager:
    def __init__(self, cache_dir="./_conv_cache", use_postgres=False, db_config=None):
        self.use_postgres = use_postgres
        if use_postgres:
            self.db_config = db_config
            self._init_postgres()
        else:
            self.cache_dir = cache_dir
            os.makedirs(self.cache_dir, exist_ok=True)  # Ensure the cache directory exists

    def _init_postgres(self):
        try:
            self.conn = psycopg2.connect(**self.db_config)
            with self.conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS conversation_cache (
                        conv_id TEXT,
                        message_id TEXT,
                        data JSONB,
                        PRIMARY KEY (conv_id, message_id)
                    )
                """)
            self.conn.commit()
        except Exception as e:
            logging.error(f"Failed to initialize PostgreSQL: {e}")
            raise

    def _get_cache_path(self, conv_id, prefix=""):
        # Create a valid filename for the cache based on the input text and an optional prefix
        filename = f"cache_{prefix}_{conv_id}.pkl"
        try:
            cache_path = os.path.join(self.cache_dir, filename)
        except Exception as e:
            logging.error(f"Failed to create cache path from directory {self.cache_dir}: {e}")
            return None
        return cache_path

    def load_from_cache(self, conv_id, prefix=""):
        """Load data from the cache using a specific prefix and order messages by timestamp."""
        if self.use_postgres:
            return self._load_from_postgres(conv_id)
        else:
            return self._load_from_file(conv_id, prefix)

    def _load_from_file(self, conv_id, prefix=""):
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
                    return {conv_id: ordered_conversation}
            except Exception as e:
                logging.error(f"Failed to load from cache {cache_path}: {e}")
                return None
        return None

    def _load_from_postgres(self, conv_id):
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT message_id, data
                    FROM conversation_cache
                    WHERE conv_id = %s
                    ORDER BY (data->>'timestamp')::timestamp
                """, (conv_id,))
                rows = cur.fetchall()
                if rows:
                    conversation_dict = OrderedDict((row[0], row[1]) for row in rows)
                    return {conv_id: conversation_dict}
        except Exception as e:
            logging.error(f"Failed to load from PostgreSQL: {e}")
        return None

    def save_to_cache(self, conv_id, new_data, prefix=""):
        """Save data to the cache using a specific prefix and update existing dictionary."""
        if self.use_postgres:
            self._save_to_postgres(conv_id, new_data)
        else:
            self._save_to_file(conv_id, new_data, prefix)

    def _save_to_file(self, conv_id, new_data, prefix=""):
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

        # Update the conversation dictionary with the new message data
        conversation_dict.update(new_data)

        # Update the existing data with the updated conversation dictionary
        existing_data[conv_id] = conversation_dict

        # Save the updated data back to the cache
        try:
            with open(cache_path, "wb") as file:
                pickle.dump(existing_data, file)
        except Exception as e:
            logging.error(f"Failed to save to cache {cache_path}: {e}")

    def _save_to_postgres(self, conv_id, new_data):
        try:
            with self.conn.cursor() as cur:
                for message_id, message_data in new_data.items():
                    cur.execute("""
                        INSERT INTO conversation_cache (conv_id, message_id, data)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (conv_id, message_id) DO UPDATE
                        SET data = EXCLUDED.data
                    """, (conv_id, message_id, Json(message_data)))
            self.conn.commit()
        except Exception as e:
            logging.error(f"Failed to save to PostgreSQL: {e}")
            self.conn.rollback()

    def clear_cache(self):
        """Clear the cache directory or PostgreSQL table."""
        if self.use_postgres:
            self._clear_postgres_cache()
        else:
            self._clear_file_cache()

    def _clear_file_cache(self):
        try:
            for filename in os.listdir(self.cache_dir):
                file_path = os.path.join(self.cache_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        except Exception as e:
            logging.error(f"Failed to clear cache directory: {e}")

    def _clear_postgres_cache(self):
        try:
            with self.conn.cursor() as cur:
                cur.execute("TRUNCATE TABLE conversation_cache")
            self.conn.commit()
        except Exception as e:
            logging.error(f"Failed to clear PostgreSQL cache: {e}")
            self.conn.rollback()
