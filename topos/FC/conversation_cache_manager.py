# cache_manager.py
import logging

import psycopg2
import json

import os
import pickle
import hashlib
import logging
from dotenv import load_dotenv
from collections import OrderedDict

class ConversationCacheManager:
    def __init__(self, cache_dir="./_conv_cache", use_postgres=False, db_config=None):
        self.cache_dir = cache_dir
        self.use_postgres = use_postgres
        self.db_config = db_config
        self.conn = None

        if not use_postgres:
            os.makedirs(cache_dir, exist_ok=True)
        elif db_config is not None:
            self._init_postgres()

    def _init_postgres(self):
        if not self.db_config:
            logging.error("Database configuration is missing")
            raise ValueError("Database configuration is required for PostgreSQL connection")

        try:
            logging.debug(f"Attempting to connect to PostgreSQL with config: {self.db_config}")
            self.conn = psycopg2.connect(**self.db_config)

            if not self.conn.closed:
                logging.info("Successfully connected to PostgreSQL")
                self._ensure_table_structure()
            else:
                logging.error("Failed to establish a valid connection to PostgreSQL")
                raise ConnectionError("Unable to establish a valid connection to PostgreSQL")
        except psycopg2.Error as e:
            logging.error(f"PostgreSQL error: {e.pgerror}")
            self.conn = None
            raise
        except Exception as e:
            logging.error(f"Failed to initialize PostgreSQL connection: {e}", exc_info=True)
            self.conn = None
            raise

        if self.conn:
            try:
                self._ensure_table_structure()
                self._check_table_structure()  # Add this line
            except Exception as e:
                logging.error(f"Failed to ensure table structure: {e}", exc_info=True)
                self.conn.close()
                self.conn = None
                raise

    def _check_table_structure(self):
        if self.conn is None:
            logging.error("PostgreSQL connection is not initialized")
            return

        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_name = 'conversation_cache'
                """)
                columns = cur.fetchall()
                logging.info("Current table structure:")
                for column in columns:
                    logging.info(f"Column: {column[0]}, Type: {column[1]}, Nullable: {column[2]}")
        except Exception as e:
            logging.error(f"Failed to check table structure: {e}", exc_info=True)

    def _ensure_table_structure(self):
        if self.conn is None:
            logging.error("PostgreSQL connection is not initialized")
            raise ConnectionError("PostgreSQL connection is not initialized")

        try:
            logging.debug("Ensuring table structure exists")
            with self.conn.cursor() as cur:
                cur.execute("DROP TABLE IF EXISTS conversation_cache")
                cur.execute("""
                    CREATE TABLE conversation_cache (
                        conv_id TEXT PRIMARY KEY,
                        message_data JSONB NOT NULL
                    )
                """)
            self.conn.commit()
            logging.info("Table structure ensured successfully")
        except Exception as e:
            logging.error(f"Failed to ensure table structure: {e}", exc_info=True)
            if self.conn:
                self.conn.rollback()
            raise

    def _ensure_table_exists(self):
        if self.conn is None:
            logging.error("PostgreSQL connection is not initialized")
            raise ConnectionError("PostgreSQL connection is not initialized")

        try:
            logging.debug("Checking if conversation_cache table exists")
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_name = 'conversation_cache'
                    )
                """)
                table_exists = cur.fetchone()[0]

                if not table_exists:
                    logging.info("conversation_cache table does not exist, creating it")
                    self._ensure_table_structure()
                else:
                    logging.debug("conversation_cache table already exists")
        except Exception as e:
            logging.error(f"Failed to check or create table: {e}", exc_info=True)
            raise

    def _get_cache_path(self, conv_id, prefix=""):
        # Create a valid filename for the cache based on the input text and an optional prefix
        filename = f"cache_{prefix}_{conv_id}.pkl"
        try:
            cache_path = os.path.join(self.cache_dir, filename)
        except Exception as e:
            logging.error(f"Failed to create cache path from directory {self.cache_dir}: {e}", exc_info=True)
            return ""
        return cache_path

    def load_from_cache(self, conv_id, prefix=""):
        """Load data from the cache using a specific prefix and order messages by timestamp."""
        if self.use_postgres:
            self._ensure_table_exists()
            return self._load_from_postgres(conv_id)
        else:
            return self._load_from_file(conv_id, prefix)

    def _load_from_file(self, conv_id, prefix=""):
        cache_path = self._get_cache_path(conv_id, prefix)
        if not cache_path:
            logging.error(f"Empty cache path for conv_id: {conv_id}")
            return None
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
                logging.error(f"Failed to load from cache {cache_path}: {e}", exc_info=True)
                return None
        return None

    def _load_from_postgres(self, conv_id):
        try:
            logging.debug(f"Attempting to load data for conv_id: {conv_id}")
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT message_data
                    FROM conversation_cache
                    WHERE conv_id = %s
                """, (conv_id,))
                row = cur.fetchone()
                if row:
                    conversation_data = row[0]  # PostgreSQL JSONB is automatically deserialized
                    logging.info(f"Successfully loaded data for conv_id: {conv_id}")
                    return {conv_id: conversation_data}
                else:
                    logging.info(f"No data found for conv_id: {conv_id}")
        except Exception as e:
            logging.error(f"Failed to load from PostgreSQL for conv_id {conv_id}: {e}", exc_info=True)
        return None

    def save_to_cache(self, conv_id, new_data, prefix=""):
        """Save data to the cache using a specific prefix and update existing dictionary."""
        if self.use_postgres:
            self._ensure_table_exists()
            self._save_to_postgres(conv_id, new_data)
        else:
            self._save_to_file(conv_id, new_data, prefix)

    def _save_to_file(self, conv_id, new_data, prefix=""):
        cache_path = self._get_cache_path(conv_id, prefix)

        if not cache_path:
            logging.error(f"Empty cache path for conv_id: {conv_id}")
            return

        # Load existing data from the cache if it exists
        try:
            with open(cache_path, "rb") as file:
                existing_data = pickle.load(file)
        except (FileNotFoundError, EOFError):
            existing_data = {conv_id: {}}
        except Exception as e:
            logging.error(f"Failed to load from cache {cache_path}: {e}", exc_info=True)
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
            logging.error(f"Failed to save to cache {cache_path}: {e}", exc_info=True)

    def _save_to_postgres(self, conv_id, new_data):
        if self.conn is None:
            logging.error("PostgreSQL connection is not initialized")
            return

        try:
            logging.debug(f"Attempting to save data for conv_id: {conv_id}")
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO conversation_cache (conv_id, message_data)
                    VALUES (%s, %s::jsonb)
                    ON CONFLICT (conv_id) DO UPDATE
                    SET message_data = EXCLUDED.message_data
                """, (conv_id, json.dumps(new_data)))
            self.conn.commit()
            logging.info(f"Successfully saved data for conv_id: {conv_id}")
        except Exception as e:
            logging.error(f"Failed to save to PostgreSQL for conv_id {conv_id}: {e}", exc_info=True)
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
            logging.info(f"Successfully cleared file cache directory: {self.cache_dir}")
        except Exception as e:
            logging.error(f"Failed to clear cache directory: {e}", exc_info=True)

    def _clear_postgres_cache(self):
        if self.conn is None:
            logging.error("PostgreSQL connection is not initialized")
            return

        try:
            logging.debug("Attempting to clear PostgreSQL cache")
            with self.conn.cursor() as cur:
                cur.execute("TRUNCATE TABLE conversation_cache")
            self.conn.commit()
            logging.info("Successfully cleared PostgreSQL cache")
        except Exception as e:
            logging.error(f"Failed to clear PostgreSQL cache: {e}", exc_info=True)
            self.conn.rollback()

    def __del__(self):
        if self.conn:
            self.conn.close()
            logging.debug("Closed PostgreSQL connection")

def _ensure_table_exists(self):
    try:
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'conversation_cache'
                )
            """)
            table_exists = cur.fetchone()[0]

            if not table_exists:
                self._init_postgres()
    except Exception as e:
        logging.error(f"Failed to check or create table: {e}")
        raise
