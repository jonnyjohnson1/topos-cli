# cache_manager.py
import logging

import psycopg2
import json
import datetime 

import os
import pickle
import hashlib
import logging
from dotenv import load_dotenv
from collections import OrderedDict

  
# Define a custom function to serialize datetime objects 
def serialize_datetime(obj): 
    if isinstance(obj, datetime.datetime): 
        return obj.isoformat() 
    raise TypeError("Type not serializable") 

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
                # self._ensure_table_structure()
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
                # self._ensure_table_structure()
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
                # Check structure of conversation_table
                cur.execute("""
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_name = 'conversation'
                """)
                conversation_columns = cur.fetchall()
                logging.info("conversation structure:")
                for column in conversation_columns:
                    logging.info(f"Column: {column[0]}, Type: {column[1]}, Nullable: {column[2]}")

                # Check structure of utterance_token_info_table
                cur.execute("""
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_name = 'utterance_token_info'
                """)
                token_columns = cur.fetchall()
                logging.info("utterance_token_info structure:")
                for column in token_columns:
                    logging.info(f"Column: {column[0]}, Type: {column[1]}, Nullable: {column[2]}")

                # Check structure of utterance_text_info_table
                cur.execute("""
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_name = 'utterance_text_info'
                """)
                text_columns = cur.fetchall()
                logging.info("utterance_text_info structure:")
                for column in text_columns:
                    logging.info(f"Column: {column[0]}, Type: {column[1]}, Nullable: {column[2]}")
        
        except Exception as e:
            logging.error(f"Failed to check table structure: {e}", exc_info=True)
            

    def _ensure_table_exists(self):
        if self.conn is None:
            logging.error("PostgreSQL connection is not initialized")
            raise ConnectionError("PostgreSQL connection is not initialized")

        try:
            logging.debug("Checking if necessary tables exist")
            with self.conn.cursor() as cur:
                # Check for conversation_table existence
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_name = 'conversation'
                    )
                """)
                conversation_table_exists = cur.fetchone()[0]

                if not conversation_table_exists:
                    logging.info("conversation does not exist, creating it")
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS conversation (
                            message_id VARCHAR PRIMARY KEY,
                            conv_id VARCHAR NOT NULL,
                            userid VARCHAR NOT NULL,
                            timestamp TIMESTAMP NOT NULL,
                            name VARCHAR,
                            role VARCHAR NOT NULL,
                            message TEXT NOT NULL
                        );
                    """)
                    logging.info("conversation created")

                # Check for utterance_token_info existence
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_name = 'utterance_token_info'
                    )
                """)
                token_table_exists = cur.fetchone()[0]

                if not token_table_exists:
                    logging.info("utterance_token_info does not exist, creating it")
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS utterance_token_info (
                            message_id VARCHAR PRIMARY KEY,
                            conv_id VARCHAR NOT NULL,
                            userid VARCHAR NOT NULL,
                            name VARCHAR,
                            role VARCHAR NOT NULL,
                            timestamp TIMESTAMP NOT NULL,
                            ents JSONB
                        );
                    """)
                    logging.info("utterance_token_info created")

                # Check for utterance_text_info existence
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_name = 'utterance_text_info'
                    )
                """)
                text_table_exists = cur.fetchone()[0]

                if not text_table_exists:
                    logging.info("utterance_text_info does not exist, creating it")
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS utterance_text_info (
                            message_id VARCHAR PRIMARY KEY,
                            conv_id VARCHAR NOT NULL,
                            userid VARCHAR NOT NULL,
                            name VARCHAR,
                            role VARCHAR NOT NULL,
                            timestamp TIMESTAMP NOT NULL,
                            moderator JSONB,
                            mod_label VARCHAR,
                            tern_sent JSONB,
                            tern_label VARCHAR,
                            emo_27 JSONB,
                            emo_27_label VARCHAR
                        );
                    """)
                    logging.info("utterance_text_info created")

                logging.debug("All necessary tables exist or were successfully created")

            # Commit the table creation if any were made
            self.conn.commit()

        except Exception as e:
            logging.error(f"Failed to check or create tables: {e}", exc_info=True)
            self.conn.rollback()
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

    def load_utterance_token_info(self, conv_id):
        # Query to load token classification data (utterance_token_info_table)
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT message_id, conv_id, userid, name, role, timestamp, ents
                FROM utterance_token_info
                WHERE conv_id = %s;
            """, (conv_id,))
            token_data = cur.fetchall()
        return token_data
    
    
    def load_utterance_text_info(self, conv_id):
        # Query to load text classification data (utterance_text_info_table)
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT message_id, conv_id, userid, name, role, timestamp, moderator, mod_label, tern_sent, tern_label, emo_27, emo_27_label
                FROM utterance_text_info
                WHERE conv_id = %s;
            """, (conv_id,))
            text_data = cur.fetchall()
        return text_data
        
        
    def _load_from_postgres(self, conv_id):
        try:
            logging.debug(f"Attempting to load data for conv_id: {conv_id}")
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT message_data
                    FROM conversation_cache
                    WHERE conv_id = %s
                """, (conv_id,))
                row = cur.fetchall() #cur.fetchone()
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
        # Incoming conversation Data
        # {'GDjCo7HieSN1': 
        #     {'role': 'user', 
        #      'timestamp': datetime.datetime(2024, 10, 25, 20, 37, 49, 881681), 
        #      'message': 'play with me mfor a minute', 
        #      'in_line': {
        #          'base_analysis': {
        #              'TIME': [{'label': 'TIME', 'text': 'a minute', 'sentiment': 0.0, 'start_position': 18, 'end_position': 26}]}
        #          }, 
        #      'commenter': {
        #          'base_analysis': {
        #              'mod_level': [{'label': 'OK', 'score': 0.2502281963825226, 'name': 'OK'}], 
        #              'tern_sent': [{'label': 'NEU', 'score': 0.8717584609985352}], 
        #              'emo_27': [{'label': 'neutral', 'score': 0.9581435322761536}]}
        #          }
        #      }}
        
        # conversation_table: conv_id, userid, timestamp, name, message_id, role, message
        # utterance_token_info_table: message_id, conv_id, userid, name, role, timestamp, 'ents' <jsonb>
        # utterance_text_info_table: message_id, conv_id, userid, name, role, timestamp, 'moderator' <jsonb>, mod_label <str>, tern_sent <jsonb>, tern_label <str>, emo_27 <jsonb>, emo_27_label <str>
       
        if self.conn is None:
            logging.error("PostgreSQL connection is not initialized")
            return

        try:
            logging.info(f"Attempting to save data for conv_id: {conv_id}")
            with self.conn.cursor() as cur:
                for message_id, message_data in new_data.items():
                    role = message_data['role']
                    timestamp = message_data['timestamp']
                    message = message_data['message']
                    userid = "unknown"  # Assuming you get this from elsewhere
                    name = "unknown"  # Assuming you get this from elsewhere
                    
                    # Insert conversation data
                    cur.execute("""
                        INSERT INTO conversation (message_id, conv_id, userid, timestamp, name, role, message)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (message_id) DO UPDATE
                        SET message = EXCLUDED.message, role = EXCLUDED.role, timestamp = EXCLUDED.timestamp;
                    """, (message_id, conv_id, userid, timestamp, name, role, message))
                    
                    # Insert token information (utterance_token_info_table)
                    if 'in_line' in message_data:
                        ents_data = message_data['in_line']['base_analysis']
                        if len(ents_data) > 0:
                            ents = json.dumps(ents_data)
                            cur.execute("""
                                INSERT INTO utterance_token_info (message_id, conv_id, userid, name, role, timestamp, ents)
                                VALUES (%s, %s, %s, %s, %s, %s, %s)
                                ON CONFLICT (message_id) DO UPDATE
                                SET ents = EXCLUDED.ents, timestamp = EXCLUDED.timestamp;
                            """, (message_id, conv_id, userid, name, role, timestamp, ents))
                    
                    # Insert text analysis information (utterance_text_info_table)
                    if 'commenter' in message_data:
                        base_analysis = message_data['commenter']['base_analysis']
                        mod_label = base_analysis['mod_level'][0]['label']
                        tern_sent = json.dumps(base_analysis['tern_sent'])
                        tern_label = base_analysis['tern_sent'][0]['label']
                        emo_27 = json.dumps(base_analysis['emo_27'])
                        emo_27_label = base_analysis['emo_27'][0]['label']

                        cur.execute("""
                            INSERT INTO utterance_text_info 
                            (message_id, conv_id, userid, name, role, timestamp, moderator, mod_label, tern_sent, tern_label, emo_27, emo_27_label)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (message_id) DO UPDATE
                            SET moderator = EXCLUDED.moderator, mod_label = EXCLUDED.mod_label, 
                                tern_sent = EXCLUDED.tern_sent, tern_label = EXCLUDED.tern_label, 
                                emo_27 = EXCLUDED.emo_27, emo_27_label = EXCLUDED.emo_27_label;
                        """, (message_id, conv_id, userid, name, role, timestamp, 
                            json.dumps(base_analysis['mod_level']), mod_label, tern_sent, tern_label, emo_27, emo_27_label))
                
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