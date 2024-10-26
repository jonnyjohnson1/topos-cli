
import os
from fastapi import APIRouter, HTTPException
from topos.services.database.conversation_cache_manager import ConversationCacheManager

from collections import Counter, defaultdict

import logging

from ....models.models import ConversationIDRequest

db_config = {
            "dbname": os.getenv("POSTGRES_DB"),
            "user": os.getenv("POSTGRES_USER"),
            "password": os.getenv("POSTGRES_PASSWORD"),
            "host": os.getenv("POSTGRES_HOST"),
            "port": os.getenv("POSTGRES_PORT")
        }

logging.info(f"Database configuration: {db_config}")

use_postgres = True
if use_postgres:
    cache_manager = ConversationCacheManager(use_postgres=True, db_config=db_config)
else:
    cache_manager = ConversationCacheManager()

router = APIRouter()

@router.post("/chat_conversation_analysis")
async def chat_conversation_analysis(request: ConversationIDRequest):
    conversation_id = request.conversation_id

    # Connect to the PostgreSQL database
    if cache_manager.use_postgres:
        try:
            # Query to load token classification data (utterance_token_info_table)
            token_data = cache_manager.load_utterance_token_info(conversation_id)
           
            # Query to load text classification data (utterance_text_info_table)
            text_data = cache_manager.load_utterance_text_info(conversation_id)

            if not token_data and not text_data:
                raise HTTPException(status_code=404, detail="Conversation not found in cache")
        
        except Exception as e:
            logging.error(f"Failed to retrieve data from PostgreSQL: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to retrieve data from cache")
        
        # Initialize counters
        named_entity_counter = Counter()
        entity_text_counter = Counter()
        emotion_counter = Counter()

        # Initialize user-based counters
        named_entity_counter_per_user = defaultdict(Counter)
        entity_text_counter_per_user = defaultdict(Counter)
        emotion_counter_per_user = defaultdict(Counter)

        print(f"\t[ conversational analysis ]")
        # Extract counts from token data
        for token_row in token_data:
            message_id, conv_id, userid, name, role, timestamp, ents = token_row
            user = name or role  # use name if available, otherwise role
            # Process named entities and base analysis
            for entity in ents:
                entity_list = ents[entity]
                for ent in entity_list:
                    entity_type = ent.get('label')
                    entity_text = ent.get('text', '')
                    named_entity_counter[entity_type] += 1
                    named_entity_counter_per_user[user][entity_type] += 1
                    entity_text_counter[entity_text] += 1
                    entity_text_counter_per_user[user][entity_text] += 1

        # Extract counts from text data
        for text_row in text_data:
            message_id, conv_id, userid, name, role, timestamp, moderator, mod_label, tern_sent, tern_label, emo_27, emo_27_label = text_row
            user = name if name != "unknown" else role  # use name if available, otherwise role
            
            # Process emotions
            for emotion in emo_27:
                emotion_label = emotion['label']
                emotion_counter[emotion_label] += 1
                emotion_counter_per_user[user][emotion_label] += 1

    else:
        # Non-Postgres handling if needed, otherwise raise an exception
        raise HTTPException(status_code=501, detail="PostgreSQL is the only supported cache manager.")

    named_entity_dict = {
        "totals": dict(named_entity_counter),
        "per_role": {user: dict(counter) for user, counter in named_entity_counter_per_user.items()}
    }
    entity_text_dict = {
        "totals": dict(entity_text_counter),
        "per_role": {user: dict(counter) for user, counter in entity_text_counter_per_user.items()}
    }
    emotion_dict = {
        "totals": dict(emotion_counter),
        "per_role": {user: dict(counter) for user, counter in emotion_counter_per_user.items()}
    }

    # Create the final dictionary
    conversation = {
        'entity_evocations': named_entity_dict,
        'entity_summons': entity_text_dict,
        'emotions27': emotion_dict
    }

    # Return the conversation or any other response needed
    return {"conversation": conversation}

