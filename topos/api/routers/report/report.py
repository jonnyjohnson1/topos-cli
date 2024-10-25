
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
    # load conversation
    conv_data = cache_manager.load_from_cache(conversation_id)

    if conv_data is None:
        raise HTTPException(status_code=404, detail="Conversation not found in cache")
    # Initialize counters
    named_entity_counter = Counter()
    entity_text_counter = Counter()
    emotion_counter = Counter()

    # Initialize user-based counters
    named_entity_counter_per_user = defaultdict(Counter)
    entity_text_counter_per_user = defaultdict(Counter)
    emotion_counter_per_user = defaultdict(Counter)

    print(f"\t[ conversational analysis ]")
    if cache_manager.use_postgres:
        # Extract counts
        for conversation_id, messages_list in conv_data.items():
            print(f"\t\t[ item :: {conversation_id} ]")
            for message_dict in messages_list:
                for cntn in message_dict:
                    for message_id, content in cntn.items():
                        # print(f"\t\t\t[ content :: {str(content)[40:]} ]")
                        # print(f"\t\t\t[ keys :: {str(content.keys())[40:]} ]")
                        role = content['role']
                        user = role
                        if role == "user" and 'user_name' in content:
                            user = content['user_name']

                        # Process named entities and base analysis
                        base_analysis = content['in_line']['base_analysis']
                        for entity_type, entities in base_analysis.items():
                            named_entity_counter[entity_type] += len(entities)
                            named_entity_counter_per_user[user][entity_type] += len(entities)
                            for entity in entities:
                                entity_text_counter[str(entity.get('text', ''))] += 1
                                entity_text_counter_per_user[user][str(entity.get('text', ''))] += 1

                        # Process emotions
                        emotions = content['commenter']['base_analysis']['emo_27']
                        for emotion in emotions:
                            emotion_counter[emotion['label']] += 1
                            emotion_counter_per_user[user][emotion['label']] += 1
    else:
        # Extract counts
        for conversation_id, messages in conv_data.items():
            print(f"\t\t[ item :: {conversation_id} ]")
            for message_id, content in messages.items():
                # print(f"\t\t\t[ content :: {str(content)[40:]} ]")
                # print(f"\t\t\t[ keys :: {str(content.keys())[40:]} ]")
                role = content['role']
                user = role
                if role == "user" and 'user_name' in content:
                    user =  content['user_name']
                base_analysis = content['in_line']['base_analysis']
                for entity_type, entities in base_analysis.items():
                    named_entity_counter[entity_type] += len(entities)
                    named_entity_counter_per_user[user][entity_type] += len(entities)
                    for entity in entities:
                        entity_text_counter[str(entity['text'])] += 1
                        entity_text_counter_per_user[user][str(entity['text'])] += 1

                emotions = content['commenter']['base_analysis']['emo_27']
                for emotion in emotions:
                    emotion_counter[emotion['label']] += 1
                    emotion_counter_per_user[user][emotion['label']] += 1

    # Evocations equals num of each entity
    # print("Named Entity Count:")
    # print(named_entity_counter)       # get the count of each entity from the conv_data

    # # Actual Items summoned
    # print("\nEntity Text Count:")
    # print(entity_text_counter)        # get the count of each summoned item from the conv_data

    # # Detected emotions in the population
    # print("\nEmotion Count:")
    # print(emotion_counter)            # also get a population count of all the emotions that were invoked in the conversation

    # print("\t\t[ emotion counter per-user :: {emotion_counter_per_user}")
    # Convert Counter objects to dictionaries
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

