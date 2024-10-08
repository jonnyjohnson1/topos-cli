import os
from fastapi import APIRouter, HTTPException, Request
import requests
from topos.FC.conversation_cache_manager import ConversationCacheManager
from collections import Counter, OrderedDict, defaultdict
from pydantic import BaseModel

from ..utilities.utils import create_conversation_string
from ..services.classification_service.base_analysis import base_text_classifier, base_token_classifier

import json
import time
from datetime import datetime
import logging

router = APIRouter()

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

@router.post("/p2p/process_message")
async def process_message(request: Request):
    try:
        payload = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    conversation_id = payload.get("conversation_id")
    message_id = payload.get("message_id")
    message = payload.get("message")
    message_history = payload.get("message_history")
    current_topic = payload.get("topic", "Unknown")
    processing_config = payload.get("processing_config", {})
    user_id = payload.get("user_id", {})
    user_name = payload.get("user_name", "user") # let's just use the username for now to use to pull in the chatroom information
    role = payload.get("role", "user")

    # Set default values if any key is missing or if processing_config is None
    default_config = {
        "showInMessageNER": True,
        "calculateInMessageNER": True,
        "showModerationTags": True,
        "calculateModerationTags": True,
        "showSidebarBaseAnalytics": True
    }

    # Update default_config with provided processing_config, if any
    config = {**default_config, **processing_config}

    # processing message functions here
    print("[ processing message :: base_analytics_functions]")
    # Fetch base, per-message token classifiers
    if config['calculateInMessageNER']:
        start_time = time.time()
        base_analysis = base_token_classifier(message)  # this is only an ner dict atm
        duration = time.time() - start_time
        print(f"\t[ base_token_classifier duration: {duration:.4f} seconds ]")

    # Fetch base, per-message text classifiers
    # Start timer for base_text_classifier
    if config['calculateModerationTags']:
        start_time = time.time()
        text_classifiers = {}
        try:
            text_classifiers = base_text_classifier(message)
        except Exception as e:
            logging.error(f"Failed to compute base_text_classifier: {e}")
        duration = time.time() - start_time
        print(f"\t[ base_text_classifier duration: {duration:.4f} seconds ]")

    conv_cache_manager = cache_manager
    dummy_data = {}  # Replace with actual processing logic
    if config['calculateModerationTags'] or config['calculateInMessageNER']:
        print(f"\t[ save to conv cache :: conversation {conversation_id}-{message_id} ]")
        try:
            dummy_data = {
                message_id :
                    {
                    'user_name': user_name,
                    'user_id': user_id,
                    'role': role,
                    'timestamp': datetime.now(),
                    'message': message
                }}
        except Exception as e:
            print("Error", e)
        if config['calculateInMessageNER']:
            dummy_data[message_id]['in_line'] = {'base_analysis': base_analysis}
        if config['calculateModerationTags']:
            dummy_data[message_id]['commenter'] = {'base_analysis': text_classifiers}
            conv_cache_manager.save_to_cache(conversation_id, dummy_data)
            # Removing the keys from the nested dictionary
        if message_id in dummy_data:
            dummy_data[message_id].pop('message', None)
            dummy_data[message_id].pop('timestamp', None)
        # Sending first batch of user message analysis back to the UI
        # return websocket.send_json({"status": "fetched_user_analysis", 'user_message': dummy_data})
    else:
        print(f"\t[ save to conv cache :: conversation {conversation_id}-{message_id} ]")
        # Saving an empty dictionary for the messag id
        conv_cache_manager.save_to_cache(conversation_id, {
            message_id :
                {
                'user_name': user_name,
                'user_id': user_id,
                'role': role,
                'message': message,
                'timestamp': datetime.now(),
            }})
        dummy_data = {
            message_id :
                {
                'user_name': user_name,
                'user_id': user_id,
                'role': role,
                'message': message,
                'timestamp': datetime.now(),
            }}  # Replace with actual processing logic

    return {"status": "fetched_user_analysis", "user_message": dummy_data}
