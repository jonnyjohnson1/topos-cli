

import os
from fastapi import APIRouter, HTTPException
from topos.FC.conversation_cache_manager import ConversationCacheManager

from ....services.generations_service.chat_gens import LLMController
from ....utilities.utils import create_conversation_string
from ....models.models import ConversationTopicsRequest

import logging

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

@router.post("/get_files")
async def create_next_messages(request: ConversationTopicsRequest):
    conversation_id = request.conversation_id
    # model specifications
    # TODO UPDATE SO ITS NOT HARDCODED
    model = request.model if request.model != None else "dolphin-llama3"
    provider = 'ollama' # defaults to ollama right now
    api_key = 'ollama'

    llm_client = LLMController(model_name=model, provider=provider, api_key=api_key)

    # load conversation
    conv_data = cache_manager.load_from_cache(conversation_id)
    if conv_data is None:
        raise HTTPException(status_code=404, detail="Conversation not found in cache")

    context = create_conversation_string(conv_data, 12)
    # print(f"\t[ generating summary :: model {model} :: subject {subject}]")

    query = f""
    # topic list first pass
    system_prompt = "PRESENT CONVERSATION:\n-------<context>" + context + "\n-------\n"
    query += """List the topics and those closely related to what this conversation traverses."""
    topic_list = llm_client.generate_response(system_prompt, query, temperature=0)
    print(topic_list)

    # return the image
    return {"response" : topic_list}
