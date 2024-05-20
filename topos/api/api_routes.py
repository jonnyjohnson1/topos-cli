# api_routes.py

import os
from fastapi import APIRouter, HTTPException
import requests
import tkinter as tk
from tkinter import filedialog
from topos.FC.conversation_cache_manager import ConversationCacheManager
router = APIRouter()

from collections import Counter, OrderedDict, defaultdict
from pydantic import BaseModel

cache_manager = ConversationCacheManager()
class ConversationIDRequest(BaseModel):
    conversation_id: str

@router.post("/chat_conversation_analysis")
async def chat_conversation_analysis(request: ConversationIDRequest):
    conversation_id = request.conversation_id
    # load conversation
    conv_data = cache_manager.load_from_cache(conversation_id)
    # Initialize counters
    named_entity_counter = Counter()
    entity_text_counter = Counter()
    emotion_counter = Counter()

    # Initialize role-based counters
    named_entity_counter_per_role = defaultdict(Counter)
    entity_text_counter_per_role = defaultdict(Counter)
    emotion_counter_per_role = defaultdict(Counter)

    # Extract counts
    for conversation_id, messages in conv_data.items():
        for message_id, content in messages.items():
            role = content['role']
            base_analysis = content['in_line']['base_analysis']
            for entity_type, entities in base_analysis.items():
                named_entity_counter[entity_type] += len(entities)
                named_entity_counter_per_role[role][entity_type] += len(entities)
                for entity in entities:
                    entity_text_counter[str(entity['text'])] += 1
                    entity_text_counter_per_role[role][str(entity['text'])] += 1
            
            emotions = content['commenter']['base_analysis']['emo_27']
            for emotion in emotions:
                emotion_counter[emotion['label']] += 1
                emotion_counter_per_role[role][emotion['label']] += 1

    # Evocations equals num of each entity
    # print("Named Entity Count:")
    # print(named_entity_counter)       # get the count of each entity from the conv_data

    # # Actual Items summoned
    # print("\nEntity Text Count:")
    # print(entity_text_counter)        # get the count of each summoned item from the conv_data

    # # Detected emotions in the population
    # print("\nEmotion Count:")
    # print(emotion_counter)            # also get a population count of all the emotions that were invoked in the conversation

    
    # Convert Counter objects to dictionaries
    named_entity_dict = {
        "totals": dict(named_entity_counter),
        "per_role": {role: dict(counter) for role, counter in named_entity_counter_per_role.items()}
    }
    entity_text_dict = {
        "totals": dict(entity_text_counter),
        "per_role": {role: dict(counter) for role, counter in entity_text_counter_per_role.items()}
    }
    emotion_dict = {
        "totals": dict(emotion_counter),
        "per_role": {role: dict(counter) for role, counter in emotion_counter_per_role.items()}
    }

    # Create the final dictionary
    conversation = {
        'entity_evocations': named_entity_dict,
        'entity_summons': entity_text_dict,
        'emotions27': emotion_dict
    }

    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found in cache")
    # Return the conversation or any other response needed
    return {"conversation": conversation}


@router.post("/list_models")
async def list_models():
    url = "http://localhost:11434/api/tags"
    try:
        result = requests.get(url)
        if result.status_code == 200:
            return {"result": result.json()}
        else:
            raise HTTPException(status_code=404, detail="Models not found")
    except requests.ConnectionError:
        raise HTTPException(status_code=500, detail="Server connection error")


@router.post("/get_files")
async def get_files():
    root = tk.Tk()
    root.withdraw()
    filetypes = [("PNG files", "*.png"), ("JPG files", "*.jpg"), ("JPEG files", "*.jpeg")]
    file_path = filedialog.askopenfilename(title="Select an image file",
                                       filetypes=(filetypes))
    print(file_path)
    
    # Use the os.path module
    system_path = os.path.abspath("/")
    print(system_path)
    bytes_list = read_file_as_bytes(file_path)
    media_type="application/json",
    
    # data = json.dumps(, ensure_ascii=False)
    # print(data[:110])
    return {"file_name" : [i for i in file_path], "bytes": bytes_list}

def read_file_as_bytes(file_path):
    try:
        with open(file_path, 'rb') as file:
            file_bytes = list(file.read())
        return file_bytes
    except FileNotFoundError:
        print("File not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None