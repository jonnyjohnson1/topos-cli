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

from ..generations.ollama_chat import generate_response

cache_manager = ConversationCacheManager()
class ConversationIDRequest(BaseModel):
    conversation_id: str


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

   
    # Return the conversation or any other response needed
    return {"conversation": conversation}


# convert to a prompt
def create_conversation_string(conversation_data, last_n_messages):
    conversation_string = ""
    for conv_id, messages in conversation_data.items():
        last_messages = list(messages.items())[-last_n_messages:]
        for msg_id, message_info in last_messages:
            role = message_info['role']
            message = message_info['message']
            conversation_string += f"{role}: {message}\n"
    return conversation_string.strip()


import torch
from diffusers import DiffusionPipeline
@router.post("/chat/conv_to_image")
async def conv_to_image(request: ConversationIDRequest):
    conversation_id = request.conversation_id

    # load conversation
    conv_data = cache_manager.load_from_cache(conversation_id)
    if conv_data is None:
        raise HTTPException(status_code=404, detail="Conversation not found in cache")

   
    model = "dolphin-llama3"
    context = create_conversation_string(conv_data, 6)
    print(context)
    print(f"\t[ converting conversation to image to text prompt: using model {model}]")
    conv_to_text_img_prompt = "Create an interesting, and compelling image-to-text prompt that can be used in a diffussor model. Be concise and convey more with the use of metaphor. Steer the image style towards Slavador Dali's fantastic, atmospheric, heroesque paintings that appeal to everyman themes."
    txt_to_img_prompt = generate_response(context, conv_to_text_img_prompt, model=model, temperature=0)
    # print(txt_to_img_prompt)
    print(f"\t[ generating a file name {model} ]")
    txt_to_img_filename = generate_response(txt_to_img_prompt, "Based on the context create an appropriate, and BRIEF, filename with no spaces. Do not use any file extensions in your name, that will be added in a later step.", model=model, temperature=0)

    # run huggingface comic diffusion
    pipeline = DiffusionPipeline.from_pretrained("ogkalu/Comic-Diffusion")
    # Move the pipeline to the GPU if available, or to MPS if on an M-Series MacBook, otherwise to CPU
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    pipeline.to(device)

    # Generate an image based on the input text
    prompt = "somewhere over the rainbow"
    print(f"\t[ generating the image using: 'ogkalu/Comic-Diffusion' ]")
    image = pipeline(txt_to_img_prompt).images[0]
    file_name = f"{txt_to_img_filename}.png"
    file_name = "".join(file_name.split())
    # Save the generated image locally
    image.save(file_name)

    # Get file bytes to pass to UI
    system_path = os.path.abspath("/")
    print(system_path)
    bytes_list = read_file_as_bytes(file_name)
    media_type = "application/json"
    
    # return the image
    return {"file_name" : file_name, "bytes": bytes_list, "prompt": txt_to_img_prompt}


class GenNextMessageOptions(BaseModel):
    conversation_id: str
    query: str
    model: str

@router.post("/gen_next_message_options")
async def create_next_messages(request: GenNextMessageOptions):
    conversation_id = request.conversation_id
    query = request.query
    model = request.model if request.model != None else "dolphin-llama3"

    # load conversation
    conv_data = cache_manager.load_from_cache(conversation_id)
    if conv_data is None:
        raise HTTPException(status_code=404, detail="Conversation not found in cache")

    context = create_conversation_string(conv_data, 12)
    print(f"\t[ generating next message options: using model {model}]")

    system_prompt = "PRESENT CONVERSATION:\n-------<context>" + context + "\n-------\n"
    system_prompt += """Given the conversation, and what the user desires to accomplish, pretend you are in their shoes, role-play, and offer 3 messages the user can use in the conversation next.
Generate options based on these parameters.

conversation.json:
{
    "tone (formal-warm; 0-10)": warm,
    "pace (leisure-fast; 0-10)": leisurely,
    "depth (simple-profound; 0-10): insightful,
    "engagement (low-high; 0-10): engaging,
    "message length (short-long; 0-10): brief
}

Rules:
Wrap each message option in a markdown codeblock.
"""
    print(system_prompt)
    print(query)
    next_message_options = generate_response(system_prompt, query, model=model, temperature=0)
    print(next_message_options)
    
    # return the image
    return {"response" : next_message_options}

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
    media_type = "application/json"
    print(type(bytes_list))
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