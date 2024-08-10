# api_routes.py

import os
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
import requests
import signal
import tkinter as tk
from tkinter import filedialog
from topos.FC.conversation_cache_manager import ConversationCacheManager
router = APIRouter()

from collections import Counter, OrderedDict, defaultdict
from pydantic import BaseModel

from ..generations.ollama_chat import generate_response
from ..utilities.utils import create_conversation_string
from ..services.ontology_service.mermaid_chart import get_mermaid_chart

cache_manager = ConversationCacheManager()
class ConversationIDRequest(BaseModel):
    conversation_id: str


@router.post("/shutdown")
def shutdown(request: Request):
    os.kill(os.getpid(), signal.SIGTERM)
    return JSONResponse(content={"message": "Server shutting down..."})


@router.get("/health")
async def health_check():
    try:
        # Perform any additional checks here if needed
        return {"status": "healthy"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {e}")


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
    # Extract counts
    for conversation_id, messages in conv_data.items():
        print(f"\t\t[ item :: {conversation_id} ]")
        for message_id, content in messages.items():
            print(f"\t\t\t[ content :: {str(content)[40:]} ]")
            print(f"\t\t\t[ keys :: {str(content.keys())[40:]} ]")
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

    print("\t\t[ emotion counter per-user :: {emotion_counter_per_user}")
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
    print(f"\t[ {system_path}")
    bytes_list = read_file_as_bytes(file_name)
    media_type = "application/json"
    
    # return the image
    return {"file_name" : file_name, "bytes": bytes_list, "prompt": txt_to_img_prompt}


class GenNextMessageOptions(BaseModel):
    conversation_id: str
    query: str
    model: str
    voice_settings: dict

@router.post("/gen_next_message_options")
async def create_next_messages(request: GenNextMessageOptions):
    conversation_id = request.conversation_id
    query = request.query
    model = request.model if request.model != None else "dolphin-llama3"
    voice_settings = request.voice_settings  if request.voice_settings != None else """{
    "tone": "analytical",
    "distance": "distant",
    "pace": "leisurely",
    "depth": "insightful",
    "engagement": "engaging",
    "message length": "brief"
}"""
    # load conversation
    conv_data = cache_manager.load_from_cache(conversation_id)
    if conv_data is None:
        raise HTTPException(status_code=404, detail="Conversation not found in cache")

    context = create_conversation_string(conv_data, 12)
    print(f"\t[ generating next message options: using model {model}]")


    conv_json = f"""
conversation.json:
{voice_settings}
"""
    print(conv_json)

    system_prompt = "PRESENT CONVERSATION:\n-------<context>" + context + "\n-------\n"
    system_prompt += """Roleplay with the current conversation, and offer 3 messages the user can speak next.
Generate options based on these parameters.
"""
    system_prompt += conv_json


    next_message_options = generate_response(system_prompt, query, model=model, temperature=0)
    print(next_message_options)
    
    # return the options
    return {"response" : next_message_options}


class ConversationTopicsRequest(BaseModel):
    conversation_id: str
    model: str

@router.post("/gen_conversation_topics")
async def create_next_messages(request: ConversationTopicsRequest):
    conversation_id = request.conversation_id
    model = request.model if request.model != None else "dolphin-llama3"

    # load conversation
    conv_data = cache_manager.load_from_cache(conversation_id)
    if conv_data is None:
        raise HTTPException(status_code=404, detail="Conversation not found in cache")

    context = create_conversation_string(conv_data, 12)
    print(f"\t[ generating summary :: model {model} :: subject {subject}]")

    query = f""
    # topic list first pass
    system_prompt = "PRESENT CONVERSATION:\n-------<context>" + context + "\n-------\n"
    query += """List the topics and those closely related to what this conversation traverses."""
    topic_list = generate_response(system_prompt, query, model=model, temperature=0)
    print(topic_list)

    # return the image
    return {"response" : topic_list}


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

@router.post("/test")
async def test():
    return "hello world"

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
    
    
class MermaidChartPayload(BaseModel):
    message: str = None
    conversation_id: str
    full_conversation: bool = False
    model: str = "dolphin-llama3"
    temperature: float = 0.04

@router.post("/generate_mermaid_chart")
async def generate_mermaid_chart(payload: MermaidChartPayload):
    try:
        conversation_id = payload.conversation_id
        full_conversation = payload.full_conversation
        model = payload.model
        temperature = payload.temperature

        if full_conversation:
            cache_manager = ConversationCacheManager()
            conv_data = cache_manager.load_from_cache(conversation_id)
            if conv_data is None:
                raise HTTPException(status_code=404, detail="Conversation not found in cache")
            print(f"\t[ generating mermaid chart :: using model {model} :: full conversation ]")
            return {"status": "generating", "response": "generating mermaid chart", 'completed': False}
            # TODO: Complete this branch if needed

        else:
            message = payload.message
            if message:
                print(f"\t[ generating mermaid chart :: using model {model} ]")
                try:
                    mermaid_string = await get_mermaid_chart(message)
                    if mermaid_string == "Failed to generate mermaid":
                        return {"status": "error", "response": mermaid_string, 'completed': True}
                    else:
                        return {"status": "completed", "response": mermaid_string, 'completed': True}
                except Exception as e:
                    return {"status": "error", "response": f"Error: {e}", 'completed': True}

    except Exception as e:
        return {"status": "error", "message": str(e)}