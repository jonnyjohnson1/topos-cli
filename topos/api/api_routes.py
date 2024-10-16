# api_routes.py

import os
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
import requests
import signal
import glob
import sys
from topos.FC.conversation_cache_manager import ConversationCacheManager
router = APIRouter()

from collections import Counter, OrderedDict, defaultdict
from pydantic import BaseModel

from ..generations.chat_gens import LLMController
from ..utilities.utils import create_conversation_string
from ..services.ontology_service.mermaid_chart import MermaidCreator

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



import torch
from diffusers import DiffusionPipeline
@router.post("/chat/conv_to_image")
async def conv_to_image(request: ConversationIDRequest):
    conversation_id = request.conversation_id

    # load conversation
    conv_data = cache_manager.load_from_cache(conversation_id)
    if conv_data is None:
        raise HTTPException(status_code=404, detail="Conversation not found in cache")


    # model specifications
    # TODO UPDATE SO ITS NOT HARDCODED
    model = "dolphin-llama3"
    provider = 'ollama' # defaults to ollama right now
    api_key = 'ollama'

    llm_client = LLMController(model_name=model, provider=provider, api_key=api_key)

    context = create_conversation_string(conv_data, 6)
    print(context)
    print(f"\t[ converting conversation to image to text prompt: using model {model}]")
    conv_to_text_img_prompt = "Create an interesting, and compelling image-to-text prompt that can be used in a diffussor model. Be concise and convey more with the use of metaphor. Steer the image style towards Slavador Dali's fantastic, atmospheric, heroesque paintings that appeal to everyman themes."
    txt_to_img_prompt = llm_client.generate_response(context, conv_to_text_img_prompt, temperature=0)
    # print(txt_to_img_prompt)
    print(f"\t[ generating a file name {model} ]")
    txt_to_img_filename = llm_client.generate_response(txt_to_img_prompt, "Based on the context create an appropriate, and BRIEF, filename with no spaces. Do not use any file extensions in your name, that will be added in a later step.", temperature=0)

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
    provider: str
    api_key: str
    model: str
    voice_settings: dict

@router.post("/gen_next_message_options")
async def create_next_messages(request: GenNextMessageOptions):
    conversation_id = request.conversation_id
    query = request.query
    print(request.provider, "/", request.model)
    print(request.api_key)
    # model specifications
    model = request.model if request.model != None else "dolphin-llama3"
    provider = request.provider if request.provider != None else 'ollama' # defaults to ollama right now
    api_key = request.api_key if request.api_key != None else 'ollama'

    llm_client = LLMController(model_name=model, provider=provider, api_key=api_key)

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


    next_message_options = llm_client.generate_response(system_prompt, query, temperature=0)
    print(next_message_options)

    # return the options
    return {"response" : next_message_options}


class ConversationTopicsRequest(BaseModel):
    conversation_id: str
    model: str

@router.post("/gen_conversation_topics")
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


@router.post("/list_models")
async def list_models(provider: str = 'ollama', api_key: str = 'ollama'):
    # Define the URLs for different providers

    list_models_urls = {
        'ollama': "http://localhost:11434/api/tags",
        'openai': "https://api.openai.com/v1/models",
        'groq': "https://api.groq.com/openai/v1/models"
    }

    if provider not in list_models_urls:
        raise HTTPException(status_code=400, detail="Unsupported provider")

    # Get the appropriate URL based on the provider
    url = list_models_urls.get(provider.lower())

    if provider.lower() == 'ollama':
        # No need for headers with Ollama
        headers = {}
    else:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    try:
        # Make the request with the appropriate headers
        result = requests.get(url, headers=headers)
        if result.status_code == 200:
            return {"result": result.json()}
        else:
            raise HTTPException(status_code=result.status_code, detail="Models not found")
    except requests.ConnectionError:
        raise HTTPException(status_code=500, detail="Server connection error")

@router.post("/test")
async def test():
    return "hello world"

@router.post("/get_files")
async def get_files():
    # Get the current working directory
    current_dir = os.getcwd()

    # List all image files in the current directory
    image_files = glob.glob(os.path.join(current_dir, "*.png")) + \
                  glob.glob(os.path.join(current_dir, "*.jpg")) + \
                  glob.glob(os.path.join(current_dir, "*.jpeg"))

    if not image_files:
        return {"error": "No image files found in the current directory."}

    # Print available files
    print("Available image files:")
    for i, file in enumerate(image_files, 1):
        print(f"{i}. {os.path.basename(file)}")

    # Get user input
    while True:
        try:
            choice = int(input("Enter the number of the file you want to select: "))
            if 1 <= choice <= len(image_files):
                file_path = image_files[choice - 1]
                break
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

    print(f"Selected file: {file_path}")

    # Use the os.path module
    system_path = os.path.abspath("/")
    print(system_path)
    bytes_list = read_file_as_bytes(file_path)
    media_type = "application/json"
    print(type(bytes_list))
    return {"file_name": [i for i in file_path], "bytes": bytes_list}

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
    provider: str = "ollama"
    api_key: str = "ollama"
    temperature: float = 0.04

@router.post("/generate_mermaid_chart")
async def generate_mermaid_chart(payload: MermaidChartPayload):
    try:
        conversation_id = payload.conversation_id
        full_conversation = payload.full_conversation
        # model specifications
        model = payload.model
        provider = payload.provider# defaults to ollama right now
        api_key = payload.api_key
        temperature = payload.temperature

        llm_client = LLMController(model_name=model, provider=provider, api_key=api_key)

        mermaid_generator = MermaidCreator(llm_client)



        if full_conversation:
            cache_manager = cache_manager
            conv_data = cache_manager.load_from_cache(conversation_id)
            if conv_data is None:
                raise HTTPException(status_code=404, detail="Conversation not found in cache")
            print(f"\t[ generating mermaid chart :: {provider}/{model} :: full conversation ]")
            return {"status": "generating", "response": "generating mermaid chart", 'completed': False}
            # TODO: Complete this branch if needed

        else:
            message = payload.message
            if message:
                print(f"\t[ generating mermaid chart :: using model {model} ]")
                try:
                    mermaid_string = await mermaid_generator.get_mermaid_chart(message)
                    print(mermaid_string)
                    if mermaid_string == "Failed to generate mermaid":
                        return {"status": "error", "response": mermaid_string, 'completed': True}
                    else:
                        return {"status": "completed", "response": mermaid_string, 'completed': True}
                except Exception as e:
                    return {"status": "error", "response": f"Error: {e}", 'completed': True}

    except Exception as e:
        return {"status": "error", "message": str(e)}
