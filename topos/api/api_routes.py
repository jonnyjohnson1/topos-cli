import os
import requests
import torch
import tkinter as tk
from tkinter import filedialog
from collections import Counter, defaultdict
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from diffusers import DiffusionPipeline

# Import custom utilities and models
from topos.FC.conversation_cache_manager import ConversationCacheManager
from ..utilities.utils import create_conversation_string


class APIRoutes:
    def __init__(self, model):
        self.cache_manager = ConversationCacheManager()
        self.model = model
        self.router = APIRouter()

        # Register routes
        self.router.post("/chat_conversation_analysis")(self.chat_conversation_analysis)
        self.router.post("/chat/conv_to_image")(self.conv_to_image)
        self.router.post("/gen_next_message_options")(self.create_next_messages)
        self.router.post("/gen_conversation_topics")(self.create_conversation_topics)
        self.router.post("/list_models")(self.list_models)
        self.router.post("/test")(self.test)
        self.router.post("/get_files")(self.get_files)

    class ConversationIDRequest(BaseModel):
        conversation_id: str

    class GenNextMessageOptions(BaseModel):
        conversation_id: str
        query: str
        model_type: str = "ollama"
        model_name: str = "dolphin-llama3"
        voice_settings: dict = {
            "tone": "analytical",
            "distance": "distant",
            "pace": "leisurely",
            "depth": "insightful",
            "engagement": "engaging",
            "message length": "brief"
        }

    class ConversationTopicsRequest(BaseModel):
        conversation_id: str
        model_type: str = "ollama"
        model_name: str = "dolphin-llama3"

    async def chat_conversation_analysis(self, request: ConversationIDRequest):
        conversation_id = request.conversation_id

        # Load conversation
        conv_data = self.cache_manager.load_from_cache(conversation_id)
        if conv_data is None:
            raise HTTPException(status_code=404, detail="Conversation not found in cache")

        # Initialize counters
        named_entity_counter = Counter()
        entity_text_counter = Counter()
        emotion_counter = Counter()
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
                user = content.get('user_name', role)

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

        # Prepare response
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

        conversation = {
            'entity_evocations': named_entity_dict,
            'entity_summons': entity_text_dict,
            'emotions27': emotion_dict
        }

        return {"conversation": conversation}

    async def conv_to_image(self, request: ConversationIDRequest):
        conversation_id = request.conversation_id

        # Load conversation
        conv_data = self.cache_manager.load_from_cache(conversation_id)
        if conv_data is None:
            raise HTTPException(status_code=404, detail="Conversation not found in cache")

        model = "dolphin-llama3"
        context = create_conversation_string(conv_data, 6)
        print(context)
        print(f"\t[ converting conversation to image to text prompt: using model {model}]")

        conv_to_text_img_prompt = (
            "Create an interesting, and compelling image-to-text prompt that can be used in a diffussor model. "
            "Be concise and convey more with the use of metaphor. Steer the image style towards Salvador Dali's fantastic, atmospheric, "
            "heroesque paintings that appeal to everyman themes."
        )
        txt_to_img_prompt = self.model.generate_response(context, conv_to_text_img_prompt)

        print(f"\t[ generating a file name {model} ]")
        txt_to_img_filename = self.model.generate_response(
            txt_to_img_prompt,
            "Based on the context create an appropriate, and BRIEF, filename with no spaces. Do not use any file extensions in your name, that will be added in a later step."
        )

        # Run HuggingFace diffusion pipeline
        pipeline = DiffusionPipeline.from_pretrained("ogkalu/Comic-Diffusion")

        # Move the pipeline to the GPU if available, or to MPS if on an M-Series MacBook, otherwise to CPU
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        pipeline.to(device)

        print(f"\t[ generating the image using: 'ogkalu/Comic-Diffusion' ]")
        image = pipeline(txt_to_img_prompt).images[0]
        file_name = f"{txt_to_img_filename}.png".replace(" ", "")

        # Save the generated image locally
        image.save(file_name)

        # Get file bytes to pass to UI
        system_path = os.path.abspath("/")
        print(f"\t[ {system_path}")
        bytes_list = self.read_file_as_bytes(file_name)

        return {"file_name": file_name, "bytes": bytes_list, "prompt": txt_to_img_prompt}

    async def create_next_messages(self, request: GenNextMessageOptions):
        conversation_id = request.conversation_id
        query = request.query
        model = request.model_name
        voice_settings = request.voice_settings

        # Load conversation
        conv_data = self.cache_manager.load_from_cache(conversation_id)
        if conv_data is None:
            raise HTTPException(status_code=404, detail="Conversation not found in cache")

        context = create_conversation_string(conv_data, 12)
        print(f"\t[ generating next message options: using model {model}]")

        conv_json = f"""
    conversation.json:
    {voice_settings}
    """
        print(conv_json)

        system_prompt = (
            f"PRESENT CONVERSATION:\n-------<context>{context}\n-------\n"
            "Roleplay with the current conversation, and offer 3 messages the user can speak next.\n"
            "Generate options based on these parameters.\n"
            + conv_json
        )

        next_message_options = self.model.generate_response(system_prompt, query)
        print(next_message_options)

        return {"response": next_message_options}

    async def create_conversation_topics(self, request: ConversationTopicsRequest):
        conversation_id = request.conversation_id
        model = request.model_name

        # Load conversation
        conv_data = self.cache_manager.load_from_cache(conversation_id)
        if conv_data is None:
            raise HTTPException(status_code=404, detail="Conversation not found in cache")

        context = create_conversation_string(conv_data, 12)
        print(f"\t[ generating summary :: model {model}]")

        query = "List the topics and those closely related to what this conversation traverses."
        system_prompt = f"PRESENT CONVERSATION:\n-------<context>{context}\n-------\n"

        topic_list = self.model.generate_response(system_prompt, query)
        print(topic_list)

        return {"response": topic_list}

    async def list_models(self):
        url = "http://localhost:11434/api/tags"
        try:
            result = requests.get(url)
            if result.status_code == 200:
                return {"result": result.json()}
            else:
                raise HTTPException(status_code=404, detail="Models not found")
        except requests.ConnectionError:
            raise HTTPException(status_code=500, detail="Server connection error")

    async def test(self):
        return "hello world"

    async def get_files(self):
        root = tk.Tk()
        root.withdraw()
        filetypes = [("PNG files", "*.png"), ("JPG files", "*.jpg"), ("JPEG files", "*.jpeg")]
        file_path = filedialog.askopenfilename(title="Select an image file", filetypes=filetypes)
        print(file_path)

        # Use the os.path module
        system_path = os.path.abspath("/")
        print(system_path)
        bytes_list = self.read_file_as_bytes(file_path)
        return {"file_name": file_path, "bytes": bytes_list}

    def read_file_as_bytes(self, file_path):
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
