import os
import logging

from fastapi import APIRouter, HTTPException

from topos.services.database.conversation_cache_manager import ConversationCacheManager
router = APIRouter()

from ....services.generations_service.chat_gens import LLMController
from ....utils.utils import create_conversation_string
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

@router.post("/chat/conv_to_image")
async def conv_to_image(request: ConversationIDRequest):
    import torch
    from diffusers import DiffusionPipeline
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

    bytes_list = read_file_as_bytes(file_name)
    media_type = "application/json"

    # return the image
    return {"file_name" : file_name, "bytes": bytes_list, "prompt": txt_to_img_prompt}

