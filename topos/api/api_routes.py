# api_routes.py

import os
from fastapi import APIRouter, HTTPException
import requests
import tkinter as tk
from tkinter import filedialog
from topos.FC.conversation_cache_manager import ConversationCacheManager
router = APIRouter()

from pydantic import BaseModel

cache_manager = ConversationCacheManager()
class ConversationIDRequest(BaseModel):
    conversation_id: str

@router.post("/chat_conversation_analysis")
async def chat_conversation_analysis(request: ConversationIDRequest):
    # TESTING :: print the conversation (We want to test that it is loading right now)
    conversation_id = request.conversation_id
    # load conversation

    conversation = cache_manager.load_from_cache(conversation_id)
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