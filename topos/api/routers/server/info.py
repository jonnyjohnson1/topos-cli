

import os
from fastapi import APIRouter, HTTPException
import requests
import glob

router = APIRouter()

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


    bytes_list = read_file_as_bytes(file_path)
    media_type = "application/json"
    print(type(bytes_list))
    return {"file_name": [i for i in file_path], "bytes": bytes_list}


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
