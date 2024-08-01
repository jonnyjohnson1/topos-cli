from ..api import api
from ..downloaders.spacy_loader import download_spacy_model
from topos.utilities.utils import get_root_directory

import requests
import threading
import webbrowser
from PIL import Image, ImageDraw
import pystray
import time
import os

API_URL = "http://0.0.0.0:13341/health"
DOCS_URL = "http://0.0.0.0:13341/docs"
ASSETS_PATH = os.path.join(get_root_directory(), "assets/topos_white.png")


def start_api():
    api.start_local_api()

def check_health(icon):
    while icon.visible:
        try:
            response = requests.get(API_URL)
            if response.status_code == 200:
                update_status(icon, "Service is running", (170, 255, 0, 255))
            else:
                update_status(icon, "Service is not running", "red")
        except requests.exceptions.RequestException as e:
            update_status(icon, f"Error: {str(e)}", "red")
        time.sleep(5)

def update_status(icon, text, color):
    icon.icon = create_image(color)

def open_docs():
    webbrowser.open_new(DOCS_URL)


def create_image(color):
    # Load the external image
    external_image = Image.open(ASSETS_PATH).convert("RGBA")
    # Resize external image to fit the icon size
    external_image = external_image.resize((34, 34), Image.Resampling.LANCZOS)
    
    # Generate an image for the system tray icon
    width = 34
    height = 34
    image = Image.new('RGBA', (width, height), (255, 255, 255, 0))  # Transparent background
    dc = ImageDraw.Draw(image)
    dc.ellipse((22, 22, 32, 32), fill=color)  # Smaller circle

    # Combine the images
    combined_image = Image.alpha_composite(external_image, image)
    
    return combined_image

def create_tray_icon():
    icon = pystray.Icon("Service Status Checker")
    icon.icon = create_image("yellow")
    icon.menu = pystray.Menu(
        pystray.MenuItem("Open API Docs", open_docs),
        pystray.MenuItem("Exit", on_exit)
    )

    def on_setup(icon):
        icon.visible = True
        # Start health check in a separate thread
        health_thread = threading.Thread(target=check_health, args=(icon,))
        health_thread.daemon = True
        health_thread.start()

    icon.run(setup=on_setup)

def on_exit(icon, item):
    icon.visible = False
    icon.stop()

def start_app():
# if __name__ == "__main__":
    # Start the API in a separate thread
    api_thread = threading.Thread(target=start_api)
    api_thread.daemon = True
    api_thread.start()

    # Create and start the tray icon on the main thread
    create_tray_icon()