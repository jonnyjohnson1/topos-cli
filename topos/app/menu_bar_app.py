from ..api import api
from ..downloaders.spacy_loader import download_spacy_model
from topos.utils.utils import get_root_directory
from ..config import get_ssl_certificates
from ..utils.check_for_update import check_for_update, update_topos

import requests
import threading
import webbrowser
from PIL import Image, ImageDraw
import pystray
import time
import os

import warnings

ASSETS_PATH = os.path.join(get_root_directory(), "assets/topos_white.png")
API_URL = "http://0.0.0.0:13341/health"
DOCS_URL = "http://0.0.0.0:13341/docs"

def start_api():
    api.start_local_api()

def start_web_app():
    global API_URL, DOCS_URL
    API_URL = "https://0.0.0.0:13341/health"
    DOCS_URL = "https://0.0.0.0:13341/docs"
    api_thread = threading.Thread(target=api.start_web_api)
    api_thread.daemon = True
    api_thread.start()
    # Create and start the tray icon on the main thread
    create_tray_icon()
    
def check_health(icon):
    certs = get_ssl_certificates()
    if not os.path.exists(certs['cert_path']):
        print(f"Certificate file not found: {certs['cert_path']}")
    if not os.path.exists(certs['key_path']):
        print(f"Key file not found: {certs['key_path']}")
    
    while icon.visible:
        try:
            with warnings.catch_warnings(): # cert=(certs['cert_path'], certs['key_path']) #for verification, but wasn't working
                warnings.filterwarnings('ignore', message='Unverified HTTPS request')
                response = requests.get(API_URL, verify=False)
            if response.status_code == 200:
                update_status(icon, "Service is running", (170, 255, 0, 255))
            else:
                update_status(icon, "Service is not running", "red")
        except requests.exceptions.RequestException as e:
            update_status(icon, f"Error: {str(e)}", "red")
        time.sleep(5)

def check_for_update():
    # Check the latest release status from the GitHub API
    # Returns True if an update is available, else False
    update_is_available = check_for_update("jonnyjohnson1", "topos-cli")
    return update_is_available

def check_update_available(icon):
    while icon.visible:
        if check_for_update():
            update_status(icon, "Update available", (255, 165, 0, 255))  # Orange indicator
            # Add "Check for Update" option if update is available
            icon.menu = pystray.Menu(
                pystray.MenuItem("Open API Docs", open_docs),
                pystray.MenuItem("Update your Topos", pull_latest_release),
                pystray.MenuItem("Exit", on_exit)
            )
        else:
            # Set the menu back to its default state without "Check for Update"
            icon.menu = pystray.Menu(
                pystray.MenuItem("Open API Docs", open_docs),
                pystray.MenuItem("Exit", on_exit)
            )
            update_status(icon, "Service is running", (170, 255, 0, 255))  # Normal green status
        time.sleep(60)  # Check every minute

def pull_latest_release():
    print("Pulling latest release...")
    update_topos()

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

def start_local_app():
    api_thread = threading.Thread(target=api.start_local_api)
    api_thread.daemon = True
    api_thread.start()
    # Create and start the tray icon on the main thread
    create_tray_icon()

def start_web_app():
    global API_URL, DOCS_URL
    API_URL = "https://0.0.0.0:13341/health"
    DOCS_URL = "https://0.0.0.0:13341/docs"
    api_thread = threading.Thread(target=api.start_web_api)
    api_thread.daemon = True
    api_thread.start()
    # Create and start the tray icon on the main thread
    create_tray_icon()