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


status_checks = {
    "health_check": False,
    "update_check": False,
}

def check_health(icon):
    """Periodically check the service health."""
    certs = get_ssl_certificates()
    if not os.path.exists(certs['cert_path']):
        print(f"Certificate file not found: {certs['cert_path']}")
    if not os.path.exists(certs['key_path']):
        print(f"Key file not found: {certs['key_path']}")

    while icon.visible:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='Unverified HTTPS request')
                response = requests.get(API_URL, verify=False)
            # Update health check status based on response
            status_checks["health_check"] = response.status_code == 200
        except requests.exceptions.RequestException as e:
            print(f"Health check error: {str(e)}")
            status_checks["health_check"] = False
        finally:
            evaluate_icon_status(icon)
        time.sleep(5)

def check_update():
    """Check if an update is available."""
    return check_for_update("jonnyjohnson1", "topos-cli")


def evaluate_icon_status(icon):
    """Evaluate and update the icon's status and menu based on checks."""
    if not status_checks["health_check"]:
        # If the service is not running
        update_status(icon, "Service is not running", "red")
    elif status_checks["update_check"]:
        # If an update is available
        update_status(icon, "Update available", (255, 165, 0, 255))  # Orange
    else:
        # If all checks pass
        update_status(icon, "Service is running", (170, 255, 0, 255))  # Green
    
    # Update the menu based on the current status
    update_menu(icon)
    
def check_update_available(icon):
    """Periodically check for updates."""
    while icon.visible:
        # Update the update check status
        status_checks["update_check"] = check_update()
        evaluate_icon_status(icon)
        time.sleep(60)

def pull_latest_release():
    print("Pulling latest release...")
    update_topos()

def update_icon(icon):
    # Start a separate thread for checking updates
    update_thread = threading.Thread(target=check_update_available, args=(icon,), daemon=True)
    update_thread.start()
    
    # Start a separate thread for checking health
    health_thread = threading.Thread(target=check_health, args=(icon,), daemon=True)
    health_thread.start()
    
def update_status(icon, text, color):
    icon.icon = create_image(color)
    # icon.notify(text)

def update_menu(icon):
    """Dynamically update the icon menu based on update status."""
    if status_checks["update_check"]:
        icon.menu = pystray.Menu(
            pystray.MenuItem("Open API Docs", open_docs),
            pystray.MenuItem("Update Topos", pull_latest_release),
            pystray.MenuItem("Exit", on_exit)
        )
    else:
        icon.menu = pystray.Menu(
            pystray.MenuItem("Open API Docs", open_docs),
            pystray.MenuItem("Exit", on_exit)
        )

# def update_status(icon, text, color):
#     icon.icon = create_image(color)
#     icon.notify(text)


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
        health_thread = threading.Thread(target=update_icon, args=(icon,))
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