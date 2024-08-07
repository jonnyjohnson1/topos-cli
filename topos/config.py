from dotenv import load_dotenv
import os
from fastapi.middleware.cors import CORSMiddleware
from topos.utilities.utils import get_root_directory


def get_openai_api_key():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\033[93mWARNING:\033[0m OPENAI_API KEY environment variable is not set.")
    return api_key


def get_ssl_certificates():
    # project_dir = get_root_directory()
    # print(project_dir)
    # "key_path": project_dir + "/key.pem",
    # "cert_path": project_dir + "/cert.pem"
    return {
        "key_path": "key.pem",
        "cert_path": "cert.pem"
    }


def setup_config(app):
    """Configure application settings."""
    load_dotenv()  # Load environment variables from a .env file

    # Set up CORS middleware for the app
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Customize as needed for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )

    # Set app metadata from environment variables or defaults
    app.title = os.getenv("APP_TITLE", "Topos Chat API")
    app.version = os.getenv("APP_VERSION", "1.0")