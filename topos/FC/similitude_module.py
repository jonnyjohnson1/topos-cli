from sentence_transformers import SentenceTransformer, util
import os


def load_model(model_name):
    try:
        # Check if the model is already downloaded by attempting to load it
        model = SentenceTransformer(model_name)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print("Failed to load model:", str(e))
        # Attempt to download the model, ensuring network issues are handled
        try:
            print("Attempting to download the model...")
            # This function automatically downloads and caches the model
            model = SentenceTransformer(model_name)
            print("Model downloaded and loaded successfully.")
            return model
        except Exception as e:
            print("An error occurred while downloading the model:", str(e))
            return None