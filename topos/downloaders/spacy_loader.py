import yaml
import os
from ..utils.utils import get_config_path

def download_spacy_model(model_selection):
    if model_selection == 'small':
        model_name = "en_core_web_sm"
    elif model_selection == 'med':
        model_name = "en_core_web_md"
    elif model_selection == 'large':
        model_name = "en_core_web_lg"
    elif model_selection == 'trf':
        model_name = "en_core_web_trf"
    else: #default
        model_name = "en_core_web_sm"

    # Define the path to the config.yaml file
    config_path = get_config_path()
    try:
        # Write updated settings to YAML file
        with open(config_path, 'w') as file:
            yaml.dump({'active_spacy_model': model_name}, file)
        print(f"'{model_name}' set as active model.")
    except Exception as e:
        print(f"An error occurred setting config.yaml: {e}")
