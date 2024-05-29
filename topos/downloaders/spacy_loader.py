import subprocess
import yaml

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
    try:
        subprocess.run(['python3', '-m', 'spacy', 'download', model_name], check=True)
        # Write updated settings to YAML file
        with open('config.yaml', 'w') as file:
            yaml.dump({'active_spacy_model': model_name}, file)
        print(f"Successfully downloaded '{model_name}' spaCy model.")
        print(f"'{model_name}' set as active model.")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading '{model_name}' spaCy model: {e}")