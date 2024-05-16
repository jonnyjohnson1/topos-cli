import subprocess

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
        subprocess.run(['python', '-m', 'spacy', 'download', model_name], check=True)
        print(f"Successfully downloaded '{model_name}' spaCy model.")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading '{model_name}' spaCy model: {e}")

if __name__ == "__main__":
    model_name = "en_core_web_sm"  # Change this to the desired spaCy model
    download_spacy_model(model_name)