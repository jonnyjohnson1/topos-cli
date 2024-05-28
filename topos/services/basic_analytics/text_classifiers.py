# Use a pipeline as a high-level helper
from transformers import pipeline


# Text moderation levels utils
def get_full_name(label):
    labels_table = {
        "S": "Sexual",
        "H": "Hate",
        "V": "Violence",
        "HR": "Harassment",
        "SH": "Self-harm",
        "S3": "Sexual/minors",
        "H2": "Hate/threatening",
        "V2": "Violence/graphic",
        "OK": "OK"
    }
    return labels_table.get(label, "Unknown label")

def get_text_moderation_levels(text):
    pipe = pipeline("text-classification", model="KoalaAI/Text-Moderation", truncation=True, max_length=128)
    labels_list = pipe(text)
    added_names = []
    for lbl in labels_list:
        lbl['name'] =  get_full_name(lbl['label'])
        added_names.append(lbl)
    return added_names

def get_text_sentiment_ternary(text):
    try:
        pipe = pipeline("text-classification", model="finiteautomata/bertweet-base-sentiment-analysis", truncation=True, max_length=128)
        try:
            results = pipe(text)
            return results
        except Exception as e:
            logging.error(f"Failed to get_text_sentiment_ternary: {e}")
            return None
    except Exception as e:
        logging.error(f"Failed to get_text_sentiment_ternary: {e}")
        return None
        

def get_text_sentiment_6(text):
    try:
        pipe = pipeline("text-classification", model="bhadresh-savani/bert-base-uncased-emotion", truncation=True, max_length=128)
        return pipe(text)
    except Exception as e:
        logging.error(f"Failed to get_text_sentiment_6: {e}")
        return None

def get_text_sentiment_27(text):
    try:
        pipe = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", truncation=True, max_length=128)
        return pipe(text)
    except Exception as e:
        logging.error(f"Failed to get_text_sentiment_27: {e}")
        return None