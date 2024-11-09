from ..basic_analytics.token_classifiers import get_ner
from ..basic_analytics.text_classifiers import get_text_moderation_levels, get_text_sentiment_ternary, get_text_sentiment_27
from ...utils.utils import is_connected

def base_token_classifier(last_message):
    """
    set of token classification options
    """
    # spacy-       token classifications
    entity_dict = get_ner(last_message)
    return entity_dict

def base_text_classifier(last_message):
    """
    set of token classification options
    """
    # transformers- text classifications
    mod_level = get_text_moderation_levels(last_message)
    tern_sent = get_text_sentiment_ternary(last_message) if is_connected else [] # TODO this line fails if not connected to internet; this was an attempt to check internet connection, but doesn't appear to work
    emo_27 = get_text_sentiment_27(last_message)
    text_class_dict = {
        "mod_level": mod_level,
        "tern_sent": tern_sent,
        "emo_27": emo_27
    }
    return text_class_dict

