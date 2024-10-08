import os

import spacy
from spacy.tokens import Token
import en_core_web_sm

def get_token_sent(token):
    '''
    helper function for spacy nlp to process sentences
    '''
    token_span = token.doc[token.i:token.i+1]
    return token_span.sent

# Now you can use `model_name` in your code
print(f"[ mem-loader :: Using spaCy model: en_core_web_sm ]")
nlp = en_core_web_sm.load()
Token.set_extension('sent', getter=get_token_sent, force = True)

def get_entity_dict(doc):
    entity_dict = {}
    for entity in doc.ents:
        if entity.label_ in entity_dict:
            entity_dict[entity.label_].append({
                        "label": str(entity.label_),
                        "text": entity.text,
                        "sentiment": entity.sentiment,
                        "start_position": entity.start_char,
                        "end_position": entity.end_char
                    })
        else:
            entity_dict[entity.label_] = [{
                        "label": str(entity.label_),
                        "text": entity.text,
                        "sentiment": entity.sentiment,
                        "start_position": entity.start_char,
                        "end_position": entity.end_char
                    }]
    return entity_dict

# Process the text with the loaded model
def get_ner(text):
    doc = nlp(text)
    entity_dict = get_entity_dict(doc)
    return entity_dict
