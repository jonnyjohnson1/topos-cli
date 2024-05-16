
# TODO Set up spacy NER function
import spacy
from spacy.tokens import Token

def get_token_sent(token):
    '''
    helper function for spacy nlp to process sentences
    '''
    token_span = token.doc[token.i:token.i+1]
    return token_span.sent

# Load the large English model
print("loading spacy model")
nlp = spacy.load("en_core_web_trf")
Token.set_extension('sent', getter=get_token_sent, force = True)

# Text to tokenize
text = "This is an example sentence to tokenize."



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
