import nltk
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize
import logging
logger = logging.getLogger(__name__)


def tokenize_sentences(new_text):
    sentences = sent_tokenize(new_text)
    number_sentence_dict = {key:val for key, val in zip(range(1, len(sentences)+1), sentences)}
    return number_sentence_dict

def number_text(text: str):
    new_text = text.replace("“", "\'")
    new_text = new_text.replace("”", "\'")
    try:
        number_sentence_dict = tokenize_sentences(new_text)
    except Exception as e:
        logger.error(e)
        return None, None


    numbered_text = ""
    for key in number_sentence_dict:
        numbered_text += "[%d] " % key
        numbered_text += number_sentence_dict[key]
    
    return numbered_text, number_sentence_dict