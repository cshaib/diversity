
import nltk 
from tqdm import tqdm

def token_patterns(data, n, top_n=10): 
    """ Pulls out ngrams patterns in the data. 

    Args:
        data (list): data to run frequency.
        n (int): n-gram length.
        top_n (int, optional): top patterns to display. Defaults to 10.

    Returns:
        list: sorted list of top n-gram patterns
    """

    # treat data as one string 
    all_data = nltk.word_tokenize(' '.join(data))

    ngrams = list(nltk.ngrams(data, n))
    # keep only unique ngrams
    ngrams = set(ngrams) 

    frequency = nltk.FreqDist(ngrams)

    sorted_frequency = sorted(frequency.items(), key=lambda kv: kv[1], reverse=True)[:top_n]

    return sorted_frequency
