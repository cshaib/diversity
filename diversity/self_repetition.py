import numpy as np
from nltk.util import ngrams

from tqdm import tqdm
from typing import List
from collections import Counter

def self_repetition_score(
        dataset: List[str],
        n: int = 4,
        verbose: bool = True
) -> float:
    """
    Calculates a self-repetition score for a dataset based on the 
    repetition of ngrams within the corpus.

    Args:
        dataset (List[str]): A list of documents (strings) to analyze.
        n (int): Size of the ngrams to check for repetition. Defaults to 4.
        verbose (bool): enable/disable show progress bar

    Returns:
        float: The self-repetition score, averaged over the dataset.
    """
    total_sum = 0
    
    # Get all unique ngrams per doc
    ngram_docs = [list(set([' '.join(ngram) for ngram in ngrams(doc.split(), 4)])) for doc in dataset]
    
    # Count occurrences of unique ngrams across whole dataset
    all_ngrams = sum(ngram_docs, [])
    ngram_counts = Counter(all_ngrams)

    for ngram_doc in tqdm(ngram_docs, desc="Calculating self-repetition score", disable=(not verbose)):
        # Find the total occurrence of an n-gram and subtract current doc's n-gram count
        # to get the count of occurrences of an n-gram in other docs
        sum_ni = sum([ngram_counts[ngram] for ngram in ngram_doc]) - len(ngram_doc)
        
        # add-one to avoid zero error
        total_sum += np.log(sum_ni + 1)
    return total_sum / len(dataset)