import numpy as np
import itertools
from diversity import memoized
from typing import List, Optional
from tqdm import tqdm
from rouge_score import rouge_scorer
from evaluate import load


def homogenization_score(
        data: List[str],
        measure: str = 'rougel',
        use_stemmer: Optional[str] = False,
        model: Optional[str] = 'distilbert-base-uncased'

) -> float:
    """ Calculates the homogenization score for a set of documents (corpus-level). 
        From https://arxiv.org/pdf/2309.05196.pdf 
     Args:
         data (List[str]): Strings to score.
         measure (str, optional): Either 'rougel' or 'bertscore'. Defaults to 'rougel'.
         use_stemmer(str, optional): Whether to use stemming in the ROUGE-L calculation. Defaults to False.
         model(str, optional): Model to use for BERTScore. Defaults to 'microsoft/deberta-xlarge-mnli'. 
     Returns:
         float: Homogenization score.
     """

    if measure == 'rougel':
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=use_stemmer)
    elif measure == 'bertscore': 
        scorer = load("bertscore")
    else: 
        raise ValueError("Scoring measure must be one of `rougel` or `bertscore`.")

    all_pairs = list(itertools.product(data, repeat=2))
    curr_str = all_pairs[0][0]
    corpus_score = 0
    doc_score = 0
    
    print('==> Scoring all pairs')
    for pair in tqdm(all_pairs):
        # single document-level homogenization score, pairs are ordered
        if pair[0] == curr_str:     
            doc_score += _calculate_score(pair, scorer, measure, model)
        else:
            corpus_score += doc_score / (len(data) - 1)
            curr_str = pair[0]
            doc_score = 0 
            doc_score += _calculate_score(pair, scorer, measure, model)
        
    # returns corpus level homogenization score 
    return round(corpus_score / len(data), 3)


@memoized
def _calculate_score(pair, scorer, measure, model):
    """ Returns the score of two strings """

    if pair[0] == pair[1]: 
        return 0
    else:
        if measure == 'rougel':
            score = scorer.score(pair[0], 
                                 pair[1])['rougeL'][-1]
        elif measure == 'bertscore': 
            score = scorer.compute(predictions=[pair[0]], 
                                   references=[pair[1]], 
                                   model_type=model)['f1'][0]
        
        return score 
