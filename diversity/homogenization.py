from typing import List, Optional
from tqdm import tqdm
from rouge_score import rouge_scorer
from evaluate import load

def homogenization_score(
        data: List[str],
        measure: str = 'rougel',
        use_stemmer: Optional[str] = False,
        model: Optional[str] = "microsoft/deberta-base-mnli",
        verbose: Optional[bool] = True,
        batch_size: Optional[int] = 64
) -> float:
    """ 
    Calculates the homogenization score for a set of documents (corpus-level). 
        From https://arxiv.org/pdf/2309.05196.pdf 
     Args:
         data (List[str]): Strings to score.
         measure (str, optional): Either 'rougel', 'bertscore', or 'bleu'. Defaults to 'rougel'.
         use_stemmer(str, optional): Whether to use stemming in the ROUGE-L calculation. Defaults to False.
         model(str, optional): Model to use for BERTScore. Defaults to 'microsoft/deberta-base-mnli'. 
         verbose(bool, optional): Whether to display progress bar. Defaults to True.
     Returns:
         float: Homogenization score.
     """

    if measure == 'rougel':
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=use_stemmer)
    elif measure == 'bertscore': 
        scorer = load("bertscore")
    elif measure == 'bleu':
        scorer = load("bleu")
    else: 
        raise ValueError("Scoring measure must be one of `rougel`, `bleu`, or `bertscore`.")

    corpus_score = 0
    
    if verbose:
        print('==> Scoring all pairs')
     
    for i, ref  in tqdm(enumerate(data), total=len(data), disable=(not verbose)):
        # Get all the other utterances to compare against a specific utterance
        preds = [x for j,x in enumerate(data) if j!=i]
        refs = [ref for _ in range(len(preds))]
        
        # Get scores over whole batch and sum it up
        if measure=='rougel':
            doc_score = sum([scorer.score(pred, ref)['rougeL'].fmeasure for pred in preds])
        elif measure=='bertscore':
            doc_score = sum(scorer.compute(predictions=preds, 
                                           references=refs, 
                                           model_type=model, 
                                           batch_size=batch_size)['f1'])
        elif measure=='bleu':
            # Need to double check that this is right
            doc_score = scorer.compute(predictions=preds, 
                                       references=[[r] for r in refs])['bleu']
        # Then average
        corpus_score += doc_score / (len(data) - 1)
    
    # case where all strings are the exact same in the list
    if corpus_score == 0: 
        corpus_score += len(data)
    
    # returns corpus level homogenization score 
    return round(corpus_score/len(data), 3)
