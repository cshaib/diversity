'''
Times and tests BERTScore, ROUGE and BLEU score computation over CNN-DailyMail
summaries.
'''

from datasets import load_dataset
from homogenization import homogenization_score
from time import perf_counter as pc
from numpy import round

if __name__ == '__main__':
    data = load_dataset("argilla/cnn-dailymail-summaries")["train"].to_pandas().highlights.sample(500, random_state=1).values.tolist()
    
    start = pc()
    bs = homogenization_score(data, measure='bertscore', verbose=True, model="distilbert-base-uncased")
    end = pc()
    
    print(f"Time taken to compute BERTScore over CNN summaries: {round(end-start,2)}\nBERTScore over 500 reference summaries: {bs}")
    
    start = pc()
    rl = homogenization_score(data, measure='rougel', verbose=True)
    end = pc()
    
    print(f"Time taken to compute Rouge over CNN summaries: {round(end-start,2)} secs\nRouge over 500 reference summaries: {rl}")
    
    start = pc()
    rl = homogenization_score(data, measure='bleu', verbose=True)
    end = pc()
    
    print(f"Time taken to compute BLEU over CNN summaries: {round(end-start,2)} secs\nBLEU over 500 reference summaries: {rl}")
