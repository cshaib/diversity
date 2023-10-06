from typing import Optional
import diversity
from diversity import get_pos, pos_patterns, token_patterns, compression_ratio
from transformers import pipeline
from datasets import load_dataset

import typer


app = typer.Typer()

@app.command()
def summarization(
        dataset: str,
        column: str,
        split: str,
        model: str = "t5-base",
        tokenizer: Optional[str] = None,
        ngram: Optional[int] = 5
):
    tokenizer = tokenizer or model

    summarizer = pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer,
        return_text=True)
    
    # load dataset (either custom CSV or dataset from HF)
    if dataset.endswith('.csv'): 
        data = load_dataset("csv", data_files=dataset)[split][:10][column]
    else: 
        data = load_dataset(dataset)
        data = data[split][:10][column]

    # generate the summaries
    outputs = summarizer(data)
    outputs = [instance['summary_text'] for instance in outputs]

    # get the token-level patterns
    patterns_token = token_patterns(outputs, ngram)
    
    # get the POS patterns 
    joined_pos, tuples = get_pos(outputs)
    ngrams_pos = token_patterns(joined_pos, ngram)

    # for the top n-gram patterns, cycle through and get the matching text
    text_matches = {}

    for pattern, _ in ngrams_pos:
        text_matches['pattern'] = pos_patterns(tuples, pattern)

    # get the compression score
    compression = compression_ratio(outputs, 'gzip')

    # TODO: function to nicely display results
    print(patterns_token)
    print(text_matches)
    print(compression)

    # TODO: compare between two models

    return 


def display_results():
    pass


if __name__ == "__main__":
    app()
