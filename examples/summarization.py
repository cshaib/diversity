from typing import Optional
from diversity import get_pos, pos_patterns
from transformers import pipeline
from datasets import load_dataset

import typer


app = typer.Typer()

@app.command()
def summarization(
        dataset: str,
        model: str = "t5-base",
        tokenizer: Optional[str] = None,
):
    tokenizer = tokenizer or model
    summarizer = pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer,
        return_text=True)

    # todo(joe): replace this with just loading a csv, take key as argument
    data = load_dataset(dataset)
    data = data['train'][:10]['text']

    outputs = summarizer(data)
    outputs = [instance['summary_text'] for instance in outputs]

    joined, tuples = get_pos(outputs)

    print(joined)


if __name__ == "__main__":
    app()