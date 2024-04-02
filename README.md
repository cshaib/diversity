
# 🎨 `diversity`

`diversity` is a package that checks for and scores repeated structures and patterns in the output of language models. 

## Installation

```sh
pip install -Ue diversity
```

## Command-line

```sh
python examples/summarization.py <DATASET CSV>
```

## Library

This library supports various scoring methods for evaluating the homogeneity and diversity of outputs. 
```python
from diversity import compression_ratio, homogenization_score, ngram_diversity_score

data_example  = [
"I enjoy walking with my cute dog for the rest of the day, but this time it was hard for me to figure out what to do with it. When I finally looked at this for a few moments, I immediately thought.",
"I enjoy walking with my cute dog. The only time I felt like walking was when I was working, so it was awesome for me. I didn't want to walk for days. I am really curious how she can walk with me", 
"I enjoy walking with my cute dog (Chama-I-I-I-I-I), and I really enjoy running. I play in a little game I play with my brother in which I take pictures of our houses."
]

cr = compression_ratio(data_example, 'gzip')
hs = homogenization_score(data_example, 'rougel')
# hs = homogenization_score(data_example, 'bertscore') 
nds = ngram_diversity_score(data_example, 4)

print(cr, hs, nds)
```

```sh
1.641 0.222 3.315
```


You can also visualize various ngram patterns using this library:
```python

# get the part-of-speech patterns and matching text for a dataset

text = [
"I enjoy walking with my cute dog for the rest of the day, but this time it was hard for me to figure out what to do with it. When I finally looked at this for a few moments, I immediately thought.",
"I enjoy walking with my cute dog. The only time I felt like walking was when I was working, so it was awesome for me. I didn't want to walk for days. I am really curious how she can walk with me", 
"I enjoy walking with my cute dog (Chama-I-I-I-I-I), and I really enjoy running. I play in a little game I play with my brother in which I take pictures of our houses."
]
n = 5 
top_n = 100

patterns = extract_patterns(text, n, top_n)
patterns
```

```sh
{'PRP VBP VBG IN PRP$': {'I enjoy walking with my'},
 'VBP VBG IN PRP$ NN': {'enjoy walking with my cute'},
 'VBG IN PRP$ NN NN': {'walking with my cute dog', 'walking with my cute dog.'},
 'IN DT JJ NN PRP': {'for a few moments, I', 'in a little game I'},
 'IN PRP$ NN NN IN': {'with my cute dog for'},
 'PRP$ NN NN IN DT': {'my cute dog for the'},
 'NN NN IN DT NN': {'cute dog for the rest'},
 'NN IN DT NN IN': {'dog for the rest of'},
 'IN DT NN IN DT': {'for the rest of the'},
 'DT NN IN DT NN': {'the rest of the day,'},
 'NN IN DT NN CC': {'rest of the day, but'},
 'IN DT NN CC DT': {'of the day, but this'},
 'DT NN CC DT NN': {'the day, but this time'},
 'NN CC DT NN PRP': {'day, but this time it'},
 'CC DT NN PRP VBD': {'but this time it was'},
 'DT NN PRP VBD JJ': {'this time it was hard'},
 'NN PRP VBD JJ IN': {'time it was hard for'},
 'PRP VBD JJ IN PRP': {'it was hard for me'},
 ...
 }
```
The pattern matches above are based on the frequency seen across the *entire* datasets, i.e., a part-of-speech pattern is only a pattern if it appears in more than 1 text in the original input. We consider the top 100 part-of-speech patterns (sorted by frequency).

Once you have the patterns dictionary, if you want to identify all the patterns contained in a single text from the input, you can do the following:

```python
idx = 2
match_patterns(text[idx], patterns)
```
```sh
[('PRP VBP VBG IN PRP$', 'I enjoy walking with my'),
 ('VBP VBG IN PRP$ NN', 'enjoy walking with my cute'),
 ('VBG IN PRP$ NN NN', 'walking with my cute dog'),
 ('IN DT JJ NN PRP', 'in a little game I'),
 ('VBP VBG IN PRP$ JJ', 'enjoy walking with my cute'),
 ('VBG IN PRP$ JJ JJ', 'walking with my cute dog'),
 ('IN PRP$ JJ JJ JJ', 'with my cute dog (Chama-I-I-I-I-I),'),
 ('PRP$ JJ JJ JJ CC', 'my cute dog (Chama-I-I-I-I-I), and'),
 ('JJ JJ JJ CC PRP', 'cute dog (Chama-I-I-I-I-I), and I'),
 ...
]
```
