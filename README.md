
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
from diversity import compression_ratio, homogenization_score, ngram_diversity_score

data_example  = [
"I enjoy walking with my cute dog for the rest of the day, but this time it was hard for me to figure out what to do with it. When I finally looked at this for a few moments, I immediately thought.",
"I enjoy walking with my cute dog. The only time I felt like walking was when I was working, so it was awesome for me. I didn't want to walk for days. I am really curious how she can walk with me", 
"I enjoy walking with my cute dog (Chama-I-I-I-I-I), and I really enjoy running. I play in a little game I play with my brother in which I take pictures of our houses."
]

# get the token-level patterns
patterns_token  =  token_patterns(outputs, ngram)

# get the POS patterns
joined_pos, tuples  =  get_pos(outputs)
ngrams_pos  =  token_patterns(joined_pos, ngram)

# for the top n-gram patterns, cycle through and get the matching text
text_matches  = {}
for  pattern, _  in  ngrams_pos:
	text_matches['pattern'] =  pos_patterns(tuples, pattern)
```
 
