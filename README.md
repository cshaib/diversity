# diversity
[![PyPI version](https://img.shields.io/pypi/v/diversity.svg)](https://pypi.org/project/diversity/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![ArXiv](https://img.shields.io/badge/arXiv-2403.00553-b31b1b.svg)](https://arxiv.org/abs/2403.00553)

### **A Python toolkit for measuring diversity in text.**

---

## Table of Contents
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Lexical Diversity Measures](#lexical-diversity-measures)
    - [`compression_ratio`](#compression_ratiotexts-methodgzip)
    - [`homogenization_score`](#homogenization_scoretexts-methodself-bleu)
    - [`ngram_diversity_score`](#ngram_diversity_scoretexts-n3)
  - [Syntactic Diversity Measures](#syntactic-diversity-measures)
    - [`extract_patterns`](#extract_patternstexts-n4-top_n5)
    - [`match_patterns`](#match_patternstext-patterns)
  - [Embedding-Based Diversity Measures](#embedding-based-diversity-measures)
    - [`remote_clique`](#remote_cliquedata-modelqwenqwen3-embedding-06b-verbo-true-batch_size64)
    - [`chamfer_dist`](#chamfer_distdata-modelqwenqwen3-embedding-06b-verbo-true-batch_size64)
  - [QUDSim (Question Under Discussion Similarity)](#qudsim-question-under-discussion-similarity)
    - [`qudsim`](#qudsimdocuments-key)
- [Citations](#citations)
- [Requirements](#requirements)
- [License](#license)
- [Contributing](#contributing)
- [Troubleshooting](#troubleshooting)
- [Support](#support)

---

## Installation

Install via pip:

```bash
pip install diversity
```

Or from source:

```bash
git clone https://github.com/cshaib/diversity.git
cd diversity
pip install .
```

----------

## Quick Start

### Lexical Diversity Measures

We provide implementations for Compression Ratio, Homogenization Score, and n-gram Diversity Score: 

```python
from diversity import (
    compression_ratio,
    homogenization_score,
    ngram_diversity_score,
)

texts = [
    "The quick brown fox jumps over the lazy dog.",
    "The quick brown fox jumps over the lazy dog again.",
    "Suddenly, the quick brown fox leaps swiftly over the sleeping dog."
]

# Compression ratio
cr = compression_ratio(texts, method='gzip')
print(f"Compression Ratio: {cr:.4f}")

# Homogenization score (Self-BLEU)
hs = homogenization_score(texts, method='self-bleu')
print(f"Homogenization (Self-BLEU): {hs:.4f}")

# N-gram diversity
ngd = ngram_diversity_score(texts, n=3)
print(f"3-gram Diversity: {ngd:.4f}")
```
#### `compression_ratio(texts, method='gzip')`

-   **Parameters:**
    -   `texts`  (list): List of text strings
    -   `method`  (str): Compression algorithm ('gzip', 'bz2', 'lzma')
-   **Returns:**  Float (0-1), higher = more repetitive

#### `homogenization_score(texts, method='self-bleu')`

-   **Parameters:**
    -   `texts`  (list): List of text strings
    -   `method`  (str): Scoring method ('self-bleu', 'rouge-l')
-   **Returns:**  Float (0-1), higher = more homogeneous

#### `ngram_diversity_score(texts, n=3)`

-   **Parameters:**
    -   `texts`  (list): List of text strings
    -   `n`  (int): N-gram size
-   **Returns:**  Float (0-1), higher = more diverse
----------

### Syntactic Diversity Measures

We also provide functions for extracting and analyze Part-of-Speech (POS) patterns to identify repetitive syntactic structures in your text: 

```python
from diversity import (
    extract_patterns,
    match_patterns
)

texts = [
    "The quick brown fox jumps over the lazy dog.",
    "The quick brown fox jumps over the lazy dog again.",
    "Suddenly, the quick brown fox leaps swiftly over the sleeping dog."
]

# POS pattern extraction
patterns = extract_patterns(texts, n=4, top_n=5)
print("Top POS patterns:", patterns)
# Example output: [(('DT', 'JJ', 'JJ', 'NN'), 15), ...]

# Match patterns in a single text
matches = match_patterns(texts[2], patterns)
print("Patterns in 3rd sentence:", matches)
# Example output: [{'pattern': ('DT', 'JJ', 'JJ', 'NN'), 'text': 'the quick brown fox', 'position': (0, 4)}]
```

#### `remote_clique(data, model='Qwen/Qwen3-Embedding-0.6B', verbose=True, batch_size=64)`

-   **Parameters:**
    
    -   `data`  (list): List of text strings to score
        
    -   `model`  (str): Embedding model to use (default:  `"Qwen/Qwen3-Embedding-0.6B"`)
        
    -   `verbose`  (bool): Whether to display progress bar (default:  `True`)
        
    -   `batch_size`  (int): Batch size for embedding (default:  `64`)
        
-   **Returns:**  `float`  — Remote Clique score (average mean pairwise distance between documents)
    

----------

#### `chamfer_dist(data, model='Qwen/Qwen3-Embedding-0.6B', verbose=True, batch_size=64)`

-   **Parameters:**
    
    -   `data`  (list): List of text strings to score
        
    -   `model`  (str): Embedding model to use (default:  `"Qwen/Qwen3-Embedding-0.6B"`)
        
    -   `verbose`  (bool): Whether to display progress bar (default:  `True`)
        
    -   `batch_size`  (int): Batch size for embedding (default:  `64`)
        
-   **Returns:**  `float`  — Chamfer distance (average minimum pairwise distance, lower when many near-duplicates are present)
----------

### Embedding-Based Diversity Measures

You can also measure semantic diversity using *embedding*-based similarity. These scores compute distances between document embeddings to quantify how spread out or clustered the texts are:

```python
from diversity.embedding import remote_clique, chamfer_dist

texts = [
    "The quick brown fox jumps over the lazy dog.",
    "A swift auburn fox vaulted a sleeping canine.",
    "I brewed coffee and read the paper."
]

# Remote Clique Score
rc = remote_clique(texts, model="Qwen/Qwen3-Embedding-0.6B")
print(f"Remote Clique: {rc:.3f}")

# Chamfer Distance
cd = chamfer_dist(texts, model="Qwen/Qwen3-Embedding-0.6B")
print(f"Chamfer Distance: {cd:.3f}")
```
#### `remote_clique(data, model='Qwen/Qwen3-Embedding-0.6B', verbose=True, batch_size=64)`

-   **data (list of str):**  Documents to score.
    
-   **model (str):**  HuggingFace/Sentence-Transformers embedding model to use (default:  `"Qwen/Qwen3-Embedding-0.6B"`).
    
-   **verbose (bool):**  Whether to show a progress bar during encoding (default:  `True`).
    
-   **batch_size (int):**  Batch size for embedding (default:  `64`).
    
-   **Returns:**  `float`  — average mean pairwise cosine distance between documents (higher = more spread out / diverse).
    

#### `chamfer_dist(data, model='Qwen/Qwen3-Embedding-0.6B', verbose=True, batch_size=64)`

-   **data (list of str):**  Documents to score.
    
-   **model (str):**  HuggingFace/Sentence-Transformers embedding model to use (default:  `"Qwen/Qwen3-Embedding-0.6B"`).
    
-   **verbose (bool):**  Whether to show a progress bar during encoding (default:  `True`).
    
-   **batch_size (int):**  Batch size for embedding (default:  `64`).
    
-   **Returns:**  `float`  — average minimum pairwise cosine distance (sensitive to near-duplicates; higher = less redundancy).

----------

### QUDSim (Question Under Discussion Similarity)

QUDSim aligns document segments based on Questions Under Discussion (QUDs) --- implicit questions that segments of text address ([QUDsim: Quantifying Discourse Similarities in LLM-Generated Text](https://arxiv.org/abs/2504.09373)). 

This function requires OpenAI API access.

```python
from diversity import qudsim

# Two documents about the same topic
document1 = "In the heart of ancient Macedonia, Philip II ascended to the throne in 359 BC..."
document2 = "The sun beat down on the rough-hewn hills of ancient Macedonia..."

# Requires OpenAI API key
import os
key = os.environ.get('OPENAI_API_KEY')  # or your API key

# Generate QUD-based alignment
alignment = qudsim([document1, document2], key=key)

# Access alignment results
results = eval(alignment)[0]  # First document pair

# View aligned segments
for source_text, target_text in results['aligned_segment_text']:
    print(f"Source: {source_text[:100]}...")
    print(f"Target: {target_text[:100]}...")
    print("---")

# View alignment scores (harmonic mean scores matrix)
scores = results['harmonic_mean_scores']
print(f"Alignment scores shape: {len(scores)}x{len(scores[0])}")

# Other available fields:
# - results['source_qud_answers']: QUDs generated for source document
# - results['target_qud_answers']: QUDs generated for target document
# - results['aligned_segments']: Indices of aligned segments
```

#### `qudsim(documents, key)`

-   **Parameters:**
    -   `documents` (list): List of texts to align
    -   `key` (str): OpenAI API key for QUD generation
    - `model` (str): LLM model to use (default: `gpt-4`)
    -  `threshold` (float): Minimum alignment score threshold (default: 0.5)
-   **Returns:**  list of alignment scores
----------
 
## Citation(s)

If you use this package, please cite:

```bibtex
@misc{shaib2025standardizingmeasurementtextdiversity,
  title={Standardizing the Measurement of Text Diversity: A Tool and a Comparative Analysis of Scores},
  author={Chantal Shaib and Joe Barrow and Jiuding Sun and Alexa F. Siu and Byron C. Wallace and Ani Nenkova},
  year={2025},
  eprint={2403.00553},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2403.00553},
}
```

If you use QUDSim, please  **also**  cite:

```bibtex
@inproceedings{
namuduri2025qudsim,
title={{QUD}sim: Quantifying Discourse Similarities in {LLM}-Generated Text},
author={Ramya Namuduri and Yating Wu and Anshun Asher Zheng and Manya Wadhwa and Greg Durrett and Junyi Jessy Li},
booktitle={Second Conference on Language Modeling},
year={2025},
url={https://openreview.net/forum?id=zFz1BJu211}
}
```

----------

## Requirements

-   Python 3.10-3.12
-   Core dependencies:
    -   `numpy`
    -   `nltk`
    -   `scikit-learn`
-   For embedding-based metrics:
    -   `sentence-transformers`
    -   `torch`
-   For QUDSim:
    -   `openai`
    -   `tqdm`

----------

## License

This package is released under the  **Apache License 2.0**.

----------

## Contributing

Contributions are welcome!  
Please open an issue or submit a pull request on GitHub.

----------
