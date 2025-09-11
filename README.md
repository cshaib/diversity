# diversity

[![PyPI version](https://img.shields.io/pypi/v/diversity.svg)](https://pypi.org/project/diversity/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![ArXiv](https://img.shields.io/badge/arXiv-2403.00553-b31b1b.svg)](https://arxiv.org/abs/2403.00553)

**A Python toolkit for measuring lexical and syntactic diversity in text outputs.**

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

---

## Quick Start

```python
from diversity import (
    compression_ratio,
    homogenization_score,
    ngram_diversity_score,
    extract_patterns,
    match_patterns
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

# POS pattern extraction
patterns = extract_patterns(texts, n=4, top_n=5)
print("Top POS patterns:", patterns)

# Match patterns in a single text
matches = match_patterns(texts[2], patterns)
print("Patterns in 3rd sentence:", matches)
```

---

## Citation

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

---

## License

This package is released under the **Apache License 2.0**.

---

## Contributing

Contributions are welcome!  
Please open an issue or submit a pull request on GitHub.

---
