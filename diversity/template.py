import re
from typing import List, Optional
from .functions import extract_patterns
from typing import Dict, Iterable, List, Optional


def template_rate(
    data = List[str], 
    templates: Optional[Dict[str, Iterable[str]]] = None,
    shard_size: int = 500,
        
) -> float:
    """ 
    Calculates the template rate (fraction of texts in a corpus that contain at least 1 template)
    for a set of documents (corpus-level), following https://arxiv.org/abs/2407.00211.  

    Args:
        data (List[str]): A list of strings to score.
        templates (dict, optional): Dictionary containing the templates extracted from the corpus. Defaults to None.
        shard_size (int, optional): Size of regex shards to compile. Defaults to 500.

    Returns:
        float: Template rate, a value between 0 and 1 indicating the fraction of texts that contain at least one template.
    """
    if not data: return 0.0
    
    if templates is None:
        # get the templates if not passed in 
        templates = extract_patterns(data)
    
    matched_text = _gather_substrings(templates)
    
    if not matched_text: return 0.0
        
    regexes = _compile_regex_shards(matched_text, shard_size=shard_size)
    match = sum(1 for doc in data if _has_any(doc, regexes))
    
    return match / len(data)


def templates_per_token(
        data: List[str],
        templates: Optional[Dict[str, Iterable[str]]] = None,
        shard_size: int = 500,
) -> List[float]:
    """ 
    Calculates the templates-per-token rate from https://arxiv.org/abs/2407.00211. 
    
    Args:
        data (List[str]):  A list of strings to score.
        templates (dict, optional): Dictionary containing the templates extracted from the corpus. Defaults to None.
        shard_size (int, optional): Size of regex shards to compile. Defaults to 500.


    Returns:
        List[float]: List of templates-per-token rates for each document in the corpus.
    """
    if not data:
        return []

    # Build templates if not provided
    if templates is None:
        templates = extract_patterns(data)

    substrings = _gather_substrings(templates)
    if not substrings:
        return [0.0] * len(data)

    # Use lookahead shards to count overlapping occurrences
    shards = _compile_regex_shards(substrings, shard_size, overlap=True)

    # Compute per-doc TPT
    tpt: List[float] = []
    for doc in data:
        word_count = len(doc.split())
        if word_count == 0:
            tpt.append(0.0)
            continue

        occ = 0
        for rx in shards:
            occ += sum(1 for _ in rx.finditer(doc))  # each match = one occurrence start
        tpt.append(occ / word_count)

    return tpt


def _compile_regex_shards(
    substrings: List[str],
    shard_size: int = 500,
    *,
    overlap: bool = False,
) -> List[re.Pattern]:
    """
    Reusable shard compiler.
    - overlap=False: plain alternation (fast existence tests).
    - overlap=True: lookahead alternation for counting overlapping matches.
    """
    regs: List[re.Pattern] = []
    
    for i in range(0, len(substrings), shard_size):
        chunk = substrings[i:i + shard_size]
        
        if not chunk:
            continue
        
        alt = "|".join(map(re.escape, chunk))
        pat = f"(?=(?:{alt}))" if overlap else f"(?:{alt})"
        
        regs.append(re.compile(pat))
        
    return regs


def _has_any(
        text: str, 
        regexes: List[re.Pattern]
    ) -> bool:
    for rx in regexes:
        # faster search
        if rx.search(text):
            return True
    return False


def _gather_substrings(
    templates: Dict[str, Iterable[str]]
) -> List[str]:
    """
    Gathers all substrings from the templates dictionary.
    
    Args:
        templates (Dict[str, Iterable[str]]): Dictionary of templates with their corresponding text matches.
        
    Returns:
        List[str]: List of unique substrings extracted from the templates.
    """
    # get the flattened values from the extracted patterns 
    matched_text = set()
    
    for v in templates.values():
        matched_text.update(v)
    
    return list(matched_text)
