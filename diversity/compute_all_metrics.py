"""
Comprehensive diversity metrics computation function.
This module computes all lexical diversity metrics plus embedding metrics for a given corpus of text.
"""

from typing import List, Optional, Dict, Any
import logging
from .compression import compression_ratio
from .homogenization import homogenization_score
from .ngram_diversity import ngram_diversity_score
from .self_repetition import self_repetition_score
from .embedding import remote_clique, chamfer_dist
from .template import template_rate, templates_per_token
from .functions import extract_patterns


def compute_all_metrics(
    corpus: List[str],
    output_format: str = "dict",
    embedding_model: Optional[str] = "Qwen/Qwen3-Embedding-0.6B",
    homogenization_measure: str = "rougel",
    compression_algorithm: str = "gzip",
    ngram_n: int = 4,
    self_repetition_n: int = 4,
    template_shard_size: int = 500,
    verbose: bool = True,
    batch_size: int = 64
) -> Dict[str, Any]:
    """
    Computes all available diversity metrics for a corpus of text.
    
    Args:
        corpus (List[str]): List of text documents to analyze
        output_format (str): Format for output - "dict", "markdown", or "latex"
        embedding_model (str): Model to use for embedding-based metrics
        homogenization_measure (str): Measure for homogenization score ("rougel", "bertscore", "bleu")
        compression_algorithm (str): Algorithm for compression ratio ("gzip", "xz")
        ngram_n (int): Maximum n-gram size for n-gram diversity
        self_repetition_n (int): N-gram size for self-repetition score
        template_shard_size (int): Shard size for template processing
        verbose (bool): Whether to show progress messages
        batch_size (int): Batch size for embedding computations
    
    Returns:
        Dict[str, Any]: Dictionary containing all computed metrics and formatted table if requested
    """
    
    if verbose:
        print("Computing diversity metrics for corpus...")
        print(f"Corpus size: {len(corpus)} documents")
    
    results = {}
    
    # Compression-based metrics
    if verbose:
        print("Computing compression ratio...")
    results["compression_ratio_gzip"] = compression_ratio(
        corpus, algorithm="gzip", verbose=False
    )
    
    if compression_algorithm == "xz":
        results["compression_ratio_xz"] = compression_ratio(
            corpus, algorithm="xz", verbose=False
        )
    
    # Homogenization score
    if verbose:
        print(f"Computing homogenization score using {homogenization_measure}...")
    results[f"homogenization_score_{homogenization_measure}"] = homogenization_score(
        corpus, measure=homogenization_measure, verbose=verbose, batch_size=batch_size
    )
    
    # N-gram diversity
    if verbose:
        print(f"Computing n-gram diversity (n={ngram_n})...")
    results["ngram_diversity"] = ngram_diversity_score(corpus, num_n=ngram_n)
    
    # Self-repetition score
    if verbose:
        print(f"Computing self-repetition score (n={self_repetition_n})...")
    results["self_repetition_score"] = self_repetition_score(
        corpus, n=self_repetition_n, verbose=verbose
    )
    
    # Embedding-based metrics
    if verbose:
        print(f"Computing embedding-based metrics using {embedding_model}...")
    
    try:
        results["remote_clique_score"] = remote_clique(
            corpus, model=embedding_model, verbose=verbose, batch_size=batch_size
        )
        results["chamfer_distance"] = chamfer_dist(
            corpus, model=embedding_model, verbose=verbose, batch_size=batch_size
        )
    except Exception as e:
        if verbose:
            print(f"⚠️  Warning: Could not compute embedding metrics - {e}")
        results["remote_clique_score"] = None
        results["chamfer_distance"] = None
    
    # Template-based metrics
    if verbose:
        print("Extracting patterns for template metrics...")
    
    try:
        patterns = extract_patterns(corpus)
        
        if verbose:
            print("Computing template rate...")
        results["template_rate"] = template_rate(
            corpus, templates=patterns, shard_size=template_shard_size
        )
        
        if verbose:
            print("Computing templates per token...")
        tpt_scores = templates_per_token(
            corpus, templates=patterns, shard_size=template_shard_size
        )
        results["avg_templates_per_token"] = sum(tpt_scores) / len(tpt_scores) if tpt_scores else 0.0
        results["templates_per_token_scores"] = tpt_scores
        
    except Exception as e:
        if verbose:
            print(f"Warning: Could not compute template metrics - {e}")
        results["template_rate"] = None
        results["avg_templates_per_token"] = None
        results["templates_per_token_scores"] = None
    
    if verbose:
        print("All metrics computed successfully!")
    
    # Format output based on requested format
    if output_format.lower() == "markdown":
        results["formatted_table"] = _format_markdown_table(results)
    elif output_format.lower() == "latex":
        results["formatted_table"] = _format_latex_table(results)
    
    return results


def _format_markdown_table(results: Dict[str, Any]) -> str:
    """Format results as a markdown table."""
    
    table = "# Diversity Metrics Results\n\n"
    table += "| Metric | Value |\n"
    table += "|--------|-------|\n"
    
    for metric, value in results.items():
        if metric in ["formatted_table", "templates_per_token_scores"]:
            continue
        
        if value is None:
            value_str = "N/A"
        elif isinstance(value, float):
            value_str = f"{value:.3f}"
        else:
            value_str = str(value)
        
        table += f"| {metric.replace('_', ' ').title()} | {value_str} |\n"
    
    return table


def _format_latex_table(results: Dict[str, Any]) -> str:
    """Format results as a LaTeX table. Requires booktabs"""
    
    table = "\\begin{table}[htbp]\n"
    table += "\\centering\n"
    table += "\\caption{Diversity Metrics Results}\n"
    table += "\\begin{tabular}{lc}\n"
    table += "\\hline\n"
    table += "\\textbf{Metric} & \\textbf{Value} \\\\\n"
    table += "\\hline\n"
    
    for metric, value in results.items():
        if metric in ["formatted_table", "templates_per_token_scores"]:
            continue
            
        if value is None:
            value_str = "N/A"
        elif isinstance(value, float):
            value_str = f"{value:.3f}"
        else:
            value_str = str(value)
        
        metric_name = metric.replace('_', ' ').title()
        table += f"{metric_name} & {value_str} \\\\\n"
        table += "\\\n"
    
    table += "\\end{tabular}\n"
    table += "\\end{table}\n"
    
    return table
