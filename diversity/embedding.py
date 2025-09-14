'''
Computes remote clique and chamfer distance on embeddings for a set of documents
to understand their semantic (in embedding space) diversity.

Following Samuel Rhys Cox et al. 2021. "Directed Diversity: Leveraging Language Embedding Distances for Collective Creativity in Crowd Ideation". In Proceedings of the 2021 CHI Conference on Human Factors in Computing Systems (CHI '21). Association for Computing Machinery, New York, NY, USA, Article 393, 1–35. https://doi.org/10.1145/3411764.3445782
'''

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances
import numpy as np
from tqdm import tqdm
from typing import List, Optional


def remote_clique(
        data: List[str],
        model: Optional[str] = 'Qwen/Qwen3-Embedding-0.6B',
        verbose: Optional[bool] = True,
        batch_size: Optional[int] = 64
) -> float:
    """
    Calculates the remote clique score for a set of documents (corpus-level).
    This is the average mean pairwise distance of a data instance to other instances.
    Args:
        data (List[str]): Strings to score.
        model(str, optional): Model to use for embedding. Defaults to 'Qwen/Qwen3-Embedding-0.6B'.
        verbose(bool, optional): Whether to display progress bar. Defaults to True.
        batch_size(int, optional): Batch size for embedding. Defaults to 64.
    Returns:
        float: Remote clique score.
    """
    model = SentenceTransformer(model)
    embeddings = model.encode(data, batch_size=batch_size, show_progress_bar=verbose)
    distances = cosine_distances(embeddings)
    mean_distances = np.mean(distances, axis=1)
    return np.mean(mean_distances).round(3)


def chamfer_dist(
        data: List[str],
        model: Optional[str] = 'Qwen/Qwen3-Embedding-0.6B',
        verbose: Optional[bool] = True,
        batch_size: Optional[int] = 64
) -> float:
    """
    Calculates the chamfer distance for a set of documents (corpus-level).
    This is the average minimum pairwise distance of a data instance to other instances.
    Args:
        data (List[str]): Strings to score.
        model(str, optional): Model to use for embedding. Defaults to 'Qwen/Qwen3-Embedding-0.6B'.
        verbose(bool, optional): Whether to display progress bar. Defaults to True.
        batch_size(int, optional): Batch size for embedding. Defaults to 64.
    Returns:
        float: Chamfer distance.
    """
    model = SentenceTransformer(model)
    embeddings = model.encode(data, batch_size=batch_size, show_progress_bar=verbose)
    distances = cosine_distances(embeddings)
    min_distances = np.min(distances + np.eye(len(distances)) * 1e9, axis=1)
    return np.mean(min_distances).round(3)
