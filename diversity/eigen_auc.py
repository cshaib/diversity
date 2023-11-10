import numpy as np
from typing import List

def eigen_auc(
        embeddings: List[np.ndarray]
) -> float:
    """Calculates the eigenAUC score for a set of token embeddings (corpus-level). 

    Args:
        embeddings (List[np.ndarray]): List of embeddings representing a single token for each datapoint.

    Returns:
        float: eigenAUC score.
    """
    
    data = np.vstack(embeddings)
    _, S, _ = np.linalg.svd(data, full_matrices=False)

    eigenvalues = S**2 
    variances = _variance_at_all_k(eigenvalues)

    auc = np.trapz(variances)

    return round(auc/len(variances), 3)


def _variance_explained(eigenvalues, k):
    """ Retrieves the variance explained by top-k eigenvalues. """
    total_variance = np.sum(eigenvalues)
    top_k_variance = np.sum(eigenvalues[:k])

    return top_k_variance / total_variance


def _variance_at_all_k(eigenvalues):
    """ Gets variances for all values of k. """
    variances = []

    for k in range(1, len(eigenvalues)):
        var = _variance_explained(eigenvalues, k)
        variances.append(var)

    return variances
