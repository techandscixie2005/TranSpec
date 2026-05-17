import logging
from typing import Dict, List, Optional

from rdkit import Chem

logger = logging.getLogger(__name__)


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """Canonicalize a SMILES string. Returns None on failure."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def compute_top_k_accuracy(
    label_smiles: str,
    candidates: List[str],
    top_k_values: Optional[List[int]] = None,
) -> Dict[str, bool]:
    """Compute Top-k hits for a single test example.

    Preserves original candidate ranking — invalid candidates remain in
    place and count as misses. Duplicates are not removed from ranking.

    Args:
        label_smiles: Ground truth SMILES string.
        candidates: Ranked candidate SMILES strings (raw from decoder).
        top_k_values: List of k values, e.g. [1, 3, 5, 10].

    Returns:
        Dict mapping ``hit_top{k}`` to bool.
    """
    if top_k_values is None:
        top_k_values = [1, 3, 5, 10]

    label_canon = canonicalize_smiles(label_smiles)
    if label_canon is None:
        return {f"hit_top{k}": False for k in top_k_values}

    # Canonicalize each candidate, preserving original order and position.
    # Invalid (None) entries remain in the list so ranking is unchanged.
    canonicalized: List[Optional[str]] = []
    for raw_smi in candidates:
        can = canonicalize_smiles(raw_smi)
        canonicalized.append(can)

    result = {}
    for k in top_k_values:
        top_k = canonicalized[:k]
        result[f"hit_top{k}"] = label_canon in top_k

    return result


def compute_all_metrics(
    label_smiles_list: List[str],
    candidates_list: List[List[str]],
    top_k_values: Optional[List[int]] = None,
) -> Dict[str, float]:
    """Compute aggregate Top-k accuracy across all test examples.

    Args:
        label_smiles_list: List of ground truth SMILES.
        candidates_list: List of ranked candidate lists (one per example).
        top_k_values: List of k values.

    Returns:
        Dict mapping ``top{k}`` to accuracy (float 0-1).
    """
    if top_k_values is None:
        top_k_values = [1, 3, 5, 10]

    n = len(label_smiles_list)
    hits = {k: 0 for k in top_k_values}

    for label, candidates in zip(label_smiles_list, candidates_list):
        result = compute_top_k_accuracy(label, candidates, top_k_values)
        for k in top_k_values:
            if result[f"hit_top{k}"]:
                hits[k] += 1

    return {f"top{k}": hits[k] / n if n > 0 else 0.0 for k in top_k_values}
