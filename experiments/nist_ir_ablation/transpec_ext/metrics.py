from typing import Dict, List, Optional, Tuple

from rdkit import Chem


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
    top_k_values: List[int] = None,
) -> Dict[str, bool]:
    """Compute Top-k hits for a single test example.

    Args:
        label_smiles: Ground truth SMILES string.
        candidates: Ranked candidate SMILES strings (raw from decoder).
        top_k_values: List of k values, e.g. [1, 3, 5, 10].

    Returns:
        Dict mapping "hit_top{k}" to bool. Invalid candidates remain in
        ranking and count as misses. Duplicate canonical SMILES: first
        occurrence counts, no reordering.
    """
    if top_k_values is None:
        top_k_values = [1, 3, 5, 10]

    label_canon = canonicalize_smiles(label_smiles)
    if label_canon is None:
        return {f"hit_top{k}": False for k in top_k_values}

    # Canonicalize each candidate, caching results
    seen_canon = set()
    # Build a list of (index, canonical) keeping only first occurrence
    unique_candidates = []
    for raw_smi in candidates:
        if raw_smi == "":
            continue
        can = canonicalize_smiles(raw_smi)
        if can is None or can in seen_canon:
            continue
        seen_canon.add(can)
        unique_candidates.append(can)

    result = {}
    for k in top_k_values:
        top_k_candidates = unique_candidates[:k]
        result[f"hit_top{k}"] = label_canon in top_k_candidates

    return result


def compute_all_metrics(
    label_smiles_list: List[str],
    candidates_list: List[List[str]],
    top_k_values: List[int] = None,
) -> Dict[str, float]:
    """Compute aggregate Top-k accuracy across all test examples.

    Args:
        label_smiles_list: List of ground truth SMILES.
        candidates_list: List of ranked candidate lists (one per example).
        top_k_values: List of k values.

    Returns:
        Dict mapping "top{k}" to accuracy (float 0-1).
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
