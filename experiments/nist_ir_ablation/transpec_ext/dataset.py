from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .vocab import PAD_ID


class NistIRAblationDataset(Dataset):
    """PyTorch Dataset for NIST IR ablation experiments.

    Loads pre-processed records and tokenized target IDs.
    """

    def __init__(
        self,
        spectra: torch.Tensor,       # (N, 3000)
        target_ids: List[torch.Tensor],  # list of 1-D LongTensor with BOS..EOS
        canonical_smiles: List[str],
        raw_smiles: List[str],
        indices: List[int],
    ):
        self.spectra = spectra
        self.target_ids = target_ids
        self.canonical_smiles = canonical_smiles
        self.raw_smiles = raw_smiles
        self.indices = indices

    def __len__(self) -> int:
        return len(self.spectra)

    def __getitem__(self, i: int) -> dict:
        return {
            "idx": self.indices[i],
            "spectrum": self.spectra[i].float(),         # (3000,)
            "target_ids": self.target_ids[i].long(),     # (seq_len,)
            "canonical_smiles": self.canonical_smiles[i],
            "raw_smiles": self.raw_smiles[i],
        }


def ablation_collate_fn(batch: List[dict]) -> dict:
    """Collate function for NistIRAblationDataset.

    Pads target_ids to uniform length, generates decoder_input,
    labels, tgt_mask, and tgt_padding_mask.
    """
    spectra = torch.stack([b["spectrum"] for b in batch])  # (B, 3000)
    # Add channel dim: (B, 1, 3000)
    spectra = spectra.unsqueeze(1)

    target_ids = [b["target_ids"] for b in batch]

    # Pad sequences
    target_ids_padded = pad_sequence(
        target_ids, batch_first=True, padding_value=PAD_ID
    )  # (B, max_seq_len)

    # Decoder input = target_ids[:, :-1]
    decoder_input = target_ids_padded[:, :-1]

    # Labels = target_ids[:, 1:]
    labels = target_ids_padded[:, 1:]

    # Target mask (batch_first compatible, causal)
    seq_len = decoder_input.size(1)
    tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len)

    # Target padding mask (True = padded position, for nn.Transformer)
    tgt_padding_mask = decoder_input == PAD_ID

    return {
        "spectrum": spectra,
        "decoder_input": decoder_input,
        "labels": labels,
        "tgt_mask": tgt_mask,
        "tgt_padding_mask": tgt_padding_mask,
        "canonical_smiles": [b["canonical_smiles"] for b in batch],
        "raw_smiles": [b["raw_smiles"] for b in batch],
        "idx": [b["idx"] for b in batch],
    }


def load_split_dataset(processed_dir: str, split: str) -> NistIRAblationDataset:
    """Load a train/valid/test split from processed data.

    Args:
        processed_dir: Root processed directory.
        split: One of "train", "valid", "test".

    Returns:
        NistIRAblationDataset for the given split.
    """
    import json
    import os

    import torch

    # Load common records
    records = torch.load(os.path.join(processed_dir, "common", "records.pt"), weights_only=False)

    # Load split indices
    with open(os.path.join(processed_dir, "common", "split_indices.json")) as f:
        split_data = json.load(f)

    split_idx = split_data[split]  # list of indices into records

    target_dir = "atom"  # default; callers can override via dataset_kwargs
    # First try atom, then spe
    atom_path = os.path.join(processed_dir, "atom", f"{split}.pt")
    spe_path = os.path.join(processed_dir, "spe", f"{split}.pt")
    if os.path.exists(atom_path):
        target_ids_list = torch.load(atom_path, weights_only=False)
    elif os.path.exists(spe_path):
        target_ids_list = torch.load(spe_path, weights_only=False)
        target_dir = "spe"
    else:
        raise FileNotFoundError(f"No tokenized data found for split '{split}'")

    # Map from global record index to local index
    record_to_position = {g: i for i, g in enumerate(split_idx)}

    # Filter to only indices in this split
    local_indices = []
    canonical_smiles = []
    raw_smiles = []
    spectra_list = []
    target_ids_filtered = []
    global_indices = []

    for i, global_i in enumerate(split_idx):
        local_indices.append(i)
        canonical_smiles.append(records["canonical_smiles"][global_i])
        raw_smiles.append(records["raw_smiles"][global_i])
        spectra_list.append(records["spectrum"][global_i])
        target_ids_filtered.append(torch.tensor(target_ids_list[i], dtype=torch.long))
        global_indices.append(global_i)

    spectra = torch.stack(spectra_list) if len(spectra_list) > 0 else torch.empty(0, 3000)

    return NistIRAblationDataset(
        spectra=spectra,
        target_ids=target_ids_filtered,
        canonical_smiles=canonical_smiles,
        raw_smiles=raw_smiles,
        indices=global_indices,
    )
