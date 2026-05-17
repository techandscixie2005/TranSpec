"""Tests for NistIRAblationDataset and ablation_collate_fn."""

import torch

from experiments.nist_ir_ablation.transpec_ext.dataset import (
    NistIRAblationDataset,
    ablation_collate_fn,
)
from experiments.nist_ir_ablation.transpec_ext.vocab import PAD_ID


def _make_dummy_dataset(n=8, seq_len=15, n_points=3000):
    """Create a small dummy dataset for testing."""
    spectra = torch.randn(n, n_points)
    target_ids = [torch.randint(4, 10, (seq_len,)).long() for _ in range(n)]
    canonical_smiles = [f"C" for _ in range(n)]
    raw_smiles = [f"C" for _ in range(n)]
    indices = list(range(n))

    # Ensure first seq has BOS at start and EOS at end
    target_ids[0][0] = 1  # BOS_ID
    target_ids[0][-1] = 2  # EOS_ID

    return NistIRAblationDataset(
        spectra=spectra,
        target_ids=target_ids,
        canonical_smiles=canonical_smiles,
        raw_smiles=raw_smiles,
        indices=indices,
    )


class TestNistIRAblationDataset:
    """Tests for the dataset class."""

    def test_len(self):
        ds = _make_dummy_dataset(n=8)
        assert len(ds) == 8

    def test_getitem_keys(self):
        ds = _make_dummy_dataset(n=4)
        item = ds[0]
        assert "idx" in item
        assert "spectrum" in item
        assert "target_ids" in item
        assert "canonical_smiles" in item
        assert "raw_smiles" in item

    def test_getitem_shapes(self):
        ds = _make_dummy_dataset(n=4, seq_len=15, n_points=3000)
        item = ds[0]
        assert item["spectrum"].shape == (3000,)
        assert item["target_ids"].dtype == torch.long
        assert item["target_ids"].dim() == 1
        assert item["idx"] == 0

    def test_getitem_last_item(self):
        ds = _make_dummy_dataset(n=8)
        item = ds[7]
        assert item["idx"] == 7

    def test_spectrum_dtype(self):
        ds = _make_dummy_dataset(n=4)
        item = ds[0]
        assert item["spectrum"].dtype == torch.float32

    def test_empty_dataset(self):
        ds = NistIRAblationDataset(
            spectra=torch.empty(0, 3000),
            target_ids=[],
            canonical_smiles=[],
            raw_smiles=[],
            indices=[],
        )
        assert len(ds) == 0

    def test_target_id_content(self):
        """Target IDs contain BOS and EOS tokens."""
        ds = _make_dummy_dataset(n=4, seq_len=20)
        item = ds[0]
        assert item["target_ids"][0] == 1  # BOS
        assert item["target_ids"][-1] == 2  # EOS


class TestAblationCollateFn:
    """Tests for the collate function."""

    def make_batch(self, n=4, seq_len=15):
        ds = _make_dummy_dataset(n=n, seq_len=seq_len)
        return [ds[i] for i in range(n)]

    def test_batch_keys(self):
        batch = self.make_batch(n=4)
        collated = ablation_collate_fn(batch)
        expected_keys = {
            "spectrum", "decoder_input", "labels",
            "tgt_mask", "tgt_padding_mask",
            "canonical_smiles", "raw_smiles", "idx",
        }
        assert set(collated.keys()) == expected_keys

    def test_spectrum_shape(self):
        batch = self.make_batch(n=4)
        collated = ablation_collate_fn(batch)
        # (B, 1, 3000)
        assert collated["spectrum"].shape == (4, 1, 3000)

    def test_spectrum_dtype(self):
        batch = self.make_batch(n=2)
        collated = ablation_collate_fn(batch)
        assert collated["spectrum"].dtype == torch.float32

    def test_decoder_input_labels_relationship(self):
        batch = self.make_batch(n=4, seq_len=15)
        collated = ablation_collate_fn(batch)
        # decoder_input = target_ids[:, :-1]
        # labels = target_ids[:, 1:]
        assert collated["decoder_input"].shape[1] == collated["labels"].shape[1]
        # decoder_input seq_len should be <= original seq_len
        assert collated["decoder_input"].size(1) <= 15

    def test_padding_value(self):
        """PAD_ID=0 is used for padding."""
        batch = self.make_batch(n=4, seq_len=15)
        collated = ablation_collate_fn(batch)
        assert collated["decoder_input"][0, 0] != PAD_ID  # first token not padded

    def test_tgt_mask_causal(self):
        """tgt_mask should be lower-triangular (causal)."""
        batch = self.make_batch(n=2, seq_len=10)
        collated = ablation_collate_fn(batch)
        mask = collated["tgt_mask"]
        seq_len = mask.size(0)
        # Upper triangle (excluding diagonal) should be -inf
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                assert mask[i, j] == float("-inf"), (
                    f"Position ({i},{j}) should be masked"
                )

    def test_tgt_mask_diagonal_finite(self):
        """Diagonal of tgt_mask should be finite (not masked)."""
        batch = self.make_batch(n=2, seq_len=10)
        collated = ablation_collate_fn(batch)
        mask = collated["tgt_mask"]
        for i in range(mask.size(0)):
            assert torch.isfinite(mask[i, i]), f"Diagonal ({i},{i}) should be visible"

    def test_tgt_padding_mask_type(self):
        """tgt_padding_mask is a boolean tensor."""
        batch = self.make_batch(n=4)
        collated = ablation_collate_fn(batch)
        assert collated["tgt_padding_mask"].dtype == torch.bool

    def test_tgt_padding_mask_shape(self):
        """tgt_padding_mask same as decoder_input shape."""
        batch = self.make_batch(n=4, seq_len=15)
        collated = ablation_collate_fn(batch)
        assert collated["tgt_padding_mask"].shape == collated["decoder_input"].shape

    def test_variable_length_sequences(self):
        """Different length target_ids get padded correctly."""
        spectra = torch.randn(3, 3000)
        target_ids = [
            torch.tensor([1, 4, 5, 6, 2]),     # len 5
            torch.tensor([1, 4, 2]),            # len 3
            torch.tensor([1, 4, 5, 6, 7, 2]),   # len 6
        ]
        ds = NistIRAblationDataset(
            spectra=spectra,
            target_ids=target_ids,
            canonical_smiles=["C", "C", "C"],
            raw_smiles=["C", "C", "C"],
            indices=[0, 1, 2],
        )
        batch = [ds[i] for i in range(3)]
        collated = ablation_collate_fn(batch)

        # decoder_input is target_ids[:, :-1]
        # Longest: 6 -> decoder_input is 5 for the last one
        assert collated["decoder_input"].shape == (3, 5)
        assert collated["labels"].shape == (3, 5)

        # Shorter sequences should have PAD_ID in decoder_input
        # Second seq: target len 3, decoder_input len 2, padded to 5
        assert collated["decoder_input"][1, 0] == 1  # BOS
        assert collated["decoder_input"][1, 1] == 4

    def test_labels_offset(self):
        """labels should be target_ids shifted: labels[t] = target_ids[t+1]."""
        spectra = torch.randn(1, 3000)
        target_ids = [torch.tensor([1, 4, 5, 6, 2])]
        ds = NistIRAblationDataset(
            spectra=spectra,
            target_ids=target_ids,
            canonical_smiles=["C"],
            raw_smiles=["C"],
            indices=[0],
        )
        batch = [ds[0]]
        collated = ablation_collate_fn(batch)
        # decoder_input = [1, 4, 5, 6], labels = [4, 5, 6, 2]
        assert collated["decoder_input"][0].tolist() == [1, 4, 5, 6]
        assert collated["labels"][0].tolist() == [4, 5, 6, 2]

    def test_smiles_metadata(self):
        """SMILES strings passed through correctly."""
        batch = self.make_batch(n=3)
        collated = ablation_collate_fn(batch)
        assert len(collated["canonical_smiles"]) == 3
        assert len(collated["raw_smiles"]) == 3
        assert len(collated["idx"]) == 3
