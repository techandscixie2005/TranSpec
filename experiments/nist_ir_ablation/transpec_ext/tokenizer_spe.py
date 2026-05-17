import json
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

from .tokenizer_atom import AtomTokenizer
from .vocab import (
    PAD_ID, BOS_ID, EOS_ID, UNK_ID,
    SPECIAL_TOKENS, FIRST_REAL_TOKEN_ID,
)

_INTERNAL_KEY_SEP = "@@"
_INTERNAL_KEY_PREFIX = "SPE::"


def _make_internal_key(left: str, right: str) -> str:
    return f"{_INTERNAL_KEY_PREFIX}{left}{_INTERNAL_KEY_SEP}{right}"


def _surface_from_internal_key(key: str) -> str:
    """Extract the surface string from an internal key."""
    # "SPE::left@@right" -> left + right (the concatenation)
    prefix_len = len(_INTERNAL_KEY_PREFIX)
    inner = key[prefix_len:]
    parts = inner.split(_INTERNAL_KEY_SEP)
    return parts[0] + parts[1]


class SPETokenizer:
    """SMILES Pair Encoding (BPE) tokenizer.

    Learns merge rules on atom-tokenized SMILES sequences.
    Atom-level tokens remain available as fallback.
    """

    def __init__(
        self,
        vocab_size: int = 100,
        min_frequency: int = 2,
        max_len: int = 256,
    ):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.max_len = max_len
        self.atom_tokenizer = AtomTokenizer(max_len=max_len)

        # Surface -> ID (includes atom tokens + merged SPE tokens)
        self._surface_to_id: Dict[str, int] = {}
        # ID -> surface
        self._id_to_surface: Dict[int, str] = {}
        # Merge rules: list of (left_surface, right_surface) in learned order
        self._merges: List[Tuple[str, str]] = []

    @property
    def pad_id(self) -> int:
        return PAD_ID

    @property
    def vocab(self) -> Dict[str, int]:
        return dict(self._surface_to_id)

    @property
    def vocab_size_real(self) -> int:
        return len(self._surface_to_id) + len(SPECIAL_TOKENS)

    @property
    def merges(self) -> List[Tuple[str, str]]:
        return list(self._merges)

    def _init_vocab_from_atom_tokens(self, smiles_list: List[str]) -> None:
        """Initialize SPE surface-to-id mapping from atom tokenizer vocab."""
        # First fit the atom tokenizer on the same data
        self.atom_tokenizer.fit(smiles_list)
        atom_vocab = self.atom_tokenizer.vocab  # token -> id (starts at 4)

        next_id = FIRST_REAL_TOKEN_ID
        self._surface_to_id = {}
        self._id_to_surface = {}

        # Copy atom token vocab surfaces
        for surface, tid in sorted(atom_vocab.items(), key=lambda x: x[1]):
            self._surface_to_id[surface] = next_id
            self._id_to_surface[next_id] = surface
            next_id += 1

    def _sequences_from_smiles(
        self, smiles_list: List[str]
    ) -> List[List[str]]:
        """Atom-tokenize a list of SMILES, returning lists of token strings."""
        sequences = []
        for smiles in smiles_list:
            tokens = self.atom_tokenizer.tokenize(smiles)
            if tokens == ["<UNK>"]:
                continue
            sequences.append(tokens)
        return sequences

    def fit(self, smiles_list: List[str]) -> "SPETokenizer":
        """Learn BPE merge rules from a list of SMILES strings.

        The atom tokenizer is fit on the same data.
        Merge rules are learned iteratively up to vocab_size merges.
        """
        # Initialize vocabulary from atom tokens
        self._init_vocab_from_atom_tokens(smiles_list)

        # Atom-tokenize all SMILES into token strings
        sequences = self._sequences_from_smiles(smiles_list)

        if not sequences:
            return self

        self._merges = []
        num_merges = self.vocab_size

        # Use Counter for pair frequencies
        for _ in range(num_merges):
            pair_counts: Counter = Counter()
            for seq in sequences:
                for i in range(len(seq) - 1):
                    pair_counts[(seq[i], seq[i + 1])] += 1

            if not pair_counts:
                break

            # Find the most frequent pair
            (best_left, best_right), best_count = pair_counts.most_common(1)[0]

            if best_count < self.min_frequency:
                break

            # Record the merge
            self._merges.append((best_left, best_right))

            # Compute new surface
            merged_surface = best_left + best_right

            # Assign a new ID
            new_id = max(self._id_to_surface.keys()) + 1 if self._id_to_surface else FIRST_REAL_TOKEN_ID
            self._surface_to_id[merged_surface] = new_id
            self._id_to_surface[new_id] = merged_surface

            # Apply the merge to all sequences
            new_sequences = []
            for seq in sequences:
                new_seq = []
                i = 0
                while i < len(seq):
                    if i < len(seq) - 1 and seq[i] == best_left and seq[i + 1] == best_right:
                        new_seq.append(merged_surface)
                        i += 2
                    else:
                        new_seq.append(seq[i])
                        i += 1
                new_sequences.append(new_seq)
            sequences = new_sequences

        return self

    def encode(
        self,
        smiles: str,
        add_special_tokens: bool = True,
        max_len: Optional[int] = None,
    ) -> List[int]:
        """Encode a SMILES string using learned BPE merges."""
        # 1. Atom-tokenize
        tokens = self.atom_tokenizer.tokenize(smiles)
        if tokens == ["<UNK>"]:
            return [BOS_ID, UNK_ID, EOS_ID] if add_special_tokens else [UNK_ID]

        # 2. Apply merge rules greedily in learned order
        for left, right in self._merges:
            merged_surface = left + right
            i = 0
            new_tokens = []
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == left and tokens[i + 1] == right:
                    new_tokens.append(merged_surface)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        # 3. Convert to IDs
        ids = []
        for token in tokens:
            if token in self._surface_to_id:
                ids.append(self._surface_to_id[token])
            else:
                ids.append(UNK_ID)

        if add_special_tokens:
            ids = [BOS_ID] + ids + [EOS_ID]

        max_len = max_len or self.max_len
        if len(ids) > max_len:
            ids = ids[:max_len]

        return ids

    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
        stop_at_eos: bool = True,
    ) -> str:
        """Decode a list of token IDs into a SMILES string."""
        tokens = []
        for tid in token_ids:
            if stop_at_eos and tid == EOS_ID:
                break
            if skip_special_tokens and tid in SPECIAL_TOKENS.values():
                continue
            surface = self._id_to_surface.get(tid)
            if surface is None:
                surface = "<UNK>"
            tokens.append(surface)

        return "".join(tokens)

    def save(self, path: str) -> None:
        """Save SPE tokenizer to a directory (creates two files)."""
        base_path = path.rstrip("/")  # path is a stem, we append suffixes

        # Save vocab
        vocab_path = f"{base_path}_vocab.json"
        vocab_data = {
            "max_len": self.max_len,
            "vocab_size": self.vocab_size,
            "min_frequency": self.min_frequency,
            "surface_to_id": self._surface_to_id,
            "atom_vocab": self.atom_tokenizer.vocab,
        }
        with open(vocab_path, "w") as f:
            json.dump(vocab_data, f, indent=2)

        # Save merges
        merges_path = f"{base_path}_merges.txt"
        with open(merges_path, "w") as f:
            f.write("#version 1.0\n")
            for left, right in self._merges:
                f.write(f"{left} {right}\n")

    @classmethod
    def load(cls, path: str) -> "SPETokenizer":
        """Load SPE tokenizer from a directory (reads two files)."""
        base_path = path.rstrip("/")

        # Load vocab
        vocab_path = f"{base_path}_vocab.json"
        with open(vocab_path) as f:
            data = json.load(f)

        tok = cls(
            vocab_size=data.get("vocab_size", 100),
            min_frequency=data.get("min_frequency", 2),
            max_len=data.get("max_len", 256),
        )

        tok._surface_to_id = {
            k: int(v) for k, v in data["surface_to_id"].items()
        }
        tok._id_to_surface = {
            int(v): k for k, v in data["surface_to_id"].items()
        }

        # Restore atom tokenizer vocab
        atom_vocab_data = data.get("atom_vocab", {})
        tok.atom_tokenizer._vocab = dict(atom_vocab_data)
        tok.atom_tokenizer._id_to_token = {
            int(v): k for k, v in atom_vocab_data.items()
        }

        # Load merges
        merges_path = f"{base_path}_merges.txt"
        tok._merges = []
        try:
            with open(merges_path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split(" ", 1)
                    if len(parts) == 2:
                        tok._merges.append((parts[0], parts[1]))
        except FileNotFoundError:
            pass

        return tok
