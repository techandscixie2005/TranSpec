import json
import re
from typing import Dict, List, Optional

from .vocab import (
    PAD_ID, BOS_ID, EOS_ID, UNK_ID,
    SPECIAL_TOKENS, FIRST_REAL_TOKEN_ID,
)

_SMILES_REGEX = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p"
    r"|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$"
    r"|\%[0-9]{2}|[0-9])"
)


class AtomTokenizer:
    """Atom-level SMILES tokenizer.

    Tokenizes SMILES strings into individual tokens (atoms, bonds,
    brackets, ring markers). Vocab IDs start at FIRST_REAL_TOKEN_ID (4).
    PAD(0), BOS(1), EOS(2), UNK(3) are reserved.
    """

    def __init__(self, max_len: int = 256):
        self.max_len = max_len
        self._vocab: Dict[str, int] = {}  # token -> id
        self._id_to_token: Dict[int, str] = {}  # id -> token

    @property
    def vocab(self) -> Dict[str, int]:
        return dict(self._vocab)

    @property
    def vocab_size(self) -> int:
        return len(self._vocab) + len(SPECIAL_TOKENS)

    @property
    def pad_id(self) -> int:
        return PAD_ID

    def tokenize(self, smiles: str) -> List[str]:
        """Tokenize a SMILES string into a list of token strings (no special tokens)."""
        tokens = _SMILES_REGEX.findall(smiles)
        reconstructed = "".join(tokens)
        if reconstructed != smiles:
            return [UNK_TOKEN]
        return tokens

    def fit(self, smiles_list: List[str]) -> "AtomTokenizer":
        vocab = {}
        next_id = FIRST_REAL_TOKEN_ID

        for smiles in smiles_list:
            tokens = self.tokenize(smiles)
            for token in tokens:
                if token not in vocab:
                    vocab[token] = next_id
                    next_id += 1

        self._vocab = vocab
        self._id_to_token = {v: k for k, v in vocab.items()}
        return self

    def encode(
        self,
        smiles: str,
        add_special_tokens: bool = True,
        max_len: Optional[int] = None,
    ) -> List[int]:
        """Encode a SMILES string into a list of token IDs.

        If add_special_tokens is True, prepends BOS_ID and appends EOS_ID.
        Truncates to max_len if specified (counting special tokens).
        """
        tokens = self.tokenize(smiles)
        ids = []
        for token in tokens:
            if token in self._vocab:
                ids.append(self._vocab[token])
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
            token = self._id_to_token.get(tid)
            if token is None:
                token = self._id_to_token.get(UNK_ID, "<UNK>")
            tokens.append(token)

        return "".join(tokens)

    def save(self, path: str) -> None:
        """Save vocabulary to a JSON file."""
        data = {
            "max_len": self.max_len,
            "vocab": self._vocab,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "AtomTokenizer":
        """Load vocabulary from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        tok = cls(max_len=data.get("max_len", 256))
        tok._vocab = data["vocab"]
        tok._id_to_token = {int(v): k for k, v in data["vocab"].items()}
        return tok
