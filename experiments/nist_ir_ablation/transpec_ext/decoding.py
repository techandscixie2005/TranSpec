from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .model_factory import AblationModel
from .vocab import BOS_ID, EOS_ID, PAD_ID


def threshold_value_search_ablation(
    model: nn.Module,
    encoder_input: torch.Tensor,
    tokenizer: Any,
    threshold_value: float = 0.01,
    candidate_limit: int = 500,
    max_len: int = 256,
    use_cnn: bool = True,
    use_mlp: bool = False,
    input_channels: int = 1,
) -> Tuple[List[str], List[float], List[List[int]]]:
    """Local ablation adaptation of threshold value search.

    Key differences from src/search_methods.py:
    - max_len from config (not hardcoded 99)
    - Uses tokenizer.decode() for SMILES reconstruction
    - BOS/EOS/PAD from constants (PAD_ID=0, BOS_ID=1, EOS_ID=2)
    - Returns (surface_smiles, probabilities, token_id_sequences)

    Args:
        model: Core Model (not AblationModel wrapper).
        encoder_input: Already Fourier-augmented if applicable (B, C, N).
        tokenizer: Tokenizer with decode() method.
        threshold_value: Probability threshold for branching.
        candidate_limit: Maximum number of candidates.
        max_len: Maximum decoding length.
        use_cnn: Whether model uses CNN frontend.
        use_mlp: Whether model uses MLP frontend.
        input_channels: Number of input channels.

    Returns:
        (candidate_smiles, probabilities, token_id_sequences)
    """
    candidates_smiles: List[str] = []
    candidates_prob: List[float] = []
    candidates_ids: List[List[int]] = []

    # Forward through CNN/MLP and encoder
    if use_cnn:
        x = model.c(encoder_input)
    if use_mlp:
        if input_channels == 1:
            x = model.m(encoder_input.reshape([encoder_input.shape[0], 10, 300]))
        elif input_channels == 2:
            x = model.m(encoder_input.reshape([encoder_input.shape[0], 20, 300]))

    x = model.encoder(x)

    # Initialize branches
    de_input_type = torch.ones((1, 1), dtype=torch.int64, device=encoder_input.device) * BOS_ID
    branches = [(de_input_type, 1.0)]
    finished: List[Tuple[List[int], float]] = []

    # Iterate over branches (list grows during iteration)
    i = 0
    while i < len(branches) and len(finished) < candidate_limit:
        de_input, cum_prob = branches[i]
        i += 1

        # Skip if already terminated (EOS emitted)
        if de_input.size(1) > 1 and de_input[0, -1].item() == EOS_ID:
            token_ids = de_input.squeeze(0)[1:-1].tolist()  # strip BOS and EOS
            finished.append((token_ids, cum_prob))
            continue

        # Decode step by step up to max_len
        current_input = de_input
        for step in range(de_input.size(1), max_len + 1):
            # Generate tgt_mask
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                current_input.size(-1), device=encoder_input.device
            )

            # Embedding + PE
            emb_out = model.embedding(current_input)
            tgt = model.pe(emb_out)

            # Decode
            tgt = model.decoder(
                tgt=tgt, memory=x,
                tgt_mask=tgt_mask, tgt_key_padding_mask=None
            )
            pre_type = model.pre_type(tgt)
            probs = F.softmax(pre_type[:, -1, :], dim=-1)  # (1, vocab_size)
            top_probs, top_indices = probs.topk(k=probs.size(-1), dim=-1)

            top_probs = top_probs.squeeze(0)  # (vocab_size,)
            top_indices = top_indices.squeeze(0)  # (vocab_size,)

            # Greedy choice
            next_token = top_indices[0].item()
            next_prob = top_probs[0].item()

            # Build new decoder input for greedy path
            next_symbol = torch.tensor(
                [[next_token]], dtype=torch.int64, device=encoder_input.device
            )
            current_input = torch.cat([current_input, next_symbol], dim=-1)

            # Check for EOS
            if next_token == EOS_ID or current_input.size(1) > max_len:
                token_ids = current_input.squeeze(0)[1:-1].tolist()  # strip BOS and EOS
                finished.append((token_ids, cum_prob * next_prob))
                break

            # Update cumulative probability for greedy path
            cum_prob = cum_prob * next_prob

            # Branch: check all non-greedy tokens above threshold
            if current_input.size(-1) <= max_len:
                for j in range(1, top_probs.size(0)):
                    branch_prob = top_probs[j].item()
                    if branch_prob >= threshold_value:
                        branch_token = top_indices[j].item()
                        branch_tensor = torch.tensor(
                            [[branch_token]], dtype=torch.int64,
                            device=encoder_input.device
                        )
                        # Build branch from current_input (without the greedy token appended)
                        # Actually, we need the branch from the state BEFORE the greedy step
                        # So we use de_input (original), then the branch token
                        # Rebuild properly:
                        branch_input = torch.cat(
                            [de_input,
                             torch.tensor([[branch_token]], dtype=torch.int64, device=encoder_input.device)],
                            dim=-1
                        )
                        branches.append((branch_input, cum_prob * branch_prob))

        # End of while: the for-loop handles iteration per branch
    else:
        # If we exhausted branches but still at the last branch
        # Check if current branch finished properly
        if i == len(branches) and current_input.size(1) <= max_len + 1:
            pass  # Already handled above

    # Also collect any unfinished branches that hit max_len
    for j in range(i, len(branches)):
        de_input, cum_prob = branches[j]
        if de_input.size(1) > 1:
            token_ids = de_input.squeeze(0)[1:].tolist()
            finished.append((token_ids, cum_prob))

    # Sort by probability descending
    finished.sort(key=lambda x: x[1], reverse=True)

    # Build results
    for token_ids, prob in finished[:candidate_limit]:
        # Decode using tokenizer
        smiles = tokenizer.decode(token_ids)
        candidates_smiles.append(smiles)
        candidates_prob.append(prob)
        candidates_ids.append(token_ids)

    return candidates_smiles, candidates_prob, candidates_ids


def decode_ablation(
    ablation_model: AblationModel,
    spectrum: torch.Tensor,
    tokenizer: Any,
    **kwargs,
) -> Tuple[List[str], List[float], List[List[int]]]:
    """Decode a single spectrum using the ablation model.

    Applies Fourier encoding (if present) exactly once, then delegates
    to threshold_value_search_ablation.

    Args:
        ablation_model: AblationModel wrapping core Model + optional Fourier.
        spectrum: Input spectrum tensor (1, 1, 3000) or (1, 3000).
        tokenizer: Tokenizer with decode() method.
        **kwargs: Passed through to threshold_value_search_ablation.

    Returns:
        (candidate_smiles, probabilities, token_id_sequences)
    """
    # Ensure shape is (1, 1, 3000)
    if spectrum.dim() == 2:
        spectrum = spectrum.unsqueeze(0)  # (1, 3000) -> (1, 1, 3000)
    if spectrum.dim() == 1:
        spectrum = spectrum.unsqueeze(0).unsqueeze(0)  # (3000,) -> (1, 1, 3000)

    # Apply Fourier encoding (if present)
    if ablation_model.fourier_encoding is not None:
        encoder_input = ablation_model.fourier_encoding(spectrum)
    else:
        encoder_input = spectrum

    # Determine input channels
    input_channels = encoder_input.size(1)

    return threshold_value_search_ablation(
        model=ablation_model.core,
        encoder_input=encoder_input,
        tokenizer=tokenizer,
        use_cnn=True,
        use_mlp=False,
        input_channels=input_channels,
        **kwargs,
    )
