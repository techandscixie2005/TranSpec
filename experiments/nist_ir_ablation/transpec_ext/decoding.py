import math
from typing import Any, List, Optional, Tuple

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
    """Threshold-value (branching beam) search for SMILES decoding.

    Processes branches in FIFO order. At each step the greedy token
    continues the current branch; all non-greedy tokens above threshold
    spawn new branches from the state *before* the greedy step.

    Candidates remain sorted by cumulative probability descending.

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
    device = encoder_input.device

    # Forward through CNN/MLP and encoder
    if use_cnn:
        x = model.c(encoder_input)
    if use_mlp:
        if input_channels == 1:
            x = model.m(encoder_input.reshape(encoder_input.shape[0], 10, 300))
        elif input_channels == 2:
            x = model.m(encoder_input.reshape(encoder_input.shape[0], 20, 300))

    x = model.encoder(x)

    # Branch queue: (decoder_input_tensor, cumulative_probability)
    start_tokens = torch.full((1, 1), BOS_ID, dtype=torch.int64, device=device)
    branches: List[Tuple[torch.Tensor, float]] = [(start_tokens, 1.0)]
    finished: List[Tuple[List[int], float]] = []

    max_branch_queue = 2000  # safety limit to prevent exponential explosion

    i = 0
    while i < len(branches) and len(finished) < candidate_limit:
        current, cum_prob = branches[i]
        i += 1

        # Skip if this branch already emitted EOS (from a prior expansion)
        if current.size(1) > 1 and current[0, -1].item() == EOS_ID:
            token_ids = current.squeeze(0)[1:-1].tolist()
            finished.append((token_ids, cum_prob))
            continue

        for _step in range(current.size(1), max_len + 1):
            # Generate tgt_mask
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                current.size(-1), device=device
            )

            # Embedding + PE
            emb_out = model.embedding(current)
            tgt = model.pe(emb_out)

            # Decode
            tgt = model.decoder(
                tgt=tgt, memory=x,
                tgt_mask=tgt_mask, tgt_key_padding_mask=None,
            )
            pre_type = model.pre_type(tgt)
            probs = F.softmax(pre_type[:, -1, :], dim=-1)  # (1, vocab_size)
            top_probs, top_indices = probs.topk(k=probs.size(-1), dim=-1)

            top_probs = top_probs.squeeze(0)  # (vocab_size,)
            top_indices = top_indices.squeeze(0)

            greedy_token = top_indices[0].item()
            greedy_prob = top_probs[0].item()

            # --- Branch before greedy step ---
            # Spawn new branches from the current state with each non-greedy
            # token whose probability exceeds threshold_value.
            if current.size(-1) <= max_len and len(branches) < max_branch_queue:
                for j in range(1, top_probs.size(0)):
                    branch_prob_val = top_probs[j].item()
                    if branch_prob_val >= threshold_value:
                        branch_token = top_indices[j].item()
                        branch_tensor = torch.tensor(
                            [[branch_token]], dtype=torch.int64, device=device,
                        )
                        branch_input = torch.cat([current, branch_tensor], dim=-1)
                        branches.append((branch_input, cum_prob * branch_prob_val))

            # --- Greedy step ---
            greedy_tensor = torch.tensor(
                [[greedy_token]], dtype=torch.int64, device=device,
            )
            current = torch.cat([current, greedy_tensor], dim=-1)
            cum_prob = cum_prob * greedy_prob

            # Check for EOS
            if greedy_token == EOS_ID or current.size(1) > max_len:
                token_ids = current.squeeze(0)[1:-1].tolist()
                finished.append((token_ids, cum_prob))
                break

    # Collect any unfinished branches that hit the queue end
    for j in range(i, len(branches)):
        tokens, prob = branches[j]
        if tokens.size(1) > 1:
            token_ids = tokens.squeeze(0)[1:].tolist()
            finished.append((token_ids, prob))

    # Sort by probability descending
    finished.sort(key=lambda x: x[1], reverse=True)

    # Build results
    candidates_smiles: List[str] = []
    candidates_prob: List[float] = []
    candidates_ids: List[List[int]] = []

    for token_ids, prob in finished[:candidate_limit]:
        smiles = tokenizer.decode(token_ids)
        candidates_smiles.append(smiles)
        candidates_prob.append(prob)
        candidates_ids.append(token_ids)

    return candidates_smiles, candidates_prob, candidates_ids


def beam_search_decode(
    model: nn.Module,
    encoder_input: torch.Tensor,
    tokenizer: Any,
    beam_size: int = 10,
    candidate_limit: int = 10,
    max_len: int = 256,
    use_cnn: bool = True,
    use_mlp: bool = False,
    input_channels: int = 1,
) -> Tuple[List[str], List[float], List[List[int]]]:
    """Fixed-width beam search for SMILES decoding.

    Keeps exactly ``beam_size`` active hypotheses at each decoding step.
    Complexity is O(beam_size * max_len) decoder calls, not exponential.
    Uses log probabilities for scoring.

    Args:
        model: Core Model (not AblationModel wrapper).
        encoder_input: Already Fourier-augmented if applicable (B, C, N).
        tokenizer: Tokenizer with decode() method.
        beam_size: Number of active hypotheses kept at each step.
        candidate_limit: Maximum number of candidates to return.
        max_len: Maximum decoding length.
        use_cnn: Whether model uses CNN frontend.
        use_mlp: Whether model uses MLP frontend.
        input_channels: Number of input channels.

    Returns:
        (candidate_smiles, probabilities, token_id_sequences)
    """
    device = encoder_input.device

    # Forward through CNN/MLP and encoder
    if use_cnn:
        x = model.c(encoder_input)
    if use_mlp:
        if input_channels == 1:
            x = model.m(encoder_input.reshape(encoder_input.shape[0], 10, 300))
        elif input_channels == 2:
            x = model.m(encoder_input.reshape(encoder_input.shape[0], 20, 300))

    memory = model.encoder(x)  # (1, seq_len, d_model)

    # Initialize beam with BOS token
    # Each entry: (log_probability, token_ids_list)
    hypotheses: List[Tuple[float, List[int]]] = [(0.0, [BOS_ID])]
    finished: List[Tuple[float, List[int]]] = []

    for _step in range(max_len):
        if not hypotheses or len(finished) >= candidate_limit:
            break

        all_candidates: List[Tuple[float, List[int]]] = []

        for log_prob, tokens in hypotheses:
            decoder_input = torch.tensor([tokens], dtype=torch.int64, device=device)
            seq_len = decoder_input.size(1)

            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                seq_len, device=device
            )

            emb_out = model.embedding(decoder_input)
            tgt = model.pe(emb_out)
            tgt = model.decoder(
                tgt=tgt, memory=memory,
                tgt_mask=tgt_mask, tgt_key_padding_mask=None,
            )
            pre_type = model.pre_type(tgt)
            logits = pre_type[0, -1, :]  # (vocab_size,)

            # Log-softmax for stable scoring
            log_probs = F.log_softmax(logits, dim=-1)

            # Top-k next tokens
            k = min(beam_size, log_probs.size(-1))
            topk_log_probs, topk_indices = log_probs.topk(k=k)

            for i in range(k):
                new_log_prob = log_prob + topk_log_probs[i].item()
                new_token = topk_indices[i].item()
                new_tokens = tokens + [new_token]
                all_candidates.append((new_log_prob, new_tokens))

        # Select top beam_size candidates globally
        all_candidates.sort(key=lambda x: x[0], reverse=True)

        # Separate finished (EOS emitted) from active
        new_hypotheses = []
        for log_prob, tokens in all_candidates:
            if tokens[-1] == EOS_ID:
                finished.append((log_prob, tokens))
            else:
                if len(new_hypotheses) < beam_size:
                    new_hypotheses.append((log_prob, tokens))

        hypotheses = [(lp, tok) for lp, tok in new_hypotheses]

    # Any remaining unfinished hypotheses become candidates
    for log_prob, tokens in hypotheses:
        finished.append((log_prob, tokens))

    # Sort by score descending
    finished.sort(key=lambda x: x[0], reverse=True)

    # Take top candidate_limit
    results = finished[:candidate_limit]

    # Build results
    candidates_smiles: List[str] = []
    candidates_prob: List[float] = []
    candidates_ids: List[List[int]] = []

    for log_prob, token_ids in results:
        # Strip BOS, EOS, PAD — they must not appear in final SMILES
        decoded_ids = [t for t in token_ids if t not in (BOS_ID, EOS_ID, PAD_ID)]
        smiles = tokenizer.decode(decoded_ids)
        candidates_smiles.append(smiles)
        candidates_prob.append(float(math.exp(log_prob)))
        candidates_ids.append(decoded_ids)

    return candidates_smiles, candidates_prob, candidates_ids


def decode_ablation(
    ablation_model: AblationModel,
    spectrum: torch.Tensor,
    tokenizer: Any,
    **kwargs,
) -> Tuple[List[str], List[float], List[List[int]]]:
    """Decode a single spectrum using the ablation model.

    Applies Fourier encoding (if present) exactly once, then delegates
    to the requested decode method (beam search or threshold-value).

    Args:
        ablation_model: AblationModel wrapping core Model + optional Fourier.
        spectrum: Input spectrum tensor (1, 1, 3000) or (1, 3000).
        tokenizer: Tokenizer with decode() method.
        **kwargs:
            decode_method: ``"beam"`` (default) or ``"threshold"``.
            beam_size: Active hypotheses kept per step (beam search).
            candidate_limit: Max candidates to return.
            max_len: Max decoding length.
            threshold_value: Probability threshold (threshold search only).

    Returns:
        (candidate_smiles, probabilities, token_id_sequences)
    """
    # Ensure shape is (1, 1, 3000)
    if spectrum.dim() == 2:
        spectrum = spectrum.unsqueeze(0)  # (1, 3000) -> (1, 1, 3000)
    elif spectrum.dim() == 1:
        spectrum = spectrum.unsqueeze(0).unsqueeze(0)  # (3000,) -> (1, 1, 3000)

    # Apply Fourier encoding (if present)
    if ablation_model.fourier_encoding is not None:
        encoder_input = ablation_model.fourier_encoding(spectrum)
    else:
        encoder_input = spectrum

    # Determine input channels
    input_channels = encoder_input.size(1)

    decode_method = kwargs.pop("decode_method", "threshold")

    if decode_method == "beam":
        beam_size = kwargs.pop("beam_size", 10)
        candidate_limit = kwargs.pop("candidate_limit", 10)
        max_len = kwargs.pop("max_len", 256)
        return beam_search_decode(
            model=ablation_model.core,
            encoder_input=encoder_input,
            tokenizer=tokenizer,
            beam_size=beam_size,
            candidate_limit=candidate_limit,
            max_len=max_len,
            use_cnn=True,
            use_mlp=False,
            input_channels=input_channels,
        )
    else:
        return threshold_value_search_ablation(
            model=ablation_model.core,
            encoder_input=encoder_input,
            tokenizer=tokenizer,
            use_cnn=True,
            use_mlp=False,
            input_channels=input_channels,
            **kwargs,
        )
