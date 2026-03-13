import pathlib
import numpy as np
from typing import Optional

def cut_string_between_bos_eos(string, bos='[CLS]', eos='[SEP]'):
    """Extract string content between BOS and EOS tokens."""
    bos_index = string.find(bos)
    eos_index = string.find(eos)
    if bos_index == -1:
        bos_index = -len(bos)
    if eos_index == -1:
        eos_index = len(string)
    return string[bos_index + len(bos):eos_index].strip()

def _load_onnxruntime():
    """Import onnxruntime lazily."""
    try:
        import onnxruntime as ort  # type: ignore
    except ImportError as error:
        raise ImportError(
            "onnxruntime is required. Install dependencies from lambda_deployment/requirements.txt"
        ) from error
    return ort

def create_ort_session(onnx_path: pathlib.Path):
    """Create an ONNX Runtime Inference Session."""
    ort = _load_onnxruntime()
    providers = ['CPUExecutionProvider']
    if 'CUDAExecutionProvider' in ort.get_available_providers():
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = 1
    options.log_severity_level = 3

    return ort.InferenceSession(str(onnx_path), sess_options=options, providers=providers)


def _softmax(logits: np.ndarray) -> np.ndarray:
    max_logits = np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    denom = np.sum(exp_logits, axis=-1, keepdims=True)
    return exp_logits / np.maximum(denom, 1e-12)


def _apply_top_k_top_p_filtering(logits: np.ndarray, top_k: int, top_p: float) -> np.ndarray:
    filtered = logits.copy()
    batch_size, vocab_size = filtered.shape

    if top_k > 0:
        k = min(top_k, vocab_size)
        if k < vocab_size:
            remove_idx = np.argpartition(filtered, -k, axis=-1)[:, :-k]
            rows = np.arange(batch_size)[:, None]
            filtered[rows, remove_idx] = -np.inf

    if top_p < 1.0:
        for i in range(batch_size):
            row = filtered[i]
            sorted_idx = np.argsort(row)[::-1]
            sorted_logits = row[sorted_idx]
            probs = _softmax(sorted_logits[None, :])[0]
            cumulative_probs = np.cumsum(probs)

            remove_mask = cumulative_probs > top_p
            remove_mask[0] = False
            row[sorted_idx[remove_mask]] = -np.inf

            # Keep at least one token selectable.
            if not np.isfinite(row).any():
                row[sorted_idx[0]] = sorted_logits[0]

    return filtered


def sample_generate_onnx(session, src_ids: np.ndarray, bos_token_id: int,
                         eos_token_id: int, pad_token_id: int, max_len: int,
                         temperature: float = 0.9, top_k: int = 50,
                         top_p: float = 0.95, repetition_penalty: float = 1.2,
                         seed: Optional[int] = None):
    """Sampling-based autoregressive generation for ONNX model."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if top_k < 0:
        raise ValueError("top_k must be >= 0")
    if not (0.0 < top_p <= 1.0):
        raise ValueError("top_p must be in (0, 1]")

    batch_size = src_ids.shape[0]
    vocab_size = session.get_outputs()[0].shape[-1]
    special_token_ids = {t for t in [pad_token_id, bos_token_id, eos_token_id] if t is not None}
    rng = np.random.default_rng(seed)

    all_tokens = np.full((batch_size, 1), bos_token_id, dtype=np.int64)
    done = np.zeros(batch_size, dtype=np.bool_)

    for _ in range(max_len - 1):
        logits = session.run(['logits'], {'src': src_ids, 'tgt': all_tokens})[0]
        next_logits = logits[:, -1, :].astype(np.float32)

        if repetition_penalty != 1.0:
            for i in range(batch_size):
                tokens_to_penalize = [t for t in all_tokens[i] if t not in special_token_ids]
                if not tokens_to_penalize:
                    continue

                for token_id in np.unique(tokens_to_penalize):
                    value = next_logits[i, token_id]
                    if value > 0:
                        next_logits[i, token_id] = value / repetition_penalty
                    else:
                        next_logits[i, token_id] = value * repetition_penalty

        next_logits = next_logits / float(temperature)

        next_logits[done] = -np.inf
        next_logits[done, eos_token_id] = 0.0

        filtered_logits = _apply_top_k_top_p_filtering(next_logits, top_k=top_k, top_p=top_p)
        probs = _softmax(filtered_logits)

        next_tokens = np.empty(batch_size, dtype=np.int64)
        for i in range(batch_size):
            if done[i]:
                next_tokens[i] = eos_token_id
                continue

            row_probs = probs[i]
            prob_sum = float(np.sum(row_probs))
            if not np.isfinite(prob_sum) or prob_sum <= 0.0:
                next_tokens[i] = int(np.argmax(filtered_logits[i]))
                continue

            row_probs = row_probs / prob_sum
            next_tokens[i] = int(rng.choice(vocab_size, p=row_probs))

        all_tokens = np.concatenate([all_tokens, next_tokens[:, None]], axis=1)
        done = np.logical_or(done, next_tokens == eos_token_id)

        if np.all(done):
            break

    if all_tokens.shape[1] < max_len:
        pad = np.full((batch_size, max_len - all_tokens.shape[1]), pad_token_id, dtype=np.int64)
        all_tokens = np.concatenate([all_tokens, pad], axis=1)

    return all_tokens[:, :max_len]

def beam_generate_onnx(session, src_ids: np.ndarray, bos_token_id: int,
                       eos_token_id: int, pad_token_id: int, max_len: int,
                       num_beams: int = 3, length_penalty: float = 1.0,
                       repetition_penalty: float = 1.3):
    """Vectorized beam search for ONNX model."""
    batch_size = src_ids.shape[0]
    vocab_size = session.get_outputs()[0].shape[-1]
    
    # Special tokens for repetition penalty
    special_token_ids = {t for t in [pad_token_id, bos_token_id, eos_token_id] if t is not None}

    # 1. Expand source for beams
    src_expanded = np.repeat(src_ids, num_beams, axis=0)

    # 2. Initialise beams
    B_beam = batch_size * num_beams
    current_tokens = np.full((B_beam, 1), bos_token_id, dtype=np.int64)
    
    beam_scores = np.zeros(B_beam, dtype=np.float32)
    # Deactivate all but the first beam per batch sentence
    beam_scores_reshaped = beam_scores.reshape(batch_size, num_beams)
    beam_scores_reshaped[:, 1:] = -1e9

    done = np.zeros(B_beam, dtype=np.bool_)
    all_tokens = current_tokens.copy()
    beam_lengths = np.ones(B_beam, dtype=np.int64)

    for step in range(max_len - 1):
        # Run ONNX session
        logits = session.run(['logits'], {'src': src_expanded, 'tgt': all_tokens})[0]
        # (B*beam, step+1, V) -> last token
        last_logits = logits[:, -1, :].astype(np.float32)
        
        # Log-softmax for beam scores
        # Stability: subtract max
        max_logits = np.max(last_logits, axis=-1, keepdims=True)
        log_probs = last_logits - max_logits - np.log(np.sum(np.exp(last_logits - max_logits), axis=-1, keepdims=True))

        if repetition_penalty != 1.0:
            for i in range(B_beam):
                tokens_to_penalize = [t for t in all_tokens[i] if t not in special_token_ids]
                if tokens_to_penalize:
                    unique_tokens = np.unique(tokens_to_penalize)
                    log_probs[i, unique_tokens] *= repetition_penalty

        # Finished beams: force EOS
        log_probs[done] = -1e9
        log_probs[done, eos_token_id] = 0.0

        # Expand and select top-k across (beam × vocab) per sentence
        next_scores = beam_scores[:, None] + log_probs  # (B*beam, V)
        next_scores = next_scores.reshape(batch_size, num_beams * vocab_size)
        
        # Get top-k indices per sentence
        top_indices = np.argsort(next_scores, axis=-1)[:, ::-1][:, :num_beams]
        top_scores = np.take_along_axis(next_scores, top_indices, axis=-1)

        beam_idx = top_indices // vocab_size    # which old beam
        token_idx = top_indices % vocab_size    # which token

        # Map to global beam indices
        offsets = np.arange(batch_size)[:, None] * num_beams
        global_idx = (offsets + beam_idx).flatten()

        beam_scores = top_scores.flatten()
        current_tokens = token_idx.flatten()[:, None]

        all_tokens = np.concatenate([all_tokens[global_idx], current_tokens], axis=1)
        
        was_done = done[global_idx]
        is_eos = (current_tokens.squeeze(-1) == eos_token_id)
        done = np.logical_or(was_done, is_eos)
        
        beam_lengths = beam_lengths[global_idx]
        beam_lengths[~was_done] += 1

        if np.all(done):
            break

    # Length Penalty
    penalized_scores = beam_scores / (beam_lengths.astype(np.float32) ** length_penalty)
    
    # Pick best beam per sentence
    best_beam_idx = np.argmax(penalized_scores.reshape(batch_size, num_beams), axis=-1)
    best_global = np.arange(batch_size) * num_beams + best_beam_idx

    result = all_tokens[best_global]
    
    if result.shape[1] < max_len:
        pad = np.full((batch_size, max_len - result.shape[1]), pad_token_id, dtype=np.int64)
        result = np.concatenate([result, pad], axis=1)
            
    return result[:, :max_len]
