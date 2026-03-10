import pathlib
import numpy as np
import os

def cut_string_between_bos_eos(string, bos='[CLS]', eos='[SEP]'):
    """Extract string content between BOS and EOS tokens."""
    bos_index = string.find(bos)
    eos_index = string.find(eos)
    if bos_index == -1:
        bos_index = -len(bos)
    if eos_index == -1:
        eos_index = len(string)
    return string[bos_index + len(bos):eos_index].strip()

def resolve_checkpoint_path(this_dir: pathlib.Path, repo_root: pathlib.Path) -> pathlib.Path:
    """Resolve path to model-pretrained.pt checkpoing."""
    candidates = [
        this_dir / 'models' / 'model-pretrained.pt',
        repo_root / 'model_training' / 'models' / 'model-pretrained.pt',
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = '\n'.join(str(path) for path in candidates)
    raise FileNotFoundError(f"Checkpoint not found. Searched:\n{searched}")

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
    return ort.InferenceSession(str(onnx_path), providers=providers)

def beam_generate_onnx(session, src_ids: np.ndarray, bos_token_id: int,
                       eos_token_id: int, pad_token_id: int, max_len: int,
                       num_beams: int = 7, length_penalty: float = 1.0,
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

def calculate_bleu_score(generated_sentence, label):
    """Sentence-level BLEU via sacrebleu (returns 0-1 scale)."""
    try:
        import sacrebleu
        score = sacrebleu.sentence_bleu(generated_sentence, [label])
        return score.score / 100.0
    except ImportError:
        print("sacrebleu not installed. BLEU score calculation unavailable.")
        return 0.0

def calculate_corpus_bleu(generated_sentences, references):
    """Corpus-level BLEU via sacrebleu (returns 0-100 scale)."""
    try:
        import sacrebleu
        score = sacrebleu.corpus_bleu(generated_sentences, [references])
        return score.score
    except ImportError:
        print("sacrebleu not installed. BLEU score calculation unavailable.")
        return 0.0
