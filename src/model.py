'''
Transformer Model from Scratch (only using Linear & Dropout)

Architecture:
    1. Encoder: pre-norm self-attention + feedforward (× N stacks)
    2. Decoder: pre-norm masked self-attn + cross-attn + feedforward (× N stacks)
    3. TransformerModel: encode → decode pipeline

Optimizations over original:
    - Separate encode()/decode() so the encoder runs once during inference
    - KV cache for autoregressive generation (decoder self-attn & cross-attn)
    - register_buffer for positional encoding & causal mask (auto .to(device))
    - Fixed attention scaling: sqrt(head_dim) instead of sqrt(embed_dim)
    - Each sub-layer uses its own dedicated LayerNorm
    - Dropout applied to attention weights (per the paper)
    - bf16 save/load utilities
    - Vectorized beam search (all beams batched as a single tensor)

NOTE: The attention scaling fix (head_dim vs embed_dim) and layer-norm fix
      change model behaviour. Existing checkpoints trained with the old code
      will produce different results. Use strict=False when loading old
      checkpoints (new register_buffer keys won't exist in old state_dicts).
'''

import torch
from torch.nn import Module, ModuleList, Linear, ReLU, Dropout, LayerNorm, Sequential, Embedding
import torch.nn.functional as F
import math


# --------------------------------------------------------------------------- #
#  Multi-Head Attention with KV Cache                                         #
# --------------------------------------------------------------------------- #
class MultiheadAttentionCustom(Module):
    """
    Multi-head scaled dot-product attention.

    Supports two caching modes controlled by ``use_cache`` / ``is_cross_attn``:
      • Self-attention cache  – new K,V are appended to the cache each step.
      • Cross-attention cache – K,V are computed once from encoder output and
                                 reused on every subsequent step.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.Q = Linear(embed_dim, embed_dim, bias=False)
        self.K = Linear(embed_dim, embed_dim, bias=False)
        self.V = Linear(embed_dim, embed_dim, bias=False)

        self.attn_dropout = Dropout(p=dropout)

        # KV cache (populated during inference only)
        self.k_cache = None
        self.v_cache = None

    def clear_cache(self):
        self.k_cache = None
        self.v_cache = None

    def forward(self, query, key, value, pad_mask=None, attn_mask=None,
                use_cache=False, is_cross_attn=False):
        batch_size = query.size(0)

        # Project query → (batch, heads, seq_q, head_dim)
        Q = self.Q(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # ---------- K, V with optional caching ----------
        if use_cache:
            if is_cross_attn and self.k_cache is not None:
                # Cross-attention: encoder K,V never change → reuse cache
                K, V = self.k_cache, self.v_cache
            else:
                K_new = self.K(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
                V_new = self.V(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

                if not is_cross_attn and self.k_cache is not None:
                    # Self-attention: append new token's K,V to history
                    K = torch.cat([self.k_cache, K_new], dim=2)
                    V = torch.cat([self.v_cache, V_new], dim=2)
                else:
                    # First step (or cross-attention first call): initialise cache
                    K, V = K_new, V_new

                self.k_cache = K
                self.v_cache = V
        else:
            # Training path – no caching
            K = self.K(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            V = self.V(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention  (paper: scale by √d_k , d_k = head_dim)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_dim)

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e4)
        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask == 0, -1e4)

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        context = torch.matmul(attn_weights, V)
        # Concatenate heads → (batch, seq, embed_dim)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        return context


# --------------------------------------------------------------------------- #
#  Encoder Layer                                                              #
# --------------------------------------------------------------------------- #
class Encoder(Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout_rate: float, hidden_layer_dim: int):
        super().__init__()

        self.multihead_attention = MultiheadAttentionCustom(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout_rate,
        )
        self.feedforward = Sequential(
            Linear(embed_dim, hidden_layer_dim, bias=True),
            ReLU(),
            Dropout(p=dropout_rate),
            Linear(hidden_layer_dim, embed_dim, bias=True),
            Dropout(p=dropout_rate),
        )

        # Each sub-layer gets its own LayerNorm (pre-norm style)
        self.layer_norm_attn = LayerNorm(embed_dim, eps=1e-05)
        self.layer_norm_ffnn = LayerNorm(embed_dim, eps=1e-05)

    def forward(self, x, pad_mask):
        # Pre-norm self-attention + residual
        normed = self.layer_norm_attn(x)
        attn_out = self.multihead_attention(query=normed, key=normed, value=normed, pad_mask=pad_mask)
        x = x + attn_out

        # Pre-norm feedforward + residual  (FIX: uses layer_norm_ffnn, not layer_norm_attn)
        normed = self.layer_norm_ffnn(x)
        ffnn_out = self.feedforward(normed)
        x = x + ffnn_out
        return x


# --------------------------------------------------------------------------- #
#  Decoder Layer                                                              #
# --------------------------------------------------------------------------- #
class Decoder(Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout_rate: float, hidden_layer_dim: int):
        super().__init__()

        self.masked_multihead_attention = MultiheadAttentionCustom(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout_rate,
        )
        self.multihead_attention = MultiheadAttentionCustom(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout_rate,
        )
        self.feedforward = Sequential(
            Linear(embed_dim, hidden_layer_dim, bias=True),
            ReLU(),
            Dropout(p=dropout_rate),
            Linear(hidden_layer_dim, embed_dim, bias=True),
            Dropout(p=dropout_rate),
        )

        # Dedicated LayerNorm per sub-layer (FIX: previous code reused layer_norm_attn for all three)
        self.layer_norm_mask_attn = LayerNorm(embed_dim, eps=1e-05)
        self.layer_norm_attn = LayerNorm(embed_dim, eps=1e-05)
        self.layer_norm_ffnn = LayerNorm(embed_dim, eps=1e-05)

    def clear_cache(self):
        self.masked_multihead_attention.clear_cache()
        self.multihead_attention.clear_cache()

    def forward(self, x, encoder_output, pad_mask=None, target_mask=None,
                use_cache=False, cross_attn_pad_mask=None):
        # 1. Pre-norm masked self-attention + residual
        normed = self.layer_norm_mask_attn(x)
        masked_attn_out = self.masked_multihead_attention(
            query=normed, key=normed, value=normed,
            pad_mask=pad_mask, attn_mask=target_mask, use_cache=use_cache,
        )
        x = x + masked_attn_out

        # 2. Pre-norm cross-attention + residual
        normed = self.layer_norm_attn(x)
        cross_mask = cross_attn_pad_mask if cross_attn_pad_mask is not None else pad_mask
        attn_out = self.multihead_attention(
            query=normed, key=encoder_output, value=encoder_output,
            pad_mask=cross_mask, use_cache=use_cache, is_cross_attn=True,
        )
        x = x + attn_out

        # 3. Pre-norm feedforward + residual
        normed = self.layer_norm_ffnn(x)
        ffnn_out = self.feedforward(normed)
        x = x + ffnn_out
        return x


# --------------------------------------------------------------------------- #
#  Transformer Model                                                          #
# --------------------------------------------------------------------------- #
class TransformerModel(Module):
    """
    Encoder-Decoder Transformer with:
      • Separate encode / decode paths (encode source once for inference)
      • KV caching for fast autoregressive decoding
      • Registered buffers for PE and causal mask (auto .to(device))
      • bf16 checkpoint save / load
      • Vectorized beam search
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout_rate: float,
                 hidden_layer_dim: int, max_len: int, vocab_size: int, stacks: int):
        super().__init__()

        self.max_len = max_len
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.hidden_layer_dim = hidden_layer_dim
        self.vocab_size = vocab_size
        self.embed_scale = math.sqrt(embed_dim)

        # Embedding layers
        self.token_embedding = Embedding(vocab_size, embed_dim)
        self.token_embedding_dec = Embedding(vocab_size, embed_dim)

        # Encoder / Decoder stacks
        self.encoders = ModuleList([
            Encoder(embed_dim, num_heads, dropout_rate, hidden_layer_dim)
            for _ in range(stacks)
        ])
        self.decoders = ModuleList([
            Decoder(embed_dim, num_heads, dropout_rate, hidden_layer_dim)
            for _ in range(stacks)
        ])

        # Fixed tensors → register_buffer (move with model, saved in state_dict)
        self.register_buffer(
            'positional_encoding',
            self._generate_positional_encoding(max_len, embed_dim),
        )
        self.register_buffer(
            'causal_mask',
            torch.tril(torch.ones(max_len, max_len)),
        )

        # Output head
        self.final_layer_norm = LayerNorm(embed_dim)
        self.output_layer = Linear(embed_dim, vocab_size, bias=True)

        # Decode-position tracker for KV cache (not saved in state_dict)
        self._decode_pos = 0

    # ------------------------------------------------------------------ #
    #  Positional encoding                                                #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _generate_positional_encoding(max_len, embed_dim):
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, max_len, embed_dim)

    # ------------------------------------------------------------------ #
    #  Mask helpers                                                       #
    # ------------------------------------------------------------------ #
    def _make_src_pad_mask(self, src):
        """2D pad mask for encoder self-attention: (B, H, S, S)."""
        mask = (src != 0)  # (B, S)
        col_mask = mask.unsqueeze(1).unsqueeze(2)   # (B, 1, 1, S)
        row_mask = mask.unsqueeze(1).unsqueeze(3)   # (B, 1, S, 1)
        return (col_mask & row_mask).expand(-1, self.num_heads, -1, -1)

    def _make_tgt_pad_mask(self, tgt):
        """Key-side pad mask for decoder self-attention: (B, H, 1, T)."""
        mask = (tgt != 0).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
        return mask.expand(-1, self.num_heads, -1, -1)

    def _make_cross_attn_pad_mask(self, src):
        """Source key-side pad mask for cross-attention: (B, H, 1, S)."""
        mask = (src != 0).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)
        return mask.expand(-1, self.num_heads, -1, -1)

    # ------------------------------------------------------------------ #
    #  Encode / Decode  (separable for efficient inference)               #
    # ------------------------------------------------------------------ #
    def encode(self, src):
        """Encode source tokens → encoder hidden states (B, S, E)."""
        pad_mask = self._make_src_pad_mask(src)
        x = self.token_embedding(src) * self.embed_scale
        x = x + self.positional_encoding[:, :src.size(1)]
        for encoder in self.encoders:
            x = encoder(x, pad_mask)
        return x

    def decode(self, tgt, encoder_output, tgt_pad_mask=None,
               cross_attn_pad_mask=None, use_cache=False):
        """
        Decode target tokens given encoder output.

        Training  → full tgt sequence, use_cache=False
        Inference → single new token per step, use_cache=True
        """
        seq_len = tgt.size(1)
        x = self.token_embedding_dec(tgt) * self.embed_scale

        # Positional encoding (track cumulative position when caching)
        if use_cache:
            x = x + self.positional_encoding[:, self._decode_pos:self._decode_pos + seq_len]
            self._decode_pos += seq_len
        else:
            x = x + self.positional_encoding[:, :seq_len]

        # Causal mask: unnecessary when using KV cache with a single token
        target_mask = (
            None if (use_cache and seq_len == 1)
            else self.causal_mask[:seq_len, :seq_len]
        )

        for decoder in self.decoders:
            x = decoder(
                x, encoder_output,
                pad_mask=tgt_pad_mask, target_mask=target_mask,
                use_cache=use_cache, cross_attn_pad_mask=cross_attn_pad_mask,
            )

        x = self.final_layer_norm(x)
        return self.output_layer(x)

    def clear_cache(self):
        """Reset all decoder KV caches and decode position counter."""
        for decoder in self.decoders:
            decoder.clear_cache()
        self._decode_pos = 0

    # ------------------------------------------------------------------ #
    #  Combined forward  (training)                                       #
    # ------------------------------------------------------------------ #
    def forward(self, src, tgt):
        """Full forward pass for teacher-forced training (no caching)."""
        tgt_pad_mask = self._make_tgt_pad_mask(tgt)
        cross_attn_pad_mask = self._make_cross_attn_pad_mask(src)

        encoder_output = self.encode(src)
        return self.decode(
            tgt, encoder_output,
            tgt_pad_mask=tgt_pad_mask,
            cross_attn_pad_mask=cross_attn_pad_mask,
            use_cache=False,
        )

    # ------------------------------------------------------------------ #
    #  bf16 save / load                                                   #
    # ------------------------------------------------------------------ #
    @staticmethod
    def save_bf16(model, path):
        """Save model checkpoint in bfloat16 (≈50 % smaller file)."""
        state_dict = {
            k: v.to(torch.bfloat16) if v.is_floating_point() else v
            for k, v in model.state_dict().items()
        }
        torch.save(state_dict, path)

    @staticmethod
    def load_bf16(model, path, device='cpu'):
        """Load a bf16 checkpoint, casting back to float32 for training."""
        state_dict = torch.load(path, map_location=device, weights_only=True)
        state_dict = {
            k: v.to(torch.float32) if v.is_floating_point() else v
            for k, v in state_dict.items()
        }
        model.load_state_dict(state_dict, strict=False)
        return model

    # ------------------------------------------------------------------ #
    #  Greedy generation with KV cache                                    #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def greedy_generate(self, src, bos_token_id, eos_token_id, max_len=None):
        """
        Fast greedy decoding: encode once → decode token-by-token with KV cache.

        Returns: (batch, ≤max_len) token ids including BOS, padded with 0.
        """
        if max_len is None:
            max_len = self.max_len

        self.eval()
        device = src.device
        batch_size = src.size(0)

        # Encode source ONCE
        encoder_output = self.encode(src)
        cross_mask = self._make_cross_attn_pad_mask(src)

        # Initialise with BOS
        current_token = torch.full((batch_size, 1), bos_token_id,
                                   dtype=torch.long, device=device)
        generated = [current_token]
        done = torch.zeros(batch_size, dtype=torch.bool, device=device)

        self.clear_cache()

        for _ in range(max_len - 1):
            logits = self.decode(current_token, encoder_output,
                                 cross_attn_pad_mask=cross_mask, use_cache=True)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

            # Pad finished sequences with 0
            next_token = next_token.masked_fill(done.unsqueeze(-1), 0)
            generated.append(next_token)

            done = done | (next_token.squeeze(-1) == eos_token_id)
            if done.all():
                break

            current_token = next_token

        self.clear_cache()

        result = torch.cat(generated, dim=1)
        # Pad to max_len
        if result.size(1) < max_len:
            pad = torch.zeros(batch_size, max_len - result.size(1),
                              dtype=torch.long, device=device)
            result = torch.cat([result, pad], dim=1)
        return result[:, :max_len]

    # ------------------------------------------------------------------ #
    #  Vectorized beam search with KV cache                               #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def beam_search(self, src, bos_token_id, eos_token_id,
                    num_beams=5, max_len=None):
        """
        Batch-vectorized beam search.

        All beams for all sentences are decoded as a single
        (batch_size × num_beams, seq) tensor – no Python loops over beams.

        Returns: (batch, ≤max_len) token ids of the best beam per sentence.
        """
        if max_len is None:
            max_len = self.max_len

        self.eval()
        device = src.device
        batch_size = src.size(0)

        # 1. Encode source once, then expand for beams
        encoder_output = self.encode(src)                                    # (B, S, E)
        cross_mask = self._make_cross_attn_pad_mask(src)                     # (B, H, 1, S)
        encoder_output = encoder_output.repeat_interleave(num_beams, dim=0)  # (B*beam, S, E)
        cross_mask = cross_mask.repeat_interleave(num_beams, dim=0)

        # 2. Initialise beams
        B_beam = batch_size * num_beams
        current_token = torch.full((B_beam, 1), bos_token_id,
                                   dtype=torch.long, device=device)
        beam_scores = torch.zeros(B_beam, device=device)
        # Deactivate all but the first beam per batch sentence
        beam_scores.view(batch_size, num_beams)[:, 1:] = -1e9

        done = torch.zeros(B_beam, dtype=torch.bool, device=device)
        all_tokens = current_token.clone()

        self.clear_cache()

        for step in range(max_len - 1):
            logits = self.decode(current_token, encoder_output,
                                 cross_attn_pad_mask=cross_mask, use_cache=True)
            log_probs = F.log_softmax(logits[:, -1, :], dim=-1)    # (B*beam, V)
            vocab_size = log_probs.size(-1)

            # Finished beams: force EOS to preserve score, -inf elsewhere
            log_probs[done] = -1e9
            log_probs[done, eos_token_id] = 0.0

            # Expand and select top-k across (beam × vocab) per sentence
            next_scores = beam_scores.unsqueeze(-1) + log_probs            # (B*beam, V)
            next_scores = next_scores.view(batch_size, num_beams * vocab_size)
            top_scores, top_indices = next_scores.topk(num_beams, dim=-1)  # (B, beam)

            beam_idx = top_indices // vocab_size    # which old beam
            token_idx = top_indices % vocab_size    # which token

            # Map to global (flattened) beam indices
            offsets = torch.arange(batch_size, device=device).unsqueeze(-1) * num_beams
            global_idx = (offsets + beam_idx).view(-1)

            # Re-order KV caches to match selected beams
            self._reorder_cache(global_idx)

            beam_scores = top_scores.view(-1)
            current_token = token_idx.view(-1, 1)

            all_tokens = torch.cat([all_tokens[global_idx], current_token], dim=1)
            done = done[global_idx] | (current_token.squeeze(-1) == eos_token_id)

            if done.all():
                break

        self.clear_cache()

        # Pick best beam per sentence
        best_beam = beam_scores.view(batch_size, num_beams).argmax(dim=-1)
        batch_idx = torch.arange(batch_size, device=device)
        best_global = batch_idx * num_beams + best_beam

        result = all_tokens[best_global]
        if result.size(1) < max_len:
            pad = torch.zeros(batch_size, max_len - result.size(1),
                              dtype=torch.long, device=device)
            result = torch.cat([result, pad], dim=1)
        return result[:, :max_len]

    def _reorder_cache(self, beam_indices):
        """Re-order all decoder KV caches to match beam selection."""
        for decoder in self.decoders:
            for attn in [decoder.masked_multihead_attention, decoder.multihead_attention]:
                if attn.k_cache is not None:
                    attn.k_cache = attn.k_cache[beam_indices]
                    attn.v_cache = attn.v_cache[beam_indices]

        

    