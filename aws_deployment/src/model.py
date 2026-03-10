'''
Transformer Model from Scratch (only using Linear, Dropout)

Architecture:
    1. Encoder: pre-norm self-attention + feedforward (× N stacks)
    2. Decoder: pre-norm masked self-attn + cross-attn + feedforward (× N stacks)
    3. TransformerModel: encode -> decode pipeline
'''

import torch
from torch.nn import Module, ModuleList, Linear, Dropout, Embedding
import torch.nn.functional as F
import math

class RMSNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + self.eps) * self.weight


class SwiGLU(Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.w1 = Linear(in_dim, hidden_dim, bias=True)
        self.w2 = Linear(in_dim, hidden_dim, bias=True)
        self.w3 = Linear(hidden_dim, in_dim, bias=True)
        self.dropout = Dropout(p=dropout)

    def forward(self, x):
        gated = F.silu(self.w1(x)) * self.w2(x)
        return self.dropout(self.w3(gated))

class FeedForward(Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        # 파이토치 내장 Sequential을 사용하여 직관적으로 구성
        self.net = torch.nn.Sequential(
            Linear(in_dim, hidden_dim),
            torch.nn.GELU(),
            Dropout(p=dropout),
            Linear(hidden_dim, in_dim),
            Dropout(p=dropout)
        )

    def forward(self, x):
        return self.net(x)

# Multi-Head Attention from Scratch (with caching)
class MultiheadAttentionCustom(Module):

    def __init__(self, embed_dim: int, num_heads: int, num_kv_heads: int,
                 dropout: float, max_len: int, rope_base: float = 10000.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.kv_group_size = num_heads // num_kv_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.rope_base = rope_base
        self.max_len = max_len

        self.Q = Linear(embed_dim, embed_dim, bias=False)
        self.K = Linear(embed_dim, num_kv_heads * self.head_dim, bias=False)
        self.V = Linear(embed_dim, num_kv_heads * self.head_dim, bias=False)

        self.out_proj = Linear(embed_dim, embed_dim, bias=True)

        # RoPE tables are precomputed once and stored as non-persistent buffers.
        inv_freq = 1.0 / (self.rope_base ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))
        positions = torch.arange(self.max_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)  # (max_len, head_dim/2)
        self.register_buffer("rope_cos", freqs.cos().view(1, self.max_len, 1, -1), persistent=False)
        self.register_buffer("rope_sin", freqs.sin().view(1, self.max_len, 1, -1), persistent=False)

        # KV cache (populated during inference only)
        self.k_cache = None
        self.v_cache = None

    def clear_cache(self):
        self.k_cache = None
        self.v_cache = None

    def _apply_rope(self, x, position_ids):
        # x: (B, L, H, D)
        if x.size(-1) % 2 != 0:
            return x

        seq_len = x.size(1)

        if position_ids is None:
            # Fast path: use slicing over precomputed RoPE tables.
            if seq_len <= self.max_len:
                cos = self.rope_cos[:, :seq_len, :, :].to(dtype=x.dtype, device=x.device)
                sin = self.rope_sin[:, :seq_len, :, :].to(dtype=x.dtype, device=x.device)
            else:
                # Fallback for sequence lengths beyond configured max_len.
                inv_freq = 1.0 / (self.rope_base ** (torch.arange(0, self.head_dim, 2, device=x.device, dtype=torch.float32) / self.head_dim))
                pos = torch.arange(seq_len, device=x.device)
                freqs = torch.outer(pos.to(torch.float32), inv_freq)
                cos = freqs.cos().view(1, seq_len, 1, -1).to(dtype=x.dtype)
                sin = freqs.sin().view(1, seq_len, 1, -1).to(dtype=x.dtype)
        else:
            max_pos = int(position_ids.max().item()) if position_ids.numel() > 0 else -1
            if position_ids.dim() == 1 and max_pos < self.max_len:
                if torch.equal(position_ids, torch.arange(seq_len, device=position_ids.device)):
                    cos = self.rope_cos[:, :seq_len, :, :].to(dtype=x.dtype, device=x.device)
                    sin = self.rope_sin[:, :seq_len, :, :].to(dtype=x.dtype, device=x.device)
                else:
                    cos = self.rope_cos[0, position_ids, 0, :].view(1, seq_len, 1, -1).to(dtype=x.dtype, device=x.device)
                    sin = self.rope_sin[0, position_ids, 0, :].view(1, seq_len, 1, -1).to(dtype=x.dtype, device=x.device)
            else:
                inv_freq = 1.0 / (self.rope_base ** (torch.arange(0, self.head_dim, 2, device=x.device, dtype=torch.float32) / self.head_dim))
                if position_ids.dim() == 1:
                    freqs = torch.outer(position_ids.to(torch.float32), inv_freq)
                    cos = freqs.cos().view(1, seq_len, 1, -1).to(dtype=x.dtype)
                    sin = freqs.sin().view(1, seq_len, 1, -1).to(dtype=x.dtype)
                else:
                    freqs = position_ids.to(torch.float32).unsqueeze(-1) * inv_freq
                    cos = freqs.cos().to(dtype=x.dtype).unsqueeze(2)
                    sin = freqs.sin().to(dtype=x.dtype).unsqueeze(2)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd = x_even * sin + x_odd * cos

        out = torch.empty_like(x)
        out[..., 0::2] = x_rot_even
        out[..., 1::2] = x_rot_odd
        return out

    def _expand_kv(self, x):
        # x: (B, H_kv, L, D) -> (B, H_q, L, D)
        if self.num_kv_heads == self.num_heads:
            return x
        B, H_kv, L, D = x.size()
        x = x.unsqueeze(2).expand(B, H_kv, self.kv_group_size, L, D)
        return x.reshape(B, self.num_heads, L, D)

    def _to_keep_mask(self, mask, device):
        if mask is None:
            return None
        if mask.dtype != torch.bool:
            mask = mask != 0
        mask = mask.to(device=device)
        if mask.dim() == 2:
            return mask.unsqueeze(0).unsqueeze(0)
        if mask.dim() == 3:
            return mask.unsqueeze(1)
        return mask

    def _merge_masks(self, pad_mask, attn_mask, q_len, device, dtype):
        keep = None

        attn_keep = self._to_keep_mask(attn_mask, device)
        if attn_keep is not None:
            keep = attn_keep

        pad_keep = self._to_keep_mask(pad_mask, device)
        if pad_keep is not None:
            keep = pad_keep if keep is None else (keep & pad_keep)

        if keep is None:
            return None

        if keep.size(-2) == 1 and q_len > 1:
            keep = keep.expand(-1, -1, q_len, -1)

        return keep

    def forward(self, query, key, value, pad_mask=None, attn_mask=None,
                use_cache=False, is_cross_attn=False,
                query_position_ids=None, key_position_ids=None):
        batch_size = query.size(0)

        q_len = query.size(1)
        Q = self.Q(query).view(batch_size, q_len, self.num_heads, self.head_dim)
        if not is_cross_attn:
            Q = self._apply_rope(Q, query_position_ids)
        Q = Q.transpose(1, 2)

        # KV calculation (stored in KV-head space for cache efficiency)
        if use_cache:
            if is_cross_attn and self.k_cache is not None:
                K_kv, V_kv = self.k_cache, self.v_cache
            else:
                kv_len = key.size(1)
                K_new = self.K(key).view(batch_size, kv_len, self.num_kv_heads, self.head_dim)
                V_new = self.V(value).view(batch_size, kv_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
                if not is_cross_attn:
                    K_new = self._apply_rope(K_new, key_position_ids)
                K_new = K_new.transpose(1, 2)

                if not is_cross_attn and self.k_cache is not None:
                    K_kv = torch.cat([self.k_cache, K_new], dim=2)
                    V_kv = torch.cat([self.v_cache, V_new], dim=2)
                else:
                    K_kv, V_kv = K_new, V_new

                self.k_cache = K_kv
                self.v_cache = V_kv
        else:
            kv_len = key.size(1)
            K_kv = self.K(key).view(batch_size, kv_len, self.num_kv_heads, self.head_dim)
            V_kv = self.V(value).view(batch_size, kv_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            if not is_cross_attn:
                K_kv = self._apply_rope(K_kv, key_position_ids)
            K_kv = K_kv.transpose(1, 2)

        K = self._expand_kv(K_kv)
        V = self._expand_kv(V_kv)

        attn_bias = self._merge_masks(
            pad_mask=pad_mask,
            attn_mask=attn_mask,
            q_len=q_len,
            device=query.device,
            dtype=Q.dtype,
        )

        # Flash Attention via PyTorch SDPA kernel when available.
        context = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_bias,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        return self.out_proj(context)

#  Encoder Layer                                                              
class Encoder(Module):
    def __init__(self, embed_dim: int, num_heads: int, num_kv_heads: int,
                 dropout_rate: float, hidden_layer_dim: int, max_len: int, rope_base: float):
        super().__init__()

        self.multihead_attention = MultiheadAttentionCustom(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout=dropout_rate,
            max_len=max_len,
            rope_base=rope_base,
        )
        self.feedforward = SwiGLU(embed_dim, hidden_layer_dim, dropout_rate)

        # Each sub-layer gets its own RMSNorm (pre-norm style)
        self.layer_norm_attn = RMSNorm(embed_dim)
        self.layer_norm_ffnn = RMSNorm(embed_dim)

    def forward(self, x, pad_mask, position_ids):
        # Pre-norm self-attention + residual
        normed = self.layer_norm_attn(x)
        attn_out = self.multihead_attention(
            query=normed,
            key=normed,
            value=normed,
            pad_mask=pad_mask,
            query_position_ids=position_ids,
            key_position_ids=position_ids,
        )
        x = x + attn_out

        # Pre-norm feedforward + residual
        normed = self.layer_norm_ffnn(x)
        ffnn_out = self.feedforward(normed)
        x = x + ffnn_out
        return x

#  Decoder Layer
class Decoder(Module):
    def __init__(self, embed_dim: int, num_heads: int, num_kv_heads: int,
                 dropout_rate: float, hidden_layer_dim: int, max_len: int, rope_base: float):
        super().__init__()

        self.masked_multihead_attention = MultiheadAttentionCustom(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout=dropout_rate,
            max_len=max_len,
            rope_base=rope_base,
        )
        self.multihead_attention = MultiheadAttentionCustom(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout=dropout_rate,
            max_len=max_len,
            rope_base=rope_base,
        )
        self.feedforward = SwiGLU(embed_dim, hidden_layer_dim, dropout_rate)

        # Dedicated RMSNorm per sub-layer
        self.layer_norm_mask_attn = RMSNorm(embed_dim)
        self.layer_norm_attn = RMSNorm(embed_dim)
        self.layer_norm_ffnn = RMSNorm(embed_dim)

    def clear_cache(self):
        self.masked_multihead_attention.clear_cache()
        self.multihead_attention.clear_cache()

    def forward(self, x, encoder_output, pad_mask=None, target_mask=None,
                use_cache=False, cross_attn_pad_mask=None,
                tgt_position_ids=None, src_position_ids=None):
        # 1. Pre-norm masked self-attention + residual
        normed = self.layer_norm_mask_attn(x)
        masked_attn_out = self.masked_multihead_attention(
            query=normed, key=normed, value=normed,
            pad_mask=pad_mask, attn_mask=target_mask, use_cache=use_cache,
            query_position_ids=tgt_position_ids,
            key_position_ids=tgt_position_ids,
        )
        x = x + masked_attn_out

        # 2. Pre-norm cross-attention + residual
        if cross_attn_pad_mask is None:
            raise ValueError("cross_attn_pad_mask must be provided for cross-attention.")
        normed = self.layer_norm_attn(x)
        attn_out = self.multihead_attention(
            query=normed, key=encoder_output, value=encoder_output,
            pad_mask=cross_attn_pad_mask, use_cache=use_cache, is_cross_attn=True,
            query_position_ids=tgt_position_ids,
            key_position_ids=src_position_ids,
        )
        x = x + attn_out

        # 3. Pre-norm feedforward + residual
        normed = self.layer_norm_ffnn(x)
        ffnn_out = self.feedforward(normed)
        x = x + ffnn_out
        return x

#  Transformer Model
class TransformerModel(Module):
    """
    Encoder-Decoder Transformer with:
      • Separate encode / decode paths (encode source once for inference)
      • KV caching for fast autoregressive decoding
      • Registered buffers for PE and causal mask (auto .to(device))
      • bf16 checkpoint save / load
      • Vectorized beam search
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout_rate: float, tokenizer=None,
                 hidden_layer_dim: int = 2048, max_len: int = 128, vocab_size: int = 32000,
                 stacks: int = 6, pad_token_id: int = 0, num_kv_heads: int = 0,
                 rope_base: float = 10000.0):
        super().__init__()

        self.max_len = max_len
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.hidden_layer_dim = hidden_layer_dim
        self.vocab_size = vocab_size
        self.embed_scale = math.sqrt(embed_dim)
        if num_kv_heads is None or num_kv_heads == 0:
            num_kv_heads = num_heads // 2 if (num_heads % 2 == 0) else num_heads
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_kv_heads = num_kv_heads

        self.pad_token_id = pad_token_id

        self.tokenizer = tokenizer
        # Embedding layers
        self.token_embedding = Embedding(vocab_size, embed_dim)
        self.token_embedding_dec = self.token_embedding  # Share source and target token embeddings (optional)

        # Encoder / Decoder stacks
        self.encoders = ModuleList([
            Encoder(embed_dim, num_heads, num_kv_heads, dropout_rate, hidden_layer_dim, max_len, rope_base)
            for _ in range(stacks)
        ])
        self.decoders = ModuleList([
            Decoder(embed_dim, num_heads, num_kv_heads, dropout_rate, hidden_layer_dim, max_len, rope_base)
            for _ in range(stacks)
        ])
        self.register_buffer(
            'causal_mask',
            torch.tril(torch.ones(max_len, max_len, dtype=torch.bool)),
        )

        # Output head
        self.final_layer_norm = RMSNorm(embed_dim)
        self.output_layer = Linear(embed_dim, vocab_size, bias=True)
        self.output_layer.weight = self.token_embedding.weight

        # Decode-position tracker for KV cache (not saved in state_dict)
        self._decode_pos = 0

        torch.nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        if self.output_layer.bias is not None:
            torch.nn.init.zeros_(self.output_layer.bias)

    #  Mask helpers
    def _make_src_pad_mask(self, src):
        return (src != self.pad_token_id).unsqueeze(1).unsqueeze(2)
    
    def _make_tgt_pad_mask(self, tgt):
        return (tgt != self.pad_token_id).unsqueeze(1).unsqueeze(2)
    
    def _make_cross_attn_pad_mask(self, src):
        return (src != self.pad_token_id).unsqueeze(1).unsqueeze(2)

    #  Encode / Decode  (separable for reusable source encoding during inference)
    def encode(self, src):
        """Encode source tokens → encoder hidden states (B, S, E)."""
        pad_mask = self._make_src_pad_mask(src)
        x = self.token_embedding(src) * self.embed_scale

        src_position_ids = torch.arange(src.size(1), device=src.device)
        for encoder in self.encoders:
            x = encoder(x, pad_mask, src_position_ids)
        return x

    def decode(self, tgt, encoder_output, tgt_pad_mask=None,
               cross_attn_pad_mask=None, use_cache=False):
        seq_len = tgt.size(1)
        x = self.token_embedding_dec(tgt) * self.embed_scale

        if use_cache:
            tgt_position_ids = torch.arange(
                self._decode_pos,
                self._decode_pos + seq_len,
                device=tgt.device,
            )
            self._decode_pos += seq_len
        else:
            tgt_position_ids = torch.arange(seq_len, device=tgt.device)

        src_position_ids = torch.arange(encoder_output.size(1), device=tgt.device)

        # Causal mask
        target_mask = (
            None if (use_cache and seq_len == 1)
            else self.causal_mask[:seq_len, :seq_len]
        )

        for decoder in self.decoders:
            x = decoder(
                x, encoder_output,
                pad_mask=tgt_pad_mask, target_mask=target_mask,
                use_cache=use_cache, cross_attn_pad_mask=cross_attn_pad_mask,
                tgt_position_ids=tgt_position_ids,
                src_position_ids=src_position_ids,
            )

        x = self.final_layer_norm(x)
        return self.output_layer(x)

    def clear_cache(self):
        for decoder in self.decoders:
            decoder.clear_cache()
        self._decode_pos = 0

    #  Combined forward  (training)
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

    #  bf16 save / load
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

    #  Greedy generation with reusable encoding
    @torch.no_grad()
    def greedy_generate(self, src, bos_token_id, eos_token_id, max_len=None):

        if max_len is None:
            max_len = self.max_len

        self.eval()
        device = src.device
        batch_size = src.size(0)

        encoder_output = self.encode(src)
        cross_mask = self._make_cross_attn_pad_mask(src)

        current_token = torch.full((batch_size, 1), bos_token_id,
                                   dtype=torch.long, device=device)
        generated = [current_token]
        done = torch.zeros(batch_size, dtype=torch.bool, device=device)

        self.clear_cache()

        for _ in range(max_len - 1):

            logits = self.decode(current_token, encoder_output,
                                 cross_attn_pad_mask=cross_mask, use_cache=True)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            next_token = next_token.masked_fill(done.unsqueeze(-1), self.pad_token_id)
            generated.append(next_token)
            done = done | (next_token.squeeze(-1) == eos_token_id)

            if done.all():
                break

            current_token = next_token

        result = torch.cat(generated, dim=1)

        if result.size(1) < max_len:
            pad = torch.full((batch_size, max_len - result.size(1)), self.pad_token_id,
                             dtype=torch.long, device=device)
            result = torch.cat([result, pad], dim=1)

        return result[:, :max_len]

    # Vectorized beam search with Length Penalty & KV cache
    # Vibe-Coded
    @torch.no_grad()
    def beam_generate(self, src, num_beams=7, max_len=None, 
                      length_penalty=1.0, repetition_penalty = 1.3):
        """
        Batch-vectorized beam search with Length Penalty.
        Returns: (batch, ≤max_len) token ids of the best beam per sentence.
        """

        if self.tokenizer is None:
            raise ValueError("tokenizer is required for beam_generate.")

        if max_len is None:
            max_len = self.max_len

        if isinstance(src, str):
            encoded = self.tokenizer(src, padding='max_length',
                                truncation=True, max_length=max_len)
            src = torch.as_tensor(
                encoded['input_ids'], dtype=torch.long, device=self.token_embedding.weight.device,
            ).unsqueeze(0)

        self.eval()
        device = self.token_embedding.weight.device
        batch_size = src.size(0)
        bos_token_id = self.tokenizer.bos_token_id
        eos_token_id = self.tokenizer.eos_token_id
        special_token_ids = {
            t for t in [
                self.pad_token_id,
                self.tokenizer.bos_token_id,
                self.tokenizer.eos_token_id,
                self.tokenizer.cls_token_id,
                self.tokenizer.sep_token_id,
            ] if t is not None
        }

        # 1. Encode source once, then expand for beams
        encoder_output = self.encode(src) # (B, S, E)
        cross_mask = self._make_cross_attn_pad_mask(src) # (B, H, 1, S)
        encoder_output = encoder_output.repeat_interleave(num_beams, dim=0) # (B*beam, S, E)
        cross_mask = cross_mask.repeat_interleave(num_beams, dim=0)

        # 2. Initialise beams
        B_beam = batch_size * num_beams
        current_token = torch.full((B_beam, 1), bos_token_id,
                                   dtype=torch.long, device=device)
        
        beam_scores = torch.zeros(B_beam, device=device)
        # Deactivate all but the first beam per batch sentence to prevent duplicates
        beam_scores.view(batch_size, num_beams)[:, 1:] = -1e9

        done = torch.zeros(B_beam, dtype=torch.bool, device=device)
        all_tokens = current_token.clone()
        
        # 실제 생성된 길이를 추적 (Length Penalty 계산용)
        beam_lengths = torch.ones(B_beam, dtype=torch.long, device=device)

        self.clear_cache()

        for step in range(max_len - 1):
            logits = self.decode(current_token, encoder_output,
                                 cross_attn_pad_mask=cross_mask, use_cache=True)
            log_probs = F.log_softmax(logits[:, -1, :], dim=-1)    # (B*beam, V)
            
            if repetition_penalty != 1.0:
                prev_token_log_probs = torch.gather(log_probs, 1, all_tokens)
                penalized_log_probs = prev_token_log_probs * repetition_penalty
                if special_token_ids:
                    special_mask = torch.zeros_like(all_tokens, dtype=torch.bool)
                    for token_id in special_token_ids:
                        special_mask |= (all_tokens == token_id)
                    penalized_log_probs = torch.where(special_mask, prev_token_log_probs, penalized_log_probs)
                log_probs.scatter_(1, all_tokens, penalized_log_probs)
            
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
            
            was_done = done[global_idx]
            is_eos = (current_token.squeeze(-1) == eos_token_id)
            done = was_done | is_eos
            
            beam_lengths = beam_lengths[global_idx]
            beam_lengths[~was_done] += 1

            if done.all():
                break

        self.clear_cache()

        # Length Penalty
        penalized_scores = beam_scores / (beam_lengths.float() ** length_penalty)
        
        # Pick best beam per sentence
        best_beam = penalized_scores.view(batch_size, num_beams).argmax(dim=-1)
        batch_idx = torch.arange(batch_size, device=device)
        best_global = batch_idx * num_beams + best_beam

        result = all_tokens[best_global]
        
        if result.size(1) < max_len:
            pad = torch.full((batch_size, max_len - result.size(1)), self.pad_token_id,
                             dtype=torch.long, device=device)
            result = torch.cat([result, pad], dim=1)
            
        return result[:, :max_len]

    def _reorder_cache(self, beam_indices):
        for decoder in self.decoders:
            for attn in [decoder.masked_multihead_attention, decoder.multihead_attention]:
                if attn.k_cache is not None:
                    attn.k_cache = attn.k_cache[beam_indices]
                    attn.v_cache = attn.v_cache[beam_indices]