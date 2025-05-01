
'''
Transformer Model from Scratch (only using Linear & Dropout)
Composition:

1. Model Class
    1) Encoder Class
        a. input embedding
        b. positional encoding
        c. multi-head attention
        d. feedforward
        e. normalization
    2) Decoder Class
        a. input embedding
        b. positional encoding
        c. masked multi-head attention
        d. multi-head attention
        e. feedforward
        f. normalization
    3) Transformer Class
        a. integrate encoders and decoders
'''

import torch
from torch.nn import Module, ModuleList, Linear, ReLU, Dropout, LayerNorm, Sequential, Embedding
import torch.nn.functional as F
import math

# Grouped Attention for K, V is used (multi query attention)
class MultiheadAttentionCustom(Module):
    def __init__(self, embed_dim:int, num_heads:int, dropout:float, batch_first:bool=True):
        super(MultiheadAttentionCustom, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Calculate all heads at once
        self.Q = Linear(in_features=embed_dim, out_features=embed_dim, bias=False)
        self.K = Linear(in_features=embed_dim, out_features=embed_dim, bias=False)
        self.V = Linear(in_features=embed_dim, out_features=embed_dim, bias=False)

        self.k_cache = None
        self.v_cache = None
        
        # simple dropout layer
        self.dropout = Dropout(p=dropout)

    def forward(self, query, key, value, pad_mask, attn_mask=None):

        # logger.debug('Query shape: {}, Key shape: {}, Value shape: {}'.format(query.shape, key.shape, value.shape))
        # logger.debug('Q shape: {}, K shape: {}, V shape: {}'.format(self.Q.shape, self.K.shape, self.V.shape))

        Q = self.Q(query)
        K = self.K(key)
        V = self.V(value)

        # logger.debug('Q shape: {}'.format(Q.shape))

        # IMPORTANT: shape matters but it's not everything. should view to incomplete dimension
        # and then transpose again to completely isolate the heads!

        Q = Q.view(query.size(0), query.size(1), self.num_heads, \
                   self.embed_dim // self.num_heads).transpose(1, 2)
        K = K.view(key.size(0), key.size(1), self.num_heads, \
                   self.embed_dim // self.num_heads).transpose(1, 2)
        V = V.view(value.size(0), value.size(1), self.num_heads, \
                   self.embed_dim // self.num_heads).transpose(1, 2)

        
        # Number of simple multiplications: (len_input)^2 * (len_heads)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.embed_dim)

        # Mask the elements that are 0 in the lookahead mask
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e4)
        scores = scores.masked_fill(pad_mask == 0, -1e4)
        
        scores = torch.softmax(scores, dim=-1)
        scores = torch.matmul(scores, V)

        context = scores.transpose(1, 2).contiguous().view(scores.size(0), -1, self.embed_dim)

        return context
        

class Encoder(Module):
    def __init__(self, embed_dim:int, num_heads:int, dropout_rate:float, hidden_layer_dim:int):
        super(Encoder, self).__init__()

        # Set hyperparams
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.hidden_layer_dim = hidden_layer_dim

        # Multi-head attention layer
        self.multihead_attention = \
        MultiheadAttentionCustom(
            embed_dim=self.embed_dim, 
            num_heads=self.num_heads, 
            dropout=self.dropout_rate,
            batch_first=True
        )

        # Feedforward network: use relu and dropout
        self.feedforward = \
        Sequential(
            Linear(in_features=self.embed_dim, out_features=self.hidden_layer_dim, bias=True),
            ReLU(),
            Dropout(p=self.dropout_rate),
            Linear(in_features=self.hidden_layer_dim, out_features=self.embed_dim, bias=True),
            Dropout(p=self.dropout_rate)
        )

        # Need two layer norms for each normalization
        self.layer_norm_attn = LayerNorm(normalized_shape=self.embed_dim, eps=1e-05)
        self.layer_norm_ffnn = LayerNorm(normalized_shape=self.embed_dim, eps=1e-05)

    def forward(self, x, pad_mask):
        # logger.debug('Input shape: {}'.format(x.shape))
        x = self.layer_norm_attn(x)
        attn_output = self.multihead_attention(query=x, key=x, value=x, pad_mask=pad_mask)
        # logger.debug('Attn output shape: {}'.format(attn_output.shape))
        attn_output = x + attn_output

        attn_output = self.layer_norm_attn(attn_output)
        ffnn_output = self.feedforward(attn_output)
        # logger.debug('FFNN output shape: {}'.format(ffnn_output.shape))
        ffnn_output = ffnn_output + attn_output
        return ffnn_output
        
    
class Decoder(Module):
    def __init__(self, embed_dim:int, num_heads:int, dropout_rate:float, hidden_layer_dim:int):
        super(Decoder, self).__init__()

        # Set Hyperparameters
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.hidden_layer_dim = hidden_layer_dim

        # Initialize layers
        self.masked_multihead_attention = \
        MultiheadAttentionCustom(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=self.dropout_rate,
            batch_first=True
        )

        self.multihead_attention = \
        MultiheadAttentionCustom(
            embed_dim=self.embed_dim, 
            num_heads=self.num_heads, 
            dropout=self.dropout_rate, 
            batch_first=True
        )

        self.feedforward = \
        Sequential(
            Linear(in_features=self.embed_dim, out_features=self.hidden_layer_dim, bias=True), 
            ReLU(),
            Dropout(p=self.dropout_rate),
            Linear(in_features=self.hidden_layer_dim, out_features=self.embed_dim, bias=True),
            Dropout(p=self.dropout_rate)
        )

        self.layer_norm_mask_attn = LayerNorm(normalized_shape=self.embed_dim, eps=1e-05)
        self.layer_norm_attn = LayerNorm(normalized_shape=self.embed_dim, eps=1e-05)
        self.layer_norm_ffnn = LayerNorm(normalized_shape=self.embed_dim, eps=1e-05)

        self.k_cache = None
        self.v_cache = None
        
    def forward(self, x, encoder_output, pad_mask, target_mask):

        # Apply masking to the first attention layer
        x = self.layer_norm_attn(x)
        masked_attn_output = self.masked_multihead_attention(query=x, key=x, value=x, pad_mask=pad_mask, attn_mask=target_mask)
        masked_attn_output = x + masked_attn_output
        
        masked_attn_output = self.layer_norm_attn(masked_attn_output)
        attn_output = self.multihead_attention(query=masked_attn_output, key=encoder_output, value=encoder_output, pad_mask=pad_mask)

        attn_output = masked_attn_output + attn_output

        attn_output = self.layer_norm_attn(attn_output)
        ffnn_output = self.feedforward(attn_output)
        ffnn_output = attn_output + ffnn_output
        return ffnn_output

'''
Composite Model
1. Get the embeddings of the input
2. Apply positional encoding
3. Apply encoders and decoders
4. Apply final linear layer
'''
class TransformerModel(Module):

    def __init__(self, embed_dim:int, num_heads:int, dropout_rate:float, hidden_layer_dim:int, max_len:int, vocab_size:int, stacks:int):

        # Set Hyperparameters
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.hidden_layer_dim = hidden_layer_dim
        self.vocab_size = vocab_size

        super(TransformerModel, self).__init__()

        # Embedding layer
        self.token_embedding = Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        self.token_embedding_dec = Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        

        # 6 encoders and decoders layer
        self.encoders = ModuleList([
            Encoder(
                embed_dim=embed_dim, 
                num_heads=num_heads,
                dropout_rate=dropout_rate, 
                hidden_layer_dim=hidden_layer_dim,
            ) for _ in range(stacks)
        ])

        self.decoders = ModuleList([
            Decoder(
                embed_dim=embed_dim, 
                num_heads=num_heads, 
                dropout_rate=dropout_rate, 
                hidden_layer_dim=hidden_layer_dim,
            ) for _ in range(stacks)
        ])

        # Pre-calculate the positional encoding
        self.positional_encoding = self._generate_positional_encoding(max_len=max_len, embed_dim=embed_dim)

        # Last linear layer right before softmax
        self.final_layer_norm = LayerNorm(self.embed_dim)
        self.output_layer = Linear(in_features=embed_dim, out_features=vocab_size, bias=True)
    
    # The positional encoding function that is called before fed into encoders or decoders
    def _generate_positional_encoding(self, max_len, embed_dim):
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe
    
    def forward(self, src, tgt):

        pad_mask_left = (src != 0).unsqueeze(1).unsqueeze(1)
        pad_mask_left = pad_mask_left.expand(-1, self.num_heads, src.size(1), src.size(1))
        pad_mask_right = pad_mask_left.transpose(-1, -2)
        pad_mask_src = pad_mask_left & pad_mask_right
        pad_mask_src = pad_mask_src.to(device=src.device)

        pad_mask_left = (tgt != 0).unsqueeze(1).unsqueeze(1)
        pad_mask_right = pad_mask_left.transpose(-1, -2)
        pad_mask_tgt = pad_mask_right
        pad_mask_tgt = pad_mask_tgt.to(device=tgt.device)


        # 1. Get the embeddings of the input
        # print(src.shape, self.token_embedding(src).shape)
        src = self.token_embedding(src)
        src *= math.sqrt(self.embed_dim)
        # logger.debug('After embedding: {}'.format(src.shape))

        # 2. Apply positional encoding
        # print(src.shape, self.positional_encoding[:, :src.size(1), :].shape)
        # logger.debug('src_dim: {}, positional_encoding_dim: {}'.format(src.shape, self.positional_encoding[:, :src.size(1), :].shape))
        src = src + self.positional_encoding[:, :src.size(1), :].to(src.device)

        # 1. Get the embeddings of the input
        tgt = self.token_embedding_dec(tgt)
        # logger.debug('After embedding: {}'.format(tgt.shape))

        # 2. Apply positional encoding
        tgt = tgt + self.positional_encoding[:, :tgt.size(1), :].to(tgt.device)
        # logger.debug('After positional encoding: {}'.format(tgt.shape))

        # Get the target mask for the decoder
        target_mask = torch.tril(torch.ones(tgt.shape[1], tgt.shape[1]), diagonal=0)
        target_mask = target_mask.to(tgt.device)
        
        # 3. Apply encoders and decoders
        for idx, encoder in enumerate(self.encoders):   
            # logger.debug(f'----------Encoder layer {idx+1}----------')
            src = encoder(src, pad_mask_src)
            # logger.debug('Encoder output[{}]: {}'.format(idx+1, src[0]))
            
        
        encoder_output = src
        
        for idx, decoder in enumerate(self.decoders):
            # logger.debug(f'----------Decoder layer {idx+1}----------')
            tgt = decoder(tgt, encoder_output, pad_mask_tgt, target_mask)
            # logger.debug('Decoder output[{}]: {}'.format(idx+1, tgt[0]))

        for idx, decoder in enumerate(self.decoders):
            decoder.multihead_attention.k_cache = None 
            decoder.multihead_attention.v_cache = None
        # 4. Get the output of the last linear layer
        result = self.final_layer_norm(tgt)
        result = self.output_layer(result)

        return result
    
    def generate_translation(self, input_sentences, n_candidates, bos_token_id, eos_token_id):
        '''
        @param input_sentences: list of strings, tokenized, batched (batch_size, max_len)
        @param n_candidates: number of translation candidates to generate
        @return: list of strings
        '''
        batch_size = len(input_sentences)
        device = next(self.parameters()).device
        
        # Get special token IDs
        start_token_id = bos_token_id
        end_token_id = eos_token_id

        # For each sentence in the batch
        all_translations = []

        for sentence_idx, input_sentence in enumerate(input_sentences):
            # Encode the input sentence
            encoder_input = torch.tensor(input_sentence, device=device)
            #encoder_outputs = self.encoder(encoder_input)
            
            # Initialize with start token
            current_beams = [(torch.tensor([[start_token_id]], device=device), 0.0)]  # (sequence tensor, log_prob)
            
            # Continue until all beams have generated an end token or reached max length
            max_length = 100  # Set a reasonable max length
            finished_beams = []
            
            while len(current_beams) > 0 and len(finished_beams) < n_candidates:
                all_next_beams = []
                
                for beam_idx, (beam_sequence, beam_score) in enumerate(current_beams):
                    # Skip if the beam is already finished
                    if beam_sequence[0, -1].item() == end_token_id:
                        finished_beams.append((beam_sequence, beam_score))
                        continue
                    
                    # Get decoder output for the current beam
                    with torch.no_grad():
                        decoder_output = self(encoder_input, beam_sequence) #self.decoder(beam_sequence, encoder_outputs)
                        logits = self.output_layer(decoder_output[:, -1, :])
                        log_probs = F.log_softmax(logits, dim=-1)
                    
                    # Get top-k candidates for the next token
                    topk_log_probs, topk_indices = log_probs.topk(n_candidates)
                    
                    # Create new beam candidates
                    for i in range(len(topk_indices)):
                        token_id = topk_indices[i].item()
                        token_log_prob = topk_log_probs[i].item()
                        
                        # Create new sequence by appending the token
                        new_sequence = torch.cat([beam_sequence, 
                                                torch.tensor([[token_id]], device=device)], dim=1)
                        new_score = beam_score + token_log_prob
                        
                        # Add to finished beams if end token, otherwise to candidates
                        if token_id == end_token_id:
                            finished_beams.append((new_sequence, new_score))
                        else:
                            all_next_beams.append((new_sequence, new_score))
                
                # If we have enough finished beams, we're done
                if len(finished_beams) >= n_candidates:
                    break
                    
                # Sort and keep top beams for next iteration
                all_next_beams = sorted(all_next_beams, key=lambda x: x, reverse=True)
                current_beams = all_next_beams[:n_candidates - len(finished_beams)]
                
                # Check for max length
                if all(beam.size(1) >= max_length for beam in current_beams):
                    # Force end all beams that reached max length
                    for beam_sequence, beam_score in current_beams:
                        new_sequence = torch.cat([beam_sequence, 
                                                torch.tensor([[end_token_id]], device=device)], dim=1)
                        finished_beams.append((new_sequence, beam_score))
                    break
            
            # Combine finished beams and remaining beams
            all_beams = finished_beams + current_beams
            all_beams = sorted(all_beams, key=lambda x: x, reverse=True)[:n_candidates]
            
            # Convert token IDs back to strings
            sentence_translations = []
            for beam_sequence, _ in all_beams:
                # Remove start and end tokens and convert to list
                token_ids = beam_sequence[0, 1:].tolist()  # Remove start token
                if end_token_id in token_ids:
                    # Truncate at end token
                    end_idx = token_ids.index(end_token_id)
                    token_ids = token_ids[:end_idx]
                
                # Convert to string
                translation = self.tokenizer.decode(token_ids)
                sentence_translations.append(translation)
            
            all_translations.append(sentence_translations)

        return all_translations

        

    