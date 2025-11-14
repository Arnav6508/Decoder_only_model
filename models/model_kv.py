import torch
import torch.nn as nn
import math

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from models.model import LayerNorm, PositionalEncoding

torch.manual_seed(100)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_attn, n_heads, dropout=0.1):
        super().__init__()
        assert d_attn % n_heads == 0, "dimension of model should be divisible by number of attention heads"

        self.d_attn = d_attn
        self.n_heads = n_heads
        self.d_k = d_attn // n_heads # Dimension of key matrix of 1 head

        self.q_linear_layer = nn.Linear(d_attn, d_attn)
        self.k_linear_layer = nn.Linear(d_attn, d_attn)
        self.v_linear_layer = nn.Linear(d_attn, d_attn)
        self.out_linear_layer = nn.Linear(d_attn, d_attn)

        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim = -1)

    def forward(self, x, kv_cache=None):
        batch_size = x.size(0)
        curr_seq_len = x.size(1) 

        # (batch, seq_len, d_model) -> (batch, n_heads, seq_len, d_k)
        q = self.q_linear_layer(x).reshape(batch_size, curr_seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_linear_layer(x).reshape(batch_size, curr_seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_linear_layer(x).reshape(batch_size, curr_seq_len, self.n_heads, self.d_k).transpose(1, 2)

        if kv_cache is not None:
            past_k, past_v = kv_cache

            # k: (batch, n_heads, seq_len_past + 1, d_k)
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)
        
        new_kv_cache = (k, v)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if kv_cache is None:
            look_ahead_mask = torch.tril(torch.ones(curr_seq_len, curr_seq_len, device=x.device))
            look_ahead_mask = look_ahead_mask.reshape(1, 1, curr_seq_len, curr_seq_len)
            scores = scores.masked_fill(look_ahead_mask == 0, float('-inf'))

        attn_weights = self.softmax(scores)
        attn_weights = self.dropout(attn_weights)

        # self_attn: (b, h, 1, d_k)
        self_attn = torch.matmul(attn_weights, v)

        # (batch, n_heads, seq_len, d_k) -> (batch, seq_len, d_model)
        self_attn = self_attn.transpose(1, 2).contiguous().reshape(batch_size, curr_seq_len, self.d_attn)
        output = self.out_linear_layer(self_attn)

        return output, attn_weights, new_kv_cache

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model 

        self.network = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.network(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, d_attn, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_attn, n_heads, dropout) 
        self.linear_layer1 = nn.Linear(d_model, d_attn)
        self.linear_layer2 = nn.Linear(d_attn, d_model)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm_layer1 = LayerNorm(d_model)
        self.norm_layer2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, kv_cache=None):
        norm_x = self.norm_layer1(x)
        lin_output = self.linear_layer1(norm_x)
        
        # Pass cache to MHA and get new cache back
        mha_output, attn_weights, new_kv_cache = self.mha(lin_output, kv_cache=kv_cache)
        
        lin_output_2 =  self.linear_layer2(mha_output)
        part1_output = x + self.dropout(lin_output_2)

        norm_mha_output = self.norm_layer2(part1_output)
        ff_output = self.feed_forward(norm_mha_output)
        out = norm_mha_output + self.dropout(ff_output)
        
        return out, attn_weights, new_kv_cache

class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, d_attn, max_seq_len, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, d_attn, dropout) for _ in range(n_layers)
        ])
        
        self.norm_layer = LayerNorm(d_model)
        self.linear_layer = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim = -1)

    def forward(self, indices, past_kv_caches=None, return_attention=False):
        token_emb = self.token_embedding(indices) 
        
        if past_kv_caches is None:
            x = self.positional_encoding(token_emb)
            past_kv_caches = [None] * len(self.transformer_blocks)
        else:
            # Get past sequence length from the cache of the first layer
            if past_kv_caches[0] is not None:
                past_seq_len = past_kv_caches[0][0].size(2)
            else:
                past_seq_len = 0
                
            current_positions = torch.arange(past_seq_len, past_seq_len + indices.size(1), device=indices.device)
            
            # Handling positions beyond max_seq_len using modulo
            max_seq_len = self.positional_encoding.pos_matrix.size(1)
            positions_mod = current_positions % max_seq_len
            
            pos_emb = self.positional_encoding.pos_matrix[0, positions_mod] 
            x = token_emb + pos_emb.unsqueeze(0)

        x = self.dropout(x)

        attn_weights_l = []
        new_kv_caches = []
        
        # Pass cache into each block and get the new cache
        for i, block in enumerate(self.transformer_blocks):
            x, attn_weights, new_kv_cache = block(x, kv_cache=past_kv_caches[i])
            attn_weights_l.append(attn_weights)
            new_kv_caches.append(new_kv_cache)
            
        x = self.norm_layer(x)
        logits = self.linear_layer(x)
        
        if return_attention:
            return logits, new_kv_caches, attn_weights_l
        else:
            return logits, new_kv_caches