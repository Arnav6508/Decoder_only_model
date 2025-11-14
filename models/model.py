import torch
import torch.nn as nn
import math

torch.manual_seed(100)

class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super().__init__()

        self.gamma = nn.Parameter(torch.ones(features)) 
        self.beta = nn.Parameter(torch.zeros(features)) 
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std_dev = x.std(-1, keepdim=True, unbiased=False)
        
        normalized_x = (x-mean)/(std_dev+self.eps)

        return self.gamma*normalized_x+self.beta

class PositionalEncoding(nn.Module):
    """
    Formula used from 'Attention is All you need paper':
        PE(pos, 2i) = sin(pos/10000^(2i / d_model))
        PE(pos, 2i+1) = cos(pos/10000^(2i / d_model))
    """
    def __init__(self, d_model, max_seq_len=5000):
        super().__init__()
        
        pos_matrix = torch.zeros(max_seq_len, d_model)

        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        i = torch.arange(0, d_model, 2, dtype=torch.float)
        div_term = 1.0/torch.pow(10000.0, i / d_model)
        
        pos_matrix[:, 0::2] = torch.sin(position * div_term) 
        pos_matrix[:, 1::2] = torch.cos(position * div_term)
  
        self.register_buffer('pos_matrix', pos_matrix.unsqueeze(0)) 

    def forward(self, x):
        return x + self.pos_matrix[:, :x.size(1)]

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

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)

        # (batch, seq_len, d_model) -> (batch, n_heads, seq_len, d_k)
        q = self.q_linear_layer(x).reshape(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_linear_layer(x).reshape(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_linear_layer(x).reshape(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # (batch, n_heads, seq_len, d_k) x (batch, n_heads, d_k, seq_len) -> (batch, n_heads, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        look_ahead_mask = torch.tril(torch.ones(seq_len, seq_len))
        look_ahead_mask = look_ahead_mask.reshape(1, 1, seq_len, seq_len)
        scores = scores.masked_fill(look_ahead_mask == 0, float('-inf'))

        attn_weights = self.softmax(scores)
        attn_weights = self.dropout(attn_weights)

        self_attn = torch.matmul(attn_weights, v)

        # (batch, n_heads, seq_len, d_k) -> (batch, seq_len, d_model)
        self_attn = self_attn.transpose(1, 2).reshape(batch_size, -1, self.d_attn)
        output = self.out_linear_layer(self_attn)

        return output, attn_weights

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
        
    def forward(self, x):
        norm_x = self.norm_layer1(x)
        lin_output = self.linear_layer1(norm_x)
        mha_output, attn_weights = self.mha(lin_output)
        lin_output_2 =  self.linear_layer2(mha_output)
        part1_output = x + self.dropout(lin_output_2)

        norm_mha_output = self.norm_layer2(part1_output)
        ff_output = self.feed_forward(norm_mha_output)
        out = norm_mha_output + self.dropout(ff_output)
        
        return out, attn_weights

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

    def forward(self, indices, return_attention=False):
        token_emb = self.token_embedding(indices) 
        x = self.positional_encoding(token_emb)
        x = self.dropout(x)

        attn_weights_l = []
        
        for block in self.transformer_blocks:
            x, attn_weights = block(x)
            attn_weights_l.append(attn_weights)
            
        x = self.norm_layer(x)
        logits = self.linear_layer(x)
        # probs = self.softmax(logits)
        
        if return_attention:
            return logits, attn_weights_l
        else:
            return logits