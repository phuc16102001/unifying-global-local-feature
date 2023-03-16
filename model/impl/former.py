import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math
import copy

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_length = 200, dropout = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_seq_length, d_model)
        for pos in range(max_seq_length):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos/(10000**(2*i/d_model)))
                pe[pos, i+1] = math.cos(pos/(10000**((2*i+1)/d_model)))
        pe = pe.unsqueeze(0) # Expand batch size dimension
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        seq_len = x.size(1) # (batch, len, d_model)
        pe = Variable(
            self.pe[:, :seq_len],
            requires_grad = False
        )
        if (x.is_cuda): pe.cuda()
        x = x * math.sqrt(self.d_model)
        x = x + pe
        x = self.dropout(x)
        return x

def attention(q, k, v, mask=None, dropout=None):
    """
    q: batch x head x seq_len x d_model
    k: batch x head x seq_len x d_model
    v: batch x head x seq_len x d_model

    mask: batch x 1 x seq_len
    output: batch x head x seq_len x d_model
    """
    d_k = q.size(-1) # last dimension
    scores = torch.matmul(q, k.transpose(-2, -1))/math.sqrt(d_k) # batch x head x seq_len x seq_len
    
    # Masking
    if (mask is not None):
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask==0, -1e9)
    
    # Softmax will convert all -inf to 0
    scores = F.softmax(scores, dim = -1) # Softmax over last dimension
    
    if (dropout is not None):
        scores = dropout(scores)
    
    output = torch.matmul(scores, v)
    return output, scores

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()
        assert(d_model % heads == 0)
        
        self.d_model = d_model
        self.heads = heads
        self.d_k = d_model//heads # In order to d_k.heads = d_model
        
        self.dropout = nn.Dropout(dropout)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v, mask):
        """
        q: bs x seq_len x d_model
        k: bs x seq_len x d_model
        v: bs x seq_len x d_model
        """
        
        bs = q.size(0)
        q = self.w_q(q).view(bs, -1, self.heads, self.d_k)
        k = self.w_k(k).view(bs, -1, self.heads, self.d_k)
        v = self.w_v(v).view(bs, -1, self.heads, self.d_k)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)        
        v = v.transpose(1, 2)

        # Attention mechanism
        output_attn, self.scores = attention(q, k, v, mask, self.dropout)
        
        concatenate_tensor = output_attn.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concatenate_tensor)
        return output

class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        
        self.alpha = nn.Parameter(torch.ones(self.d_model))
        self.bias = nn.Parameter(torch.zeros(self.d_model))
        
    def forward(self, x):
        num = (x - x.mean(dim = -1, keepdim = True))
        denom = (x.std(dim=-1, keepdim=True) + self.eps) 
        out = self.alpha * (num / denom) + self.bias
        return out

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        
        self.d_ff = d_ff
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        x = self.linear_1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(heads, d_model, dropout)
        self.norm_1 = Norm(d_model)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.norm_2 = Norm(d_model)
    
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        x_norm = self.norm_1(x)
        x_attn = self.attn(x_norm, x_norm, x_norm, mask)
        
        x = x + self.dropout_1(x_attn)
        x_norm = self.norm_2(x)
        x_ff = self.ff(x_norm)

        x = x + self.dropout_2(x_ff)
        return x

def get_clones(module, n):
    module_list = []
    for _ in range(n):
        module_list.append(copy.deepcopy(module))
    return nn.ModuleList(module_list)

class Encoder(nn.Module):
    def __init__(self, d_model, n, heads, dropout=0.1):
        super().__init__()
        
        self.n = n
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.encoder_layers = get_clones(EncoderLayer(d_model, heads, dropout), n)
        self.norm = Norm(d_model)
    
    def forward(self, x, mask):
        x = self.pe(x)
        for i in range(self.n):
            x = self.encoder_layers[i](x, mask)
        x = self.norm(x)
        return x