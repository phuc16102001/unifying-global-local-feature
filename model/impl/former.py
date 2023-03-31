import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math
from ..utils import get_clones

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
    def __init__(self, d_model, heads, d_ff=2048, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, heads, batch_first=True, dropout=dropout)
        
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)

        self.ff_1 = FeedForward(d_model, d_ff=d_ff, dropout=dropout)
        self.ff_2 = FeedForward(d_model, d_ff=d_ff, dropout=dropout)
    
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        
    def forward(self, x, mask=None, key_padding_mask=None):
        x_attn = self.attn(x, x, x, attn_mask=mask, key_padding_mask=key_padding_mask)[0]
        cnt_nan = torch.sum(torch.isnan(x))
        print("After attn", cnt_nan)
        assert(int(cnt_nan.item())==0)
        if (key_padding_mask is not None):
            x_attn = x_attn.masked_fill(key_padding_mask.unsqueeze(-1), 0)
        cnt_nan = torch.sum(torch.isnan(x))
        print("After mask 1", cnt_nan)
        assert(int(cnt_nan.item())==0)
        x = x + self.dropout_1(x_attn)

        # Handle all zeros before norm
        esp = 1e-9
        x = self.norm_1(x+esp)
        cnt_nan = torch.sum(torch.isnan(x))
        print("After norm 1", cnt_nan)
        assert(int(cnt_nan.item())==0)

        x_linear = self.ff_1(x)
        x_linear = F.relu(x_linear)
        x_linear = self.dropout_2(x_linear)
        x_linear = self.ff_2(x_linear)

        x = x + self.dropout_3(x_linear)
        
        cnt_nan = torch.sum(torch.isnan(x))
        print("Before norm 2", cnt_nan)
        assert(int(cnt_nan.item())==0)
        x = self.norm_2((x.to(torch.float16)+esp))
        
        cnt_nan = torch.sum(torch.isnan(x))
        print("Before mask 2", cnt_nan)
        assert(int(cnt_nan.item())==0)
        if (key_padding_mask is not None):
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0)
        cnt_nan = torch.sum(torch.isnan(x))
        print("After mask 2", cnt_nan)
        assert(int(cnt_nan.item())==0)
        return x

class Encoder(nn.Module):
    def __init__(self, d_model, n, heads, dropout=0.1, use_pe=True):
        super().__init__()
        
        self._n = n
    
        self._use_pe = use_pe
        if (use_pe):
           self.pe = PositionalEncoder(d_model, dropout=dropout)
    
        self.encoder_layers = get_clones(
            EncoderLayer(d_model, heads, dropout=dropout), self._n)
        self.norm = Norm(d_model)
    
    def forward(self, x, mask=None, key_padding_mask=None):
        if (self._use_pe):
            x = self.pe(x)
        print("before encode", torch.sum(torch.isnan(x)))
        for i in range(self._n):
            x = self.encoder_layers[i](
                x, mask=mask, key_padding_mask=key_padding_mask)
        print("after encode", torch.sum(torch.isnan(x)))
        x = self.norm(x)
        return x