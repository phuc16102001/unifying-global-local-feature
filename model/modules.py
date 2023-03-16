import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import SingleStageTCN
from .impl.asformer import MyTransformer
from .impl.former import Encoder


class FCPrediction(nn.Module):

    def __init__(self, feat_dim, num_classes):
        super().__init__()
        self._fc_out = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        batch_size, clip_len, _ = x.shape
        return self._fc_out(x.reshape(batch_size * clip_len, -1)).view(
            batch_size, clip_len, -1)


class GRUPrediction(nn.Module):

    def __init__(self, feat_dim, num_classes, hidden_dim, num_layers=1):
        super().__init__()
        self._gru = nn.GRU(
            feat_dim, hidden_dim, num_layers=num_layers, batch_first=True,
            bidirectional=True)
        self._fc_out = FCPrediction(2 * hidden_dim, num_classes)
        self._dropout = nn.Dropout()

    def forward(self, x):
        y, _ = self._gru(x)
        return self._fc_out(self._dropout(y))


class TCNPrediction(nn.Module):

    def __init__(self, feat_dim, num_classes, num_stages=1, num_layers=5):
        super().__init__()

        self._tcn = SingleStageTCN(
            feat_dim, 256, num_classes, num_layers, True)
        self._stages = None
        if num_stages > 1:
            self._stages = nn.ModuleList([SingleStageTCN(
                num_classes, 256, num_classes, num_layers, True)
                for _ in range(num_stages - 1)])

    def forward(self, x):
        x = self._tcn(x)
        if self._stages is None:
            return x
        else:
            outputs = [x]
            for stage in self._stages:
                x = stage(F.softmax(x, dim=2))
                outputs.append(x)
            return torch.stack(outputs, dim=0)


class ASFormerPrediction(nn.Module):

    def __init__(self, feat_dim, num_classes, num_decoders=3, num_layers=5):
        super().__init__()

        r1, r2 = 2, 2
        num_f_maps = 64
        self._net = MyTransformer(
            num_decoders, num_layers, r1, r2, num_f_maps, feat_dim,
            num_classes, channel_masking_rate=0.3)

    def forward(self, x):
        B, T, D = x.shape
        return self._net(
            x.permute(0, 2, 1), torch.ones((B, 1, T), device=x.device)
        ).permute(0, 1, 3, 2)


class VanillaEncoderPrediction(nn.Module):
    def __init__(self, hidden_dim, num_classes, num_encoders=3, heads=8, dropout=0.1):
        super().__init__()
        self._encoder = Encoder(hidden_dim, num_encoders, heads, dropout)
        self._out = nn.Linear(hidden_dim, num_classes)
        self._dropout = nn.Dropout()
        
    def forward(self, x):
        
        # Forward with mask
        # x: batch x frames x dim
        # mask: batch x 1 x frames

        B, T, D = x.shape
        mask = torch.ones((B, 1, T), device=x.device)
        e_out = self._encoder(x, mask)
        out = self._out(self._dropout(e_out))
        return out
    
class ObjectFusion(nn.Module):
    def __init__(self, env_dim, env_hidden_dim, obj_dim, obj_hidden_dim, max_obj):
        super().__init__()
        
        self._env_dim = env_dim
        self._env_hidden_dim = env_hidden_dim
        self._obj_dim = obj_dim
        self._obj_hidden_dim = obj_hidden_dim
        self._max_obj = max_obj

        self._env_linear = nn.Linear(env_dim, env_hidden_dim)
        self._obj_linear = nn.Linear(obj_dim, obj_hidden_dim)

    def fuse_obj(self, env_feat, obj_feat, obj_mask):
        batch_size, clip_len, max_obj, hidden_dim = obj_feat.size()

        # Broadcast all env feat to obj feat
        obj_env_feat = torch.unsqueeze(env_feat, 2) + obj_feat  


    # Fuse object feature to environment feature
    # env feature size: batch x frames x env_dim
    # obj feature size: batch x frames x max_obj x obj_dim
    # obj mask size: batch x frames x max_obj
    def forward(self, env_feat, obj_feat, obj_mask):
        env_feat = self._env_linear(env_feat)
        obj_feat = self._obj_linear(obj_feat)
        