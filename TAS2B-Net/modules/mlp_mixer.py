from functools import partial
import torch
from torch.nn import functional as F
from torch import nn, Tensor

from timm.models.layers import trunc_normal_, get_act_layer, DropPath
from itertools import repeat
import collections

from modules.norm import NormLayer


# From PyTorch internals
def _ntuple(n):

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse

to_2tuple = _ntuple(2)

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class FFN(nn.Module):

    def __init__(self,
                 embed_dims,
                 dim_ffd,
                 num_fcs=2,
                 act_type='gelu',
                 norm_type='layernorm',
                 ffn_drop=0.,
                 drop_path=0.,
                 init_values=None,
                 **kwargs):
        super(FFN, self).__init__()
        # Implementation of Feedforward model
        assert num_fcs >= 2 , f'num_fcs should be no less than 2. got {num_fcs}.'

        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(
                nn.Sequential(
                    nn.Linear(in_channels, dim_ffd),
                    get_act_layer(act_type)(),
                    nn.Dropout(ffn_drop)
                ))
            in_channels = dim_ffd
        layers.append(nn.Linear(dim_ffd, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = nn.Sequential(*layers)

        self.norm = NormLayer(norm_type, embed_dims)

        self.drop_path = DropPath(drop_prob=drop_path) if drop_path > 0. else nn.Identity()
        self.ls = LayerScale(embed_dims, init_values=init_values) if init_values else nn.Identity()

        self.apply(self._reset_parameters)

    def _reset_parameters(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02, a=-0.02, b=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, tgt: Tensor):
        tgt2 = self.norm(tgt)
        tgt2 = self.layers(tgt2)
        tgt = tgt + self.drop_path(self.ls(tgt2))
        return tgt

class GatedMlp(nn.Module):
    """ MLP as used in gMLP
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_type='gelu',
            gate_layer=None,
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = get_act_layer(act_type)()
        self.drop1 = nn.Dropout(drop_probs[0])
        if gate_layer is not None:
            assert hidden_features % 2 == 0
            self.gate = gate_layer(hidden_features)
            hidden_features = hidden_features // 2  # FIXME base reduction on gate property?
        else:
            self.gate = nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

        self.apply(self._reset_parameters)
        if gate_layer is not None:
            self.gate.init_weights()

    def _reset_parameters(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02, a=-0.02, b=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.gate(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MLPMixerLayer(nn.Module):
    def __init__(self,
                 num_tokens,
                 embed_dims,
                 token_expansion,
                 channel_expansion,
                 norm_type='layernorm',
                 drop_path=0.,
                 drop_out=0.,
                 **kwargs):

        super(MLPMixerLayer, self).__init__()

        token_mix_dims = int(token_expansion * embed_dims)
        channel_mix_dims = int(channel_expansion * embed_dims)

        self.token_mixer = nn.Sequential(
            nn.Linear(num_tokens, token_mix_dims),
            nn.GELU(),
            nn.Dropout(drop_out),
            nn.Linear(token_mix_dims, num_tokens),
            nn.Dropout(drop_out)
        )

        self.channel_mixer = nn.Sequential(
            nn.Linear(embed_dims, channel_mix_dims),
            nn.GELU(),
            nn.Dropout(drop_out),
            nn.Linear(channel_mix_dims, embed_dims),
            nn.Dropout(drop_out)
        )

        self.drop_path1 = DropPath(drop_prob=drop_path)
        self.drop_path2 = DropPath(drop_prob=drop_path)

        self.norm1 = NormLayer(norm_type, embed_dims)
        self.norm2 = NormLayer(norm_type, embed_dims)

    def forward(self, x):
        x = x + self.drop_path1(self.token_mixer(self.norm1(x).transpose(1,2)).transpose(1,2))
        x = x + self.drop_path2(self.channel_mixer(self.norm2(x)))
        return x


class MLPMixer(nn.Module):
    def __init__(self,
                 num_tokens,
                 embed_dims,
                 token_expansion=0.5,
                 channel_expansion=4.0,
                 depth=1,
                 norm_type='layernorm',
                 drop_path=0.,
                 drop_out=0.,
                 **kwargs):
        super(MLPMixer, self).__init__()
        layers = [
            MLPMixerLayer(num_tokens, embed_dims, token_expansion, channel_expansion,
                          norm_type, drop_path, drop_out)
            for _ in range(depth)
        ]
        self.layers = nn.Sequential(*layers)

        self.apply(self._reset_parameters)

    def _reset_parameters(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02, a=-0.02, b=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        return self.layers(x)


class SpatialGatingUnit(nn.Module):
    """ Spatial Gating Unit
    Based on: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
    """
    def __init__(self, dim, seq_len, norm_type='layernorm'):
        super().__init__()
        gate_dim = dim // 2
        self.norm = NormLayer(norm_type, gate_dim)
        self.proj = nn.Linear(seq_len, seq_len)

    def init_weights(self):
        # special init for the projection gate, called as override by base model init
        nn.init.normal_(self.proj.weight, std=1e-6)
        nn.init.ones_(self.proj.bias)

    def forward(self, x):
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)
        v = self.proj(v.transpose(-1, -2))
        return u * v.transpose(-1, -2)


class SpatialGatingBlock(nn.Module):
    """ Residual Block w/ Spatial Gating
    Based on: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
    """
    def __init__(self, dim, seq_len, mlp_ratio=4, mlp_layer=GatedMlp,
                 norm_type='layernorm', act_type='gelu',
                 drop=0., drop_path=0.):
        super().__init__()
        channel_dim = int(dim * mlp_ratio)
        self.norm = NormLayer(norm_type, dim)
        sgu = partial(SpatialGatingUnit, seq_len=seq_len, norm_type=norm_type)
        self.mlp_channels = mlp_layer(dim, channel_dim, act_type=act_type, gate_layer=sgu, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.mlp_channels(self.norm(x)))
        return x
