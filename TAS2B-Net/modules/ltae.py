import copy

import numpy as np
import torch
import torch.nn as nn


class LTAE2d(nn.Module):
    def __init__(
        self,
        in_channels=128,
        n_head=16,
        d_k=4,
        mlp=[128, 128],
        dropout=0.2,
        d_model=256,
        T=1000,
        return_att=False,
        positional_encoding=True,
    ):
        """
    用于图像时间序列的轻量级时间注意编码器 (L-TAE)。基于注意的序列编码，将图像序列映射到单个特征图。
    共享 L-TAE 应用于图像序列的所有像素位置。
        Args:
            in_channels (int): 输入embedding的通道数
            n_head (int): 注意头的数量.
            d_k (int): key和query向量的维度
            mlp (List[int]): 处理注意头的连接输出的 MLP 层的宽度
            dropout (float): dropout
            d_model (int, optional): 如果指定，输入张量将首先由完全连接的层处理以将它们投影到维度为 d_model 的特征空间中
            T (int): 用于位置编码的周期
            return_att (bool): 如果为 true,模块将返回attention masks以及embeddings(默认为 False)
            positional_encoding (bool): 如果为 False,则不使用位置编码(默认为 True)
        """
        super(LTAE2d, self).__init__()
        self.in_channels = in_channels
        self.mlp = copy.deepcopy(mlp)
        self.return_att = return_att
        self.n_head = n_head

        if d_model is not None:  # 若需要首先将输入的C维变为d_model
            self.d_model = d_model
            self.inconv = nn.Conv1d(in_channels, d_model, 1)
        else:
            self.d_model = in_channels
            self.inconv = None
        assert self.mlp[0] == self.d_model

        if positional_encoding:   # 若需要进行位置编码
            self.positional_encoder = PositionalEncoder(self.d_model // n_head, T=T, repeat=n_head)
        else:
            self.positional_encoder = None

        self.attention_heads = MultiHeadAttention(n_head=n_head, d_k=d_k, d_in=self.d_model)
        # self.in_norm = nn.GroupNorm(num_groups=n_head,num_channels=self.d_model)
        self.out_norm = nn.GroupNorm(num_groups=n_head,num_channels=mlp[-1])
        self.out_dim = mlp[-1]

        # 建立MLP
        layers = []
        for i in range(len(self.mlp) - 1):
            layers.extend([nn.Linear(self.mlp[i], self.mlp[i + 1]),nn.BatchNorm1d(self.mlp[i + 1]),nn.ReLU()])
        self.mlp = nn.Sequential(*layers)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, batch_positions=None, pad_mask=None, return_comp=False):
        sz_b, seq_len, d, h, w = x.shape   # 首先拿到Unet最底部的特征图 [B, T, C, H, W]
        # 生成pad_mask
        if pad_mask is not None:
            pad_mask = (
                pad_mask.unsqueeze(-1)
                .repeat((1, 1, h))
                .unsqueeze(-1)
                .repeat((1, 1, 1, w))
            )  # pad_mask [B, T, H, W]
            pad_mask = (pad_mask.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)) 
               # pad_mask[B*H*W, T]

        out = x.permute(0, 3, 4, 1, 2).contiguous().view(sz_b * h * w, seq_len, d)  # 改变形状  out [B*H*W, T, C]
        # out = self.in_norm(out.permute(0, 2, 1)).permute(0, 2, 1)     # norm一下，形状不变 out [B*H*W, T, C]

        if self.inconv is not None:
            out = self.inconv(out.permute(0, 2, 1)).permute(0, 2, 1)                # 卷积一下，改变C维   out[B*H*W, T, d_model]
        # 以上就是对输入x进行了一个简单的norm和conv操作，形状变为[B*H*W, T, d_model] 3D
        if self.positional_encoder is not None:
            bp = (
                batch_positions.unsqueeze(-1)
                .repeat((1, 1, h))
                .unsqueeze(-1)
                .repeat((1, 1, 1, w))
            )  # bp [B, T, H, W]

            bp = bp.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)       # bp [B*H*W, T]
        
            out = out + self.positional_encoder(bp)       # out[B*H*W, T, d_model]  3D
        # in_norm -> conv -> pe ->attention -> out_norm
        out, attn = self.attention_heads(out, pad_mask=pad_mask)   # 进入多头注意力机制，这下知道了为什么要将H,W和B合并成1维了，因为本来多头注意力就是本来是应用于NLP的，本来就是没有HW的位置
        # out  [HEADS=16, B*H*W, d_model//HEADS]  3D
        # attn [HEADS=16, B*H*W, T]               3D

        out = (out.permute(1, 0, 2).contiguous().view(sz_b * h * w, -1))  
        # Concatenate heads  [B*H*W, HEADS*d_model//HEADS] = [B*H*W, d_model]  2D

        out = self.dropout(self.mlp(out))   # out[B*H*W, C= mlp[-1]=128]  2D
        out = self.out_norm(out) if self.out_norm is not None else out  # shape不变
        out = out.view(sz_b, h, w, -1).permute(0, 3, 1, 2)  
        # out[B, C=128, H=16, W=16]  变成4D，自从x进入了多头注意力机制，T维度就被消去了

        attn = attn.view(self.n_head, sz_b, h, w, seq_len).permute(0, 1, 4, 2, 3)  
        # attn [HEADS=16, B, T, H=16, W=16]

        if self.return_att:
            return out, attn
        else:
            return out


# 传统的多头注意力机制
class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module
    Modified from github.com/jadore801120/attention-is-all-you-need-pytorch
    """

    def __init__(self, n_head, d_k, d_in):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in

        self.Q = nn.Parameter(torch.zeros((n_head, d_k))).requires_grad_(True)
        nn.init.normal_(self.Q, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.fc1_k = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_k.weight, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))

    def forward(self, v, pad_mask=None, return_comp=False):
        # v [B*H*W, T, d_model]
        # n_head=16, d_k=4, d_in=d_model=256
        d_k, d_in, n_head = self.d_k, self.d_in, self.n_head
        sz_b, seq_len, _ = v.size()

        q = torch.stack([self.Q for _ in range(sz_b)], dim=1).view(-1, d_k)  # [(HEADS*B*H*W), d_k] 每个头有一个 d_k 维的查询向量
        # k [B*H*W, T, d_model]->[B*H*W, T, HEADS*d_k]
        k = self.fc1_k(v).view(sz_b, seq_len, n_head, d_k) # k [B*H*W, T, HEADS, d_k]
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k)     # k [HEADS*(B*H*W), T, d_k]

        if pad_mask is not None:
            pad_mask = pad_mask.repeat((n_head, 1))  # replicate pad_mask for each head (nxb) x lk

        v = torch.stack(v.split(v.shape[-1] // n_head, dim=-1)).view(n_head * sz_b, seq_len, -1)
        # v [B*H*W, T, d_model] -> [HEADS*(B*H*W), T, d_model//HEADS]
        if return_comp:
            output, attn, comp = self.attention(q, k, v, pad_mask=pad_mask, return_comp=return_comp)
            # output [HEADS*(B*H*W), 1, d_model//HEADS]
            # attn [HEADS*(B*H*W), 1, T]
        else:
            output, attn = self.attention(q, k, v, pad_mask=pad_mask, return_comp=return_comp)
        attn = attn.view(n_head, sz_b, 1, seq_len) # attn [HEADS, B*H*W, 1, T]
        attn = attn.squeeze(dim=2)                 # attn [HEADS, B*H*W, T]

        output = output.view(n_head, sz_b, 1, d_in // n_head)  # output [HEADS, B*H*W, 1, d_model//HEADS]
        output = output.squeeze(dim=2)                         # output [HEADS, B*H*W, d_model//HEADS]

        if return_comp:
            return output, attn, comp
        else:
            return output, attn

# MultiHeadAttention的子函数
class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention
    Modified from github.com/jadore801120/attention-is-all-you-need-pytorch
    """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, pad_mask=None, return_comp=False):
        # q [HEADS*(B*H*W), d_k]
        # k [HEADS*(B*H*W), T, d_k]
        # v [HEADS*(B*H*W), T, d_model//HEADS]
        attn = torch.matmul(q.unsqueeze(1), k.transpose(1, 2)) # attn [HEADS*(B*H*W), 1, T]
        # [HEADS*(B*H*W), 1, d_k]*[HEADS*(B*H*W), d_k, T] = [HEADS*(B*H*W), 1, T] = [8192,1,5]
        attn = attn / self.temperature
        if pad_mask is not None:
            attn = attn.masked_fill(pad_mask.unsqueeze(1), -1e3)
        if return_comp:
            comp = attn
        # compat = attn
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)  
        # [HEADS*(B*H*W), 1, T] * [HEADS*(B*H*W), T, d_model//HEADS] = [HEADS*(B*H*W), 1, d_model//HEADS]
        # 此时T维度被彻底消除
        if return_comp:
            return output, attn, comp
        else:
            return output, attn

# dates搞了那么火热，到头来只是为了计算PE
class PositionalEncoder(nn.Module):
    def __init__(self, d, T=1000, repeat=None, offset=0):
        super(PositionalEncoder, self).__init__()
        self.d = d
        self.T = T
        self.repeat = repeat
        self.denom = torch.pow(T, 2 * (torch.arange(offset, offset + d).float() // 2) / d)
        self.updated_location = False

    def forward(self, batch_positions):
        if not self.updated_location:
            self.denom = self.denom.to(batch_positions.device)
            self.updated_location = True
        sinusoid_table = (batch_positions[:, :, None] / self.denom[None, None, :])  # B x T x C
        sinusoid_table[:, :, 0::2] = torch.sin(sinusoid_table[:, :, 0::2])  # dim 2i
        sinusoid_table[:, :, 1::2] = torch.cos(sinusoid_table[:, :, 1::2])  # dim 2i+1

        if self.repeat is not None:
            sinusoid_table = torch.cat([sinusoid_table for _ in range(self.repeat)], dim=-1)

        return sinusoid_table   # [B, T, d]