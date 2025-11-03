"""
Mostly copy-paste from GPViT
https://github.com/ChenhongyiYang/GPViT
Modified by X.Cai
"""
from typing import Tuple, List, Dict, Optional

import torch
from torch.nn import functional as F
from torch import nn, Tensor

from timm.models.layers import trunc_normal_, get_act_layer, DropPath

from modules.xsoftmax import XSoftmax
from modules.norm import NormLayer
from modules.mlp_mixer import MLPMixer, FFN



class MultiHeadAttention(nn.Module):
    def __init__(self,
                 embed_dims,        # 输入特征的维度
                 num_heads,         # 注意力头的数量
                 kdim=None,         # k 和 v 的维度，默认情况下与embed_dims相同
                 vdim=None,
                 q_proj=True,       # 决定是否使用投影层对输入的qkv进行映射
                 k_proj=True,
                 v_proj=True,
                 proj_after_attn=True, # 若为True，则在注意力计算之后进行投影
                 qkv_bias=False,       # qkv的投影是否包含偏置，为上面qkv的映射服务
                 qk_scale=None,        # 缩放q和k的因子，默认为 head_dim**-0.5
                 attn_drop=0.,         # 注意力矩阵的 dropout
                 proj_drop=0.,         # 投影层的 dropouts
                 **kwargs
                 ):
        super(MultiHeadAttention, self).__init__()

        self.embed_dims = embed_dims               # 总维度
        self.head_dim = embed_dims // num_heads    # 每个头的维度
        assert self.head_dim * num_heads == self.embed_dims, \
            f"embed_dims: {embed_dims} must be divisible by num_heads: {num_heads}"
        self.num_heads = num_heads                 # 注意力头的个数
        self.qkv_bias = qkv_bias                   # qkv投影层是否使用偏置
        self.scale_factor = qk_scale or self.head_dim ** -0.5   # 缩放因子
        
        # 三个投影层
        self.q_proj = nn.Linear(embed_dims, embed_dims, bias=qkv_bias) if q_proj else None
        self.k_proj = nn.Linear(kdim if kdim is not None else embed_dims,
                                embed_dims, bias=qkv_bias) if k_proj else None
        self.v_proj = nn.Linear(vdim if vdim is not None else embed_dims,
                                embed_dims, bias=qkv_bias) if v_proj else None
        # 在注意力计算之后进行投影
        if proj_after_attn:
            self.proj = nn.Sequential(
                nn.Linear(embed_dims, embed_dims),
                nn.Dropout(proj_drop)
            )
        else:
            self.proj = None

        self.attn_drop = nn.Dropout(attn_drop)

        self.apply(self._reset_parameters)
    # 参数初始化
    def _reset_parameters(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02, a=-0.02, b=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self,
                q: Tensor,        # 查询
                k: Tensor,        # 键
                v: Tensor,        # 值
                key_padding_mask: Optional[Tensor] = None,# 指示哪些位置在计算注意力时应该被忽略的掩码（某些像素点是被忽略的，故不参加注意力计算）
                attn_mask: Optional[Tensor] = None,       # 用于遮蔽掉某些时间步之间的关系（某些时间步是padding的，也要忽略）
                attn_bias: Optional[Tensor] = None,       # 对注意力得分进行偏置处理
                return_raw_similarities: bool = False,    # 否返回原始的相似性分数（即 q 和 k 的点积）
                kth_cluster: Optional[int] = None,        # 用于在非训练模式下进行集群选择
                **kwargs):

        tgt_len, bsz, _ = q.shape    
        src_len = k.shape[0]

        q = self.q_proj(q) if self.q_proj is not None else q
        k = self.k_proj(k) if self.k_proj is not None else k
        v = self.v_proj(v) if self.v_proj is not None else v

        # --> [batch, num_heads, seq_len, head_dim]
        q = q.contiguous().view(tgt_len, bsz, self.num_heads, -1).permute(1, 2, 0, 3) * self.scale_factor
        k = k.contiguous().view(src_len, bsz, self.num_heads, -1).permute(1, 2, 0, 3)
        v = v.contiguous().view(src_len, bsz, self.num_heads, -1).permute(1, 2, 0, 3)

        scaled_dot_prod = torch.einsum('b h i d , b h j d -> b h i j', q, k)

        if key_padding_mask is not None:
            # where True indicates elements that will be ignored in the softmax calculation
            assert key_padding_mask.shape == (bsz, src_len), \
                f"expected shape of key_padding_mask is {bsz, src_len}, but got {key_padding_mask.shape}"
            mask = key_padding_mask.view(bsz, 1, 1, src_len).expand(
                -1, self.num_heads, tgt_len, -1).bool()
        else:
            mask = torch.zeros_like(scaled_dot_prod).bool()

        if attn_mask is not None:
            if attn_mask.dim() == 3:
                assert attn_mask.shape == (bsz, tgt_len, src_len), \
                    f'expected shape: {bsz, tgt_len, src_len} but got {attn_mask.shape}'
                attn_mask = attn_mask.view(bsz, 1, tgt_len, src_len)
            elif attn_mask.dim() == 4:
                assert attn_mask.shape == (bsz, self.num_heads, tgt_len, src_len), \
                    f'expected shape: {bsz, self.num_heads, tgt_len, src_len} but got {attn_mask.shape}'
            else:
                raise ValueError(f'attn_mask dims are expected to be 3 or 4 but got {attn_mask.shape}')
            mask = mask.bool() | attn_mask.bool()

        if attn_bias is not None:
            assert attn_bias.shape == scaled_dot_prod.shape, \
                f"expected shape of attn_bias is {bsz, self.num_heads, tgt_len, src_len}, but got {attn_bias.shape}"
            scaled_dot_prod = scaled_dot_prod + attn_bias

        if return_raw_similarities:
            raw_similarities = scaled_dot_prod.clone()

        scaled_dot_prod = (
            scaled_dot_prod -
            scaled_dot_prod.max(dim=-1, keepdim=True).values.detach()).to(q)
        attn = XSoftmax.apply(scaled_dot_prod, mask, -1)    # 带有掩码的Softmax函数
        attn = self.attn_drop(attn)

        if kth_cluster is not None and not self.training:
            attn_values, attn_indices = attn.sort(dim=-1, descending=True)
            attn = torch.zeros_like(attn_values).scatter_(
                -1, attn_indices[..., kth_cluster].unsqueeze(dim=-1), 1.)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = out.permute(2, 0, 1, 3).contiguous().view(tgt_len, bsz, -1)
        out = self.proj(out) if self.proj is not None else out

        if return_raw_similarities:
            return out, attn, raw_similarities
        else:
            return out, attn

    def extra_repr(self,):
        return f"embed_dims={self.embed_dims}, num_heads={self.num_heads}, head_dim={self.head_dim}, use_bias={self.qkv_bias}"

class LightGroupAttnBlock(nn.Module):
    """
    Lightweight Cross Attention for Updating an External Memory Module.
    """
    def __init__(self,
                 embed_dims,
                 num_heads,
                 norm_type='layernorm',
                 qkv_bias=False,
                 qk_scale=None,
                 proj_drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 **kwargs):
        super(LightGroupAttnBlock, self).__init__()

        self.multihead_attn = MultiHeadAttention(
            embed_dims,
            num_heads,
            q_proj=False,
            k_proj=True,
            v_proj=False,
            proj_after_attn=False,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop
        )

        self.norm_query = NormLayer(norm_type, embed_dims)
        self.norm_key   = NormLayer(norm_type, embed_dims)

        self.drop_path = DropPath(drop_prob=drop_path) if drop_path > 0. else nn.Identity()

        self.apply(self._reset_parameters)

    def _reset_parameters(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02, a=-0.02, b=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def with_pos_embed(self, tensor, pos: Optional[Tensor] = None):
        return tensor if pos is None else tensor + pos

    def forward(self,
                query,            # q
                memory,           # k
                memory_mask: Optional[Tensor] = None,   # attn_mask
                memory_key_padding_mask: Optional[Tensor] = None,
                memory_pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                attn_bias: Optional[Tensor] = None,
                return_raw_similarities: bool = False,
                **kwargs):

        q = self.norm_query(query)  # qk归一化
        k = self.norm_key(memory)
        attn_out = self.multihead_attn(
            q=self.with_pos_embed(q, query_pos),   # q进行位置编码 
            k=self.with_pos_embed(k, memory_pos),  # k进行位置编码
            v=k,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            attn_bias=attn_bias,
            return_raw_similarities=return_raw_similarities,
            **kwargs
        )

        query = query + self.drop_path(attn_out[0])

        if return_raw_similarities:
            return query, *attn_out[1:]
        else:
            return query, attn_out[1]


class FullAttnCatBlock(nn.Module):

    def __init__(self,
                 embed_dims,
                 num_heads,
                 ffn_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 act_type='gelu',
                 norm_type='layernorm',
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 **kwargs):
        super(FullAttnCatBlock, self).__init__()

        self.multihead_attn = MultiHeadAttention(
            embed_dims,
            num_heads,
            q_proj=True,
            k_proj=True,
            v_proj=True,
            proj_after_attn=True,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop
        )

        self.norm_query = NormLayer(norm_type, embed_dims)
        self.norm_key = NormLayer(norm_type, embed_dims)

        self.ffn = FFN(
            embed_dims,
            dim_ffd=int(embed_dims * ffn_ratio),
            num_fcs=2,
            act_type=act_type,
            norm_type=norm_type,
            ffn_drop=drop,
            drop_path=drop_path,
        )

        self.proj = nn.Linear(embed_dims * 2, embed_dims, bias=True)

        self.drop_path = DropPath(drop_prob=drop_path) if drop_path > 0. else nn.Identity()

        self.apply(self._reset_parameters)

    def _reset_parameters(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02, a=-0.02, b=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def with_pos_embed(self, tensor, pos: Optional[Tensor] = None):
        return tensor if pos is None else tensor + pos

    def forward(self,
                query,
                memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                memory_pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                attn_bias: Optional[Tensor] = None,
                kth_cluster: Optional[int] = None,
                **kwargs):

        q = self.norm_query(query)
        k = self.norm_key(memory)
        q2, attn = self.multihead_attn(
            q=self.with_pos_embed(q, query_pos),
            k=self.with_pos_embed(k, memory_pos),
            v=k,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            attn_bias=attn_bias,
            kth_cluster=kth_cluster,
            **kwargs
        )
        query = torch.cat([query, self.drop_path(q2)], dim=-1)
        query = self.ffn(self.proj(query))

        return query, attn


class GPBlock(nn.Module):
    """
    Feature Update by Exchanging Information (Cross-Attention)   跨注意力和外部记忆模块进行信息交换
    with an External Memory Module.
    """
    def __init__(self, config, **kwargs):

        super(GPBlock, self).__init__()

        spec_dict = {k.lower(): v for k, v in dict(config.GPBLOCK).items()} # 拿到config关于GPBLOCK的配置
        spec_dict.update(**kwargs) # overwrite
        embed_dims = spec_dict['embed_dims']                # embedding的维度    [128, 128]
        num_group_tokens = spec_dict['num_group_tokens']    # 每个阶段的token数量  [8, 8] 
        add_pos_token = spec_dict['add_pos_token']           # 是否添加位置编码  True
        num_heads = spec_dict['num_heads']                   # 多头注意力的头数  8
        act_type = spec_dict['act_type']                     # 激活函数类型  gelu
        norm_type = spec_dict['norm_type']                   # 归一化类型  layernorm
        ffn_ratio = spec_dict['ffn_ratio']                   # 前馈网络的扩展比例  4.0
        qkv_bias = spec_dict['qkv_bias']                     # 是否使用qkv偏置  True
        drop = spec_dict['drop']                             # 0.1
        attn_drop = spec_dict['attn_drop']
        drop_path = spec_dict['drop_path']                   # drop path rate 0.1
        mixer_depth = spec_dict['mixer_depth']
        mixer_token_expansion = spec_dict['mixer_token_expansion']
        mixer_channel_expansion = spec_dict['mixer_channel_expansion']
        untied_pos_encode = spec_dict['untied_pos_encode']
        pe_dim = spec_dict['pe_dim']

        self.group_token_feat_nwd = nn.Parameter(torch.empty(num_group_tokens, embed_dims)) 
        # num_group_tokens × embed_dims 大小的可训练参数，用于存储组 token 特征。 [8,128] num_group_tokens相当于batch_size
        trunc_normal_(self.group_token_feat_nwd, std=.02, a=-.02, b=.02)   # 初始化

        if add_pos_token:
            self.group_token_pos_nwd = nn.Parameter(torch.empty(num_group_tokens, embed_dims)) 
            # num_group_tokens × embed_dims 大小的可训练参数，用于存储组 token 位置编码。
            trunc_normal_(self.group_token_pos_nwd, std=.02, a=-.02, b=.02)  # 初始化
            # 如果 untied_pos_encode 为 True，则需要额外的位置编码参数。
            if untied_pos_encode: 
                self.pos_scale = float(embed_dims / num_heads * 2) ** -0.5
                self.pos_q_norm = NormLayer(norm_type, embed_dims)
                self.pos_k_norm = NormLayer(norm_type, pe_dim)
                self.pos_k_proj = nn.Linear(pe_dim, embed_dims)
            else:
                self.pos_proj = nn.Sequential(
                    NormLayer(norm_type, pe_dim),
                    nn.Linear(pe_dim, embed_dims),
                )
        else:
            self.group_token_pos_nwd = None

        _group_attn_cfg = dict(
            embed_dims=embed_dims,
            num_heads=num_heads,                                    # 独有
            norm_type=norm_type,
            qkv_bias=qkv_bias,                                      # 独有
            qk_scale=self.pos_scale if untied_pos_encode else None, # 独有
            proj_drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
        )

        self.group_layer = LightGroupAttnBlock(**_group_attn_cfg)

        _mixer_cfg = dict(
            num_tokens=num_group_tokens,               # 独有
            embed_dims=embed_dims,
            token_expansion=mixer_token_expansion,    # 独有
            channel_expansion=mixer_channel_expansion,# 独有
            depth=mixer_depth,                        # 独有
            norm_type=norm_type,
            drop_path=drop_path,
            drop_out=drop,
        )

        self.mixer = MLPMixer(**_mixer_cfg)

        _ungroup_attn_cfg = dict(
            embed_dims=embed_dims,
            num_heads=num_heads,
            ffn_ratio=ffn_ratio,                      # 独有
            qkv_bias=qkv_bias,
            qk_scale=self.pos_scale if untied_pos_encode else None,
            act_type=act_type,
            norm_type=norm_type,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
        )

        self.ungroup_layer = FullAttnCatBlock(**_ungroup_attn_cfg)

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_group_tokens = num_group_tokens
        self.untied_pos_encode = untied_pos_encode

        self.apply(self._reset_parameters)

    def _reset_parameters(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02, a=-0.02, b=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)
    # # 基于q查询位置编码和k键位置编码 计算一个注意力偏置的矩阵
    # # 用于增强 Transformer 或注意力机制的计算。
    def get_attn_pos_bias(self, pos_q, pos_k):
        # pos_q [T, B, C]  
        tgt_len, bsz, _ = pos_q.shape
        src_len = pos_k.shape[0]
        # 归一化--[T,B,H,C/H]--->[B,H,T,C/H] --乘一个缩放因子
        pos_q = self.pos_q_norm(pos_q).view(tgt_len, bsz, self.num_heads, -1).permute(1, 2, 0, 3) * self.pos_scale
        # 归一化--投影--[T, B,H,C/H]--->[B,H,T,C/H]??
        pos_k = self.pos_k_proj(self.pos_k_norm(pos_k)).view(
            src_len, bsz, self.num_heads, -1).permute(1, 2, 0, 3)
        attn_pos_bias = torch.einsum('bhic, bhjc -> bhij', pos_q, pos_k)

        return attn_pos_bias

    def forward(self, x: Tensor, pos: Optional[Tensor] = None, key_padding_mask: Optional[Tensor] = None, kth_cluster: Optional[int] = None, **kwargs) -> Tuple[Tensor]:
        """
        Args:
            x: [T, B, C]
            pos: [T, B, C]
            key_padding_mask: [B, T]
        """

        B = x.shape[1]

        # [T, B, C]
        ungroup_tokens = x

        group_token_feat = self.group_token_feat_nwd.unsqueeze(1).expand(-1, B, -1) # [8, B, 128] 

        # --------------------------------------------------------------------------
        group_token_pos = self.group_token_pos_nwd.unsqueeze(1).expand(-1, B, -1) \
            if self.group_token_pos_nwd is not None else None
        if group_token_pos is not None:
            if self.untied_pos_encode:
                attn_pos_bias = self.get_attn_pos_bias(group_token_pos, pos)
            else:
                pos = self.pos_proj(pos)
        else:
            self.untied_pos_encode = False
            pos = None
        # --------------------------------------------------------------------------
 


        group_token_feat = self.group_layer(
            group_token_feat,    # 可学习参数 ，形状为[8, B, 128] q
            ungroup_tokens,      # [T, B, C]                     k,也叫memory
            memory_key_padding_mask=key_padding_mask,
            memory_pos=pos                  if not self.untied_pos_encode else None,
            query_pos=group_token_pos       if not self.untied_pos_encode else None,
            attn_bias=attn_pos_bias         if     self.untied_pos_encode else None,
        )[0]

        # update
        group_token_feat = self.mixer(group_token_feat.transpose(0, 1)).transpose(0, 1)

        # distribute
        ungroup_tokens, attn = self.ungroup_layer(
            ungroup_tokens, 
            group_token_feat,
            memory_mask=key_padding_mask.unsqueeze(-1).expand(-1, -1, self.num_group_tokens) if key_padding_mask is not None else None,
            memory_pos=group_token_pos                if not self.untied_pos_encode else None,
            query_pos=pos                             if not self.untied_pos_encode else None,
            attn_bias=attn_pos_bias.transpose(-2, -1) if     self.untied_pos_encode else None,
            kth_cluster=kth_cluster,
        )

        return ungroup_tokens, attn.mean(dim=1)

