from typing import List, Optional
import numpy as np
import torch
from torch import nn, Tensor


from modules.linear import LinearLayer
from modules.GPBlock import GPBlock


class Exchanger(nn.Module):
    def __init__(self, config, **kwargs):
        super(Exchanger, self).__init__()

        spec_dict = {k.lower(): v for k, v in dict(config.EXCHANGER).items()}
        # 通过传入的 config 配置文件中的 EXCHANGER 字段，将配置转换为小写键的字典
        embed_dims: List[int] = spec_dict['embed_dims'] # 每个阶段的嵌入维度
        num_token_list = spec_dict['num_token_list']    # 每个阶段的 token 数量
        num_heads_list = spec_dict['num_heads_list']    # 每个阶段的头数
        drop_path_rate = spec_dict['drop_path_rate']    # drop path rate
        mlp_norm = spec_dict['mlp_norm']                # 定义MLP层的归一化类型
        mlp_act = spec_dict['mlp_act']                  # 定义MLP层的激活函数类型
        in_dim = kwargs['in_channels']                  # 输入数据的通道数
        pe_dim = kwargs['pe_dim']                       # 位置编码的维度

        num_stages = len(num_token_list)               # 阶段数  2
        assert len(embed_dims) == len(num_heads_list) == num_stages
        dpr = np.linspace(0, drop_path_rate, num_stages)

        self.mlp_layers = nn.ModuleList()
        self.temp_encoder_blocks = nn.ModuleList()

        # 创建 num_stages 个 LinearLayer，用于转换特征维度
        # 创建 num_stages 个 GPBlock，它的作用可能是进行特征提取或变换。
        for i in range(num_stages):

            self.mlp_layers.append(
                LinearLayer(
                    in_dim=in_dim if i == 0 else embed_dims[i-1],
                    out_dim=embed_dims[i],
                    norm_type=mlp_norm,
                    act_type=mlp_act,
                    bias=False))

            self.temp_encoder_blocks.append(
                GPBlock(config,
                    **{
                        'embed_dims': embed_dims[i],
                        'num_group_tokens': num_token_list[i],
                        'num_heads': num_heads_list[i],
                        'drop_path': dpr[i],
                        'pe_dim': pe_dim,
                     }))

        self.num_stages = num_stages
        self.out_dim = embed_dims[-1] # 最后一个阶段的输出维度
    # 输入x[B, T, C] B=BHW
    def forward(self, x: Tensor, temp_pos: Tensor, temp_mask: Optional[Tensor] = None, return_attn: bool = False, kth_cluster: Optional[int] = None, **kwargs) -> Tensor:
        """
        Args Shape:
            x: [B, T, C]
            temp_pos: [T, B, C]
            temp_mask: [B, T]
        """
        attn_list = []
        feat_list = []

        for i in range(self.num_stages):
            x = self.mlp_layers[i](x)  # 进入mlp层
            x = x.transpose(0, 1)      # [B, T, C] -> [T, B, C]
            # 进入GPBlock(时间编码模块)
            x, attn = self.temp_encoder_blocks[i](x, temp_pos, temp_mask, kth_cluster=kth_cluster) 
           
            if return_attn:
                attn_list.append(attn)
            x = x.transpose(0, 1)     # [T, B, C] -> [B, T, C] 再换回来，以便进入下一次循环
            if return_attn:
                feat_list.append(x)

        if return_attn:
            return x, torch.stack(attn_list, dim=1), torch.stack(feat_list, dim=1)
        else:
            return x, None, None

if __name__ == '__main__':
    config = {
        'EXCHANGER': {
            'embed_dims': [128, 128],
            'num_token_list': [64, 64],
            'num_heads_list': [8, 8],
            'drop_path_rate': 0.1,
            'mlp_norm': 'ln',
            'mlp_act': 'gelu'
        }
    }
    # config = Config(**config)
    exchanger = Exchanger(config, in_channels=3, pe_dim=128)
    x = torch.randn(2, 64, 3)
    temp_pos = torch.randn(64, 2, 128)
    temp_mask = torch.randn(2, 64)
    out = exchanger(x, temp_pos, temp_mask)
    print(out[0].shape)
    print(out[1].shape)
    print(out[2].shape)
