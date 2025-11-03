import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn

def get_freq_indices(method):
    assert method in ['top1','top2','top4','top8','top16','top32',
                      'bot1','bot2','bot4','bot8','bot16','bot32',
                      'low1','low2','low4','low8','low16','low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0,0,6,0,0,1,1,4,5,1,3,0,0,0,3,2,4,6,3,5,5,2,6,5,5,3,3,4,2,2,6,1]
        all_top_indices_y = [0,1,0,5,2,0,2,0,0,6,0,4,6,3,5,2,6,3,3,3,5,1,1,2,4,2,1,1,3,0,5,3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0,0,1,1,0,2,2,1,2,0,3,4,0,1,3,0,1,2,3,4,5,0,1,2,3,4,5,6,1,2,3,4]
        all_low_indices_y = [0,1,0,1,2,0,1,2,2,3,0,0,4,3,1,5,4,3,2,1,0,6,5,4,3,2,1,0,6,5,4,3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6,1,3,3,2,4,1,2,4,4,5,1,4,6,2,5,6,1,6,2,2,4,3,3,5,5,6,2,5,5,3,6]
        all_bot_indices_y = [6,4,4,6,6,3,1,4,4,5,6,5,2,2,5,1,4,3,5,0,3,1,1,2,4,2,1,1,5,3,3,3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y



class DCT(nn.Module):
    def __init__(self,in_channels, dct_h, dct_w, frequency_branches=16,frequency_selection='top'):
        super(DCT, self).__init__()

        assert frequency_branches in [1, 2, 4, 8, 16, 32]  # 决定选取多少个频率分量，越大表示提取的频率信息越丰富。
        frequency_selection = frequency_selection + str(frequency_branches)  # 择哪些频率分量（比如 top16 表示选取前 16 个最高频成分）
 
        self.num_freq = frequency_branches
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(frequency_selection)  # 取离散余弦变换（DCT）的频率索引，决定使用哪些 DCT 频率分量。
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]

        assert len(mapper_x) == len(mapper_y)  

        for freq_idx in range(frequency_branches):
            self.register_buffer('dct_weight_{}'.format(freq_idx), self.get_dct_filter(dct_h, dct_w, mapper_x[freq_idx], mapper_y[freq_idx], in_channels))
    
    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, in_channels):
        dct_filter = torch.zeros(in_channels, tile_size_x, tile_size_y)

        for t_x in range(tile_size_x):
            for t_y in range(tile_size_y):
                dct_filter[:, t_x, t_y] = self.build_filter(t_x, mapper_x, tile_size_x) * self.build_filter(t_y, mapper_y, tile_size_y)

        return dct_filter

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)
    

# DCT-Enhanced Frequency Channel Attention
# 通过 DCT 变换增强频率通道注意力
class DFCA(nn.Module):
    def __init__(self, in_channels, dct_h, dct_w, reduction=16):
        super(DFCA, self).__init__()
        self.dct_h = dct_h
        self.dct_w = dct_w
        self.dct   = DCT(in_channels, dct_h, dct_w, frequency_branches=16, frequency_selection='top')
        self.num_freq = 16

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, stride=1, padding=0, bias=False))

        self.average_channel_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_channel_pooling     = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        batch_size, C, H, W = x.shape

        x_pooled = x  

        # 若输入的H和W与 DCT 期望的大小不同，就进行自适应平均池化，将输入大小调整到 dct_h x dct_w
        if H != self.dct_h or W != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))

        multi_spectral_feature_avg, multi_spectral_feature_max, multi_spectral_feature_min = 0, 0, 0
        # 对每个频率分量进行 DCT 变换
        for name, params in self.dct.state_dict().items():
            if 'dct_weight' in name:
                x_pooled_spectral = x_pooled * params  # [B, C, H, W] * [C, H, W] = [B, C, H, W] 对应像素相乘
                multi_spectral_feature_avg += self.average_channel_pooling(x_pooled_spectral)  # 对每个频率分量进行平均池化
                multi_spectral_feature_max += self.max_channel_pooling(x_pooled_spectral)      # 对每个频率分量进行最大池化
                multi_spectral_feature_min += -self.max_channel_pooling(-x_pooled_spectral)    # 对每个频率分量进行最小池化
            
        # 取平均值，确保不同频率通道的贡献一致
        multi_spectral_feature_avg = multi_spectral_feature_avg / self.num_freq
        multi_spectral_feature_max = multi_spectral_feature_max / self.num_freq
        multi_spectral_feature_min = multi_spectral_feature_min / self.num_freq

        # 对avg、max、min进行全连接层映射，得到通道注意力图
        multi_spectral_avg_map = self.fc(multi_spectral_feature_avg).view(batch_size, C, 1, 1)
        multi_spectral_max_map = self.fc(multi_spectral_feature_max).view(batch_size, C, 1, 1)
        multi_spectral_min_map = self.fc(multi_spectral_feature_min).view(batch_size, C, 1, 1)
        # 通过 Sigmoid 归一化，将特征图与注意力图相乘，增强有用信息
        multi_spectral_attention_map = F.sigmoid(multi_spectral_avg_map + multi_spectral_max_map + multi_spectral_min_map)

        return x * multi_spectral_attention_map.expand_as(x)

# HybridAttentionFusionBlock
class HAF(nn.Module):
    def __init__(self,
                 in_channels,
                 scale_branches=2,          # 多尺度分支数，决定有多少个不同尺度的卷积操作
                 frequency_branches=16,     # 多频率分支数，决定如何进行频率注意力建模  top16
                 frequency_selection='top', # 频率通道选择策略
                 block_repetition=1,        # 表示该模块是否需要重复
                 min_channel=64,            # 限制最小通道数
                 min_resolution=6,          # 限制最小分辨率
                 groups=32):                # 卷积的分组数，影响计算效率和特征提取能力
        super(HAF, self).__init__()

        self.scale_branches = scale_branches
        self.frequency_branches = frequency_branches
        self.block_repetition = block_repetition
        self.min_channel = min_channel
        self.min_resolution = min_resolution
        # -------------------多尺度卷积-----------------------
        self.multi_scale_branches = nn.ModuleList([])
        for scale_idx in range(scale_branches):  # scale_idx=0,1
            inter_channel = in_channels // 2**scale_idx
            if inter_channel < self.min_channel: inter_channel = self.min_channel

            self.multi_scale_branches.append(nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1 + scale_idx, dilation=1 + scale_idx, groups=groups, bias=False), # 33卷积，并进行分组卷积
                nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, inter_channel, kernel_size=1, stride=1, padding=0, bias=False), #  11卷积
                nn.BatchNorm2d(inter_channel), nn.ReLU(inplace=True)
            ))
        # ---------------------------------------------------

        c2wh = dict([(32, 96), (64, 48), (128, 24), (320, 12), (512, 6)])  # 前者是通道数，后者是对应的分辨率
        self.multi_frequency_branches = nn.ModuleList([])
        self.multi_frequency_branches_conv1 = nn.ModuleList([])
        self.multi_frequency_branches_conv2 = nn.ModuleList([])
        self.alpha_list = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(scale_branches)])
        self.beta_list = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(scale_branches)])

        # --------------------多频率通道注意力--------------------------------
        for scale_idx in range(scale_branches):
            inter_channel = in_channels // 2**scale_idx
            if inter_channel < self.min_channel: inter_channel = self.min_channel

            if frequency_branches > 0:
                self.multi_frequency_branches.append(
                    nn.Sequential(
                        DFCA(inter_channel, c2wh[in_channels], c2wh[in_channels], reduction=16)))
            # 1×1 卷积降维，然后通过 Sigmoid 归一化生成一个空间注意力图，用于衡量每个位置的重要性
            self.multi_frequency_branches_conv1.append(
                nn.Sequential(
                    nn.Conv2d(inter_channel, 1, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.Sigmoid()))
            # 将频率信息、空间注意力图 和 原始特征 进行融合
            self.multi_frequency_branches_conv2.append(
                nn.Sequential(
                    nn.Conv2d(inter_channel, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True)))
            
        

    def forward(self, x):
        feature_aggregation = 0
        for scale_idx in range(self.scale_branches):
            # 池化降低分辨率，scale_idx=0不做池化，scale_idx=1时，用2×2池化，分辨率变成 1/2以此类推
            feature = F.avg_pool2d(x, kernel_size=2 ** scale_idx, stride=2 ** scale_idx, padding=0) if int(x.shape[2] // 2 ** scale_idx) >= self.min_resolution else x
            # 如果池化后 feature 的分辨率小于 self.min_resolution，就直接使用 x，避免过度缩小导致信息丢失
            # 通过多尺度卷积提取特征
            feature = self.multi_scale_branches[scale_idx](feature)
            # ----------------------------------------------------------------------------------
            if self.frequency_branches > 0:    # 使用多频率注意力增强特征
                feature = self.multi_frequency_branches[scale_idx](feature) 
            spatial_attention_map = self.multi_frequency_branches_conv1[scale_idx](feature)  # 计算空间注意力图
            # 计算最终特征
            feature = self.multi_frequency_branches_conv2[scale_idx](feature * (1 - spatial_attention_map) * self.alpha_list[scale_idx] + feature * spatial_attention_map * self.beta_list[scale_idx])
            # 恢复原始分辨率
            feature_aggregation += F.interpolate(feature, size=None, scale_factor=2**scale_idx, mode='bilinear', align_corners=None) if (x.shape[2] != feature.shape[2]) or (x.shape[3] != feature.shape[3]) else feature
        
        feature_aggregation /= self.scale_branches # 归一化，确保所有尺度特征平均分配权重，防止尺度间不平衡。        
        feature_aggregation += x                   # 残差连接
        

        return feature_aggregation


    
# if __name__ == "__main__":

#     mfmsa = MFMSAttentionBlock(in_channels=512)
#     x = torch.randn(2, 512, 6, 6)
#     y = mfmsa(x)
#     print(y.shape)  # torch.Size([1, 64, 64, 64])

if __name__ == "__main__":

    model = DCT(in_channels=64,dct_h=48, dct_w=48)
    # print(model.state_dict())
    print(type(model.state_dict()))
    print(model.state_dict().keys())
    print(model.state_dict()['dct_weight_0'].shape)  # [64, 48, 48]
