import torch
from torch import nn
import torch.nn.functional as F

# Cascaded Context-Enhanced Segmentation Head
class CCE_Head(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CCE_Head, self).__init__()
        self.sca=SpatialContextAggregator(in_channels, num_classes, 1)
        self.ce=CEHead(in_channels, in_channels, pool='att', fusions=['channel_mul'])
        self.fin_seg_head = nn.Conv2d(in_channels, num_classes, 1)

    
    def forward(self, x):
        context = self.sca(x)              # [B, C, N, 1]
        object_context = self.ce(x, context)+ x             # [B, C, H, W]
        output = self.fin_seg_head(object_context)           # [B, N, H, W]

        return output


# SCA 空间上下文聚合模块
class SpatialContextAggregator(nn.Module):
  
    def __init__(self, in_channels, num_classes, scale):
        super(SpatialContextAggregator, self).__init__()
        self.map_conv      = nn.Conv2d(in_channels, num_classes, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.distance_conv = nn.Conv2d(in_channels, num_classes, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.boundary_conv = nn.Conv2d(in_channels, num_classes, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.scale = scale

    def forward(self, feats):
        # feats: [B, C, H, W] 特征图
        
        map      = self.map_conv(feats)                                    # 计算主分割图
        distance = self.distance_conv(feats) * torch.sigmoid(map)         # 计算距离图
        boundary = self.boundary_conv(feats) * torch.sigmoid(distance)    # 计算边界图
        # 级联增强，但不改变分辨率
        distance = distance + torch.sigmoid(boundary)      # 使distance在边界区域有更强的特征
        probs = map + torch.sigmoid(distance)              # 让分割图受边界特征的影响
        # probs: B, N=num_classes, H, W 概率分布
        B, N, H, W = probs.size()                          # 拿到B和num_classes
        C = feats.size(1)                                  # 拿到特征图的channel
        probs = probs.view(B, N, -1)                       # probs->[B, N, H*W]
        feats = feats.view(B, C, -1)                       # feats->[B, C, H*W]
        feats = feats.permute(0, 2, 1)                     # feats->[B, H*W, C]
        probs = F.softmax(self.scale * probs, dim=2)       # softmax使probs变成权重矩阵，用于计算特征加权求和probs->[B, N, H*W]

        ocr_context = torch.matmul(probs, feats)           # [B, N, H*W] * [B, H*W, C] = [B, N, C]
        ocr_context = ocr_context.permute(0, 2, 1).contiguous().unsqueeze(3) # [B, C, N, 1]
        return ocr_context

# ContextualEnhancementHead  基于上下文的特征增强，适用于注意力池化（att）与全局特征融合任务。
class CEHead(nn.Module):
    def __init__(self, in_channel, mid_channel, pool, fusions):
        super(CEHead, self).__init__()
        assert pool in ['avg', 'att']                                        # 采用平均池化或者注意力池化
        assert all([f in ['channel_add', 'channel_mul'] for f in fusions])   # 通道加性注意力或者通道乘性注意力
        assert len(fusions) > 0, 'at least one fusion should be used'
        self.in_channel = in_channel
        self.mid_channel = mid_channel
        self.pool = pool
        self.fusions = fusions

        if 'att' in pool:
            self.conv_mask = nn.Conv2d(self.in_channel, 1, kernel_size=1)   # 11卷积层，用于生成注意力权重
            self.softmax = nn.Softmax(dim=2)                                # 归一化注意力权重，进行加权求和
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)                         # 平均池化，输出的空间尺寸为 1×1
        
        if 'channel_add' in fusions:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.in_channel, self.mid_channel, kernel_size=1),
                nn.LayerNorm([self.mid_channel, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.mid_channel, self.in_channel, kernel_size=1)
            )
        else:
            self.channel_add_conv = None
        

        if 'channel_mul' in fusions:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.in_channel, self.mid_channel, kernel_size=1),
                nn.LayerNorm([self.mid_channel, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.mid_channel, self.in_channel, kernel_size=1)
            )
        else:
            self.channel_mul_conv = None
        
        self.reset_parameters()
    
    # 初始化卷积层参数
    def reset_parameters(self):
        if self.pool == 'att':
            kaiming_init(self.conv_mask, mode='fan_in')
            self.conv_mask.inited = True

        if self.channel_add_conv is not None:
            last_zero_init(self.channel_add_conv)
        if self.channel_mul_conv is not None:
            last_zero_init(self.channel_mul_conv)

    def spatial_pool(self, x):
        B, C, H, W = x.size()
        if self.pool == 'att':
            input_x = x
            input_x = input_x.view(B, C, H*W)                 # [B, C, H*W]
            input_x = input_x.unsqueeze(1)                    # [B, 1, C, H*W]

            context_mask = self.conv_mask(x)                  # [B, C=1, H, W]
            context_mask = context_mask.view(B, 1, H*W)       # [B, 1, H*W]
            context_mask = self.softmax(context_mask)         # [B, 1, H*W]
            context_mask = context_mask.unsqueeze(3)          # [B, 1, H*W, 1]
            context = torch.matmul(input_x, context_mask)     # [B, 1, C, H*W]*[B, 1, H*W, 1] = [B, 1, C, 1]
            context = context.view(B, C, 1, 1)                # [B, C, 1, 1]
        else:

            context = self.avg_pool(x)                        # [B, C, 1, 1]

        return context

    def forward(self, x, y):
        # x [B, C=64, H, W]             没有进分割头的最后输出
        # y [B, C, N, 1] 由SpatialContextAggregator生成
        context = self.spatial_pool(y)  # [B, C, 1, 1]

        if self.channel_mul_conv is not None:
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context)) # [N, C, 1, 1]
            out = x * channel_mul_term #  [B, C=64, H, W]*[N, C, 1, 1] =[N, D, H, W]
        else:
            out = x
        if self.channel_add_conv is not None:
            channel_add_term = self.channel_add_conv(context)# [N, D, 1, 1]
            out = out + channel_add_term

        return out

def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.kaiming_uniform_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
        m[-1].inited = True
    else:
        constant_init(m, val=0)
        m.inited = True

if __name__ == '__main__':
    model   = CCE_Head(64, 19) # 输入通道数，输出类别数
    x = torch.randn(2, 64, 48, 48)
    out = model(x)
    print(out.shape)