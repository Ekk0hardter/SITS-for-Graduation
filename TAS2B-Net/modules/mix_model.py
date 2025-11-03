# 本文件进行了如下修改
# FADB模块去除PGFN; MFMSA和SA,MSCB合并成全新模块
# 去除了ascadedSubDecoderBinary进行上采样，采用原始的EUCB
# 将ascadedSubDecoderBinary和DPGHEAD合并成一个模块
import torch
from torch import nn

from timm.layers import trunc_normal_, get_act_layer
from modules.mix_decoder import EKKO_DECODER
from modules.seg_head import CCE_Head
# from mix_decoder import EKKO_DECODER
# from seg_head import CCE_Head


class ConvModule(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
            norm_layer=nn.BatchNorm2d,
            act_type='relu',
    ):
        super(ConvModule, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)
        self.norm = norm_layer(out_channels)
        self.act = get_act_layer(act_type)()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

# ksp=311,不改变特征图尺寸的卷积块
class BasicConvBlock(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            num_convs=2,
            stride=1,
            dilation=1,
            norm_layer=nn.BatchNorm2d,
            act_type='relu',
    ):
        super(BasicConvBlock, self).__init__()

        convs = []
        for i in range(num_convs):
            convs.append(
                ConvModule(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=stride if i == 0 else 1,
                    dilation=1 if i == 0 else dilation,
                    padding=1 if i == 0 else dilation,
                    norm_layer=norm_layer,
                    act_type=act_type
                )
            )

        self.convs = nn.Sequential(*convs)

    def forward(self, x):
        out = self.convs(x)
        return out


class DeconvModule(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        norm_layer=nn.BatchNorm2d,
        act_type='relu',
        *,
        kernel_size=4,
        scale_factor=2
    ):
        super(DeconvModule, self).__init__()

        assert (kernel_size - scale_factor >= 0) and\
               (kernel_size - scale_factor) % 2 == 0,\
               f'kernel_size should be greater than or equal to scale_factor '\
               f'and (kernel_size - scale_factor) should be even numbers, '\
               f'while the kernel size is {kernel_size} and scale_factor is '\
               f'{scale_factor}.'

        stride = scale_factor
        padding = (kernel_size - scale_factor) // 2
        deconv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

        norm = norm_layer(out_channels)
        activate = get_act_layer(act_type)()

        self.deconv_upsamping = nn.Sequential(deconv, norm, activate)

    def forward(self, x):

        out = self.deconv_upsamping(x)
        return out


class InterpConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_layer=nn.BatchNorm2d,
                 act_type='relu',
                 *,
                 conv_first=False,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 upsample_cfg=dict(
                     scale_factor=2, mode='bilinear', align_corners=False)):
        super(InterpConv, self).__init__()

        conv = ConvModule(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            norm_layer=norm_layer,
            act_type=act_type
        )
        upsample = nn.Upsample(**upsample_cfg)
        if conv_first:
            self.interp_upsample = nn.Sequential(conv, upsample)
        else:
            self.interp_upsample = nn.Sequential(upsample, conv)

    def forward(self, x):
        out = self.interp_upsample(x)
        return out


class UpConvBlock(nn.Module):

    def __init__(self, conv_block, in_channels, skip_channels, out_channels, num_convs=2, stride=1, dilation=1, norm_layer=nn.BatchNorm2d, act_type='relu', upsample_layer=InterpConv):
        super(UpConvBlock, self).__init__()

        self.conv_block = conv_block(
            in_channels=2 * skip_channels,
            out_channels=out_channels,
            num_convs=num_convs,
            stride=stride,
            dilation=dilation,
            norm_layer=norm_layer,
            act_type=act_type)

        self.upsample = upsample_layer(
            in_channels=in_channels,
            out_channels=skip_channels,
            norm_layer=norm_layer,
            act_type=act_type)

    def forward(self, skip, x):

        x = self.upsample(x)
        out = torch.cat([skip, x], dim=1)
        out = self.conv_block(out)

        return out



class EKKO(nn.Module):
    
    def __init__(self, config, **kwargs):
        super(EKKO, self).__init__()

        spec_dict = {k.lower(): v for k, v in dict(config.UNET).items()}  # 拿到config中关于unet的各种参数
        spec_dict.update(**kwargs)                                        # (若外部更新了参数)更新参数
        in_channel    = spec_dict['in_channel']         # 输入通道数                         
        base_channels = spec_dict['base_channels']      # 基础通道数
        num_stages    = spec_dict['num_stages']         # 网络的阶段数(上采样，下采样的次数)  4
        strides       = spec_dict['strides']            # [1, 1, 1, 1]
        enc_num_convs = spec_dict['enc_num_convs']      # 编码器的卷积次数 [2, 2, 2, 2] 4次编码，每次编码两次卷积
        dec_num_convs = spec_dict['dec_num_convs']      # 解码器的卷积次数 [2, 2, 2]    3次解码，每次解码两次卷积
        downsamples   = spec_dict['downsamples']        # 下采样[True, True, True],意为三次下采样
        enc_dilations = spec_dict['enc_dilations']      # 编码器的膨胀率  [1, 1, 1, 1]  4次编码，每次膨胀率为1
        dec_dilations = spec_dict['dec_dilations']      # 解码器的膨胀率  [1, 1, 1]     3次解码，每次膨胀率为1
        norm_type     = spec_dict['norm_type']          # 归一化类型  默认bn
        act_type      = spec_dict['act_type']           # 激活函数类型  默认gelu
        upsample_type = spec_dict['upsample_type']      # 上采样的类型 interp 使用插值进行上采样
        in_channels   = spec_dict['in_channels']        # 每个下采样后的通道数 [64, 128, 320, 512]
 

        self.num_stages = num_stages
        self.strides = strides
        self.downsamples = downsamples
        self.base_channels = base_channels

        if norm_type == 'bn':
            norm_layer = nn.BatchNorm2d
        else:
            raise NotImplementedError

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.out_dims = []

        self.emcad = EKKO_DECODER(channels=[512, 320, 128, 64], kernel_sizes=[1, 3, 5], expansion_factor=2, dw_parallel=True, add=True, lgag_ks=3, activation='relu')

        self.head = CCE_Head(in_channels[0], 19)

        # 空间编码器
        for i in range(num_stages):  # i=0,1,2,3
            enc_conv_block = []   # 编码器的下采样卷积块
            if i != 0:            # i=1,2,3
                if strides[i] == 1 and downsamples[i - 1]:
                    enc_conv_block.append(nn.MaxPool2d(kernel_size=2))  # 降采样
                
            # Unet的第一次卷积(非下采样)
            enc_conv_block.append(
                BasicConvBlock(
                    in_channels=in_channel,
                    out_channels=in_channels[i],
                    num_convs=enc_num_convs[i],
                    stride=strides[i],
                    dilation=enc_dilations[i],
                    norm_layer=norm_layer,
                    act_type=act_type,
                ))
            

            self.encoder.append((nn.Sequential(*enc_conv_block)))
            in_channel = in_channels[i]
            self.out_dims.append(in_channel)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            trunc_normal_(m.weight, std=0.02, a=-0.02, b=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        self._check_input_divisible(x)
        # spatial encoder
        enc_outs = []
        for enc in self.encoder:
            x = enc(x)
            enc_outs.append(x)

        # spatial decoder
        x1, x2, x3, x4 = enc_outs
        dec_outs = self.emcad(x4, [x3, x2, x1])
        
        # head
        output = self.head(dec_outs[3])

        return output

    # 检查输入 x 的h和w 是否能被整个encoder的下采样倍率整除
    def _check_input_divisible(self, x):
        h, w = x.shape[-2:]
        whole_downsample_rate = 1
        for i in range(1, self.num_stages):
            if self.strides[i] == 2 or self.downsamples[i - 1]:
                whole_downsample_rate *= 2
        assert (h % whole_downsample_rate == 0) \
            and (w % whole_downsample_rate == 0),\
            f'The input image size {(h, w)} should be divisible by the whole '\
            f'downsample rate {whole_downsample_rate}, when num_stages is '\
            f'{self.num_stages}, strides is {self.strides}, and downsamples '\
            f'is {self.downsamples}.'


if __name__ == '__main__':
    import torch
    from config import _C as config

    model = EKKO(config, **{'in_channel': 128})
    x = torch.randn(2, 128, 48, 48)
    y = model(x)
    print(y.shape)   # [B,num_classes,H,W] 