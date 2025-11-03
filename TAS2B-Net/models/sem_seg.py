import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import torchvision.transforms.functional as TF
from typing import List, Tuple, Dict

from modules.exchanger import Exchanger          # 时间编码器和空间编码器（Unet既是空间编码器又是空间解码器）
from modules.ltae  import LTAE2d                 # 轻量级时间注意编码器

# 两个可选的空间编解码器
from modules.unet import UNet
from modules.mix_model import EKKO

from modules.tpe import TemporalPositionalEncoding   # 时间位置编码
from modules.loss import FocalCELoss

class Segmentor(nn.Module):
    def __init__(self, config, **kwargs):
        super(Segmentor, self).__init__()

        self.mode = kwargs['mode'] # mode 控制模型运行模式 (train/val/test)
        spec_dict = {k.lower(): v for k, v in dict(config.SEGMENTOR).items()}  # 拿到语义分割下面的配置参数
        modality = config.DATASET.MODALITY[0] # 数据集的模态 这里应该使用的是光学数据  S2

        # 时间 位置编码需要的参数
        pos_encode_type = spec_dict['pos_encode_type']     # 时间位置编码器的类型  PE模块  默认为default
        with_gdd_pos = spec_dict['with_gdd_pos']
        pe_dim = spec_dict['pe_dim']
        pe_t = spec_dict['pe_t']
        max_temp_len = spec_dict['max_temp_len']

        # 空间编码器需要的参数
        space_encoder_type = spec_dict['space_encoder_type'] # 空间编码器的类型  默认为unet
        in_dim = config.DATASET.INPUT_DIM[0]      # 输入数据的通道数
        num_classes = config.DATASET.NUM_CLASSES  # 分类的类别数
        ignore_index = config.LOSS.IGNORE_INDEX   # 忽略的索引
        loss_type = config.LOSS.TYPE              # 损失函数的类型


        self.temp_pos_encode = TemporalPositionalEncoding(pos_encode_type, pe_dim, T=pe_t, with_gdd_pos=with_gdd_pos, max_len=max_temp_len,)

        self.temp_encoder1 = Exchanger(config, **{'in_channels': in_dim, 'pe_dim': pe_dim}) # 时间编码器
        self.temp_encoder2 = LTAE2d(in_channels=in_dim, d_model=pe_dim, T=pe_t, return_att=True, positional_encoding=True) # 时间编码器2
        self.fusion = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False)
        temp_out_dim = self.temp_encoder2.out_dim

        if space_encoder_type == 'unet':                                                   # 空间编码器
            self.space_encoder = UNet(config, **{'in_channels': temp_out_dim})
            last_dim = self.space_encoder.out_dims[0]
            self.cls_head = nn.Conv2d(last_dim, num_classes, 1)
            if loss_type == 'crossentropy':
                self.ce_loss = nn.CrossEntropyLoss(
                    ignore_index=ignore_index,
                    reduction='mean',
                    label_smoothing=config.LOSS.SMOOTH_FACTOR
                )
            elif loss_type == 'focal_ce':
                gamma = config.LOSS.FOCAL[1]
                self.ce_loss = FocalCELoss(gamma=gamma, size_average=True, ignore_index=ignore_index)
                r = 0.01
                nn.init.constant_(self.cls_head.bias, -1.0 * math.log((1 - r) / r))
            else:
                raise NotImplementedError
        elif space_encoder_type == 'ekko':
            self.space_encoder = EKKO(config, **{'in_channel': temp_out_dim})
            last_dim = self.space_encoder.out_dims[0]
            self.cls_head = nn.Conv2d(last_dim, num_classes, 1)
            if loss_type == 'crossentropy':
                self.ce_loss = nn.CrossEntropyLoss(
                    ignore_index=ignore_index,
                    reduction='mean',
                    label_smoothing=config.LOSS.SMOOTH_FACTOR
                )
            elif loss_type == 'focal_ce':
                gamma = config.LOSS.FOCAL[1]
                self.ce_loss = FocalCELoss(gamma=gamma, size_average=True, ignore_index=ignore_index)
                r = 0.01
                nn.init.constant_(self.cls_head.bias, -1.0 * math.log((1 - r) / r))
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        self.modality = modality
        self.space_encoder_type = space_encoder_type
        self.ignore_index = ignore_index
        self.num_classes = num_classes

    # 解析输入数据
    def parse_inputs(self, inputs: Dict[str, Tensor], modality: str):

        data = inputs[f'data_{modality}'][:, :-1, ...].float()                    # 去掉第二维（T维）的最后一个时间步
        img_mask = inputs[f'data_{modality}'][:, -1, ...].bool()                  # 将第二维最后一个值叫做img_mask [B,1,C,H,W]  
        date_pos = inputs[f'date_positions_{modality}'][:, 0, ...].long()
        temporal_mask = inputs[f'date_positions_{modality}'][:, 1, ...].bool()    # 提取时间信息，取出T维第一个时间步
        labels = inputs['label'][:, 0, ...].long()                                # [B, H, W] 提取标签信息
        spatial_pad_masks = inputs['label'][:, -1, ...].bool()
        labels = labels.masked_fill(spatial_pad_masks, self.ignore_index)

        return data, img_mask, date_pos, temporal_mask, labels
    
    # 根据不同的空间解码器选择不同的损失函数
    def losses(self, outputs: Dict[str, Tensor], targets: Tensor, **kwargs):

        preds = outputs['preds']
        if preds.shape[-2:] != targets.shape[-2:]:
            preds = F.interpolate(preds, size=targets.shape[-2:], mode='bilinear', align_corners=False)
        ce_loss = self.ce_loss(preds, targets)
        losses = {'loss_ce': ce_loss}

        return losses

    def forward(self,inputs: Dict[str, Tensor],**kwargs) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:

        outputs = {}

        data, img_mask, date_pos, temporal_mask, labels = self.parse_inputs(inputs, self.modality)  # 解析输入数据

        B, _, T, H, W = data.shape   # 原始形状是[B, C, T, H, W]
        num_patches = H * W          # patch size 1
        x = data.reshape(B, -1, T, num_patches).contiguous().permute(0, 3, 2, 1)\
            .contiguous().view(B * num_patches, T, -1)                                       # x [B, C, T, H, W]->[BHW, T, C]  
        img_mask = img_mask.view(B, T, num_patches).transpose(1, 2).contiguous().view(B * num_patches, T) # img_mask[B, T, H, W]->[BHW, T]
        temp_pos = self.temp_pos_encode(date_pos, key_padding_mask=temporal_mask)                         # temp_pos [B,T,C]
        temp_pos = temp_pos.repeat_interleave(num_patches, dim=0).transpose(0, 1)                         # temp_pos [B, T, C]->[T, BHW, C]

        out_GPB = self.temp_encoder1(x, temp_pos, img_mask)[0]          # 时间编码器1  out_GPB  [BHW, T, C]
        # -----------------------------------------------------------------------------------------------
        input = data.permute(0, 2, 1, 3, 4)  # input [B, T, C, H, W]
        out_ltae, attn = self.temp_encoder2(input, batch_positions=date_pos, pad_mask=temporal_mask) 
        # 时间编码器2 out_ltae[B, C, H, W]  attn [Heads,B,T, H, W]
        # 开始消去out的T维
        out_GPB = out_GPB.view(B, H, W, T, -1).permute(0, 3, 4, 1, 2)   # out_GPB [BHW, T, C]->[B, T, C, H, W]
        x = Temporal_Aggregator(out_GPB, temporal_mask, attn)           # out_GPB [B, C, H, W]



        if self.space_encoder_type == 'maskformer':
            out: Dict[str, Tensor] = self.space_encoder(x, (H, W))
            outputs.update(**out)
        elif self.space_encoder_type == 'EMCADNET':
            outputs['preds'] = self.space_encoder(x)[-1]
        elif self.space_encoder_type == 'ekko':
            outputs['preds'] = self.space_encoder(x)
        else:
            feats: List[Tensor] = self.space_encoder(x)
            outputs['preds'] = self.cls_head(feats[-1] if self.space_encoder_type == 'unet' else feats)

        outputs['labels'] = labels

        if self.mode != 'test':
            losses = self.losses(outputs, labels) # 计算损失
        else:
            losses = {}

        return outputs, losses




    def multi_crop_inference(self, inputs: Dict[str, Tensor], crop_size):
        # 将原始输入拆分为多个补丁并拼接预测
        # CUDA OOM out of memory的解决方法
        # 不适用于 MaskFormer
        x = inputs.pop(f'data_{self.modality}')
        B = x.shape[0]
        orig_h, orig_w = x.shape[-2:]
        assert B == 1, f'only supports batchsize 1'
        assert crop_size[0] < orig_h and crop_size[1] < orig_w, \
            f'crop size should be smaller than input size {orig_h, orig_w}, but got {crop_size}'

        stride_h = int(crop_size[0] * 2. / 3.)
        stride_w = int(crop_size[1] * 2. / 3.)
        nrows = int(np.ceil((orig_h - crop_size[0]) / stride_h)) + 1
        ncols = int(np.ceil((orig_w - crop_size[1]) / stride_w)) + 1
        final_pred = torch.zeros((B, self.num_classes, orig_h, orig_w)).to(x)
        count = torch.zeros((B, 1, orig_h, orig_w)).to(x)

        for i in range(nrows):
            for j in range(ncols):
                h0 = i * stride_h
                w0 = j * stride_w
                h1 = min(h0 + crop_size[0], orig_h)
                w1 = min(w0 + crop_size[1], orig_w)
                patch = x[..., h0:h1, w0:w1]
                if patch.shape[-2] < crop_size[0] or patch.shape[-1] < crop_size[1]:
                    img = patch[:, :-1, ...]
                    mask = patch[:, -1, ...].unsqueeze(dim=1)
                    img = pad_if_smaller(img, crop_size, fill=0.)
                    mask = pad_if_smaller(mask, crop_size, fill=1)
                    patch = torch.cat([img, mask], dim=1)
                inputs[f'data_{self.modality}'] = patch
                outputs, _ = self.forward(inputs)
                preds = outputs['preds']
                if preds.shape[-2:] != crop_size:
                    preds = F.interpolate(preds, size=crop_size, mode='bilinear', align_corners=False)
                final_pred[..., h0:h1, w0:w1] += preds[..., :h1-h0, :w1-w0]
                count[..., h0:h1, w0:w1] += 1

        final_pred = final_pred / count
        labels = outputs['labels']
        return {'preds': final_pred, 'labels': labels}



def pad_if_smaller(data, size, fill=0):
    h, w = data.shape[-2:]
    if h < size[0] or w < size[1]:
        pad_h = size[0] - h if h < size[0] else 0
        pad_w = size[1] - w if w < size[1] else 0
        data = TF.pad(data, (0, 0, pad_w, pad_h), fill=fill)

    return data


def Temporal_Aggregator(x, pad_mask=None, attn_mask=None):
    n_heads, b, t, h, w = attn_mask.shape
    attn = attn_mask.view(n_heads * b, t, h, w)  # attn = [HEADS*B, T, H, W]

    # 保证attn的大小与x的输入大小一致
    if x.shape[-2] > w:
        attn = nn.Upsample(
            size=x.shape[-2:], mode="bilinear", align_corners=False
        )(attn)
    else:
        attn = nn.AvgPool2d(kernel_size=w // x.shape[-2])(attn)

    attn = attn.view(n_heads, b, t, *x.shape[-2:]) # 恢复 attn = [HEADS, B, T, H, W]
    attn = attn * (~pad_mask).float()[None, :, :, None, None]  # 将padding的位置掩盖掉，避免对无效的填充区域进行加权   [HEADS, B, T, H, W] * [1, B, T, 1, 1] = [HEADS, B, T, H, W]  本质上是缩小了T维

    out = torch.stack(x.chunk(n_heads, dim=2))  # x  输入数据 [HEADS, B, T, C/HEADS, H, W]
    out = attn[:, :, :, None, :, :] * out      
    # [HEADS, B, T, 1, H, W] * [HEADS, B, T, C/HEADS, H, W] = [HEADS, B, T, C/HEADS, H, W]
    out = out.sum(dim=2)  # sum on temporal dim -> [HEADS, B, C/HEADS, H, W]
    out = torch.cat([group for group in out], dim=1)  # -> [B, C, H, W]
    return out