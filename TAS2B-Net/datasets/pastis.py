from typing import Tuple
import datetime as dt
import json
import os
import json

import numpy as np
import pandas as pd
import geopandas as gpd

import torch
from torch.utils import data


class PASTIS(data.Dataset):
    def __init__(
        self,
        config,
        nfolds=[1, 2, 3],
        reference_date='2018-09-01',
        mode='train',
        **kwargs
    ):
        super(PASTIS, self).__init__()

        root = config.DATASET.ROOT                 # 拿到数据集的根目录，分割任务就是PASTIS-R(非pixelset)
        self.modality = config.DATASET.MODALITY    # 拿到数据集的模态list
        self.task_type = config.DATASET.TASK_TYPE  # 拿到任务类型，分割任务就是sem_seg
        self.ignore_index = config.LOSS.IGNORE_INDEX # -1
        """
        The void label is reserved for out-of-scope parcels, either because their crop type
        is not in nomenclature or their overlap with the selected square patch is too small.
        空标签仅用于超出范围的地块，因为它们的作物类型不在命名法中，或者它们与所选方形斑块的重叠太小。
        """
        self.void_label = 19  # 无效标签的编号是19  表示那些不属于任何已知作物类别的区域
        self.data_folders = [os.path.join(root, f'DATA_{s}') for s in self.modality] # 拿到数据文件夹路径
        self.semantic_label_folder = os.path.join(root, 'ANNOTATIONS')               # 拿到语义标签文件夹路径
        self.instance_label_folder = os.path.join(root, 'INSTANCE_ANNOTATIONS')      # 拿到实例标签文件夹路径

        self.max_val = 32767
        self.reference_date = dt.datetime(*map(int, reference_date.split("-")))  # 通过解析传入的日期字符串来创建日期对象，表示参考日期。
        self.temp_drop_rate = config.DATASET.TEMP_DROP_RATE # 时间编码器的dropout rate
        self.random_crop = config.TRAIN.RANDOM_CROP         # 是否进行随机裁剪 默认为true
        self.crop_size = config.TRAIN.CROP_SIZE             # 裁剪的尺寸(32,32)
        self.z_norm = config.DATASET.Z_NORM                 # 是否进行z-score标准化 默认为true

        self.metadata = gpd.read_file(os.path.join(root, "metadata.geojson")) # 读取元数据


        if nfolds is not None:
            self.metadata = self.metadata[self.metadata.Fold.isin(nfolds)]
            # nfolds 参数来筛选出包含特定折叠数据的元数据


        self.stats = {}
        for s in self.modality:  # S2,S1A
            stats_pth = os.path.join(root, f'NORM_{s}_patch.json')  # 拿到均值和标准差
            stats_df = pd.read_json(stats_pth)
            selected_folds = nfolds if nfolds is not None else list(range(1, 6))
            mean = np.stack(
                [stats_df[f'Fold_{f}']['mean'] for f in selected_folds]
            ).astype(np.float32).mean(axis=0)[None, :, None, None]
            std = np.stack(
                [stats_df[f'Fold_{f}']['std'] for f in selected_folds]
            ).astype(np.float32).mean(axis=0)[None, :, None, None]
            self.stats[s] = np.stack([mean, std])
        # 拿到mean和std

        self.mode = mode

    def __len__(self):
        return len(self.metadata)

    @staticmethod
    # 这个静态方法用于生成随机裁剪的参数
    def get_random_crop_params(
        img_size: Tuple[int, int], output_size: Tuple[int, int],
    ) -> Tuple[int, int, int, int]:

        h, w = img_size
        th, tw = output_size

        if h < th or w < tw:
            raise ValueError(f"Required crop size {(th, tw)} is larger than input image size {(h, w)}")

        if w == tw and h == th:
            return 0, 0, h, w

        i = torch.randint(0, h - th + 1, size=(1,)).item()
        j = torch.randint(0, w - tw + 1, size=(1,)).item()
        return i, j, th, tw

    def __getitem__(self, index):

        row = self.metadata.iloc[index]
        # sample字典
        sample = {}
        for s, folder, tdr in zip(self.modality, self.data_folders, self.temp_drop_rate):
            data = np.load(os.path.join(folder, f'{s}_{row.id}.npy'))
            date_positions = self.prepare_dates(json.loads(row[f'dates-{s}']))
            data = np.clip(data, 0, self.max_val).astype(np.float32)  # 将加载的数据限制在0到self.max_val的范围内，防止出现异常值

            # random sample timesteps
            T = data.shape[0]  # 拿到T维度

            # 若不是测试模式，则随机丢弃一些时间步
            if self.mode != 'test':
                ntimesteps = int(T * (1 - np.random.uniform(low=tdr[0], high=tdr[1])))  # low=0.2, high=0.4
                time_idx = sorted(np.random.choice(T, ntimesteps, replace=False))
                data = data[time_idx, ...]
                date_positions = date_positions[time_idx]
            # 进行norm归一化
            if self.z_norm:
                data = data - self.stats[s][0]  # mean = self.stats[s][0]  减去均值
                data = data / self.stats[s][1]  # std = self.stats[s][1]  除以标准差
            else:
                data = data / self.max_val
            
            # 将数据转换为tensor，并存储在sample字典中
            sample[f'data_{s}'] = torch.from_numpy(data).transpose(0, 1) # --> [C, T, H, W] for pad_collate
            sample[f'date_positions_{s}'] = torch.from_numpy(date_positions + 1)[None, :]

        if self.task_type == 'sem_seg':
            # 加载语义标签
            pixel_semantic_annotation = np.load(os.path.join(self.semantic_label_folder, f'TARGET_{row.id}.npy'))[0].astype(np.int32)  #拿到gt
            # 将 void_label 类别的像素值替换为 ignore_index
            pixel_semantic_annotation[pixel_semantic_annotation == self.void_label] = self.ignore_index
            sample['label'] = torch.from_numpy(pixel_semantic_annotation[None])  # 将标签存储到sample['label']中


        # 需要使用随即裁剪并不是测试模式
        # 128*128的图像裁剪成32*32的图像,只有一张哦
        if self.random_crop and self.mode != 'test':
            # workaround for CUDA OOM
            img_size = sample[f'data_{self.modality[0]}'].shape[-2:] # 获取data的最后两个维度，即HW

            i, j, th, tw = self.get_random_crop_params(img_size, self.crop_size) # 传入HW和裁剪的目标尺寸，即(32,32)
            # i,j是裁剪的起始位置，th,tw是裁剪区域的高度和宽度

            for s in self.modality:
                img = sample[f'data_{s}']
                sample[f'data_{s}'] = img[..., i:i+th, j:j+tw]

            label = sample['label']
            sample['label'] = label[:, i:i+th, j:j+tw]

        sample['field_id'] = torch.tensor(row.ID_PATCH)

        return sample
    

    def prepare_dates(self, date_dict):
        d = pd.DataFrame().from_dict(date_dict, orient='index')
        d = d[0].apply(
            lambda x: (
                dt.datetime(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:]))
                - self.reference_date
            ).days
        )
        return d.values

if __name__ =='__main__':
    from config import _C as config
    dataset = PASTIS(config, nfolds=[1, 2, 3], mode='train')
    print(len(dataset))
    print(type(dataset))
    sample = dataset[0]
    # print(sample)
    print(type(sample))
    print(sample.keys())
    print(sample['data_S2'].shape)           # 原始data的tensor(随机裁剪后的)形状为[1, T, C, H, W]
    print(sample['date_positions_S2'].shape) # 时间位置，表示的是距离参考日期的天数
    # 类似这种tensor([[ 39,  49,  54,  59,  69,  74,  79,  84, 139, 144, 169, 174, 224, 234,
    #269, 279, 294, 299, 309, 314, 319, 334, 344, 349, 359, 369, 379, 389,404]])  形状为(1,T)
    print(sample['label'].shape)    # 标签 形状为(1, H, W)
    print(sample['field_id'])       # 提取该地块的唯一标识符