"""
modified default_collate from the official pytorch repo
https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py
by X.Cai
"""

import collections
import re
from typing import List, Optional
from torch import  Tensor
import numpy as np

import torch


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return tuple(maxes)

# 封装了张量和掩码的类
class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor] = None):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

# 是将一个张量列表（tensor_list）转换为一个 NestedTensor 对象
def nested_tensor_from_tensor_list(tensor_list: List[Tensor], pad_value=0.):
    # TODO make this more general
    # tensor shape: [C, ...]
    # mask: where True indicates the padded positions
    bsz = len(tensor_list)   # 这个batch的有几个样本
    max_size = _max_by_axis([list(img.shape) for img in tensor_list])
    batch_shape = (bsz,) + max_size
    max_shape = max_size[1:]
    dtype = tensor_list[0].dtype
    device = tensor_list[0].device
    tensor = torch.full(batch_shape, pad_value, dtype=dtype, device=device)
    mask = torch.ones((bsz,) + max_shape, dtype=torch.bool, device=device)
    for img, pad_img, m in zip(tensor_list, tensor, mask):
        slices = [slice(0, s) for s in img.shape]
        pad_img[slices].copy_(img)
        m[slices[1:]] = False
    return NestedTensor(tensor, mask)

default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")

np_str_obj_array_pattern = re.compile(r"[SaUO]")


def pad_collate(batch, pad_value=0.):
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        if len(elem.shape) > 0:
            nested_tensors = nested_tensor_from_tensor_list(batch, pad_value=pad_value)
            tensors, masks = nested_tensors.decompose()
            return torch.cat([tensors, masks.to(dtype=tensors.dtype).unsqueeze(dim=1)], dim=1)
        return torch.stack(batch, dim=0)
    elif (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):
        if elem_type.__name__ == "ndarray" or elem_type.__name__ == "memmap":
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError("Format not managed : {}".format(elem.dtype))

            return pad_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, collections.abc.Mapping):
        return {key: pad_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(*(pad_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("each element in list of batch should be of equal size")
        transposed = zip(*batch)
        return [pad_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


def unfold_reshape(img, patch_size):

    if len(img.shape) == 4:
        C, T, H, W = img.shape
        img = img.unfold(2, size=patch_size, step=patch_size).unfold(3, size=patch_size, step=patch_size)
        img = img.reshape(C, T, -1, patch_size, patch_size).permute(2, 0, 1, 3, 4).contiguous()
    elif len(img.shape) == 3:
        C, H, W = img.shape
        img = img.unfold(1, size=patch_size, step=patch_size).unfold(2, size=patch_size, step=patch_size)
        img = img.reshape(C, -1, patch_size, patch_size).transpose(0, 1).contiguous()

    return img


def pad_collate_split_stack(batch, split_size, pad_value=0.):
    # ugly workaround for CUDA OOM
    # only support dict format
    # FIXME ensure the divisibility
    batch_new = {}
    batch = pad_collate(batch, pad_value=pad_value)
    orig_h, orig_w = batch['label'].shape[-2:]
    add_bsz = (orig_h // split_size) * (orig_w // split_size)
    for k, v in batch.items():
        l_data = []
        l_mask = []
        if k == 'label' or k.startswith('data'):
            for i in range(v.shape[0]):
                l_data.append(unfold_reshape(v[i, :-1], split_size))
                l_mask.append(unfold_reshape(v[i, -1].unsqueeze(dim=0), split_size))
            new_v = torch.cat(l_data, dim=0)
            mask = torch.cat(l_mask, dim=0)
            batch_new[k] = torch.cat([new_v, mask], dim=1)
        elif v.ndim == 3:
            mask = v[:, -1, ...].unsqueeze(dim=1).repeat_interleave(add_bsz, dim=0)
            new_v = v[:, :-1].repeat_interleave(add_bsz, dim=0)
            batch_new[k] = torch.cat([new_v, mask], dim=1)
        else:
            batch_new[k] = v.repeat_interleave(add_bsz, dim=0)

    return batch_new

