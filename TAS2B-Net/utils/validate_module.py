# ------------------------------------------------------------------------------
# Licensed under the MIT License.
# ------------------------------------------------------------------------------
import torch
from torch.nn import functional as F

from utils.dist_helper import *
from utils.metrics import  IoU
import collections
from torch import Tensor

def recursive2device(x, device):
    if isinstance(x, Tensor):
        return x.to(device, non_blocking=True)
    elif isinstance(x, collections.abc.Mapping):
        return {k: recursive2device(x[k], device) for k in x}
    elif isinstance(x, collections.abc.Sequence):
        return [recursive2device(elem, device) for elem in x]
    else:
        raise TypeError(f'Support Tensor, Dict, List, but found unrecognized input type: {type(x)}')
    


def validate(config, testloader, model, writer_dict, device='cpu'):

    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Validation:'

    num_classes = config.DATASET.NUM_CLASSES
    ignore_index = config.LOSS.IGNORE_INDEX


    
    eval_metric = IoU(num_classes, cm_device='cpu', ignore_index=ignore_index)


    with torch.no_grad():
        for batch in metric_logger.log_every(testloader, config.PRINT_FREQ, header):

            batch = recursive2device(batch, device)
            outputs, losses = model(batch)

            preds = outputs['preds'] # [b, c, h, w]
            if preds.ndim == 4:
                if preds.shape[-2:] != outputs['labels'].shape[-2:]:
                    preds = F.interpolate(
                        preds, size=outputs['labels'].shape[-2:], mode='bilinear', align_corners=False
                    )
                preds = preds.softmax(dim=1).argmax(dim=1)


            metric_logger.update(**losses)
            tot_loss = 0.
            for l in losses.values():
                tot_loss += l
            metric_logger.update(TotalLoss=tot_loss)

            eval_metric.add(preds, outputs['labels'])

        metric_logger.synchronize_between_processes()
        log(metric_logger)
        scores = eval_metric.value()

    if writer_dict:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_loss', metric_logger.TotalLoss.global_avg, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

    return metric_logger.TotalLoss.global_avg, scores
