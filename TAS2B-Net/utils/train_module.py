# ------------------------------------------------------------------------------
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

# from utils import dist_helper, recursive2device
from  .dist_helper import MetricLogger, log, SmoothedValue
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



def train(config, epoch, max_iters, trainloader, optimizer, lr_scheduler, model, writer_dict, device='cpu'):
    # Training
    model.train()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.8f}'))
    header = f'Epoch: [{epoch}]'

    for batch in metric_logger.log_every(trainloader, config.PRINT_FREQ, header):
        optimizer.zero_grad()

        batch = recursive2device(batch, device)
        outputs, losses = model(batch)

        metric_logger.update(**losses)

        tot_loss = 0.
        for l in losses.values():
            tot_loss += l

        tot_loss.backward()

        optimizer.step()
        lr_scheduler.step()

        metric_logger.update(TotalLoss=tot_loss)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    log(metric_logger)

    if writer_dict:
        writer = writer_dict['writer']
        global_steps = writer_dict['train_global_steps']
        writer.add_scalar('train_loss', metric_logger.TotalLoss.global_avg, global_steps)
        writer_dict['train_global_steps'] = global_steps + 1

