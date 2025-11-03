# ------------------------------------------------------------------------------
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

import argparse
import os
import pprint
import time
import datetime
from pathlib import Path
import random
from functools import partial
import collections
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter



from datasets.pastis import PASTIS
from models.sem_seg import  Segmentor
from utils.train_module import train
from utils.validate_module import validate
from utils.dist_helper import *
from utils.logger import create_logger
from utils.optimizer import get_optimizer
from utils.lr_scheduler import get_lr_scheduler
from utils.pad_collate import  pad_collate

from config import _C as config




def main():

    device = torch.device(config.DEVICE)

    if config.SEED is not None:
        seed = config.SEED + get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        cudnn.deterministic = True

    # cudnn related setting
    cudnn.benchmark = True

    n_fold = config.N_FOLD - 1 if config.N_FOLD is not None else config.DATASET.N_FOLD
    fold_sequence = [
        [[1, 2, 3], [4], [5]],
        [[2, 3, 4], [5], [1]],
        [[3, 4, 5], [1], [2]],
        [[4, 5, 1], [2], [3]],
        [[5, 1, 2], [3], [4]],
    ]

    # Distributed Training: print logs on the first worker
    verbose = True if is_main_process() else False

    if verbose:
        print("create logger ...")
        logger, final_output_dir, tb_log_dir = create_logger(config, phase=f'train_{n_fold}', n_fold=n_fold,)
        logger.info(config)

    # write Tensorboard logs on the first worker
    writer_dict = {'writer': SummaryWriter(tb_log_dir), 'train_global_steps': 0, 'valid_global_steps': 0,} if is_main_process() else None

    # prepare data
    train_fold, val_fold, _ = fold_sequence[n_fold]
    train_dataset = PASTIS(config, nfolds=train_fold, mode='train')
    val_dataset   = PASTIS(config, nfolds=val_fold, mode='val')

    # build model
    model = Segmentor(config, mode='train_val') 

    if verbose:
        model.eval()
        logger.info(model)   
        tot_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000.
        logger.info(f">>> total params: {tot_params:.2f}Mb")

    
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    val_sampler = torch.utils.data.SequentialSampler(val_dataset)

    print(f'The length of trainloader on each GPU is: {len(train_sampler)}')
    print(f'The length of valloader on each GPU is: {len(val_sampler)}')

    kwargs = {'num_workers': config.WORKERS, 'pin_memory': True}
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.TRAIN.BATCH_SIZE_PER_GPU, shuffle=False,drop_last=True, sampler=train_sampler, collate_fn=partial(pad_collate, pad_value=config.DATASET.PAD_VALUE), **kwargs)
        

    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=config.TRAIN.BATCH_SIZE_PER_GPU, shuffle=False,drop_last=True, sampler=val_sampler, collate_fn=partial(pad_collate, pad_value=config.DATASET.PAD_VALUE), **kwargs)


    # Load weights from pre-trained models
    if config.FUNE_TUNE:
        assert os.path.isfile(config.MODEL.PRETRAINED), f'the path of pretrained model {config.MODEL.PRETRAINED}" is not valid!!'
        model_state_file = config.MODEL.PRETRAINED
        log(f'=> Loading model from {model_state_file}')
        pretrained_dict = torch.load(model_state_file, map_location='cpu')['state_dict']
        model_dict = model.state_dict()
        # check loaded parameters size/shape compatibility
        for k in pretrained_dict:
            if k in model_dict:
                if pretrained_dict[k].shape != model_dict[k].shape:
                    log(f'=> Skip loading parameter {k}, required shape {model_dict[k].shape}, loaded shape {pretrained_dict[k].shape}.')
                    pretrained_dict[k] = model_dict[k]
            else:
                log(f'=> Drop parameter {k}.')
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if k in model_dict.keys()}
        if verbose:
            for k in pretrained_dict.keys():
                logger.info(f'=> Loading {k} from pretrained model')
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    best = 0.
    last_epoch = 0
    # Load weights from checkpoint
    # 加载模型的检查点（checkpoint），以便在训练过程中恢复之前的状态，避免从头开始训练
    if config.RESUME:
        model_state_file = os.path.join(final_output_dir, 'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file, map_location='cpu')
            best = checkpoint['best']
            last_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if verbose:
                logger.info(f"=> loaded checkpoint (epoch {last_epoch})")

    
    model = model.to(device)
    lr_scaler = get_world_size()
    optimizer = get_optimizer(config, model, lr_scaler)
    epoch_iters = int(train_dataset.__len__() / config.TRAIN.BATCH_SIZE_PER_GPU / get_world_size())
    end_epoch = config.TRAIN.END_EPOCH
    num_iters = config.TRAIN.END_EPOCH * epoch_iters

    log(f'epoch_iters: {epoch_iters}, max_iters: {num_iters}')

    # learning rate scheduler
    lr_scheduler_dict = {
        'optimizer': optimizer,
        'milestones': [int(s * num_iters) for s in config.LR.LR_STEP],
        'gamma': config.LR.LR_FACTOR,
        'max_iters': num_iters,
        'last_epoch': last_epoch,
        'epoch_iters': epoch_iters,
        'warmup_iters': int(config.LR.WARMUP_ITERS_RATIO * num_iters),
        'warmup_factor': config.LR.WARMUP_FACTOR,
    }
    lr_scheduler = get_lr_scheduler(config.LR.LR_SCHEDULER, **lr_scheduler_dict)

    model_without_ddp = model


    start_time = time.time()
    task_type = config.DATASET.TASK_TYPE
    # 正式开始训练
    for epoch in range(last_epoch, end_epoch):
        
        # 训练单个epoch
        train(config, epoch, num_iters, trainloader, optimizer, lr_scheduler, model, writer_dict, device)
        # 验证模型性能
        valid_loss, scores = validate(config, valloader, model, writer_dict, device)

        # 在主进程中保存检查点并打印日志。
        if get_rank() == 0:
            best_cur = scores['mIoU']
            scores_msg = []
            for k, v in scores.items():
                if isinstance(v, np.ndarray):
                    v = np.array2string(v.flatten(), precision=4, separator=', ')
                elif isinstance(v, collections.abc.Sequence):
                    v = '[' + ', '.join(map(lambda x: f'{x:.4f}', v)) + ']'
                elif isinstance(v, float):
                    v = f'{v:.4f}'
                scores_msg.append(f'{k}={v}')

            msg = f'\nValidation Loss: {valid_loss:.4f}, ' + '\nMetrics: \n' + ', '.join(scores_msg)
            logger.info(msg)

            logger.info('=> saving checkpoint to {}'.format(
                final_output_dir + '/checkpoint.pth.tar'))
            torch.save(
                {'epoch': epoch + 1,
                 'best': best,
                 'val_loss': valid_loss,
                 'state_dict': model_without_ddp.state_dict(),
                 'optimizer': optimizer.state_dict()},
                os.path.join(final_output_dir, 'checkpoint.pth.tar')
            )

            writer_dict['writer'].add_scalars(
                "metrics",
                dict([(k, v) for k, v in scores.items() if np.isscalar(v)]),
                global_step=epoch
            )

            if best < best_cur:
                best = best_cur
                torch.save({'epoch': epoch + 1,
                            'best': best,
                            'val_loss': valid_loss,
                            'metrics': scores,
                            'state_dict': model_without_ddp.state_dict()},
                           os.path.join(final_output_dir, 'best.pth'))

            if epoch == end_epoch - 1:
                torch.save({'metrics': scores,
                            'state_dict': model_without_ddp.state_dict()},
                           os.path.join(final_output_dir, 'final_state.pth'))

                writer_dict['writer'].close()
                end_time = time.time() - start_time
                tot_time = str(datetime.timedelta(seconds=int(end_time)))
                logger.info(f'Training Time: {tot_time}')
                logger.info('Done!')

    # 在训练完成后，读取最佳模型，并打印其存储的最佳模型评估信息。格式化日志
    if get_rank() == 0:
        model_state_file = os.path.join(final_output_dir, 'best.pth')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file, map_location='cpu')
            epoch = checkpoint['epoch']
            val_loss = checkpoint['val_loss']
            val_metrics = checkpoint['metrics']

        scores_msg = f"\nEpoch: {epoch}\n"
        scores_msg += "Validation Metrics: \n"
        scores_msg += ", ".join([f"{k}={v:.4f}" for (k, v) in val_metrics.items() if np.isscalar(v)])
        msg = f'\nValidation Loss: {val_loss:.4f}, ' + scores_msg
        logger.info(msg)


if __name__ == '__main__':
    main()
