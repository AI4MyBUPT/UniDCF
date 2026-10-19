import os, sys
# online package
import torch
# optimizer
import torch.optim as optim
from timm.scheduler import CosineLRScheduler
# dataloader
from datasets import build_dataset_from_cfg
from models import build_model_from_cfg
# utils
from utils.logger import *
from utils.misc import *


def dataset_builder(args, config):
    dataset = build_dataset_from_cfg(config._base_, config.others)
    shuffle = config.others.subset == 'train'
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle = shuffle)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = config.others.bs if shuffle else 1,
                                            num_workers = int(args.num_workers),
                                            drop_last = config.others.subset == 'train',
                                            worker_init_fn = worker_init_fn,
                                            sampler = sampler)
    else:
        sampler = None
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.others.bs if shuffle else 1,
                                                shuffle = shuffle, 
                                                drop_last = config.others.subset == 'train',
                                                num_workers = int(args.num_workers),
                                                worker_init_fn=worker_init_fn)
    return sampler, dataloader


def model_builder(config):
    model = build_model_from_cfg(config)
    return model


def build_optimizer(base_model, config):
    opti_config = config.optimizer
    if opti_config.type == 'AdamW':
        def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
            decay = []
            no_decay = []
            for name, param in model.module.named_parameters():
                if not param.requires_grad:
                    continue  # frozen weights
                if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
                    no_decay.append(param)
                else:
                    decay.append(param)
            return [
                {'params': no_decay, 'weight_decay': 0.},
                {'params': decay, 'weight_decay': weight_decay}]
        param_groups = add_weight_decay(base_model, weight_decay=opti_config.kwargs.weight_decay)
        optimizer = optim.AdamW(param_groups, **opti_config.kwargs)
    elif opti_config.type == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, base_model.parameters()), **opti_config.kwargs)
    elif opti_config.type == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, base_model.parameters()), **opti_config.kwargs)
    else:
        raise NotImplementedError()

    return optimizer


def build_scheduler(base_model, optimizer, config, last_epoch=-1):
    sche_config = config.scheduler
    if sche_config.type == 'LambdaLR':
        scheduler = build_lambda_sche(optimizer, sche_config.kwargs, last_epoch=last_epoch)  # misc.py
    elif sche_config.type == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, last_epoch=last_epoch, **sche_config.kwargs)
    elif sche_config.type == 'GradualWarmup':
        scheduler_steplr = torch.optim.lr_scheduler.StepLR(optimizer, last_epoch=last_epoch, **sche_config.kwargs_1)
        scheduler = GradualWarmupScheduler(optimizer, after_scheduler=scheduler_steplr, **sche_config.kwargs_2)
    elif sche_config.type == 'CosLR':
        scheduler = CosineLRScheduler(optimizer,
                t_initial=sche_config.kwargs.t_max,
                lr_min=sche_config.kwargs.min_lr,
                warmup_t=sche_config.kwargs.initial_epochs,
                t_in_epochs=True)
    else:
        raise NotImplementedError()
    
    if config.get('bnmscheduler') is not None:
        bnsche_config = config.bnmscheduler
        if bnsche_config.type == 'Lambda':
            bnscheduler = build_lambda_bnsche(base_model, bnsche_config.kwargs)  # misc.py
        scheduler = [scheduler, bnscheduler]
    
    return scheduler


def resume_model(base_model, args, logger = None, current_stage='stage1', consider_metric='CDL1'):
    """
    恢复模型训练
    优先级：
    1. ckpt-{stage}-last.pth (最新的阶段checkpoint)
    2. ckpt-last.pth (旧版本兼容)
    """
    # 优先查找带stage前缀的checkpoint
    ckpt_path = os.path.join(args.experiment_path, f'ckpt-{current_stage}-last.pth')
    if not os.path.exists(ckpt_path):
        # 兼容旧版本：查找不带stage的checkpoint
        ckpt_path = os.path.join(args.experiment_path, 'ckpt-last.pth')
        if not os.path.exists(ckpt_path):
            print_log(f'[RESUME INFO] No checkpoint file found. Tried:', logger = logger)
            print_log(f'  - {args.experiment_path}/ckpt-{current_stage}-last.pth', logger = logger)
            print_log(f'  - {args.experiment_path}/ckpt-last.pth', logger = logger)
            print_log(f'[RESUME INFO] Starting training from scratch...', logger = logger)
            return 0, None
    
    print_log(f'[RESUME INFO] Loading model weights from {ckpt_path}...', logger = logger )

    # load state dict
    map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank}
    state_dict = torch.load(ckpt_path, map_location=map_location)
    
    # parameter resume of base model
    # 只去掉开头的 "module." 前缀（DDP/DataParallel添加的），不要替换中间的 "module."
    base_ckpt = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state_dict['base_model'].items()}
    
    # 处理DDP包装的情况
    if hasattr(base_model, 'module'):
        base_model.module.load_state_dict(base_ckpt, strict=False)
    else:
        base_model.load_state_dict(base_ckpt, strict=False)

    # parameter
    start_epoch = state_dict['epoch'] + 1
    best_metrics = state_dict.get('best_metrics', None)
    if best_metrics is not None and not isinstance(best_metrics, dict):
        best_metrics = best_metrics.state_dict()

    # 将 dict 转回 Metrics 对象，以便 better_than 正常工作
    if isinstance(best_metrics, dict) and len(best_metrics) > 0:
        from utils.metrics import Metrics
        best_metrics = Metrics(consider_metric, best_metrics)

    print_log(f'[RESUME INFO] Successfully resumed from epoch {state_dict["epoch"]}', logger = logger)
    print_log(f'[RESUME INFO] Will start training from epoch {start_epoch}', logger = logger)
    if best_metrics:
        print_log(f'[RESUME INFO] Best metrics so far: {str(best_metrics):s}', logger = logger)
    
    return start_epoch, best_metrics


def resume_optimizer(optimizer, args, logger = None, current_stage='stage1'):
    """
    恢复优化器状态
    优先级：
    1. ckpt-{stage}-last.pth (最新的阶段checkpoint)
    2. ckpt-last.pth (旧版本兼容)
    """
    # 优先查找带stage前缀的checkpoint
    ckpt_path = os.path.join(args.experiment_path, f'ckpt-{current_stage}-last.pth')
    if not os.path.exists(ckpt_path):
        # 兼容旧版本
        ckpt_path = os.path.join(args.experiment_path, 'ckpt-last.pth')
        if not os.path.exists(ckpt_path):
            print_log(f'[RESUME INFO] No optimizer checkpoint found', logger = logger)
            return
    
    print_log(f'[RESUME INFO] Loading optimizer from {ckpt_path}...', logger = logger )
    # load state dict
    state_dict = torch.load(ckpt_path, map_location='cpu')
    
    # optimizer
    if 'optimizer' in state_dict:
        try:
            optimizer.load_state_dict(state_dict['optimizer'])
            print_log(f'[RESUME INFO] Optimizer state restored', logger = logger)
        except Exception as e:
            print_log(f'[RESUME INFO] Failed to load optimizer state: {e}', logger = logger)
            print_log(f'[RESUME INFO] Will use fresh optimizer', logger = logger)


def save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, prefix, args, logger = None):
    if args.local_rank == 0:
        # 检查模型是否被 DataParallel 或 DistributedDataParallel 包装
        if hasattr(base_model, 'module'):
            model_state_dict = base_model.module.state_dict()
        else:
            model_state_dict = base_model.state_dict()
        
        torch.save({
                    'base_model' : model_state_dict,
                    'optimizer' : optimizer.state_dict(),
                    'epoch' : epoch,
                    'metrics' : metrics.state_dict() if metrics is not None else dict(),
                    'best_metrics' : best_metrics.state_dict() if best_metrics is not None else dict(),
                    }, os.path.join(args.experiment_path, prefix + '.pth'))
        print_log(f"Save checkpoint at {os.path.join(args.experiment_path, prefix + '.pth')}", logger = logger)
        
        # 返回保存的路径
        return os.path.join(args.experiment_path, prefix + '.pth')
    return None


def load_model(base_model, ckpt_path, logger = None):
    if not os.path.exists(ckpt_path):
        raise NotImplementedError('no checkpoint file from path %s...' % ckpt_path)
    print_log(f'Loading weights from {ckpt_path}...', logger = logger )

    # load state dict
    state_dict = torch.load(ckpt_path, map_location='cpu')
    # parameter resume of base model
    if state_dict.get('model') is not None:
        base_ckpt = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state_dict['model'].items()}
    elif state_dict.get('base_model') is not None:
        base_ckpt = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state_dict['base_model'].items()}
    else:
        raise RuntimeError('mismatch of ckpt weight')
    base_model.load_state_dict(base_ckpt)

    epoch = -1
    if state_dict.get('epoch') is not None:
        epoch = state_dict['epoch']
    if state_dict.get('metrics') is not None:
        metrics = state_dict['metrics']
        if not isinstance(metrics, dict):
            metrics = metrics.state_dict()
    else:
        metrics = 'No Metrics'
    return 