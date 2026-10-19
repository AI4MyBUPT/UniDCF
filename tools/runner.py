import torch
import torch.nn as nn
import os
import json
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
from utils.metrics import Metrics
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2


# ========== 训练模式的模块配置 ==========
TRAINING_MODE_CONFIG = {
    'completion_only': {'completion': True,  'denoise': False},
    'denoise_only':    {'completion': False, 'denoise': True},
    'joint_cd':        {'completion': True,  'denoise': True},
}


def setup_training_stage(base_model, config, current_stage, prev_stage_best_path=None, logger=None):
    """
    根据配置设置训练阶段，支持3种训练模式 (completion_only, denoise_only, joint_cd)
    """
    if not hasattr(config, 'training_stages') or current_stage not in config.training_stages:
        print_log(f'Stage {current_stage} not found in config', logger=logger)
        return None
    
    stage_config = config.training_stages[current_stage]
    training_mode = stage_config.get('mode', 'completion_only')
    
    print_log('=' * 50, logger=logger)
    print_log(f'Setting up training stage: {current_stage}', logger=logger)
    print_log(f'Training mode: {training_mode}', logger=logger)
    print_log(f'Description: {stage_config.get("description", "N/A")}', logger=logger)
    
    # 显示模块训练状态
    mode_config = TRAINING_MODE_CONFIG.get(training_mode, {})
    print_log(f'Module status:', logger=logger)
    print_log(f'  - Completion: {"TRAIN" if mode_config.get("completion") else "FROZEN"}', logger=logger)
    print_log(f'  - Denoise:    {"TRAIN" if mode_config.get("denoise") else "FROZEN/SKIP"}', logger=logger)
    print_log('=' * 50, logger=logger)
    
    # 获取模型
    model = base_model.module if hasattr(base_model, 'module') else base_model
    
    # 设置模型训练模式
    model.set_training_mode(training_mode)
    
    # ========== 加载各模块的预训练权重 ==========
    # 加载补全模块
    completion_ckpt = stage_config.get('load_completion_ckpt')
    if completion_ckpt and os.path.exists(completion_ckpt):
        print_log(f'Loading completion weights from: {completion_ckpt}', logger=logger)
        load_completion_checkpoint(base_model, completion_ckpt, logger=logger)
    
    # 加载去噪模块
    denoise_ckpt = stage_config.get('load_denoise_ckpt')
    if denoise_ckpt and os.path.exists(denoise_ckpt):
        print_log(f'Loading denoise weights from: {denoise_ckpt}', logger=logger)
        load_denoise_checkpoint(base_model, denoise_ckpt, logger=logger)
    
    # 加载通用checkpoint（优先级最低）
    if prev_stage_best_path is not None and os.path.exists(prev_stage_best_path):
        print_log(f'Auto-loading from previous stage: {prev_stage_best_path}', logger=logger)
        load_checkpoint_flexible(base_model, prev_stage_best_path, logger=logger)
    elif stage_config.get('load_ckpt') is not None:
        load_path = stage_config['load_ckpt']
        if os.path.exists(load_path):
            print_log(f'Loading checkpoint from config: {load_path}', logger=logger)
            load_checkpoint_flexible(base_model, load_path, logger=logger)
    
    return stage_config


def load_checkpoint_flexible(base_model, ckpt_path, logger=None):
    """灵活加载检查点（strict=False）"""
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    if 'base_model' in checkpoint:
        state_dict = checkpoint['base_model']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        key = k[len("module."):] if k.startswith("module.") else k
        cleaned_state_dict[key] = v
    
    model = base_model.module if hasattr(base_model, 'module') else base_model
    missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=False)
    
    print_log(f'Loaded checkpoint from {ckpt_path} (flexible mode)', logger=logger)
    if missing_keys:
        print_log(f'Missing keys: {len(missing_keys)} (first 5: {missing_keys[:5]})', logger=logger)
    if unexpected_keys:
        print_log(f'Unexpected keys: {len(unexpected_keys)}', logger=logger)


def load_completion_checkpoint(base_model, ckpt_path, logger=None):
    """
    只加载补全模块的权重
    支持两种格式：
    1. 新格式: completion_module.xxx (来自 UniDCF)
    2. 旧格式: completion_xxx (来自旧模型，需要转换为 completion_module.xxx)
    """
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    if 'base_model' in checkpoint:
        state_dict = checkpoint['base_model']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    model = base_model.module if hasattr(base_model, 'module') else base_model
    completion_state_dict = {}
    for k, v in state_dict.items():
        key = k[len("module."):] if k.startswith("module.") else k
        
        # 兼容旧版本检查点: 转换参数名格式
        # 旧格式: completion_xxx -> 新格式: completion_module.xxx
        if key.startswith('completion_') and not key.startswith('completion_module.'):
            key = key.replace('completion_', 'completion_module.', 1)
        
        if key.startswith('completion_module.'):
            completion_state_dict[key] = v
    
    if len(completion_state_dict) > 0:
        missing_keys, unexpected_keys = model.load_state_dict(completion_state_dict, strict=False)
        print_log(f'Loaded {len(completion_state_dict)} completion weights from {ckpt_path}', logger=logger)
        if len(missing_keys) > 0:
            print_log(f'  Missing keys: {len(missing_keys)} (first 3: {missing_keys[:3]})', logger=logger)
        if len(unexpected_keys) > 0:
            print_log(f'  Unexpected keys: {len(unexpected_keys)} (first 3: {unexpected_keys[:3]})', logger=logger)
    else:
        print_log(f'Warning: No completion weights found in {ckpt_path}', logger=logger)


def load_denoise_checkpoint(base_model, ckpt_path, logger=None):
    """
    只加载去噪模块的权重
    支持两种格式：
    1. 新格式: denoise_module.xxx (来自 UniDCF)
    2. 旧格式: denoise_xxx (来自旧模型，需要转换为 denoise_module.xxx)
    """
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    if 'base_model' in checkpoint:
        state_dict = checkpoint['base_model']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    model = base_model.module if hasattr(base_model, 'module') else base_model
    
    denoise_state_dict = {}
    for k, v in state_dict.items():
        key = k[len("module."):] if k.startswith("module.") else k
        
        # 兼容旧版本检查点: 转换参数名格式
        # 旧格式: denoise_xxx -> 新格式: denoise_module.xxx
        if key.startswith('denoise_') and not key.startswith('denoise_module.'):
            key = key.replace('denoise_', 'denoise_module.', 1)
        
        if key.startswith('denoise_module.'):
            denoise_state_dict[key] = v
    
    if len(denoise_state_dict) > 0:
        missing_keys, unexpected_keys = model.load_state_dict(denoise_state_dict, strict=False)
        print_log(f'Loaded {len(denoise_state_dict)} denoise weights from {ckpt_path}', logger=logger)
        if len(missing_keys) > 0:
            print_log(f'  Missing keys: {len(missing_keys)} (first 3: {missing_keys[:3]})', logger=logger)
        if len(unexpected_keys) > 0:
            print_log(f'  Unexpected keys: {len(unexpected_keys)} (first 3: {unexpected_keys[:3]})', logger=logger)
    else:
        print_log(f'Warning: No denoise_module weights found in {ckpt_path}', logger=logger)


def train_single_stage(base_model, optimizer, scheduler, train_sampler, train_dataloader, 
                       test_dataloader, args, config, current_stage, 
                       train_writer, val_writer, logger,
                       start_epoch=0, best_metrics=None):
    """
    训练单个阶段，支持3种训练模式 (completion_only, denoise_only, joint_cd)
    
    Args:
        start_epoch: 起始epoch（用于resume）
        best_metrics: 之前的最佳指标（用于resume）
    
    Returns:
        best_ckpt_path: 该阶段最佳模型的路径
    """
    stage_config = config.training_stages[current_stage]
    max_epoch = stage_config.get('epochs', config.max_epoch)
    training_mode = stage_config.get('mode', 'completion_only')
    
    # 判断当前模式是否需要UDF数据 (此项目不包含UDF)
    mode_config = TRAINING_MODE_CONFIG.get(training_mode, {})
    need_udf = False  # 固定为False,因为UniDCF_denoise不包含UDF模块
    
    print_log(f'\n{"="*50}', logger=logger)
    print_log(f'Starting training for {current_stage}', logger=logger)
    print_log(f'Training mode: {training_mode}', logger=logger)
    print_log(f'Max epochs: {max_epoch}', logger=logger)
    print_log(f'Start epoch: {start_epoch}', logger=logger)
    print_log(f'Learning rate: {optimizer.param_groups[0]["lr"]}', logger=logger)
    print_log(f'Need UDF data: {need_udf}', logger=logger)
    if best_metrics is not None:
        print_log(f'Resumed best metrics: {best_metrics}', logger=logger)
    print_log(f'{"="*50}\n', logger=logger)
    
    # Criterion
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()
    
    # 使用传入的best_metrics，如果没有则为None
    metrics = None
    best_ckpt_path = None
    
    # 获取模型用于更新training_iter
    model = base_model.module if hasattr(base_model, 'module') else base_model

    base_model.zero_grad(set_to_none=True)
    global_step = start_epoch * len(train_dataloader)
    
    for epoch in range(start_epoch, max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        base_model.train()
        
        # ⚠️ 根据训练模式冻结相应模块
        if hasattr(model, 'training_mode'):
            if model.training_mode == 'denoise_only':
                # denoise_only: 补全模块应该被冻结
                if hasattr(model, 'completion_module'):
                    model.completion_module.eval()
            elif model.training_mode == 'completion_only':
                # completion_only: 去噪模块应该被冻结或不使用
                if hasattr(model, 'denoise_module'):
                    model.denoise_module.eval()
            # joint_cd 模式: 两个模块都保持训练模式,不需要特殊处理

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        
        # 根据模式设置loss名称 (不包含UDF)
        losses = AverageMeter(['SparseLoss', 'DenseLoss'])

        num_iter = 0
        n_batches = len(train_dataloader)
        
        for idx, (_, _, data) in enumerate(train_dataloader):
            data_time.update(time.time() - batch_start_time)
        
            dataset_name = config.dataset.train._base_.NAME
            if dataset_name == 'ZYTOOTH':
                partial = data[0].cuda()
                gt = data[1].cuda()
                partial_ima = data[2].cuda()
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            num_iter += 1
            global_step += 1
            
            # 前向传播 (不传递UDF数据)
            ret = base_model(partial, partial_ima)
            
            # 计算损失 (不包含UDF)
            sparse_loss, dense_loss = model.get_loss(ret, gt)
            
            # ========== 根据训练模式组合用于反向传播的损失 ==========
            if training_mode == 'denoise_only':
                # 去噪独立训练：只用去噪损失（dense_loss已包含denoise_cd）
                _loss = dense_loss
            elif training_mode in ['completion_only', 'joint_cd']:
                # 补全或联合训练
                _loss = sparse_loss + dense_loss
            else:
                raise ValueError(f"Unknown training mode: {training_mode}")
            
            # 检查损失是否需要梯度
            if not _loss.requires_grad:
                print_log(f'Warning: Loss does not require grad in {training_mode} mode, check model configuration', logger=logger)
                continue

            _loss.backward()

            if num_iter == config.step_per_update:
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), 
                                               getattr(config, 'grad_norm_clip', 10), norm_type=2)
                num_iter = 0
                optimizer.step()
                base_model.zero_grad(set_to_none=True)

            if args.distributed:
                # 只 reduce 有效的损失
                sparse_loss = dist_utils.reduce_tensor(sparse_loss, args)
                dense_loss = dist_utils.reduce_tensor(dense_loss, args)
            
            # 更新损失记录（使用 detach 避免梯度问题）
            losses.update([
                sparse_loss.detach().item() * 1000, 
                dense_loss.detach().item() * 1000
            ])

            if args.distributed:
                torch.cuda.synchronize()

            n_itr = epoch * n_batches + idx
            if train_writer is not None:
                train_writer.add_scalar(f'{current_stage}/Loss/Batch/Sparse', sparse_loss.item() * 1000, n_itr)
                train_writer.add_scalar(f'{current_stage}/Loss/Batch/Dense', dense_loss.item() * 1000, n_itr)
                
                # 记录详细损失
                if hasattr(model, 'loss_items'):
                    for loss_name, loss_value in model.loss_items.items():
                        val = loss_value.item() * 1000 if torch.is_tensor(loss_value) else loss_value * 1000
                        train_writer.add_scalar(f'{current_stage}/Loss/Batch/{loss_name}', val, n_itr)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 500 == 0:
                log_msg = f'[{current_stage}] [Epoch {epoch}/{max_epoch}][Batch {idx + 1}/{n_batches}] ' \
                          f'BatchTime={batch_time.val():.3f}s DataTime={data_time.val():.3f}s ' \
                          f'Losses={[f"{l:.4f}" for l in losses.val()]} lr={optimizer.param_groups[0]["lr"]:.6f}'
                
                if hasattr(model, 'loss_items'):
                    loss_details = ' | '.join([
                        f'{k}: {v.item()*1000 if torch.is_tensor(v) else v*1000:.4f}' 
                        for k, v in model.loss_items.items()
                    ])
                    log_msg += f'\n  Details: {loss_details}'
                
                print_log(log_msg, logger=logger)

            if config.scheduler.type == 'GradualWarmup':
                if n_itr < config.scheduler.kwargs_2.total_epoch:
                    scheduler.step()

        if isinstance(scheduler, list):
            for item in scheduler:
                item.step()
        else:
            scheduler.step()
        
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar(f'{current_stage}/Loss/Epoch/Sparse', losses.avg(0), epoch)
            train_writer.add_scalar(f'{current_stage}/Loss/Epoch/Dense', losses.avg(1), epoch)
        
        print_log(f'[{current_stage}] EPOCH: {epoch} EpochTime={epoch_end_time - epoch_start_time:.3f}s ' \
                  f'Losses={[f"{l:.4f}" for l in losses.avg()]}', logger=logger)

        if epoch % args.val_freq == 0:
            metrics = validate(base_model, test_dataloader, epoch, ChamferDisL1, ChamferDisL2, 
                             val_writer, args, config, current_stage, logger=logger)

            if metrics.better_than(best_metrics):
                best_metrics = metrics
                ckpt_name = f'ckpt-{current_stage}-best'
                best_ckpt_path = builder.save_checkpoint(
                    base_model, optimizer, epoch, metrics, best_metrics, 
                    ckpt_name, args, logger=logger
                )
                print_log(f'✓ Saved best checkpoint: {ckpt_name}', logger=logger)
        
        # 保存最新模型
        ckpt_name = f'ckpt-{current_stage}-last'
        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 
                               ckpt_name, args, logger=logger)
        
        # 保存最后几个epoch
        if epoch > 10 and epoch % 5 == 0:
            ckpt_name = f'ckpt-{current_stage}-epoch-{epoch:03d}'
            builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 
                                   ckpt_name, args, logger=logger)
    
    print_log(f'\n{"="*50}', logger=logger)
    print_log(f'✓ Stage {current_stage} ({training_mode}) completed!', logger=logger)
    if best_metrics is not None:
        print_log(f'Best metrics: {best_metrics}', logger=logger)
    print_log(f'Best model: {best_ckpt_path}', logger=logger)
    print_log(f'{"="*50}\n', logger=logger)
    
    return best_ckpt_path


def run_net(args, config, train_writer=None, val_writer=None):
    """
    主训练函数：支持单阶段和多阶段训练，支持3种训练模式 (completion_only, denoise_only, joint_cd)
    """
    logger = get_logger(args.log_name)

    # 构建数据集
    (train_sampler, train_dataloader), (_, test_dataloader) = \
        builder.dataset_builder(args, config.dataset.train), \
        builder.dataset_builder(args, config.dataset.val)
    
    # 构建模型
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)
    
    # 打印模型配置
    print_log('='*50, logger=logger)
    print_log('Model Configuration:', logger=logger)
    print_log(f'  - Model: {config.model.NAME}', logger=logger)
    print_log(f'  - Enable Denoise: {getattr(config.model, "enable_denoise", False)}', logger=logger)
    print_log('='*50, logger=logger)
    
    # 检查是否运行所有阶段
    run_all_stages = getattr(config, 'run_all_stages', False)
    
    if run_all_stages:
        # ========== 多阶段自动训练模式 ==========
        print_log('\n' + '='*50, logger=logger)
        print_log('🚀 MULTI-STAGE TRAINING MODE', logger=logger)
        print_log('='*50 + '\n', logger=logger)
        
        stage_order = getattr(config, 'stage_order', ['stage1', 'stage2', 'stage3'])
        print_log(f'Will train stages in order: {stage_order}', logger=logger)
        
        # 显示每个阶段的训练模式
        for stage in stage_order:
            if stage in config.training_stages:
                mode = config.training_stages[stage].get('mode', 'unknown')
                print_log(f'  - {stage}: {mode}', logger=logger)
        
        prev_best_path = None
        
        for stage_idx, current_stage in enumerate(stage_order):
            print_log(f'\n{"#"*50}', logger=logger)
            print_log(f'# STAGE {stage_idx + 1}/{len(stage_order)}: {current_stage}', logger=logger)
            print_log(f'{"#"*50}\n', logger=logger)
            
            # 设置当前阶段
            stage_config = setup_training_stage(base_model, config, current_stage, 
                                               prev_best_path, logger)
            if stage_config is None:
                print_log(f'ERROR: Stage {current_stage} configuration not found!', logger=logger)
                continue
            
            # 更新学习率
            stage_lr = stage_config.get('lr', config.optimizer.kwargs.lr)
            config.optimizer.kwargs.lr = stage_lr
            
            # DDP/DataParallel（第一次）
            if stage_idx == 0:
                if args.distributed:
                    if args.sync_bn:
                        base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
                    base_model = nn.parallel.DistributedDataParallel(
                        base_model, 
                        device_ids=[args.local_rank % torch.cuda.device_count()], 
                        find_unused_parameters=True
                    )
                else:
                    base_model = nn.DataParallel(base_model).cuda()
            
            # 重新构建优化器和调度器
            optimizer = builder.build_optimizer(base_model, config)
            scheduler = builder.build_scheduler(base_model, optimizer, config, last_epoch=-1)
            
            # 训练当前阶段
            prev_best_path = train_single_stage(
                base_model, optimizer, scheduler, train_sampler, train_dataloader,
                test_dataloader, args, config, current_stage,
                train_writer, val_writer, logger
            )
        
        print_log('\n' + '='*50, logger=logger)
        print_log('🎉 ALL STAGES COMPLETED!', logger=logger)
        print_log('='*50 + '\n', logger=logger)
        
    else:
        # ========== 单阶段训练模式 ==========
        current_stage = getattr(config, 'current_stage', 'stage1')
        print_log(f'\n🎯 SINGLE-STAGE TRAINING MODE: {current_stage}\n', logger=logger)
        
        # 初始化变量
        start_epoch = 0
        best_metrics = None
        
        # 设置训练阶段（在resume之前，确保模型结构正确）
        stage_config = setup_training_stage(base_model, config, current_stage, None, logger)
        
        if stage_config is not None:
            config.max_epoch = stage_config.get('epochs', config.max_epoch)
            stage_lr = stage_config.get('lr', config.optimizer.kwargs.lr)
            config.optimizer.kwargs.lr = stage_lr
        else:
            current_stage = 'default'
        
        # DDP/DataParallel（在resume之前包装，确保state_dict格式一致）
        if args.distributed:
            if args.sync_bn:
                base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            base_model = nn.parallel.DistributedDataParallel(
                base_model,
                device_ids=[args.local_rank % torch.cuda.device_count()],
                find_unused_parameters=True
            )
        else:
            base_model = nn.DataParallel(base_model).cuda()
        
        # 构建优化器和调度器
        optimizer = builder.build_optimizer(base_model, config)
        
        # Resume或加载检查点（在DDP包装和优化器创建之后）
        if args.resume:
            start_epoch, best_metrics = builder.resume_model(base_model, args, logger=logger, current_stage=current_stage, consider_metric=config.consider_metric)
            builder.resume_optimizer(optimizer, args, logger=logger, current_stage=current_stage)
            # 调度器从resume的epoch开始
            scheduler = builder.build_scheduler(base_model, optimizer, config, last_epoch=start_epoch-1)
        elif args.start_ckpts is not None:
            builder.load_model(base_model, args.start_ckpts, logger=logger)
            scheduler = builder.build_scheduler(base_model, optimizer, config, last_epoch=-1)
        else:
            scheduler = builder.build_scheduler(base_model, optimizer, config, last_epoch=-1)
        
        # 训练（传入start_epoch和best_metrics）
        train_single_stage(
            base_model, optimizer, scheduler, train_sampler, train_dataloader,
            test_dataloader, args, config, current_stage,
            train_writer, val_writer, logger,
            start_epoch=start_epoch, best_metrics=best_metrics
        )
    
    if train_writer is not None and val_writer is not None:
        train_writer.close()
        val_writer.close()


def validate(base_model, test_dataloader, epoch, ChamferDisL1, ChamferDisL2, 
            val_writer, args, config, current_stage, logger=None):
    """验证函数"""
    print_log(f"[{current_stage}] Validating epoch {epoch}...", logger=logger)
    base_model.eval()

    test_losses = AverageMeter(['SparseLossL1', 'SparseLossL2', 'DenseLossL1', 'DenseLossL2'])
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    n_samples = len(test_dataloader)
    interval = max(n_samples // 10, 1)

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
            model_id = model_ids[0]

            dataset_name = config.dataset.val._base_.NAME
            if dataset_name == 'ZYTOOTH':
                partial = data[0].cuda()
                gt = data[1].cuda()
                partial_ima = data[2].cuda()
            else:
                raise NotImplementedError(f'Validation does not support {dataset_name}')

            ret = base_model(partial, partial_ima)
            
            # 根据返回类型获取点云
            if isinstance(ret, dict):
                coarse_points = ret['coarse']
                # 优先使用去噪后的点云
                if 'denoised' in ret:
                    dense_points = ret['denoised']
                else:
                    dense_points = ret['fine']
            else:
                coarse_points = ret[0]
                dense_points = ret[-1]

            sparse_loss_l1 = ChamferDisL1(coarse_points, gt)
            sparse_loss_l2 = ChamferDisL2(coarse_points, gt)
            dense_loss_l1 = ChamferDisL1(dense_points, gt)
            dense_loss_l2 = ChamferDisL2(dense_points, gt)

            if args.distributed:
                sparse_loss_l1 = dist_utils.reduce_tensor(sparse_loss_l1, args)
                sparse_loss_l2 = dist_utils.reduce_tensor(sparse_loss_l2, args)
                dense_loss_l1 = dist_utils.reduce_tensor(dense_loss_l1, args)
                dense_loss_l2 = dist_utils.reduce_tensor(dense_loss_l2, args)

            test_losses.update([
                sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000,
                dense_loss_l1.item() * 1000, dense_loss_l2.item() * 1000
            ])

            _metrics = Metrics.get(dense_points, gt)
            if args.distributed:
                _metrics = [dist_utils.reduce_tensor(_metric, args).item() for _metric in _metrics]
            else:
                _metrics = [_metric.item() for _metric in _metrics]

            for _taxonomy_id in taxonomy_ids:
                if _taxonomy_id not in category_metrics:
                    category_metrics[_taxonomy_id] = AverageMeter(Metrics.names())
                category_metrics[_taxonomy_id].update(_metrics)
            
            if (idx + 1) % interval == 0:
                print_log(f'[{current_stage}] Val[{idx + 1}/{n_samples}] ' \
                         f'Losses={[f"{l:.4f}" for l in test_losses.val()]} ' \
                         f'Metrics={[f"{m:.4f}" for m in _metrics]}', logger=logger)
        
        for _, v in category_metrics.items():
            test_metrics.update(v.avg())
        
        print_log(f'[{current_stage}] EPOCH {epoch} Validation Metrics: ' \
                 f'{[f"{m:.4f}" for m in test_metrics.avg()]}', logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

    # Print testing results
    shapenet_dict_path = './data/shapenet_synset_dict.json'
    if os.path.exists(shapenet_dict_path):
        shapenet_dict = json.load(open(shapenet_dict_path, 'r'))
    else:
        shapenet_dict = {}
    print_log('============================ TEST RESULTS ============================',logger=logger)
    msg = ''
    msg += 'Taxonomy\t'
    msg += '#Sample\t'
    for metric in test_metrics.items:
        msg += metric + '\t'
    msg += '#ModelName\t'
    print_log(msg, logger=logger)

    for taxonomy_id in category_metrics:
        msg = ''
        msg += (taxonomy_id + '\t')
        msg += (str(category_metrics[taxonomy_id].count(0)) + '\t')
        for value in category_metrics[taxonomy_id].avg():
            msg += '%.3f \t' % value
        msg += shapenet_dict.get(taxonomy_id, taxonomy_id) + '\t'
        print_log(msg, logger=logger)

    msg = ''
    msg += 'Overall\t\t'
    for value in test_metrics.avg():
        msg += '%.3f \t' % value
    print_log(msg, logger=logger)

    # TensorBoard记录
    if val_writer is not None:
        val_writer.add_scalar(f'{current_stage}/Loss/Epoch/Sparse', test_losses.avg(0), epoch)
        val_writer.add_scalar(f'{current_stage}/Loss/Epoch/Dense', test_losses.avg(2), epoch)
        for i, metric in enumerate(test_metrics.items):
            val_writer.add_scalar(f'{current_stage}/Metric/{metric}', test_metrics.avg(i), epoch)

    return Metrics(config.consider_metric, test_metrics.avg())


def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger = logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)
 
    base_model = builder.model_builder(config.model)
    builder.load_model(base_model, args.ckpts, logger = logger)
    if args.use_gpu:
        base_model.to(args.local_rank)

    if args.distributed:
        raise NotImplementedError()

    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()

    test(base_model, test_dataloader, ChamferDisL1, ChamferDisL2, args, config, logger=logger)


def test(base_model, test_dataloader, ChamferDisL1, ChamferDisL2, args, config, logger=None):
    base_model.eval()

    test_losses = AverageMeter(['SparseLossL1', 'SparseLossL2', 'DenseLossL1', 'DenseLossL2'])
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    n_samples = len(test_dataloader)

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
            model_id = model_ids[0]

            dataset_name = config.dataset.test._base_.NAME
            if dataset_name == 'ZYTOOTH':
                partial = data[0].cuda()
                gt = data[1].cuda()
                partial_ima = data[2].cuda()

                ret = base_model(partial, partial_ima)
                
                # 根据返回类型获取点云
                if isinstance(ret, dict):
                    coarse_points = ret['coarse']
                    if 'denoised' in ret:
                        dense_points = ret['denoised']
                    else:
                        dense_points = ret['fine']
                else:
                    coarse_points = ret[0]
                    dense_points = ret[-1]

                sparse_loss_l1 = ChamferDisL1(coarse_points, gt)
                sparse_loss_l2 = ChamferDisL2(coarse_points, gt)
                dense_loss_l1 = ChamferDisL1(dense_points, gt)
                dense_loss_l2 = ChamferDisL2(dense_points, gt)

                test_losses.update([
                    sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000,
                    dense_loss_l1.item() * 1000, dense_loss_l2.item() * 1000
                ])

                _metrics = Metrics.get(dense_points, gt, require_emd=True)

                if taxonomy_id not in category_metrics:
                    category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
                category_metrics[taxonomy_id].update(_metrics)
            else:
                raise NotImplementedError(f'Test phase do not support {dataset_name}')

            if (idx+1) % 200 == 0:
                print_log('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics = %s' %
                            (idx + 1, n_samples, taxonomy_id, model_id, 
                             ['%.4f' % l for l in test_losses.val()], 
                             ['%.4f' % m for m in _metrics]), logger=logger)
        
        if dataset_name == 'KITTI':
            return
        for _,v in category_metrics.items():
            test_metrics.update(v.avg())
        print_log('[TEST] Metrics = %s' % (['%.4f' % m for m in test_metrics.avg()]), logger=logger)

    # Print testing results
    shapenet_dict_path = './data/shapenet_synset_dict.json'
    if os.path.exists(shapenet_dict_path):
        shapenet_dict = json.load(open(shapenet_dict_path, 'r'))
    else:
        shapenet_dict = {}
    print_log('============================ TEST RESULTS ============================',logger=logger)
    msg = ''
    msg += 'Taxonomy\t'
    msg += '#Sample\t'
    for metric in test_metrics.items:
        msg += metric + '\t'
    msg += '#ModelName\t'
    print_log(msg, logger=logger)

    for taxonomy_id in category_metrics:
        msg = ''
        msg += (taxonomy_id + '\t')
        msg += (str(category_metrics[taxonomy_id].count(0)) + '\t')
        for value in category_metrics[taxonomy_id].avg():
            msg += '%.3f \t' % value
        msg += shapenet_dict.get(taxonomy_id, taxonomy_id) + '\t'
        print_log(msg, logger=logger)

    msg = ''
    msg += 'Overall \t\t'
    for value in test_metrics.avg():
        msg += '%.3f \t' % value
    print_log(msg, logger=logger)
    return 