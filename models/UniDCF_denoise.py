##############################################################
# UniDCF_denoise: 模块化点云补全+去噪模型
# 特点：补全和去噪模块完全分离，支持分阶段训练
##############################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import DropPath, trunc_normal_

from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from extensions import emd
from extensions.emd import emd_module

from utils import misc
from utils.logger import print_log

from .build import MODELS, build_model_from_cfg
from .Transformer_utils import *
from .UniDCF_util import (
    DGCNN_Grouper, SimpleEncoder, PointTransformerEncoderEntry,
    PointTransformerDecoderEntry, Fold, SimpleRebuildFCLayer,
    DinoViTEncoder, ViewFeatureFusionModule
)
from .DenoiseHeadV2 import FeatureConditionedDenoiseHead


class PointCloudCompletionModule(nn.Module):
    """
    独立的点云补全模块
    可单独训练，输出 coarse 和 fine 点云
    """
    def __init__(self, config):
        super().__init__()
        encoder_config = config.encoder_config
        decoder_config = config.decoder_config
        self.center_num = getattr(config, 'center_num', [512, 128])
        self.encoder_type = config.encoder_type
        
        in_chans = 3
        self.num_query = query_num = config.num_query
        global_feature_dim = config.global_feature_dim
        self.trans_dim = decoder_config.embed_dim
        
        # Encoder配置
        if self.encoder_type == 'graph':
            self.grouper = DGCNN_Grouper(k=16)
        else:
            self.grouper = SimpleEncoder(k=32, embed_dims=512)
            
        self.pos_embed = nn.Sequential(
            nn.Linear(in_chans, 128),
            nn.GELU(),
            nn.Linear(128, encoder_config.embed_dim)
        )
        
        self.input_proj = nn.Sequential(
            nn.Linear(self.grouper.num_features, 512),
            nn.GELU(),
            nn.Linear(512, encoder_config.embed_dim)
        )
        
        # Transformer Encoder
        self.encoder = PointTransformerEncoderEntry(encoder_config)
        
        # 图像特征提取器（可选）
        self.use_image = getattr(config, 'use_image', False)
        if self.use_image:
            dino_path = getattr(config, 'dino_path', "/path/to/dinov2_vitb14_pretrain.pth")
            self.img_feature_extractor = DinoViTEncoder(dino_path)
            self.view_feature_fusion = ViewFeatureFusionModule(embed_dim=1024)
        
        # 全局特征提取
        self.increase_dim = nn.Sequential(
            nn.Linear(encoder_config.embed_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, global_feature_dim)
        )
        
        # Coarse点云生成
        self.coarse_pred = nn.Sequential(
            nn.Linear(global_feature_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, 3 * query_num)
        )
        
        # Query生成
        self.mlp_query = nn.Sequential(
            nn.Linear(global_feature_dim + 3, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, decoder_config.embed_dim)
        )
        
        # Memory连接
        if decoder_config.embed_dim == encoder_config.embed_dim:
            self.mem_link = nn.Identity()
        else:
            self.mem_link = nn.Linear(encoder_config.embed_dim, decoder_config.embed_dim)
            
        # Transformer Decoder
        self.decoder = PointTransformerDecoderEntry(decoder_config)
        
        # Query排序
        self.query_ranking = nn.Sequential(
            nn.Linear(3, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # Fine点云生成
        self.decoder_type = config.decoder_type
        self.num_points = getattr(config, 'num_points', None)
        self.fold_step = 8
        
        if self.decoder_type == 'fold':
            self.factor = self.fold_step ** 2
            self.decode_head = Fold(self.trans_dim, step=self.fold_step, hidden_dim=256)
        else:
            if self.num_points is not None:
                self.factor = self.num_points // self.num_query
                assert self.num_points % self.num_query == 0
                self.decode_head = SimpleRebuildFCLayer(
                    self.trans_dim * 2, step=self.num_points // self.num_query
                )
            else:
                self.factor = self.fold_step ** 2
                self.decode_head = SimpleRebuildFCLayer(
                    self.trans_dim * 2, step=self.fold_step ** 2
                )
                
        self.increase_dim_decode = nn.Sequential(
            nn.Conv1d(self.trans_dim, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1)
        )
        
        self.reduce_map = nn.Linear(self.trans_dim + 1027, self.trans_dim)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, xyz, image=None):
        """
        补全模块前向传播
        Args:
            xyz: [B, N, 3] 输入部分点云
            image: [B, 3, 256, 256] 可选的图像输入
        Returns:
            coarse_points: [B, M, 3] 粗糙点云
            fine_points: [B, N_fine, 3] 精细点云
            completion_features: dict 包含中间特征，用于去噪模块
        """
        B = xyz.size(0)
        
        # 1) 点云编码
        coor, f = self.grouper(xyz, self.center_num)  # [B, n, c]
        pe = self.pos_embed(coor)
        x = self.input_proj(f)
        x = self.encoder(x + pe, coor)  # [B, n, embed_dim]
        
        # 2) 全局特征
        pc_feat = self.increase_dim(x)  # [B, n, 1024]
        pc_feat_global = torch.max(pc_feat, dim=1)[0]  # [B, 1024]
        
        # 3) 图像特征融合（如果使用）
        if self.use_image and image is not None:
            depth_feat = self.img_feature_extractor(image)  # [B, 1024]
            # 需要增加序列维度以匹配 MultiheadAttention 的要求
            fused_features = self.view_feature_fusion(
                pc_feat_global.unsqueeze(1),  # [B, 1, 1024]
                depth_feat.unsqueeze(1)  # [B, 1, 1024] - 添加这个 unsqueeze
            ).squeeze(1)  # 输出 [B, 1, 1024] -> squeeze 到 [B, 1024]
        else:
            fused_features = pc_feat_global
            
        # 4) 生成coarse点云
        coarse = self.coarse_pred(fused_features).reshape(B, -1, 3)
        
        # 5) Query选择
        mem = self.mem_link(x)
        query_ranking = self.query_ranking(coarse)
        idx = torch.argsort(query_ranking, dim=1, descending=True)
        coarse = torch.gather(
            coarse, 1, 
            idx[:, :self.num_query].expand(-1, -1, coarse.size(-1))
        )
        
        # 6) 生成query特征
        q = self.mlp_query(
            torch.cat([
                fused_features.unsqueeze(1).expand(-1, coarse.size(1), -1),
                coarse
            ], dim=-1)
        )
        
        # 7) Decoder
        q = self.decoder(q=q, v=mem, q_pos=coarse, v_pos=coor)
        
        # 8) 生成fine点云
        M = q.size(1)
        global_feature = self.increase_dim_decode(q.transpose(1, 2)).transpose(1, 2)
        global_feature = torch.max(global_feature, dim=1)[0]
        
        rebuild_feature = torch.cat([
            global_feature.unsqueeze(-2).expand(-1, M, -1),
            q,
            coarse
        ], dim=-1)
        
        if self.decoder_type == 'fold':
            rebuild_feature = self.reduce_map(rebuild_feature.reshape(B * M, -1))
            relative_xyz = self.decode_head(rebuild_feature).reshape(B, M, 3, -1)
            fine_points = (relative_xyz + coarse.unsqueeze(-1)).transpose(2, 3)
        else:
            rebuild_feature = self.reduce_map(rebuild_feature)
            relative_xyz = self.decode_head(rebuild_feature)
            fine_points = relative_xyz + coarse.unsqueeze(-2)
            
        fine_points = fine_points.reshape(B, -1, 3).contiguous()
        
        # 9) 返回特征用于去噪模块
        completion_features = {
            'query_features': q,  # [B, M, trans_dim]
            'global_feature': global_feature,  # [B, 1024]
            'coarse_points': coarse,  # [B, M, 3]
            'encoder_features': x,  # [B, n, embed_dim]
        }
        
        return coarse, fine_points, completion_features


class PointCloudDenoiseModule(nn.Module):
    """
    独立的点云去噪模块
    可单独训练，也可接收补全模块的特征进行联合优化
    """
    def __init__(self, config):
        super().__init__()
        self.trans_dim = config.decoder_config.embed_dim
        
        # 去噪版本选择
        self.denoise_version = getattr(config, 'denoise_version', 'v2')
        
        self.denoise_head = FeatureConditionedDenoiseHead(
            k=getattr(config, 'denoise_k', 16),
            in_channels=3,
            feat_dim=self.trans_dim,
            global_dim=1024,
            hidden=getattr(config, 'denoise_hidden', 128),
            out_scale=getattr(config, 'denoise_out_scale', 0.03),
            use_confidence=getattr(config, 'denoise_use_confidence', True),
            use_global=True
        )
        
        # 特征投影层（用于将补全特征适配到去噪输入）
        self.use_completion_features = getattr(config, 'use_completion_features', True)
        if self.use_completion_features and self.denoise_version != 'v1':
            self.feat_adapter = nn.Sequential(
                nn.Linear(self.trans_dim, self.trans_dim),
                nn.LayerNorm(self.trans_dim),
                nn.GELU()
            )
        
    def forward(self, points, completion_features=None):
        """
        去噪模块前向传播
        Args:
            points: [B, N, 3] 输入点云（通常是补全后的fine点云）
            completion_features: dict 来自补全模块的特征（可选）
        Returns:
            denoised_points: [B, N, 3] 去噪后的点云
            offsets: [B, N, 3] 点的偏移量
            confidence: [B, N, 1] 置信度（可选）
        """
        B, N, _ = points.size()
        
        if self.denoise_version == 'v1':
            # v1版本只使用坐标
            denoised_points, offsets, conf = self.denoise_head(points)
        else:
            # v2版本使用特征
            if completion_features is not None and self.use_completion_features:
                # 从补全特征中提取
                query_features = completion_features.get('query_features')  # [B, M, trans_dim]
                global_feature = completion_features.get('global_feature')  # [B, 1024]
                
                # 需要将query_features扩展到N个点
                # 假设points是从M个query展开的，每个query对应 factor个点
                M = query_features.size(1)
                factor = N // M
                if N % M == 0:
                    # 简单复制
                    point_features = query_features.unsqueeze(2).expand(-1, -1, factor, -1)
                    point_features = point_features.reshape(B, N, -1)
                else:
                    # 使用插值
                    point_features = query_features.mean(dim=1, keepdim=True).expand(-1, N, -1)
                
                # 特征适配
                if hasattr(self, 'feat_adapter'):
                    point_features = self.feat_adapter(point_features)
                
                denoised_points, offsets, conf = self.denoise_head(
                    points,
                    features=point_features,
                    global_feat=global_feature
                )
            else:
                # 不使用补全特征，直接去噪
                denoised_points, offsets, conf = self.denoise_head(points)
                
        return denoised_points, offsets, conf


@MODELS.register_module()
class UniDCF_denoise(nn.Module):
    """
    统一的点云补全+去噪框架
    特点：
    1. 补全和去噪模块完全分离
    2. 支持三种训练模式：
       - completion_only: 只训练补全模块
       - denoise_only: 只训练去噪模块（冻结补全）
       - joint: 联合训练两个模块
    3. 支持灵活的特征共享策略
    """
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        
        # 训练模式
        self.training_mode = getattr(config, 'training_mode', 'joint_cd')
        assert self.training_mode in ['completion_only', 'denoise_only', 'joint_cd']
        
        # 补全模块
        self.completion_module = PointCloudCompletionModule(config)
        
        # 去噪模块
        self.enable_denoise = getattr(config, 'enable_denoise', True)
        if self.enable_denoise:
            self.denoise_module = PointCloudDenoiseModule(config)
        
        # 损失函数
        self.build_loss_func()
        
        # 损失权重
        self.w_coarse = getattr(config, 'w_coarse', 1.0)
        self.w_fine = getattr(config, 'w_fine', 1.0)
        self.w_denoise = getattr(config, 'w_denoise', 1.0)
        self.w_smooth = getattr(config, 'w_smooth', 0.0)  # 默认关闭
        self.w_offset = getattr(config, 'w_offset', 0.0)  # 默认关闭
        
    def build_loss_func(self):
        self.loss_func = ChamferDistanceL1()
        self.loss_func_l2 = ChamferDistanceL2()
        
    def freeze_completion(self):
        """
        冻结补全模块参数
        注意：DINOv2预训练模型会被完全冻结（包括原本开放的最后两层）
        """
        for param in self.completion_module.parameters():
            param.requires_grad = False
        
        # 确保DINOv2也被完全冻结
        if self.completion_module.use_image and hasattr(self.completion_module, 'img_feature_extractor'):
            self.completion_module.img_feature_extractor.freeze_all()
        
        print_log("Completion module frozen (including DINOv2)", logger='MODEL')
        
    def unfreeze_completion(self):
        """
        解冻补全模块参数
        注意：DINOv2预训练模型只解冻最后两层（保持预训练策略）
        """
        for param in self.completion_module.parameters():
            param.requires_grad = True
        
        # DINOv2恢复标准冻结策略（只开放最后两层）
        if self.completion_module.use_image and hasattr(self.completion_module, 'img_feature_extractor'):
            self.completion_module.img_feature_extractor.freeze_backbone_except_last_layers()
        
        print_log("Completion module unfrozen (DINOv2: last 2 layers trainable)", logger='MODEL')
        
    def freeze_denoise(self):
        """冻结去噪模块参数"""
        if self.enable_denoise:
            for param in self.denoise_module.parameters():
                param.requires_grad = False
            print_log("Denoise module frozen", logger='MODEL')
            
    def unfreeze_denoise(self):
        """解冻去噪模块参数"""
        if self.enable_denoise:
            for param in self.denoise_module.parameters():
                param.requires_grad = True
            print_log("Denoise module unfrozen", logger='MODEL')
    
    def set_training_mode(self, mode):
        """
        设置训练模式
        Args:
            mode: 'completion_only', 'denoise_only', 'joint_cd'
        """
        self.training_mode = mode
        
        if mode == 'completion_only':
            self.unfreeze_completion()
            if self.enable_denoise:
                self.freeze_denoise()
            print_log("Set to completion_only mode", logger='MODEL')
        elif mode == 'denoise_only':
            self.freeze_completion()
            self.unfreeze_denoise()
            print_log("Set to denoise_only mode", logger='MODEL')
        elif mode == 'joint_cd':
            self.unfreeze_completion()
            self.unfreeze_denoise()
            print_log("Set to joint_cd training mode", logger='MODEL')
        else:
            raise ValueError(f"Unknown training mode: {mode}")
        
        # 打印DINOv2状态（如果使用图像特征）
        if self.completion_module.use_image and hasattr(self.completion_module, 'img_feature_extractor'):
            n_trainable = self.completion_module.img_feature_extractor.get_trainable_params()
            total_params = sum(p.numel() for p in self.completion_module.img_feature_extractor.parameters())
            print_log(f"DINOv2 status: {n_trainable}/{total_params} parameters trainable", logger='MODEL')
    
    def forward(self, xyz, image=None):
        """
        前向传播
        Args:
            xyz: [B, N, 3] 输入部分点云
            image: [B, 3, 256, 256] 可选的图像输入
        Returns:
            根据training_mode返回不同的输出：
            - completion_only: (coarse, fine)
            - denoise_only/joint: (coarse, fine, denoised)
        """
        # 1) 补全阶段
        # 只有denoise_only模式需要冻结补全模块
        if self.training_mode == 'denoise_only':
            # 补全模块被冻结：不计算梯度
            self.completion_module.eval()  # 强制eval模式
            with torch.no_grad():
                coarse, fine, completion_features = self.completion_module(xyz, image)
        else:
            # completion_only 和 joint_cd: 补全模块参与训练
            coarse, fine, completion_features = self.completion_module(xyz, image)
        
        # 2) 去噪阶段
        if self.enable_denoise and self.training_mode != 'completion_only':
            # 根据training_mode决定是否使用补全特征
            if self.training_mode == 'denoise_only':
                # 去噪专用：从补全分支detach特征
                detached_features = {
                    k: v.detach() if torch.is_tensor(v) else v 
                    for k, v in completion_features.items()
                }
                denoised, offsets, conf = self.denoise_module(
                    fine.detach(), 
                    detached_features
                )
            elif self.training_mode == 'joint_cd':
                # 联合训练：特征保持梯度,允许梯度从去噪回传到补全
                denoised, offsets, conf = self.denoise_module(
                    fine, 
                    completion_features
                )
            else:
                raise ValueError(f"Unknown training mode: {self.training_mode}")
            
            # 保存中间结果用于可视化
            self.last_offsets = offsets
            self.last_conf = conf.squeeze(-1) if conf is not None else None
            
            return coarse, fine, denoised
        else:
            # 仅补全模式
            return coarse, fine
    
    def get_loss(self, ret, gt):
        """
        计算损失
        Args:
            ret: forward的返回值
            gt: [B, N, 3] 真实完整点云
        Returns:
            loss_coarse: coarse点云损失
            loss_fine: fine点云损失（包含去噪损失）
        """
        if len(ret) == 2:
            # 仅补全模式
            coarse, fine = ret
            denoised = None
        else:
            # 包含去噪
            coarse, fine, denoised = ret
        
        # 基础补全损失
        loss_coarse = self.loss_func(coarse, gt) * self.w_coarse
        loss_fine_base = self.loss_func(fine, gt) * self.w_fine
        
        loss_details = {
            'coarse_cd': loss_coarse,
            'fine_cd': loss_fine_base,
        }
        
        # 去噪损失
        if denoised is not None:
            loss_denoise = self.loss_func(denoised, gt) * self.w_denoise
            loss_details['denoise_cd'] = loss_denoise
            
            # 平滑损失（可选）
            if self.w_smooth > 0 and hasattr(self, 'last_offsets'):
                loss_smooth = self.compute_smooth_loss(
                    self.last_offsets, 
                    fine
                ) * self.w_smooth
                loss_details['smooth'] = loss_smooth
            else:
                loss_smooth = 0.0
            
            # 偏移量正则（可选）
            if self.w_offset > 0 and hasattr(self, 'last_offsets'):
                loss_offset = (self.last_offsets ** 2).mean() * self.w_offset
                loss_details['offset_reg'] = loss_offset
            else:
                loss_offset = 0.0
            
            # 根据训练模式组合损失
            if self.training_mode == 'completion_only':
                loss_fine_total = loss_fine_base
            elif self.training_mode == 'denoise_only':
                # 只优化去噪
                loss_fine_total = loss_denoise + loss_smooth + loss_offset
            elif self.training_mode == 'joint_cd':
                # 联合训练补全+去噪
                loss_fine_total = loss_fine_base + loss_denoise + loss_smooth + loss_offset
            else:
                raise ValueError(f"Unknown training mode: {self.training_mode}")
        else:
            loss_fine_total = loss_fine_base
        
        self.loss_items = loss_details
        
        return loss_coarse, loss_fine_total
    
    def compute_smooth_loss(self, offsets, points, k=16):
        """
        计算偏移量的平滑损失（完全向量化版本）
        Args:
            offsets: [B, N, 3] 点的偏移量
            points: [B, N, 3] 点的坐标
            k: 邻域大小
        Returns:
            smooth_loss: scalar
        """
        B, N, _ = points.size()
        
        # 找k近邻
        idx = knn_point(k, points, points)  # [B, N, k]
        
        # 创建 batch 索引用于高级索引
        batch_idx = torch.arange(B, device=offsets.device).view(B, 1, 1).expand(B, N, k)
        
        # 使用高级索引收集邻居的 offsets（完全向量化，无循环）
        neighbor_offsets = offsets[batch_idx, idx]  # [B, N, k, 3]
        
        # 计算偏移量差异
        offset_diff = offsets.unsqueeze(2) - neighbor_offsets  # [B, N, k, 3]
        smooth_loss = (offset_diff ** 2).mean()
        
        return smooth_loss