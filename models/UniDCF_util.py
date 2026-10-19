##############################################################
# % Author: Castle
# % Date:01/12/2022
###############################################################

import torch
import torch.nn as nn
import torchvision.models as models
from functools import partial
from timm.models.layers import trunc_normal_

from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from extensions.emd import emd_module
from .build import MODELS
from models.Transformer_utils import *
from utils import misc

from dinov2.eval.utils import ModelWithIntermediateLayers
from dinov2.hub.backbones import dinov2_vitb14

# 从 AdaPoinTr 导入与其完全相同的基础组件，避免重复定义
from .AdaPoinTr import (
    SelfAttnBlockApi,
    CrossAttnBlockApi,
    TransformerEncoder,
    TransformerDecoder,
    PointTransformerEncoder,
    PointTransformerDecoder,
    PointTransformerEncoderEntry,
    PointTransformerDecoderEntry,
    DGCNN_Grouper,
    Encoder,
    SimpleEncoder,
    Fold,
    SimpleRebuildFCLayer,
)


######################################## UniDCF 专有模块 ########################################

class MultiLayerVisionTransformer(nn.Module):
    """
    多层图像 Encoder (ViT)：patch embed + L 层 Transformer Block。
    用于将投影图像编码为序列特征。
    """
    def __init__(self, img_size=256, patch_size=16, in_chans=3,
                 embed_dim=256, depth=6, num_heads=4, mlp_ratio=4.0):
        super().__init__()

        # patch embedding：将图像切分为 patch 并映射到 embed_dim
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # 多层 Transformer Block
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                batch_first=True
            )
            for _ in range(depth)
        ])

        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x):
        """
        Args:
            x: [B, H, W, C]（注意通道在最后）
        Returns:
            [B, 1 + num_patches, embed_dim]（cls_token + patch tokens）
        """
        B, H, W, C = x.shape
        # patch embed 要求输入为 [B, C, H, W]
        x = x.permute(0, 3, 1, 2)
        x = self.patch_embed(x)          # [B, embed_dim, H', W']
        x = x.flatten(2).transpose(1, 2) # [B, num_patches, embed_dim]

        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)       # [B, 1+num_patches, embed_dim]
        x = x + self.pos_embed[:, :x.size(1), :]

        for blk in self.blocks:
            x = blk(x)
        return x


class ViewFeatureFusionModule(nn.Module):
    """
    视图特征融合模块：以点云特征为 query，图像特征为 key/value，
    通过交叉注意力将两种模态的特征融合。
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )
        self.attn_layer = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)

    def forward(self, point_features, image_features):
        """
        Args:
            point_features:  [B, N, embed_dim] 点云特征
            image_features:  [B, num_patches, embed_dim] 图像特征
        Returns:
            fused_features:  [B, N, embed_dim]
        """
        if image_features.dim() < 2:
            image_features = image_features.unsqueeze(0)
        view_feature = self.mlp(image_features)
        attn_output, _ = self.attn_layer(point_features, view_feature, view_feature)
        # 残差连接，保留原始点云特征
        fused_features = attn_output + point_features
        return fused_features


class DinoViTEncoder(nn.Module):
    """
    基于 DINOv2 ViT-B/14 的图像编码器。
    默认冻结主干网络，仅解冻最后两个 Transformer block 和 projection 层。
    """
    def __init__(self, pretrained_path: str, device=None):
        super().__init__()

        self.base = dinov2_vitb14(weights={'LVD142M': pretrained_path})
        self.embed_dim = self.base.embed_dim  # ViT-B/14: 768

        autocast_ctx = partial(torch.amp.autocast, 'cuda', enabled=True, dtype=torch.float16)
        self.feature_model = ModelWithIntermediateLayers(
            self.base, n_last_blocks=1, autocast_ctx=autocast_ctx
        )

        # 将 DINOv2 输出维度映射到 1024
        self.projection = nn.Linear(self.embed_dim, 1024)

        self.freeze_backbone_except_last_layers()

    def freeze_backbone_except_last_layers(self):
        """冻结主干，仅保留最后两个 block（blocks.10/11）和 projection 可训练。"""
        for param in self.feature_model.parameters():
            param.requires_grad = False
        for name, param in self.base.named_parameters():
            if 'blocks.10' in name or 'blocks.11' in name:
                param.requires_grad = True
        for param in self.projection.parameters():
            param.requires_grad = True

    def freeze_all(self):
        """完全冻结 DINOv2（含 projection），用于特定训练阶段。"""
        for param in self.feature_model.parameters():
            param.requires_grad = False
        for param in self.projection.parameters():
            param.requires_grad = False

    def get_trainable_params(self):
        """返回可训练参数数量（用于调试）。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            [B, 1024] float32 图像全局特征
        """
        with torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
            features = self.feature_model(x)
            feature_list = features[0]
            out = self.projection(feature_list[-1])
        return out.to(dtype=torch.float32)


######################################## UniDCF_PCTransformer ########################################

class UniDCF_PCTransformer(nn.Module):
    """
    UniDCF 版本的 PCTransformer。
    在 AdaPoinTr.PCTransformer 基础上增加了：
      - DINOv2 图像编码器（DinoViTEncoder）
      - 视图特征融合模块（ViewFeatureFusionModule）
      - 图像-点云联合全局特征生成
    forward 接口为 (xyz, image) 而非仅 (xyz)。
    """
    def __init__(self, config):
        super().__init__()
        encoder_config = config.encoder_config
        decoder_config = config.decoder_config
        self.center_num = getattr(config, 'center_num', [512, 128])
        self.encoder_type = config.encoder_type
        assert self.encoder_type in ['graph', 'pn'], f'unexpected encoder_type {self.encoder_type}'

        in_chans = 3
        self.num_query = query_num = config.num_query
        global_feature_dim = config.global_feature_dim

        print_log(f'Transformer with config {config}', logger='MODEL')

        # 点云局部特征提取（DGCNN 或 SimpleEncoder）
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

        # 点云 Transformer 编码器
        self.encoder = PointTransformerEncoderEntry(encoder_config)

        # DINOv2 图像编码器 + 视图融合
        dino_path = getattr(config, 'dino_path', '/path/to/dinov2_vitb14_pretrain.pth')
        self.img_feature_extractor = DinoViTEncoder(dino_path)
        self.view_feature_fusion = ViewFeatureFusionModule(embed_dim=1024)

        # 将点云全局特征（1024）与图像全局特征（1024）融合为 global_feature_dim
        self.increase_dim = nn.Sequential(
            nn.Linear(encoder_config.embed_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, global_feature_dim)
        )

        # 粗点云预测头
        self.coarse_pred = nn.Sequential(
            nn.Linear(global_feature_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, 3 * query_num)
        )
        self.mlp_query = nn.Sequential(
            nn.Linear(global_feature_dim + 3, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, decoder_config.embed_dim)
        )

        if decoder_config.embed_dim == encoder_config.embed_dim:
            self.mem_link = nn.Identity()
        else:
            self.mem_link = nn.Linear(encoder_config.embed_dim, decoder_config.embed_dim)

        # Transformer 解码器
        self.decoder = PointTransformerDecoderEntry(decoder_config)

        # query 排序网络（按空间位置打分）
        self.query_ranking = nn.Sequential(
            nn.Linear(3, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, xyz, image):
        """
        Args:
            xyz:   [B, N, 3] 输入点云
            image: [B, 3, H, W] 对应投影图像
        Returns:
            q:      [B, num_query, embed_dim] decoder 输出特征
            coarse: [B, num_query, 3] 粗点云坐标
        """
        bs = xyz.size(0)

        # 1) 点云局部特征提取与编码
        coor, f = self.grouper(xyz, self.center_num)
        pe = self.pos_embed(coor)
        x = self.input_proj(f)
        x = self.encoder(x + pe, coor)

        # 2) 点云全局特征
        pc_feat = self.increase_dim(x)
        pc_feat = torch.max(pc_feat, dim=1)[0]  # [B, global_feature_dim]

        # 3) 图像特征提取与融合
        depth_feat = self.img_feature_extractor(image)
        fused_features = self.view_feature_fusion(pc_feat, depth_feat)

        # 4) 生成粗点云
        coarse = self.coarse_pred(fused_features).reshape(bs, -1, 3)

        # 5) memory（编码器输出）
        mem = self.mem_link(x)

        # 6) query 选点（按排名取 top-num_query）
        query_ranking = self.query_ranking(coarse)
        idx = torch.argsort(query_ranking, dim=1, descending=True)
        coarse = torch.gather(coarse, 1, idx[:, :self.num_query].expand(-1, -1, coarse.size(-1)))

        # 7) 生成 query 并经 decoder 精化
        q = self.mlp_query(torch.cat([
            pc_feat.unsqueeze(1).expand(-1, coarse.size(1), -1),
            coarse
        ], dim=-1))
        q = self.decoder(q=q, v=mem, q_pos=coarse, v_pos=coor)

        return q, coarse


######################################## UniDCF ########################################

@MODELS.register_module()
class UniDCF(nn.Module):
    """
    UniDCF 点云补全模型。
    在 AdaPoinTr 基础上引入图像模态输入，使用 UniDCF_PCTransformer 作为骨干，
    损失函数额外加入 EMD（Earth Mover's Distance）。
    """
    def __init__(self, config, **kwargs):
        super().__init__()
        self.trans_dim = config.decoder_config.embed_dim
        self.num_query = config.num_query
        self.num_points = getattr(config, 'num_points', None)

        self.decoder_type = config.decoder_type
        assert self.decoder_type in ['fold', 'fc'], f'unexpected decoder_type {self.decoder_type}'

        self.fold_step = 8
        self.base_model = UniDCF_PCTransformer(config)

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

        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1)
        )
        self.reduce_map = nn.Linear(self.trans_dim + 1027, self.trans_dim)
        self.build_loss_func()

    def build_loss_func(self):
        self.loss_func = ChamferDistanceL1()
        self.loss_func_l2 = ChamferDistanceL2()
        self.dist = emd_module.emdModule()

    def get_loss(self, ret, gt):
        """
        Args:
            ret: (pred_coarse, pred_fine) 模型输出
            gt:  [B, N, 3] ground truth 点云
        Returns:
            loss_coarse, loss_fine（Chamfer L1）
        """
        pred_coarse, pred_fine = ret
        assert pred_fine.size(1) == gt.size(1)
        loss_coarse = self.loss_func(pred_coarse, gt)
        loss_fine = self.loss_func(pred_fine, gt)
        return loss_coarse, loss_fine

    def forward(self, xyz, image):
        """
        Args:
            xyz:   [B, N, 3] 输入残缺点云
            image: [B, 3, H, W] 对应投影图像
        Returns:
            (coarse_point_cloud, rebuild_points)
        """
        q, coarse_point_cloud = self.base_model(xyz, image)
        B, M, C = q.shape

        # 全局特征聚合
        global_feature = self.increase_dim(q.transpose(1, 2)).transpose(1, 2)  # [B, M, 1024]
        global_feature = torch.max(global_feature, dim=1)[0]                   # [B, 1024]

        # 拼接全局特征、局部 query 特征、粗点云坐标
        rebuild_feature = torch.cat([
            global_feature.unsqueeze(-2).expand(-1, M, -1),
            q,
            coarse_point_cloud
        ], dim=-1)  # [B, M, 1027 + C]

        # 细粒度点云重建（fold 或 fc 两种解码方式）
        if self.decoder_type == 'fold':
            rebuild_feature = self.reduce_map(rebuild_feature.reshape(B * M, -1))
            relative_xyz = self.decode_head(rebuild_feature).reshape(B, M, 3, -1)
            rebuild_points = (relative_xyz + coarse_point_cloud.unsqueeze(-1)).transpose(2, 3)
        else:
            rebuild_feature = self.reduce_map(rebuild_feature)
            relative_xyz = self.decode_head(rebuild_feature)
            rebuild_points = relative_xyz + coarse_point_cloud.unsqueeze(-2)

        rebuild_points = rebuild_points.reshape(B, -1, 3).contiguous()

        assert rebuild_points.size(1) == self.num_query * self.factor
        assert coarse_point_cloud.size(1) == self.num_query

        return coarse_point_cloud, rebuild_points
