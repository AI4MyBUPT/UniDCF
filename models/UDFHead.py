"""
UDF预测头：利用点云坐标 + 局部特征 + 全局特征 预测无符号距离场
支持 Eikonal 梯度约束
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .dgcnn_group import square_distance

class PositionalEncoder(nn.Module):
    """位置编码器：将3D坐标映射到高维特征空间"""
    def __init__(self, input_dim=3, embed_dim=64, num_freqs=10):
        super().__init__()
        self.num_freqs = num_freqs
        self.embed_dim = embed_dim
        
        # Fourier特征编码
        freq_bands = 2.0 ** torch.linspace(0, num_freqs-1, num_freqs)
        self.register_buffer('freq_bands', freq_bands)
        
        # MLP映射到目标维度
        fourier_dim = input_dim * num_freqs * 2  # sin + cos
        self.mlp = nn.Sequential(
            nn.Linear(fourier_dim + input_dim, embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim * 2, embed_dim)
        )
    
    def forward(self, coords):
        """
        Args:
            coords: [B, N, 3] 点云坐标
        Returns:
            pos_feat: [B, N, embed_dim] 位置编码特征
        """
        B, N, C = coords.shape
        
        # Fourier特征: [sin(2^k * pi * x), cos(2^k * pi * x)]
        coords_scaled = coords.unsqueeze(-1) * self.freq_bands.view(1, 1, 1, -1) * 3.14159
        sin_feat = torch.sin(coords_scaled).reshape(B, N, -1)
        cos_feat = torch.cos(coords_scaled).reshape(B, N, -1)
        
        fourier_feat = torch.cat([sin_feat, cos_feat, coords], dim=-1)
        pos_feat = self.mlp(fourier_feat)
        
        return pos_feat


class FineFeatInterpolator(nn.Module):
    """细粒度特征插值器：为每个fine点生成独特特征"""
    def __init__(self, coarse_dim, pos_dim, output_dim, k=8, temp=10.0):
        """
        Args:
            coarse_dim: coarse特征维度（如384）
            pos_dim: 位置编码维度（如64）
            output_dim: 输出特征维度（如128）
            k: KNN邻居数
            temp: Softmax温度参数（控制平滑度）
        """
        super().__init__()
        self.k = k
        self.temp = temp  # 新增：温度参数
        
        # 特征融合MLP
        input_dim = coarse_dim + pos_dim + 3
        
        self.fusion_mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, output_dim)
        )
        
        # K-NN聚合注意力（可选）
        self.attn_weight = nn.Sequential(
            nn.Linear(output_dim + 3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )
    
    def forward(self, fine_points, coarse_points, coarse_features, pos_encoding):
        """
        Args:
            fine_points: [B, N_fine, 3] 精细点坐标
            coarse_points: [B, N_coarse, 3] 粗糙点坐标
            coarse_features: [B, N_coarse, C] 粗糙点特征
            pos_encoding: [B, N_fine, D] 精细点位置编码
        
        Returns:
            fine_features: [B, N_fine, output_dim] 插值后的精细特征
        """
        B, N_fine, _ = fine_points.shape
        N_coarse = coarse_points.size(1)
        
        # 1. 找到每个fine点的K个最近coarse点
        with torch.no_grad():
            # 计算距离矩阵 [B, N_fine, N_coarse]
            dist = square_distance(fine_points, coarse_points)
            # 取最近的K个
            knn_dist, knn_idx = torch.topk(dist, k=min(self.k, N_coarse), 
                                           dim=-1, largest=False, sorted=True)
            
            # ========== 改进：使用Softmax权重代替IDW ==========
            # knn_dist: [B, N_fine, K] 平方距离
            # 取平方根得到欧氏距离（可选，也可以直接用平方距离）
            knn_dist_sqrt = torch.sqrt(knn_dist + 1e-8)
            
            # Softmax权重: exp(-beta * dist) / sum(exp(-beta * dist))
            # beta = 1/temp，temp越大权重越平滑
            weights = torch.softmax(-knn_dist_sqrt / self.temp, dim=-1)  # [B, N_fine, K]
            # ================================================
        
        # 2. 聚合K个邻居的特征
        knn_idx_exp = knn_idx.unsqueeze(-1).expand(-1, -1, -1, coarse_features.size(-1))
        knn_features = torch.gather(
            coarse_features.unsqueeze(1).expand(-1, N_fine, -1, -1),
            dim=2,
            index=knn_idx_exp
        )  # [B, N_fine, K, C]
        
        # 加权聚合
        aggregated_feat = torch.sum(
            knn_features * weights.unsqueeze(-1),
            dim=2
        )  # [B, N_fine, C]
        
        # 3. 计算相对位置（到最近coarse点）
        nearest_idx = knn_idx[:, :, 0:1]  # [B, N_fine, 1]
        nearest_idx_exp = nearest_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
        
        nearest_coarse_pos = torch.gather(
            coarse_points.unsqueeze(1).expand(-1, N_fine, -1, -1),
            dim=2,
            index=nearest_idx_exp
        ).squeeze(2)  # [B, N_fine, 3]
        
        relative_pos = fine_points - nearest_coarse_pos  # [B, N_fine, 3]
        
        # 4. 融合所有特征
        fused_input = torch.cat([
            aggregated_feat,
            pos_encoding,
            relative_pos
        ], dim=-1)
        
        fine_features = self.fusion_mlp(fused_input)
        
        return fine_features


class FeatureConditionedUDFHead(nn.Module):
    """
    特征条件化的UDF预测头
    
    利用：
    1. 查询点坐标 (query_points)
    2. 表面点云坐标 (surface_points) - 用于KNN局部几何
    3. 表面点云特征 (surface_features) - Decoder输出的丰富语义特征
    4. 全局特征 (global_feature) - 物体级别的语义条件
    
    输入:
        - query_points: [B, N_q, 3] 查询点坐标
        - surface_points: [B, N_s, 3] 表面点云坐标
        - surface_features: [B, N_s, C_feat] 表面点云特征（来自Decoder）
        - global_feature: [B, C_global] 全局特征（来自Encoder）
    
    输出:
        - udf_values: [B, N_q] 预测的UDF值
    """
    
    def __init__(self, point_dim=3, feat_dim=384, global_dim=1024, hidden_dim=256, 
                 num_layers=6, k=16, use_skip=True):
        super().__init__()
        self.k = k
        self.use_skip = use_skip
        self.hidden_dim = hidden_dim
        
        # ========== 1. 局部几何编码器 ==========
        # 编码查询点与邻居点的相对几何关系
        self.local_geo_encoder = nn.Sequential(
            nn.Linear(point_dim * 2, hidden_dim),  # 相对坐标 + 查询点坐标
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # ========== 2. 局部特征聚合器 ==========
        # 聚合KNN邻居的语义特征
        self.local_feat_encoder = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # 几何-特征融合
        self.geo_feat_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # ========== 3. 全局条件编码器 ==========
        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # ========== 4. 查询点位置编码 ==========
        # 使用正弦位置编码增强坐标表示
        self.pos_freq = 10  # 频率数量
        pos_enc_dim = point_dim * (2 * self.pos_freq + 1)  # 3 * 21 = 63
        self.query_encoder = nn.Sequential(
            nn.Linear(pos_enc_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # ========== 5. UDF回归网络 (带Skip Connection) ==========
        # 输入: query_feat + local_feat + global_feat
        mlp_input_dim = hidden_dim * 3
        self.skip_layer_idx = num_layers // 2
        
        self.mlp_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                in_dim = mlp_input_dim
            elif i == self.skip_layer_idx and use_skip:
                in_dim = hidden_dim + mlp_input_dim  # Skip connection
            else:
                in_dim = hidden_dim
            
            self.mlp_layers.append(nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True) if i < num_layers - 1 else nn.Identity()
            ))
        
        # 最终输出层
        self.output_layer = nn.Linear(hidden_dim, 1)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def positional_encoding(self, x):
        """
        正弦位置编码
        Args:
            x: [B, N, 3] 坐标
        Returns:
            [B, N, 3 * (2 * freq + 1)] 位置编码
        """
        freqs = 2.0 ** torch.linspace(0, self.pos_freq - 1, self.pos_freq, device=x.device)
        freqs = freqs.view(1, 1, 1, -1)  # [1, 1, 1, freq]
        
        x_expanded = x.unsqueeze(-1)  # [B, N, 3, 1]
        x_freq = x_expanded * freqs * torch.pi  # [B, N, 3, freq]
        
        # 拼接 [x, sin(x), cos(x), sin(2x), cos(2x), ...]
        pe = torch.cat([
            x,
            torch.sin(x_freq).reshape(*x.shape[:-1], -1),
            torch.cos(x_freq).reshape(*x.shape[:-1], -1)
        ], dim=-1)
        
        return pe
    
    def forward(self, query_points, surface_points, surface_features=None, global_feature=None):
        """
        Args:
            query_points: [B, N_q, 3] 查询点坐标
            surface_points: [B, N_s, 3] 表面点云坐标
            surface_features: [B, N_s, C_feat] 表面点特征（可选）
            global_feature: [B, C_global] 全局特征（可选）
        
        Returns:
            udf_values: [B, N_q] 预测的UDF值
        """
        B, N_q, _ = query_points.shape
        N_s = surface_points.shape[1]
        
        # ========== 1. 查询点位置编码 ==========
        query_pe = self.positional_encoding(query_points)  # [B, N_q, pe_dim]
        query_feat = self.query_encoder(query_pe)  # [B, N_q, hidden_dim]
        
        # ========== 2. KNN搜索 ==========
        with torch.no_grad():
            dist_matrix = torch.cdist(query_points, surface_points)  # [B, N_q, N_s]
            _, knn_idx = dist_matrix.topk(self.k, dim=-1, largest=False)  # [B, N_q, k]
        
        # ========== 3. 局部几何特征 ==========
        # 获取KNN邻居坐标
        knn_idx_expand = knn_idx.unsqueeze(-1).expand(-1, -1, -1, 3)  # [B, N_q, k, 3]
        surface_expand = surface_points.unsqueeze(1).expand(-1, N_q, -1, -1)  # [B, N_q, N_s, 3]
        knn_points = torch.gather(surface_expand, 2, knn_idx_expand)  # [B, N_q, k, 3]
        
        # 相对坐标
        query_expand = query_points.unsqueeze(2).expand(-1, -1, self.k, -1)  # [B, N_q, k, 3]
        relative_pos = knn_points - query_expand  # [B, N_q, k, 3]
        
        # 编码局部几何
        geo_input = torch.cat([relative_pos, query_expand], dim=-1)  # [B, N_q, k, 6]
        local_geo = self.local_geo_encoder(geo_input)  # [B, N_q, k, hidden]
        local_geo = local_geo.max(dim=2)[0]  # [B, N_q, hidden] - Max pooling
        
        # ========== 4. 局部语义特征 ==========
        if surface_features is not None:
            C_feat = surface_features.shape[-1]
            knn_idx_feat = knn_idx.unsqueeze(-1).expand(-1, -1, -1, C_feat)  # [B, N_q, k, C]
            surface_feat_expand = surface_features.unsqueeze(1).expand(-1, N_q, -1, -1)  # [B, N_q, N_s, C]
            knn_features = torch.gather(surface_feat_expand, 2, knn_idx_feat)  # [B, N_q, k, C]
            
            # 编码局部特征
            local_feat = self.local_feat_encoder(knn_features)  # [B, N_q, k, hidden]
            local_feat = local_feat.max(dim=2)[0]  # [B, N_q, hidden]
            
            # 融合几何和特征
            local_combined = self.geo_feat_fusion(torch.cat([local_geo, local_feat], dim=-1))
        else:
            local_combined = local_geo  # 无特征时只用几何
        
        # ========== 5. 全局条件 ==========
        if global_feature is not None:
            global_feat = self.global_encoder(global_feature)  # [B, hidden]
            global_feat = global_feat.unsqueeze(1).expand(-1, N_q, -1)  # [B, N_q, hidden]
        else:
            global_feat = torch.zeros(B, N_q, self.hidden_dim, device=query_points.device)
        
        # ========== 6. MLP预测 ==========
        combined_feat = torch.cat([query_feat, local_combined, global_feat], dim=-1)  # [B, N_q, hidden*3]
        skip_feat = combined_feat
        
        x = combined_feat
        for i, layer in enumerate(self.mlp_layers):
            if i == self.skip_layer_idx and self.use_skip:
                x = torch.cat([x, skip_feat], dim=-1)
            x = layer(x)
        
        # 输出UDF值（确保非负）
        udf_values = self.output_layer(x).squeeze(-1)  # [B, N_q]
        udf_values = F.softplus(udf_values)  # 确保UDF > 0
        
        return udf_values
    
    def forward_with_grad(self, query_points, surface_points, surface_features=None, global_feature=None):
        """
        带梯度计算的前向传播（用于Eikonal Loss）
        
        Returns:
            udf_values: [B, N_q]
            udf_grad: [B, N_q, 3] UDF对查询点坐标的梯度
        """
        query_points.requires_grad_(True)
        
        udf_values = self.forward(query_points, surface_points, surface_features, global_feature)
        
        # 计算UDF对查询点的梯度
        udf_grad = torch.autograd.grad(
            outputs=udf_values,
            inputs=query_points,
            grad_outputs=torch.ones_like(udf_values),
            create_graph=True,
            retain_graph=True
        )[0]  # [B, N_q, 3]
        
        return udf_values, udf_grad


# 为了向后兼容，保留原始类名
class UDFHead(FeatureConditionedUDFHead):
    """向后兼容的别名"""
    pass
