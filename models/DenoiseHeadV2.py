import torch
import torch.nn as nn
import torch.nn.functional as F
from .dgcnn_group import knn_point, square_distance


def get_graph_feature_hybrid(coor: torch.Tensor, feat: torch.Tensor, k: int = 16):
    """
    构造混合邻域特征：坐标差 + 特征差 + 中心特征
    coor: [B, 3, N] 坐标
    feat: [B, C, N] 语义特征
    return: [B, 2*C+3, N, k]  拼接 [相对坐标, f_j-f_i, f_i]
    """
    B, C_feat, N = feat.size()
    
    with torch.no_grad():
        idx = knn_point(k+1, xyz=coor.transpose(1,2).contiguous(), 
                        new_xyz=coor.transpose(1,2).contiguous())
        idx = idx[:, :, 1:]  # 去掉自身
        idx_base = torch.arange(0, B, device=feat.device).view(-1,1,1) * N
        idx_flat = (idx + idx_base).reshape(-1)
    
    # 邻域坐标
    coor_t = coor.transpose(2,1).contiguous()  # [B, N, 3]
    neighbor_coor = coor_t.view(B*N, 3)[idx_flat, :].view(B, N, k, 3)
    coor_center = coor_t.unsqueeze(2).expand(-1, -1, k, -1)
    relative_coor = neighbor_coor - coor_center  # [B, N, k, 3]
    
    # 邻域特征
    feat_t = feat.transpose(2,1).contiguous()  # [B, N, C]
    neighbor_feat = feat_t.view(B*N, C_feat)[idx_flat, :].view(B, N, k, C_feat)
    feat_center = feat_t.unsqueeze(2).expand(-1, -1, k, -1)
    relative_feat = neighbor_feat - feat_center  # [B, N, k, C]
    
    # 拼接：相对坐标 + 相对特征 + 中心特征
    edge = torch.cat([relative_coor, relative_feat, feat_center], dim=3)  # [B, N, k, 3+2C]
    return edge.permute(0, 3, 1, 2).contiguous()  # [B, 3+2C, N, k]


class FeatureConditionedDenoiseHead(nn.Module):
    """
    特征条件去噪头：融合补全阶段的语义特征
    
    输入:
        points: [B, N, 3] 补全后的点云坐标
        features: [B, N, C] 补全阶段提取的特征 (可选)
        global_feat: [B, C_global] 全局特征 (可选)
    
    输出:
        denoised_points: [B, N, 3]
        offsets: [B, N, 3]
        confidence: [B, N, 1]
    """
    def __init__(self, k=16, in_channels=3, feat_dim=None, global_dim=None,
                 hidden=128, out_scale=0.03, use_confidence=True, use_global=True):
        super().__init__()
        self.k = k
        self.out_scale = out_scale
        self.use_confidence = use_confidence
        self.use_global = use_global
        self.feat_dim = feat_dim
        self.global_dim = global_dim
        #print("feat_dim384: ", feat_dim)
        # ===== 特征投影 =====
        if feat_dim is not None:
            #print("=============")
            # 将补全特征投影到统一维度
            self.feat_proj = nn.Sequential(
                nn.Conv1d(feat_dim, hidden, 1, bias=False),
                # nn.BatchNorm1d(hidden),     # 问题：随机初始化的BN可能产生极端值
                # 替换 BN1d：用 LayerNorm 或 GroupNorm
                # 这里用 GroupNorm，对 1D conv 特别常见
                nn.GroupNorm(num_groups=8, num_channels=hidden),
                nn.LeakyReLU(0.2)
            )
        else:
            self.feat_proj = None
            
        # ===== 全局特征融合 =====
        if use_global and global_dim is not None:
            self.global_proj = nn.Sequential(
                nn.Linear(global_dim, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, hidden)
            )
        else:
            self.global_proj = None
        
        # ===== 混合EdgeConv =====
        # EdgeConv1: 输入是 [相对坐标(3) + 相对特征(C) + 中心特征(C)]
        edge_in_dim = 3 + 2 * (hidden if feat_dim else in_channels)
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(edge_in_dim, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.LeakyReLU(0.2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(2*hidden, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.LeakyReLU(0.2)
        )
        
        # ===== 特征融合 =====
        # f1 + f2 + (proj_feat) + (global_feat)
        fuse_dim = hidden * 2
        if feat_dim:
            fuse_dim += hidden
        if use_global and global_dim:
            fuse_dim += hidden
            
        self.mlp = nn.Sequential(
            nn.Conv1d(fuse_dim, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Conv1d(256, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2)
        )
        
        # ===== 输出层 =====
        self.out_offset = nn.Conv1d(64, 3, 1)
        
        if use_confidence:
            self.out_conf = nn.Sequential(
                nn.Conv1d(64, 1, 1),
                nn.Sigmoid()
            )
    
    def forward(self, points, features=None, global_feat=None):
        """
        points: [B, N, 3]
        features: [B, N, C] 或 None (补全阶段的逐点特征)
        global_feat: [B, C_g] 或 None (全局形状特征)
        """

        B, N, _ = points.size()
        # 调试模式：仅在训练时检查 NaN/Inf
        if self.training:
            assert not torch.isnan(points).any(), "NaN in points BEFORE denoise_head"
            assert not torch.isinf(points).any(), "Inf in points BEFORE denoise_head"
            if features is not None:
                assert not torch.isnan(features).any(), "NaN in features BEFORE feat_proj"
                assert not torch.isinf(features).any(), "Inf in features BEFORE feat_proj"
            if self.feat_proj is not None:
                conv = self.feat_proj[0]
                assert not torch.isnan(conv.weight).any(), "NaN in feat_proj Conv weight"
                if conv.bias is not None:
                    assert not torch.isnan(conv.bias).any(), "NaN in feat_proj Conv bias"

        coor = points.transpose(1,2).contiguous()  # [B, 3, N]
            
        # ===== 特征预处理 =====
        if features is not None and self.feat_proj is not None:
            feat_in = features.transpose(1,2).contiguous()  # [B, C, N]
            proj_feat = self.feat_proj(feat_in)  # [B, hidden, N]
        else:
            proj_feat = coor  # 退化为只用坐标

        # ===== 混合EdgeConv =====
        # EdgeConv1: 在坐标+特征上构图
        e1 = get_graph_feature_hybrid(coor, proj_feat, k=self.k)  # [B, 3+2*hidden, N, k]
        f1 = self.conv1(e1).max(dim=-1)[0]  # [B, hidden, N]

        # EdgeConv2: 继续迭代
        e2 = get_graph_feature_hybrid(coor, f1, k=self.k)
        # 这里输入维度是 3+2*hidden，但我们只想用特征部分
        # 简化：直接在 f1 上做标准EdgeConv
        with torch.no_grad():
            idx = knn_point(self.k+1, xyz=coor.transpose(1,2).contiguous(), 
                           new_xyz=coor.transpose(1,2).contiguous())[:,:,1:]
            assert idx.min() >= 0 and idx.max() < N, f"KNN idx out of range: min={idx.min()}, max={idx.max()}, N={N}"
            idx_base = torch.arange(0, B, device=coor.device).view(-1,1,1)*N
            idx_flat = (idx + idx_base).reshape(-1)
        
        f1_t = f1.transpose(2,1).contiguous()
        neighbor_f1 = f1_t.view(B*N, f1.size(1))[idx_flat,:].view(B,N,self.k,f1.size(1))
        center_f1 = f1_t.unsqueeze(2).expand(-1,-1,self.k,-1)
        edge_f1 = torch.cat([neighbor_f1 - center_f1, center_f1], dim=3)
        edge_f1 = edge_f1.permute(0,3,1,2).contiguous()
        f2 = self.conv2(edge_f1).max(dim=-1)[0]  # [B, hidden, N]
        
        # ===== 特征融合 =====
        feat_list = [f1, f2]
        
        if features is not None and self.feat_proj is not None:
            feat_list.append(proj_feat)
            
        if self.use_global and global_feat is not None and self.global_proj is not None:
            g = self.global_proj(global_feat)  # [B, hidden]
            g_expand = g.unsqueeze(-1).expand(-1, -1, N)  # [B, hidden, N]
            feat_list.append(g_expand)
        
        f_fused = torch.cat(feat_list, dim=1)  # [B, fuse_dim, N]
        feat_final = self.mlp(f_fused)  # [B, 64, N]
        
        # ===== 偏移与置信度 =====
        offset = torch.tanh(self.out_offset(feat_final)) * self.out_scale
        offset = offset.transpose(1,2).contiguous()  # [B, N, 3]
        
        if self.use_confidence:
            conf = self.out_conf(feat_final).transpose(1,2).contiguous()  # [B, N, 1]
            denoised = points + offset * conf
            return denoised, offset, conf
        else:
            denoised = points + offset
            return denoised, offset, None


class IterativeDenoiseHead(nn.Module):
    """
    迭代去噪头：模拟 Langevin Dynamics 的多步优化
    每步根据当前点云重新计算邻域和偏移
    """
    def __init__(self, k=16, in_channels=3, feat_dim=None, global_dim=None,
                 hidden=128, out_scale=0.03, num_iters=3, step_decay=0.9,
                 use_confidence=True):
        super().__init__()
        self.num_iters = num_iters
        self.step_decay = step_decay
        self.use_confidence = use_confidence
        
        # 共享的去噪模块
        self.denoise_module = FeatureConditionedDenoiseHead(
            k=k, in_channels=in_channels, feat_dim=feat_dim, global_dim=global_dim,
            hidden=hidden, out_scale=out_scale, use_confidence=use_confidence
        )
    
    def forward(self, points, features=None, global_feat=None):
        """
        多步迭代去噪
        points: [B, N, 3]
        """
        current = points
        trajectory = [points.clone()]
        
        for i in range(self.num_iters):
            denoised, offset, conf = self.denoise_module(current, features, global_feat)
            
            # 带衰减的更新
            alpha = self.step_decay ** i
            if self.use_confidence and conf is not None:
                current = current + alpha * offset * conf
            else:
                current = current + alpha * offset
            
            trajectory.append(current.clone())
        
        # 返回最终结果和轨迹
        final_offset = current - points
        final_conf = conf if self.use_confidence else None
        
        return current, final_offset, final_conf, trajectory


class ScoreMatchingDenoiseHead(nn.Module):
    """
    基于Score Matching思想的去噪头
    不直接预测偏移，而是预测"梯度场"（指向干净流形的方向）
    
    灵感来源: Score-Denoise (ICCV 2021)
    但简化为单步 + 特征条件版本
    """
    def __init__(self, k=32, feat_dim=384, global_dim=1024, 
                 hidden=128, num_blocks=3, use_confidence=True):
        super().__init__()
        self.k = k
        self.use_confidence = use_confidence
        
        # 特征提取 (类似 Score-Denoise 的 FeatureNet)
        self.feature_net = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, hidden, 1)
        )
        
        # 补全特征投影
        if feat_dim:
            self.feat_proj = nn.Linear(feat_dim, hidden)
        else:
            self.feat_proj = None
            
        # 全局特征投影
        if global_dim:
            self.global_proj = nn.Linear(global_dim, hidden)
        else:
            self.global_proj = None
        
        # Score网络 (条件MLP)
        cond_dim = hidden  # 点特征
        if feat_dim:
            cond_dim += hidden
        if global_dim:
            cond_dim += hidden
            
        self.score_net = nn.ModuleList()
        for i in range(num_blocks):
            if i == 0:
                # 第一个block：输入是 3+cond_dim
                self.score_net.append(nn.Sequential(
                    nn.Conv1d(3 + cond_dim, hidden, 1),
                    nn.BatchNorm1d(hidden),
                    nn.ReLU(),
                    nn.Conv1d(hidden, hidden, 1),
                    nn.BatchNorm1d(hidden),
                    nn.ReLU()
                ))
            else:
                # 后续block：输入是 hidden
                self.score_net.append(nn.Sequential(
                    nn.Conv1d(hidden, hidden, 1),
                    nn.BatchNorm1d(hidden),
                    nn.ReLU(),
                    nn.Conv1d(hidden, hidden, 1),
                    nn.BatchNorm1d(hidden),
                    nn.ReLU()
                ))
        
        self.score_out = nn.Conv1d(hidden, 3, 1)
        
        if use_confidence:
            self.conf_net = nn.Sequential(
                nn.Conv1d(hidden, 1, 1),
                nn.Sigmoid()
            )
    
    def forward(self, points, features=None, global_feat=None):
        """
        points: [B, N, 3]
        features: [B, N, C] 补全特征
        global_feat: [B, C_g] 全局特征
        
        输出: (denoised, offsets, confidence)
        """
        B, N, _ = points.size()
        x = points.transpose(1,2).contiguous()  # [B, 3, N]
        
        # ===== 提取点云自身特征 =====
        point_feat = self.feature_net(x)  # [B, hidden, N]
        
        # ===== 构建条件向量 =====
        cond_list = [point_feat]
        
        if features is not None and self.feat_proj is not None:
            comp_feat = self.feat_proj(features)  # [B, N, hidden]
            cond_list.append(comp_feat.transpose(1,2).contiguous())
        
        if global_feat is not None and self.global_proj is not None:
            g = self.global_proj(global_feat)  # [B, hidden]
            cond_list.append(g.unsqueeze(-1).expand(-1, -1, N))
        
        cond = torch.cat(cond_list, dim=1)  # [B, cond_dim, N]
        
        # ===== Score网络 (预测梯度) =====
        # 输入：相对坐标 + 条件特征
        # 简化版：直接用绝对坐标 + 条件
        score_in = torch.cat([x, cond], dim=1)  # [B, 3+cond_dim, N]
        
        # 第一个block处理输入，后续block使用残差连接
        h = self.score_net[0](score_in)  # [B, hidden, N]
        for i in range(1, len(self.score_net)):
            h = self.score_net[i](h) + h  # 残差连接
        
        score = self.score_out(h)  # [B, 3, N]
        score = score.transpose(1,2).contiguous()  # [B, N, 3]
        
        # ===== 置信度 =====
        if self.use_confidence:
            conf = self.conf_net(h).transpose(1,2).contiguous()  # [B, N, 1]
        else:
            conf = None
        
        # ===== 去噪 (Langevin-style update) =====
        step_size = 0.02  # 可调
        if conf is not None:
            offset = step_size * score * conf
        else:
            offset = step_size * score
        
        denoised = points + offset
        
        return denoised, offset, conf

