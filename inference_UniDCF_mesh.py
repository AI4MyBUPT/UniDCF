##############################################################
# UniDCF + SAP 联合推理脚本
# 流程: 输入残缺点云 -> UniDCF补全 -> SAP重建网格
# Author: Chunxia Ren
##############################################################
import argparse
import os
import sys
import time

import numpy as np
import open3d as o3d
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../'))

from tools import builder
from utils.config import cfg_from_yaml_file
from datasets.io import IO
from datasets.data_transforms import Compose

from sap.dpsr import DPSR
from sap.model import Encode2Points
from sap.utils import (
    export_mesh,
    load_config,
    load_model_manual,
    mc_from_psr,
    scale2onet,
)


# ======================== UniDCF 预处理 ========================
TRANSFORM = Compose([
    {
        'callback': 'DeterministicSamplePoints',
        'parameters': {'n_points': 14336},
        'objects': ['input']
    },
    {
        'callback': 'TransIma',
        'parameters': {'size': 224},
        'objects': ['partialima']
    },
    {
        'callback': 'ToTensor',
        'objects': ['input']
    }
])


# ======================== SAP 辅助函数 ========================
def normalize_bbox(points):
    """将点云归一化到 [-0.5, 0.5] 范围。"""
    points = points.astype(np.float32)
    p_min = points.min(axis=0)
    p_max = points.max(axis=0)
    center = (p_min + p_max) / 2.0
    scale = np.max(p_max - p_min)
    if scale <= 0:
        raise ValueError("Point cloud bounding box scale is invalid.")
    points_norm = (points - center) / scale
    return points_norm.astype(np.float32), center.astype(np.float32), np.float32(scale)


def to_sap_input(points_onet, padding=1.2):
    """将 ONet 空间点云转换到 SAP 的 [0, 1) 坐标范围。"""
    points_sap = points_onet / padding + 0.5
    points_sap = np.clip(points_sap, 1e-6, 0.99)
    return points_sap.astype(np.float32)


def resample_points(points, target_n, seed=0):
    """将点云重采样到目标点数。"""
    if target_n is None or target_n <= 0 or points.shape[0] == target_n:
        return points.astype(np.float32)
    rng = np.random.default_rng(seed)
    replace = points.shape[0] < target_n
    indices = rng.choice(points.shape[0], size=target_n, replace=replace)
    return points[indices].astype(np.float32)


# ======================== 参数解析 ========================
def get_args():
    parser = argparse.ArgumentParser(
        description='UniDCF point cloud completion + SAP mesh reconstruction')

    # UniDCF 参数
    parser.add_argument('--model_config', type=str,
                        default=os.path.join(BASE_DIR, 'cfgs/UniDCF_models/UniDCF.yaml'),
                        help='UniDCF YAML config file path')
    parser.add_argument('--model_checkpoint', type=str,
                        default='/data/experiments/UniDCF/UniDCF.pth',
                        help='UniDCF pretrained checkpoint path')
    parser.add_argument('--pc_root', type=str, default='/data/data/pcd',
                        help='Directory containing input .ply files')
    parser.add_argument('--ima_root', type=str, default='/data/data/ima',
                        help='Directory containing projection images (_x/_y/_z.png)')
    parser.add_argument('--pc', type=str, default='',
                        help='Single .ply file path')

    # SAP 参数
    parser.add_argument('--sap_config', type=str,
                        default=os.path.join(BASE_DIR, 'sap/configs/dsap.yaml'),
                        help='SAP model config YAML path')
    parser.add_argument('--sap_checkpoint', type=str,
                        default=os.path.join(BASE_DIR, 'sap/save_model/dsap.pt'),
                        help='SAP model checkpoint path')
    parser.add_argument('--sap_target_points', type=int, default=-1,
                        help='SAP input point count (<=0 uses config value)')

    # 输出参数
    parser.add_argument('--out_root', type=str,
                        default='/code/inference_mesh_result/',
                        help='Output directory for mesh files')
    parser.add_argument('--save_completion', action='store_true', default=False,
                        help='Also save the intermediate completion point cloud')

    # 通用参数
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Inference device')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed for SAP resampling')

    args = parser.parse_args()

    assert args.model_config is not None
    assert args.model_checkpoint is not None
    assert (args.pc != '') or (args.pc_root != ''), \
        'Either --pc or --pc_root must be provided'

    return args


# ======================== UniDCF 推理 ========================
def load_inputs(pc_path, ima_root):
    """读取点云和三视图投影图像。"""
    stem = os.path.splitext(os.path.basename(pc_path))[0]
    x_path = os.path.join(ima_root, f'{stem}_x.png')
    y_path = os.path.join(ima_root, f'{stem}_y.png')
    z_path = os.path.join(ima_root, f'{stem}_z.png')

    pc_ndarray = IO.get(pc_path).astype(np.float32)
    ima_np = np.stack([
        IO.get(x_path).astype(np.float32),
        IO.get(y_path).astype(np.float32),
        IO.get(z_path).astype(np.float32),
    ], axis=-1)

    return pc_ndarray, ima_np


@torch.no_grad()
def completion_inference(model, pc_path, ima_root, device):
    """UniDCF 补全推理，返回补全后的点云 numpy 数组。"""
    if not pc_path.endswith('.ply'):
        print(f'[SKIP] Not a .ply file: {pc_path}')
        return None

    pc_ndarray, ima_np = load_inputs(pc_path, ima_root)

    sample = TRANSFORM({'input': pc_ndarray, 'partialima': ima_np})
    pc_tensor = sample['input'].unsqueeze(0).to(device)
    ima_tensor = sample['partialima'].unsqueeze(0).to(device)

    ret = model(pc_tensor, ima_tensor)
    dense_points = ret[-1].squeeze(0).cpu().numpy()

    return dense_points


# ======================== SAP 网格重建 ========================
@torch.no_grad()
def mesh_reconstruction(sap_model, dpsr, points_np, sap_cfg, device, seed=1):
    """
    使用 SAP 模型将补全后的点云重建为三角网格。

    Args:
        sap_model: 加载好的 SAP Encode2Points 模型
        dpsr: DPSR 模块
        points_np: (N, 3) numpy 点云数组
        sap_cfg: SAP 配置字典
        device: 推理设备
        seed: 随机种子

    Returns:
        verts: (V, 3) 网格顶点
        faces: (F, 3) 网格面
    """
    # 归一化到 ONet 空间
    points_onet, center, scale = normalize_bbox(points_np)

    # 重采样到 SAP 所需点数
    target_n = sap_cfg['data']['pointcloud_n']
    points_onet = resample_points(points_onet, target_n, seed=seed)

    # 转换到 SAP 坐标空间
    points_sap = to_sap_input(points_onet, padding=1.2)

    # SAP 推理
    inputs = torch.from_numpy(points_sap[None]).to(device)
    pred_points, pred_normals = sap_model(inputs)

    # DPSR + Marching Cubes
    psr_grid = dpsr(pred_points, pred_normals)
    zero_level = sap_cfg['data']['zero_level']
    verts, faces, _ = mc_from_psr(psr_grid, zero_level=zero_level)

    # 还原到原始坐标空间
    verts_onet = scale2onet(verts)
    if isinstance(verts_onet, torch.Tensor):
        verts_onet = verts_onet.detach().cpu().numpy()
    if isinstance(faces, torch.Tensor):
        faces = faces.detach().cpu().numpy()

    verts_original = verts_onet.astype(np.float32) * scale + center

    return verts_original, faces


def save_point_cloud(points, save_path):
    """保存点云为 .ply 文件。"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    o3d.io.write_point_cloud(save_path, pcd)


def save_mesh(verts, faces, save_path):
    """保存网格为 .ply 文件。"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    o3d.io.write_triangle_mesh(save_path, mesh)


# ======================== 主函数 ========================
def main():
    args = get_args()
    device = args.device.lower()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---------- 加载 UniDCF 模型 ----------
    print('[1/4] Loading UniDCF model...')
    unidcf_config = cfg_from_yaml_file(args.model_config)
    unidcf_model = builder.model_builder(unidcf_config.model)
    builder.load_model(unidcf_model, args.model_checkpoint)
    unidcf_model.to(device)
    unidcf_model.eval()
    print('      UniDCF model loaded.')

    # ---------- 加载 SAP 模型 ----------
    print('[2/4] Loading SAP model...')
    sap_default_config = os.path.join(BASE_DIR, 'sap/configs/default.yaml')
    sap_cfg = load_config(args.sap_config, sap_default_config)

    if args.sap_target_points > 0:
        sap_cfg['data']['pointcloud_n'] = args.sap_target_points

    sap_model = Encode2Points(sap_cfg).to(device)
    sap_state = torch.load(args.sap_checkpoint, map_location=device)
    load_model_manual(sap_state['state_dict'], sap_model)
    sap_model.eval()

    # 初始化 DPSR
    psr_resolution = sap_cfg['generation'].get('psr_resolution', 0) or sap_cfg['model']['grid_res']
    psr_sigma = sap_cfg['generation'].get('psr_sigma', 0) or sap_cfg['model']['psr_sigma']
    dpsr = DPSR(
        res=(psr_resolution, psr_resolution, psr_resolution),
        sig=psr_sigma,
    ).to(device)
    print('      SAP model loaded.')

    # ---------- 准备输出目录 ----------
    os.makedirs(args.out_root, exist_ok=True)

    # ---------- 执行推理 ----------
    print('[3/4] Running inference pipeline...')

    if args.pc_root:
        pc_file_list = sorted(f for f in os.listdir(args.pc_root) if f.endswith('.ply'))
        print(f'      Found {len(pc_file_list)} .ply file(s)')
    else:
        pc_file_list = [os.path.basename(args.pc)]
        args.pc_root = os.path.dirname(args.pc)

    results = []
    for pc_file in pc_file_list:
        pc_path = os.path.join(args.pc_root, pc_file)
        stem = os.path.splitext(pc_file)[0]

        t_start = time.time()

        # Step A: UniDCF 补全
        completed_points = completion_inference(
            unidcf_model, pc_path, args.ima_root, device)
        if completed_points is None:
            continue
        t_completion = time.time()

        # Step B: SAP 网格重建
        verts, faces = mesh_reconstruction(
            sap_model, dpsr, completed_points, sap_cfg, device, seed=args.seed)
        t_mesh = time.time()

        # Step C: 保存结果
        mesh_path = os.path.join(args.out_root, f'{stem}_UniDCF_mesh.ply')
        save_mesh(verts, faces, mesh_path)

        if args.save_completion:
            pc_out_path = os.path.join(args.out_root, f'{stem}_UniDCF_completion.ply')
            save_point_cloud(completed_points, pc_out_path)

        total_time = t_mesh - t_start
        results.append({
            'file': pc_file,
            'verts': verts.shape[0],
            'faces': faces.shape[0],
            'time_completion': t_completion - t_start,
            'time_mesh': t_mesh - t_completion,
            'time_total': total_time,
        })
        print(f'      {pc_file} -> {mesh_path}  '
              f'(V={verts.shape[0]}, F={faces.shape[0]}, '
              f'completion={t_completion - t_start:.2f}s, '
              f'mesh={t_mesh - t_completion:.2f}s)')

    # ---------- 汇总 ----------
    print(f'[4/4] Done! Processed {len(results)} file(s).')
    if results:
        avg_time = np.mean([r['time_total'] for r in results])
        print(f'      Average time per sample: {avg_time:.2f}s')
        print(f'      Output directory: {args.out_root}')


if __name__ == '__main__':
    main()
