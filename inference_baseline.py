##############################################################
# % Author: Chunxia Ren
# % Date:31/03/2026
###############################################################
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


# 推理时固定的预处理流水线：确定性采样 -> 转 Tensor
TRANSFORM = Compose([
    {
        'callback': 'DeterministicSamplePoints',
        'parameters': {'n_points': 14336},
        'objects': ['input']
    },
    {
        'callback': 'ToTensor',
        'objects': ['input']
    }
])


def get_args():
    parser = argparse.ArgumentParser(description='UniDCF single-dataset inference')
    parser.add_argument('--model_config', type=str,
                        default='/code/cfgs/baseline_model/AdaPoinTr.yaml',
                        help='YAML config file path')
    parser.add_argument('--model_checkpoint', type=str,
                        default='/data/experiments/baseline/AdaPoinTr.pth',
                        help='Pretrained checkpoint path')
    parser.add_argument('--pc_root', type=str,
                        default='/data/data/pcd',
                        help='Directory containing input .ply files')
    parser.add_argument('--pc', type=str, default='',
                        help='Single .ply file path (used when --pc_root is not set)')
    parser.add_argument('--out_pc_root', type=str,
                        default='/code/inference_result/',
                        help='Root directory for output .ply files')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Inference device (e.g. cuda:0 or cpu)')
    args = parser.parse_args()

    assert args.model_config is not None, 'model_config must be specified'
    assert args.model_checkpoint is not None, 'model_checkpoint must be specified'
    assert (args.pc != '') or (args.pc_root != ''), \
        'Either --pc or --pc_root must be provided'

    return args


def load_inputs(pc_path):
    """读取单个样本的点云。

    Args:
        pc_path:  输入点云文件路径（.ply）

    Returns:
        pc_ndarray: (N, 3) float32 点云数组
    """
    pc_ndarray = IO.get(pc_path).astype(np.float32)
    return pc_ndarray


def save_point_cloud(points, save_path):
    """将 (N, 3) numpy 数组保存为 .ply 文件。"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    o3d.io.write_point_cloud(save_path, pcd)


@torch.no_grad()
def inference_single(model, pc_path, out_path, device):
    """对单个点云样本执行补全推理并保存结果。

    Args:
        model:    已加载权重的模型（eval 模式）
        pc_path:  输入 .ply 文件路径
        out_path: 输出 .ply 文件保存路径
        device:   推理设备字符串
    """
    if not pc_path.endswith('.ply'):
        print(f'[SKIP] Not a .ply file: {pc_path}')
        return

    pc_ndarray = load_inputs(pc_path)

    # 预处理：采样、转 Tensor
    sample = TRANSFORM({'input': pc_ndarray})
    pc_tensor  = sample['input'].unsqueeze(0).to(device)

    # 前向推理；ret[-1] 为最终 dense 补全结果
    ret = model(pc_tensor)
    dense_points = ret[-1].squeeze(0).cpu().numpy()

    save_point_cloud(dense_points, out_path)


def build_model(args, config):
    """构建模型并加载预训练权重。"""
    model = builder.model_builder(config.model)
    builder.load_model(model, args.model_checkpoint)
    model.to(args.device.lower())
    model.eval()
    return model


def main():
    args = get_args()
    config = cfg_from_yaml_file(args.model_config)
    model = build_model(args, config)
    device = args.device.lower()

    os.makedirs(args.out_pc_root, exist_ok=True)

    if args.pc_root:
        # 批量推理：遍历 pc_root 下所有 .ply 文件
        pc_file_list = sorted(f for f in os.listdir(args.pc_root) if f.endswith('.ply'))
        print(f'Found {len(pc_file_list)} ply file(s) in {args.pc_root}')

        for pc_file in pc_file_list:
            pc_path  = os.path.join(args.pc_root, pc_file)
            out_path = os.path.join(args.out_pc_root, pc_file)
            t0 = time.time()
            inference_single(model, pc_path, out_path, device)
            print(f'  {pc_file} -> {out_path}  ({time.time() - t0:.2f}s)')
    else:
        # 单文件推理
        pc_file  = os.path.basename(args.pc)
        out_path = os.path.join(args.out_pc_root, pc_file)
        t0 = time.time()
        inference_single(model, args.pc, out_path, device)
        print(f'  {pc_file} -> {out_path}  ({time.time() - t0:.2f}s)')


if __name__ == '__main__':
    main()
