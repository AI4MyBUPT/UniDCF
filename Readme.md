# UniDCF: A Foundation Model for Comprehensive Dentocraniofacial Hard Tissue Reconstruction

This repository contains the official PyTorch implementation of **UniDCF**, a foundation model for comprehensive dentocraniofacial (DCF) hard tissue reconstruction from partial point clouds, with optional multi-view projection image guidance.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Pretrained Checkpoints](#pretrained-checkpoints)
- [Inference](#inference)
  - [UniDCF Point Cloud Completion](#unidcf-point-cloud-completion)
  - [Baseline Model Point Cloud Completion](#baseline-model-point-cloud-completion)
  - [UniDCF + SAP Mesh Reconstruction](#unidcf--sap-mesh-reconstruction)
  - [Baseline + SAP Mesh Reconstruction](#baseline--sap-mesh-reconstruction)
  - [SAP Standalone Mesh Reconstruction](#sap-standalone-mesh-reconstruction)
  - [Applying to External Datasets](#applying-to-external-datasets)
- [Training](#training)
- [Baseline Models](#baseline-models)
- [Metrics](#metrics)
- [Configuration Reference](#configuration-reference)
- [Citation](#citation)

---

## Overview

This repository provides the official inference package for the manuscript:

**UniDCF: A Foundation Model for Comprehensive Dentocraniofacial Hard Tissue Reconstruction**
Authors: Chunxia Ren et al.

UniDCF is a domain-specific pretrained reconstruction model for dentocraniofacial hard-tissue reconstruction. It is designed to reconstruct complete 3D hard-tissue structures from partial dentocraniofacial inputs, including teeth/crowns, jaw/alveolar bone, and skull-related structures.

This repository contains the code and resources required to run UniDCF inference, including:

* point-cloud completion from partial input point clouds;
* multimodal inference using point clouds and multi-view projection images;
* point-cloud denoising and refinement;
* point-cloud-to-mesh reconstruction and mesh export;
* pretrained model checkpoints and configuration files;
* example input data and expected outputs;
* scripts for inference.

The package is intended to support reproducible research and technical evaluation of the UniDCF reconstruction pipeline. It is not intended for direct clinical use without further professional review, CAD/CAM validation, manufacturability assessment, and institution-specific quality control.

---

## Repository Structure

```
codeocaen/
├── code/
│   ├── main.py                        # Main training / evaluation entry point
│   ├── inference_UniDCF.py            # UniDCF point cloud completion inference
│   ├── inference_baseline.py          # Baseline model point cloud completion inference
│   ├── inference_UniDCF_mesh.py       # UniDCF completion + SAP mesh reconstruction
│   ├── inference_baseline_mesh.py     # Baseline completion + SAP mesh reconstruction
│   ├── cfgs/
│   │   ├── UniDCF_models/
│   │   │   └── UniDCF.yaml            # UniDCF model & training config
│   │   ├── baseline_model/
│   │   │   ├── AdaPoinTr.yaml
│   │   │   ├── PoinTr.yaml
│   │   │   ├── GRNet.yaml
│   │   │   ├── PCN.yaml
│   │   │   └── SnowFlakeNet.yaml
│   │   └── dataset_configs/
│   │       └── data.yaml              # Dataset paths and settings
│   ├── models/
│   │   ├── UniDCF_util.py             
│   │   ├── UniDCF_denoise.py          
│   │   ├── AdaPoinTr.py              
│   │   ├── PoinTr.py                 
│   │   ├── GRNet.py                   
│   │   ├── PCN.py                     
│   │   ├── SnowFlakeNet.py            
│   │   ├── Transformer_utils.py       
│   │   ├── DenoiseHeadV2.py           
│   │   └── build.py                   
│   ├── sap/                           # SAP mesh reconstruction module
│   │   ├── __init__.py
│   │   ├── dpsr.py                    
│   │   ├── model.py                   
│   │   ├── utils.py                   
│   │   ├── configs/
│   │   │   ├── default.yaml           # SAP default config
│   │   │   └── dsap.yaml             # SAP model config
│   │   └── network/
│   │       ├── __init__.py
│   │       ├── encoder.py            
│   │       ├── decoder.py            
│   │       ├── unet3d.py             
│   │       └── utils.py              
│   ├── datasets/
│   │   ├── io.py                      
│   │   └── data_transforms.py         
│   ├── tools/
│   │   ├── builder.py                
│   │   └── runner.py                  
│   ├── utils/
│   │   ├── config.py                 
│   │   ├── metrics.py                
│   │   ├── misc.py                   
│   │   ├── parser.py                 
│   │   └── logger.py                 
│   ├── extensions/
│   │   ├── chamfer_dist/             
│   │   ├── emd/                      
│   │   ├── gridding/                
│   │   └── cubic_feature_sampling/    
│   └── dinov2/                       
└── data/
    ├── data/
    │   ├── pcd/                       # Input partial point clouds (.ply)
    │   └── ima/                       # Projection images (<stem>_x/y/z.png)
    ├── experiments/
    │   └── UniDCF/
    │       ├── UniDCF.pth             # UniDCF pretrained checkpoint
    │       └── dinov2_vitb14_pretrain.pth  # DINOv2 pretrained weights
    ├── inference_result/              # Point cloud completion output
    └── inference_result_mesh/         # Mesh reconstruction output
```

---

## Requirements

- Python >= 3.8
- PyTorch >= 1.11 (with CUDA support)
- CUDA >= 11.3

Core Python dependencies:

```
torch
torchvision
open3d
timm
tensorboardX
scipy
numpy
PyYAML
trimesh
scikit-image
torch_scatter
```

---

## Installation

### 1. Clone the repository

```bash
git clone <repository_url>
cd codeocaen/code
```

### 2. Install Python dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install open3d timm tensorboardX scipy numpy pyyaml trimesh scikit-image
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

### 3. Build CUDA extensions

All custom CUDA extensions must be compiled before training or inference:

```bash
# Chamfer Distance
cd extensions/chamfer_dist && python setup.py install && cd ../..

# Earth Mover's Distance
cd extensions/emd && python setup.py install && cd ../..

# Gridding
cd extensions/gridding && python setup.py install && cd ../..

# Cubic Feature Sampling
cd extensions/cubic_feature_sampling && python setup.py install && cd ../..
```

### 4. Verify installation

```bash
cd code
python -c "import models; print('All models loaded successfully.')"
```

---

## Data Preparation

### Dataset format (ZYTOOTH)

The model expects the following directory layout:

```
<data_root>/
├── pcd/
│   ├── <scan_id>.ply    # Partial point cloud
├── ima/
│   ├── <case_id>_x/y/z.png  # Three-view projection images
```

---

## Pretrained Checkpoints

| Model | Description | Path |
|---|---|---|
| UniDCF | Full model (completion + denoising + image guidance) | `data/experiments/UniDCF/UniDCF.pth` |
| DINOv2 ViT-B/14 | Image backbone pretrained weights | `data/experiments/UniDCF/dinov2_vitb14_pretrain.pth` |
| SAP (DSAP) | Shape As Points mesh reconstruction model | `code/sap/save_model/dsap.pt` |

> **Note for reproducibility:** Checkpoints for all baseline models (AdaPoinTr, PoinTr, GRNet, PCN, SnowFlakeNet) trained on the ZYTOOTH dataset will be released alongside the paper.

---

## Inference

This project provides a complete inference pipeline supporting two output modes:

1. **Point cloud completion**: Input a partial point cloud, output a dense completed point cloud (`.ply`)
2. **Point cloud completion + mesh reconstruction**: Input a partial point cloud, first complete it then reconstruct a triangle mesh (`.ply` mesh)

Mesh reconstruction is based on the [Shape As Points (NeurIPS 2021)](https://arxiv.org/abs/2106.03452) SAP model, which generates triangle meshes from oriented point clouds via DPSR (Differentiable Poisson Surface Reconstruction) + Marching Cubes.

### Inference Scripts Overview

| Script | Function | Input | Output |
|--------|----------|-------|--------|
| `inference_UniDCF.py` | UniDCF point cloud completion | Point cloud + tri-view images | Completed point cloud |
| `inference_baseline.py` | Baseline model point cloud completion | Point cloud | Completed point cloud |
| `inference_UniDCF_mesh.py` | UniDCF completion + SAP mesh reconstruction | Point cloud + tri-view images | Mesh (+ optional completed point cloud) |
| `inference_baseline_mesh.py` | Baseline completion + SAP mesh reconstruction | Point cloud | Mesh (+ optional completed point cloud) |

---

### UniDCF Point Cloud Completion

Performs point cloud completion only, outputs a dense point cloud:

```bash
cd code

# Single file inference
CUDA_VISIBLE_DEVICES=0 python inference_UniDCF.py \
  --model_config cfgs/UniDCF_models/UniDCF.yaml \
  --model_checkpoint /path/to/UniDCF.pth \
  --pc /path/to/input/case001.ply \
  --ima_root /path/to/input/ima/ \
  --out_pc_root /path/to/output/ \
  --device cuda:0
```

**Input requirements:**
- Point cloud: `.ply` file with XYZ coordinates (any point count; automatically resampled to 14,336 internally)
- Images: Three PNG files named `<stem>_x.png`, `<stem>_y.png`, `<stem>_z.png`, stored in `ima_root`

**Output:**
- `out_pc_root/<stem>.ply`, containing 2,048 completed points

---

### Baseline Model Point Cloud Completion

Baseline models (AdaPoinTr, PoinTr, GRNet, PCN, SnowFlakeNet) require only point cloud input, without projection images:

```bash
cd code

# Single file inference
CUDA_VISIBLE_DEVICES=0 python inference_baseline.py \
  --model_config cfgs/baseline_model/AdaPoinTr.yaml \
  --model_checkpoint /path/to/AdaPoinTr.pth \
  --pc /path/to/input/case001.ply \
  --out_pc_root /path/to/output/ \
  --device cuda:0
```

Switch between baseline models by changing `--model_config` and `--model_checkpoint`:

| Model | Config |
|---|---|
| AdaPoinTr | `cfgs/baseline_model/AdaPoinTr.yaml` |
| PoinTr | `cfgs/baseline_model/PoinTr.yaml` |
| GRNet | `cfgs/baseline_model/GRNet.yaml` |
| PCN | `cfgs/baseline_model/PCN.yaml` |
| SnowFlakeNet | `cfgs/baseline_model/SnowFlakeNet.yaml` |

---

### UniDCF + SAP Mesh Reconstruction

End-to-end pipeline: UniDCF completes the point cloud, then SAP automatically reconstructs a triangle mesh:

```bash
cd code

# Batch inference (output meshes)
CUDA_VISIBLE_DEVICES=0 python inference_UniDCF_mesh.py \
  --model_config cfgs/UniDCF_models/UniDCF.yaml \
  --model_checkpoint /path/to/UniDCF.pth \
  --pc_root /path/to/input/pcd/ \
  --ima_root /path/to/input/ima/ \
  --out_root /path/to/output_mesh/ \
  --device cuda:0

# Single file inference + save intermediate completed point cloud
CUDA_VISIBLE_DEVICES=0 python inference_UniDCF_mesh.py \
  --model_config cfgs/UniDCF_models/UniDCF.yaml \
  --model_checkpoint /path/to/UniDCF.pth \
  --pc /path/to/input/case001.ply \
  --ima_root /path/to/input/ima/ \
  --out_root /path/to/output_mesh/ \
  --save_completion \
  --device cuda:0
```

**Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--model_config` | UniDCF config file | `cfgs/UniDCF_models/UniDCF.yaml` |
| `--model_checkpoint` | UniDCF pretrained weights | - |
| `--pc_root` | Input point cloud directory | - |
| `--pc` | Single input point cloud path | - |
| `--ima_root` | Tri-view projection image directory | - |
| `--sap_config` | SAP model config | `sap/configs/dsap.yaml` |
| `--sap_checkpoint` | SAP pretrained weights | `sap/save_model/dsap.pt` |
| `--sap_target_points` | SAP input point count (<=0 uses config value 2048) | -1 |
| `--out_root` | Output mesh directory | `data/inference_result_mesh/` |
| `--save_completion` | Also save completed point cloud | False |
| `--device` | Inference device | `cuda:0` |
| `--seed` | Random seed | 1 |

**Output:**
- `out_root/<stem>_mesh.ply` — Triangle mesh file
- `out_root/<stem>_completion.ply` — Completed point cloud (requires `--save_completion`)

**Example output log:**
```
[1/4] Loading UniDCF model...
      UniDCF model loaded.
[2/4] Loading SAP model...
      SAP model loaded.
[3/4] Running inference pipeline...
      Found 10 .ply file(s)
      case001.ply -> output_mesh/case001_mesh.ply  (V=53466, F=106924, completion=0.35s, mesh=0.12s)
      case002.ply -> output_mesh/case002_mesh.ply  (V=48230, F=96452, completion=0.33s, mesh=0.11s)
      ...
[4/4] Done! Processed 10 file(s).
      Average time per sample: 0.47s
      Output directory: output_mesh/
```

---

### Baseline + SAP Mesh Reconstruction

Same pipeline as the UniDCF version, but uses a baseline model for point cloud completion (no image input required):

```bash
cd code

# Single file inference
CUDA_VISIBLE_DEVICES=0 python inference_baseline_mesh.py \
  --model_config cfgs/baseline_model/AdaPoinTr.yaml \
  --model_checkpoint /path/to/AdaPoinTr.pth \
  --pc /path/to/input/case001.ply \
  --out_root /path/to/output_mesh/ \
  --device cuda:0
```

---

### Visualize results

```python
import open3d as o3d

# Visualize mesh
mesh = o3d.io.read_triangle_mesh("/your/output/mesh/case001_mesh.ply")
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh])

# Visualize completed point cloud
pcd = o3d.io.read_point_cloud("/your/output/mesh/case001_completion.ply")
o3d.visualization.draw_geometries([pcd])
```

---

## Training

All training is launched via `main.py` from the `code/` directory. UniDCF uses a multi-stage training strategy.

### Quick Start

```bash
cd code

CUDA_VISIBLE_DEVICES=0 python main.py \
  --config cfgs/UniDCF_models/UniDCF.yaml \

```

## Baseline Models

| Model | Config |
|---|---|
| **AdaPoinTr** | `cfgs/baseline_model/AdaPoinTr.yaml` |
| **PoinTr** | `cfgs/baseline_model/PoinTr.yaml` |
| **GRNet** | `cfgs/baseline_model/GRNet.yaml` |
| **PCN** | `cfgs/baseline_model/PCN.yaml` |
| **SnowFlakeNet** | `cfgs/baseline_model/SnowFlakeNet.yaml` |

---

## Metrics

| Metric | Description | Better |
|---|---|---|
| **CDL1** | Chamfer Distance L1 (×10³) | Lower |
| **CDL2** | Chamfer Distance L2 (×10³) | Lower |
| **F-Score** | F-Score at threshold 0.01 | Higher |
| **EMDistance** | Earth Mover's Distance (×10³) | Lower |

---

## Citation

If you find this work useful in your research, please cite:

```bibtex
@article{ren2025unidcf,
  title={UniDCF: A Foundation Model for Comprehensive Dentocraniofacial Hard Tissue Reconstruction},
  author={Ren, Chunxia and Zhu, Ning and Lai, Yue and Chen, Gui and Wang, Ruijie and Hu, Yangyi and Liu, Suyao and Mao, Shuwen and Su, Hong and Zhang, Yu and others},
  journal={arXiv preprint arXiv:2508.11728},
  year={2025}
}
```

The baseline models are based on:

```bibtex
@article{AdaPoinTr,
    author={Yu, Xumin and Rao, Yongming and Wang, Ziyi and Lu, Jiwen and Zhou, Jie},
    title={{ AdaPoinTr: Diverse Point Cloud Completion With Adaptive Geometry-Aware Transformers }},
    journal={ IEEE Transactions on Pattern Analysis \& Machine Intelligence },
    volume={45},
    number={12},
    pages={14114-14130},
    year={2023}
}

@inproceedings{GRNet,
    author = {Xie, Haozhe and Yao, Hongxun and Zhou, Shangchen and Mao, Jiageng and Zhang, Shengping and Sun, Wenxiu},
    title = {GRNet: Gridding Residual Network for Dense Point Cloud Completion},
    numpages = {17},
    booktitle = {Computer Vision – ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part IX},
    pages = {365–381},
    address = {Berlin, Heidelberg},
    publisher = {Springer-Verlag},
    year = {2020}
}

@inproceedings{PCN,
  author={Yuan, Wentao and Khot, Tejas and Held, David and Mertz, Christoph and Hebert, Martial},
  title={PCN: Point Completion Network}, 
  volume={},
  booktitle={2018 International Conference on 3D Vision (3DV)}, 
  pages={728-737},
  year={2018}
}

@inproceedings{PoinTr,
  author={Yu, Xumin and Rao, Yongming and Wang, Ziyi and Liu, Zuyan and Lu, Jiwen and Zhou, Jie},
  title={PoinTr: Diverse Point Cloud Completion with Geometry-Aware Transformers}, 
  volume={},
  booktitle={2021 IEEE/CVF International Conference on Computer Vision (ICCV)}, 
  pages={12478-12487},
  year={2021}
}

@inproceedings {SnowflakeNet,
    author = { Xiang, Peng and Wen, Xin and Liu, Yu-Shen and Cao, Yan-Pei and Wan, Pengfei and Zheng, Wen and Han, Zhizhong },
    title = {{ SnowflakeNet: Point Cloud Completion by Snowflake Point Deconvolution with Skip-Transformer }},
    volume = {},
    booktitle = { 2021 IEEE/CVF International Conference on Computer Vision (ICCV) },
    pages = {5479-5489},
    address = {Los Alamitos, CA, USA},
    publisher = {IEEE Computer Society},
    year = {2021}
}
```

The mesh reconstruction module is based on Shape As Points:

```bibtex
@article{peng2021shape,
  title={Shape as points: A differentiable poisson solver},
  author={Peng, Songyou and Jiang, Chiyu and Liao, Yiyi and Niemeyer, Michael and Pollefeys, Marc and Geiger, Andreas},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={13032--13044},
  year={2021}
}
```

---

## License

This project is released for research purposes. Please refer to the LICENSE file for details.
