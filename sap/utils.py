"""Utility functions for SAP inference pipeline."""

import os
import yaml
import numpy as np
import torch
from collections import OrderedDict
from skimage import measure
import open3d as o3d


##################################################
# DPSR helper functions

def fftfreqs(res, dtype=torch.float32, exact=True):
    n_dims = len(res)
    freqs = []
    for dim in range(n_dims - 1):
        r_ = res[dim]
        freq = np.fft.fftfreq(r_, d=1/r_)
        freqs.append(torch.tensor(freq, dtype=dtype))
    r_ = res[-1]
    if exact:
        freqs.append(torch.tensor(np.fft.rfftfreq(r_, d=1/r_), dtype=dtype))
    else:
        freqs.append(torch.tensor(np.fft.rfftfreq(r_, d=1/r_)[:-1], dtype=dtype))
    omega = torch.meshgrid(freqs)
    omega = list(omega)
    omega = torch.stack(omega, dim=-1)
    return omega


def img(x, deg=1):
    """Multiply tensor x by i ** deg (assume last dim: real/imag)."""
    deg %= 4
    if deg == 0:
        res = x
    elif deg == 1:
        res = x[..., [1, 0]]
        res[..., 0] = -res[..., 0]
    elif deg == 2:
        res = -x
    elif deg == 3:
        res = x[..., [1, 0]]
        res[..., 1] = -res[..., 1]
    return res


def spec_gaussian_filter(res, sig):
    omega = fftfreqs(res, dtype=torch.float64)
    dis = torch.sqrt(torch.sum(omega ** 2, dim=-1))
    filter_ = torch.exp(-0.5*((sig*2*dis/res[0])**2)).unsqueeze(-1).unsqueeze(-1)
    filter_.requires_grad = False
    return filter_


def grid_interp(grid, pts, batched=True):
    """
    :param grid: tensor of shape (batch, *size, in_features)
    :param pts: tensor of shape (batch, num_points, dim) within range (0, 1)
    :return values at query points
    """
    if not batched:
        grid = grid.unsqueeze(0)
        pts = pts.unsqueeze(0)
    dim = pts.shape[-1]
    bs = grid.shape[0]
    size = torch.tensor(grid.shape[1:-1]).to(grid.device).type(pts.dtype)
    cubesize = 1.0 / size

    ind0 = torch.floor(pts / cubesize).long()
    ind1 = torch.fmod(torch.ceil(pts / cubesize), size).long()
    ind01 = torch.stack((ind0, ind1), dim=0)
    tmp = torch.tensor([0, 1], dtype=torch.long)
    com_ = torch.stack(torch.meshgrid(tuple([tmp] * dim)), dim=-1).view(-1, dim)
    dim_ = torch.arange(dim).repeat(com_.shape[0], 1)
    ind_ = ind01[com_, ..., dim_]
    ind_n = ind_.permute(2, 3, 0, 1)
    ind_b = torch.arange(bs).expand(ind_n.shape[1], ind_n.shape[2], bs).permute(2, 0, 1)

    if dim == 2:
        lat = grid.clone()[ind_b, ind_n[..., 0], ind_n[..., 1]]
    else:
        lat = grid.clone()[ind_b, ind_n[..., 0], ind_n[..., 1], ind_n[..., 2]]

    xyz0 = ind0.type(cubesize.dtype) * cubesize
    xyz1 = (ind0.type(cubesize.dtype) + 1) * cubesize
    xyz01 = torch.stack((xyz0, xyz1), dim=0)
    pos = xyz01[com_, ..., dim_].permute(2, 3, 0, 1)
    pos_ = xyz01[1-com_, ..., dim_].permute(2, 3, 0, 1)
    pos_ = pos_.type(pts.dtype)
    dxyz_ = torch.abs(pts.unsqueeze(-2) - pos_) / cubesize
    weights = torch.prod(dxyz_, dim=-1, keepdim=False)
    query_values = torch.sum(lat * weights.unsqueeze(-1), dim=-2)
    if not batched:
        query_values = query_values.squeeze(0)
    return query_values


def point_rasterize(pts, vals, size):
    """
    :param pts: point coords, tensor of shape (batch, num_points, dim) within range (0, 1)
    :param vals: point values, tensor of shape (batch, num_points, features)
    :param size: len(size)=dim tuple for grid size
    :return rasterized values (batch, features, res0, res1, res2)
    """
    dim = pts.shape[-1]
    assert(pts.shape[:2] == vals.shape[:2])
    assert(pts.shape[2] == dim)
    size_list = list(size)
    size = torch.tensor(size).to(pts.device).float()
    cubesize = 1.0 / size
    bs = pts.shape[0]
    nf = vals.shape[-1]
    npts = pts.shape[1]
    dev = pts.device

    ind0 = torch.floor(pts / cubesize).long()
    ind1 = torch.fmod(torch.ceil(pts / cubesize), size).long()
    ind01 = torch.stack((ind0, ind1), dim=0)
    tmp = torch.tensor([0, 1], dtype=torch.long)
    com_ = torch.stack(torch.meshgrid(tuple([tmp] * dim)), dim=-1).view(-1, dim)
    dim_ = torch.arange(dim).repeat(com_.shape[0], 1)
    ind_ = ind01[com_, ..., dim_]
    ind_n = ind_.permute(2, 3, 0, 1)
    ind_b = torch.arange(bs, device=dev).expand(ind_n.shape[1], ind_n.shape[2], bs).permute(2, 0, 1)

    xyz0 = ind0.type(cubesize.dtype) * cubesize
    xyz1 = (ind0.type(cubesize.dtype) + 1) * cubesize
    xyz01 = torch.stack((xyz0, xyz1), dim=0)
    pos = xyz01[com_, ..., dim_].permute(2, 3, 0, 1)
    pos_ = xyz01[1-com_, ..., dim_].permute(2, 3, 0, 1)
    pos_ = pos_.type(pts.dtype)
    dxyz_ = torch.abs(pts.unsqueeze(-2) - pos_) / cubesize
    weights = torch.prod(dxyz_, dim=-1, keepdim=False)

    ind_b = ind_b.unsqueeze(-1).unsqueeze(-1)
    ind_n = ind_n.unsqueeze(-2)
    ind_f = torch.arange(nf, device=dev).view(1, 1, 1, nf, 1)

    ind_b = ind_b.expand(bs, npts, 2**dim, nf, 1)
    ind_n = ind_n.expand(bs, npts, 2**dim, nf, dim).to(dev)
    ind_f = ind_f.expand(bs, npts, 2**dim, nf, 1)
    inds = torch.cat([ind_b, ind_f, ind_n], dim=-1)

    vals = weights.unsqueeze(-1) * vals.unsqueeze(-2)

    inds = inds.view(-1, dim+2).permute(1, 0).long()
    vals = vals.reshape(-1)
    raster = scatter_to_grid(inds.permute(1, 0), vals, [bs, nf] + size_list)

    return raster


def scatter_to_grid(inds, vals, size):
    dims = inds.shape[1]
    assert(inds.shape[0] == vals.shape[0])
    assert(len(size) == dims)
    dev = vals.device
    result = torch.zeros(*size, device=dev).view(-1).type(vals.dtype)
    fac = [np.prod(size[i+1:]) for i in range(len(size)-1)] + [1]
    fac = torch.tensor(fac, device=dev).type(inds.dtype)
    inds_fold = torch.sum(inds*fac, dim=-1)
    result.scatter_add_(0, inds_fold, vals)
    result = result.view(*size)
    return result


##################################################
# General utility functions

def load_model_manual(state_dict, model):
    """Load state_dict handling module. prefix mismatch."""
    new_state_dict = OrderedDict()
    is_model_parallel = isinstance(model, torch.nn.DataParallel)
    for k, v in state_dict.items():
        if k.startswith('module.') != is_model_parallel:
            if k.startswith('module.'):
                k = k[7:]
            else:
                k = 'module.' + k
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict)


def mc_from_psr(psr_grid, pytorchify=False, real_scale=False, zero_level=0):
    """Run marching cubes from PSR grid."""
    batch_size = psr_grid.shape[0]
    s = psr_grid.shape[-1]
    psr_grid_numpy = psr_grid.squeeze().detach().cpu().numpy()

    if batch_size > 1:
        verts, faces, normals = [], [], []
        for i in range(batch_size):
            verts_cur, faces_cur, normals_cur, _ = measure.marching_cubes(
                psr_grid_numpy[i], level=zero_level)
            verts.append(verts_cur)
            faces.append(faces_cur)
            normals.append(normals_cur)
        verts = np.stack(verts, axis=0)
        faces = np.stack(faces, axis=0)
        normals = np.stack(normals, axis=0)
    else:
        try:
            verts, faces, normals, _ = measure.marching_cubes(
                psr_grid_numpy, level=zero_level)
        except Exception:
            verts, faces, normals, _ = measure.marching_cubes(psr_grid_numpy)

    if real_scale:
        verts = verts / (s - 1)
    else:
        verts = verts / s

    if pytorchify:
        device = psr_grid.device
        verts = torch.Tensor(np.ascontiguousarray(verts)).to(device)
        faces = torch.Tensor(np.ascontiguousarray(faces)).to(device)
        normals = torch.Tensor(np.ascontiguousarray(-normals)).to(device)

    return verts, faces, normals


def load_config(path, default_path=None):
    """Load config file with optional default fallback."""
    with open(path, 'r') as f:
        cfg_special = yaml.load(f, Loader=yaml.Loader)

    inherit_from = cfg_special.get('inherit_from')

    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.Loader)
    else:
        cfg = dict()

    update_recursive(cfg, cfg_special)
    return cfg


def update_recursive(dict1, dict2):
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


def export_pointcloud(name, points, normals=None):
    if len(points.shape) > 2:
        points = points[0]
        if normals is not None:
            normals = normals[0]
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
        if normals is not None:
            normals = normals.detach().cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    o3d.io.write_point_cloud(name, pcd)


def export_mesh(name, v, f):
    if len(v.shape) > 2:
        v, f = v[0], f[0]
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu().numpy()
        f = f.detach().cpu().numpy()
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(v)
    mesh.triangles = o3d.utility.Vector3iVector(f)
    o3d.io.write_triangle_mesh(name, mesh)


def scale2onet(p, scale=1.2):
    """Scale point cloud from SAP [0,1) to ONet range."""
    return (p - 0.5) * scale
