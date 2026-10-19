"""SAP model: Encode2Points for point cloud to oriented points prediction."""

import time
import torch
import torch.nn as nn
from sap.network import encoder_dict, decoder_dict
from sap.network.utils import map2local


class Encode2Points(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        encoder = cfg['model']['encoder']
        decoder = cfg['model']['decoder']
        dim = cfg['data']['dim']
        c_dim = cfg['model']['c_dim']
        encoder_kwargs = cfg['model']['encoder_kwargs']
        if encoder_kwargs is None:
            encoder_kwargs = {}
        decoder_kwargs = cfg['model']['decoder_kwargs']
        padding = cfg['data']['padding']
        self.predict_normal = cfg['model']['predict_normal']
        self.predict_offset = cfg['model']['predict_offset']

        out_dim = 3
        out_dim_offset = 3
        num_offset = cfg['data']['num_offset']
        if num_offset > 1:
            out_dim_offset = out_dim * num_offset
        self.num_offset = num_offset

        self.map2local = None
        local_mapping = None
        if cfg['model']['local_coord']:
            if 'unet' in encoder_kwargs.keys():
                unit_size = 1 / encoder_kwargs['plane_resolution']
            else:
                unit_size = 1 / encoder_kwargs['grid_resolution']
            local_mapping = map2local(unit_size)

        self.encoder = encoder_dict[encoder](
            dim=dim, c_dim=c_dim, map2local=local_mapping,
            **encoder_kwargs
        )

        if self.predict_normal:
            self.decoder_normal = decoder_dict[decoder](
                dim=dim, c_dim=c_dim, out_dim=out_dim,
                **decoder_kwargs)
        if self.predict_offset:
            self.decoder_offset = decoder_dict[decoder](
                dim=dim, c_dim=c_dim, out_dim=out_dim_offset,
                map2local=local_mapping,
                **decoder_kwargs)
            self.s_off = cfg['model']['s_offset']

    def forward(self, p):
        """
        Args:
            p (tensor): input unoriented points, shape (batch, N, 3)
        Returns:
            points (tensor): predicted point positions
            normals (tensor): predicted normals
        """
        batch_size = p.size(0)
        points = p.clone()

        c = self.encoder(p)

        if self.predict_offset:
            offset = self.decoder_offset(p, c)
            if self.num_offset > 1:
                points = points.repeat(1, 1, self.num_offset).reshape(batch_size, -1, 3)
            points = points + self.s_off * offset
        else:
            points = p

        if self.predict_normal:
            normals = self.decoder_normal(points, c)

        points = torch.clamp(points, 0.0, 0.99)
        if self.cfg['model']['normal_normalize']:
            normals = normals / (normals.norm(dim=-1, keepdim=True) + 1e-8)

        return points, normals
