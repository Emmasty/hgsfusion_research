import torch
import torch.nn as nn

from .vfe_template import VFETemplate
from .image_vfe_modules import ffn, f2v
from .image_vfe import ImageVFE
from .pillar_vfe import Radar7PillarVFE
from .simple_sampler import SimpleSampler
from .radar_occupancy import RadarOccupancy
from .radar_occupancy_2d_v2 import RadarOccupancy2DV2

radar_occupancy = {
    'RadarOccupancy': RadarOccupancy,
    'RadarOccupancy2DV2': RadarOccupancy2DV2
}

feature_sampler = {
    'SimpleSampler': SimpleSampler,
}

class FusionVFE(nn.Module):
    def __init__(self, 
                 model_cfg, 
                 ImageVFE:ImageVFE, 
                 RadarVFE:Radar7PillarVFE, 
                 point_cloud_range=None,
                 voxel_size=None):
        
        super().__init__()
        self.model_cfg = model_cfg
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size

        self.vfe_module_list = [ImageVFE, RadarVFE]
        self.add_module('ImageVFE', ImageVFE)
        self.add_module('RadarVFE', RadarVFE)

        if self.model_cfg.get('ImageFeatureSampler', None) is not None:
            sampler_name = self.model_cfg.ImageFeatureSampler.NAME
            if sampler_name == 'GaussianSampler':
                try:
                    from .feature_sampler import GaussianSampler
                except ModuleNotFoundError as exc:
                    raise ModuleNotFoundError(
                        'ImageFeatureSampler=GaussianSampler requires the missing '
                        'pcdet.models.backbones_3d.vfe.feature_sampler module.'
                    ) from exc
                sampler_cls = GaussianSampler
            else:
                sampler_cls = feature_sampler[sampler_name]
            self.image_features_sampler = sampler_cls(
                                                    self.model_cfg.ImageFeatureSampler,
                                                    point_cloud_range=point_cloud_range,
                                                    voxel_size=self.voxel_size,
                                                    use_virtual=self.model_cfg.RadarBackbone.READER.get('USE_VIRTUAL_POINT', False)
                )
            self.add_module('ImageFeatureSampler', self.image_features_sampler)
            self.use_image_sampler = True
        else:
            self.use_image_sampler = False

        if self.model_cfg.get('RadarOccupancy', None) is not None:
            occupancy_name = self.model_cfg.RadarOccupancy.NAME
            if occupancy_name == 'RadarOccupancy2D':
                try:
                    from .radar_occupancy_2d import RadarOccupancy2D
                except ModuleNotFoundError as exc:
                    raise ModuleNotFoundError(
                        'RadarOccupancy2D is configured, but the '
                        'pcdet.models.backbones_3d.vfe.radar_occupancy_2d module is missing.'
                    ) from exc
                occupancy_cls = RadarOccupancy2D
            else:
                occupancy_cls = radar_occupancy[occupancy_name]
            self.radar_occupancy = occupancy_cls(
                model_cfg=self.model_cfg.RadarOccupancy,
                point_cloud_range=point_cloud_range,
                voxel_size=voxel_size,
                occupancy_init = self.model_cfg.RadarOccupancy.get('occupancy_init', 0.01),
                radar_backbone=self.model_cfg.RadarOccupancy.get('RADAR_BACKBONE', 'pillarnet'),
                use_mask=self.model_cfg.RadarOccupancy.get('MASK', False)
                )
            self.add_module('RadarOccupancy', self.radar_occupancy)
            self.use_radar_occupancy = True
        else:
            self.use_radar_occupancy = False

        if self.model_cfg.get('ForegroundSampler', None) is not None:
            try:
                from .foreground_sampler import ForegroundSampler
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    'ForegroundSampler is configured, but the '
                    'pcdet.models.backbones_3d.vfe.foreground_sampler module is missing.'
                ) from exc
            self.fore_sampler = ForegroundSampler(
                model_cfg=self.model_cfg.ForegroundSampler,
                point_cloud_range=point_cloud_range,
                voxel_size=voxel_size
            )
            self.use_fore_sampler = True
        else:
            self.use_fore_sampler = False

    def get_output_feature_dim(self):
        output_feature_dim = 0
        for vfe in self.vfe_module_list:
            output_feature_dim += vfe.get_output_feature_dim()
        return output_feature_dim

    def forward(self, batch_dict):
        for vfe in self.vfe_module_list:
            batch_dict = vfe(batch_dict)
        if self.use_image_sampler:
            batch_dict = self.image_features_sampler(batch_dict)
        if self.use_radar_occupancy:
            batch_dict = self.radar_occupancy(batch_dict)
        if self.use_fore_sampler:
            batch_dict = self.fore_sampler(batch_dict)
        return batch_dict

    def get_loss(self):
        loss, tb_dict = self.vfe_module_list[0].ffn.get_loss()
        return loss, tb_dict
    
    def get_occ_loss(self):
        if self.use_radar_occupancy:
            return self.radar_occupancy.get_loss()
        else:
            raise NotImplementedError
