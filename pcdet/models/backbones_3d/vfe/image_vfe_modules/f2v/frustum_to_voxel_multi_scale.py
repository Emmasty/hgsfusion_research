import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .frustum_grid_generator import FrustumGridGenerator
from .sampler import Sampler


class FrustumToVoxelMultiScale(nn.Module):

    def __init__(self, model_cfg, grid_size, pc_range, disc_cfg, 
                 use_depth=True, feature_num=None, fuse_mode=None,
                 in_channels=None, out_channels=None, **kwargs):
        """
        Initializes module to transform frustum features to voxel features via 3D transformation and sampling
        Args:
            model_cfg: EasyDict, Module configuration
            grid_size: [X, Y, Z], Voxel grid size
            pc_range: [x_min, y_min, z_min, x_max, y_max, z_max], Voxelization point cloud range (m)
            disc_cfg: EasyDict, Depth discretiziation configuration
        """
        assert feature_num is not None
        super().__init__()
        self.model_cfg = model_cfg
        self.grid_size = grid_size
        self.pc_range = pc_range
        self.disc_cfg = disc_cfg
        self.bev_augment = self.model_cfg.get('BEV_AUG', False)
        self.grid_generator = FrustumGridGenerator(grid_size=grid_size,
                                                   pc_range=pc_range,
                                                   disc_cfg=disc_cfg,
                                                   bev_aug=self.bev_augment)
        self.sampler = Sampler(**model_cfg.SAMPLER)
        self.use_depth = use_depth
        self.feature_num = feature_num
        self.fuse_mode = fuse_mode
        self.in_channels = in_channels * feature_num
        self.out_channels = out_channels
        self.use_checkpoint = self.model_cfg.get('USE_CHECKPOINT', False)
        if self.fuse_mode == 'CONCAT':
            assert self.in_channels is not None
            assert self.out_channels is not None
            self.channel_reduce = nn.Sequential(
                nn.Conv3d(self.in_channels, self.out_channels, (1, 1, 1)),
                nn.BatchNorm3d(num_features=self.out_channels),
                nn.ReLU(inplace=True)
            )

    def sample_and_project_feature(self, grid, feature, weight, bias):
        if not self.use_depth:
            feature = feature.unsqueeze(2)
        sampled = self.sampler(input_features=feature, grid=grid).permute(0, 1, 4, 3, 2)
        return F.conv3d(sampled, weight, bias=bias)

    def sample_and_project_concat(self, grid, *features):
        """Checkpoint each scale so backward rebuilds one voxel volume at a time."""
        conv = self.channel_reduce[0]
        reduced = None
        channel_offset = 0
        use_checkpoint = self.use_checkpoint and self.training and torch.is_grad_enabled()

        for feature in features:
            channel_count = feature.shape[1]
            weight = conv.weight[:, channel_offset:channel_offset + channel_count]
            bias = conv.bias if reduced is None else None
            if use_checkpoint:
                part = checkpoint(
                    self.sample_and_project_feature,
                    grid,
                    feature,
                    weight,
                    bias,
                    preserve_rng_state=False,
                )
            else:
                part = self.sample_and_project_feature(grid, feature, weight, bias)

            if reduced is None:
                reduced = part
            else:
                reduced.add_(part)
                del part
            channel_offset += channel_count

        if channel_offset != conv.in_channels:
            raise RuntimeError(
                f'Expected {conv.in_channels} concatenated channels, got {channel_offset}'
            )
        return reduced

    def finish_concat_features(self, reduced):
        # BatchNorm must stay outside checkpointing: recomputation during
        # backward must not update its running statistics a second time.
        norm, activation = self.channel_reduce[1], self.channel_reduce[2]
        return activation(norm(reduced))

    def forward(self, batch_dict):
        """
        Generates voxel features via 3D transformation and sampling
        Args:
            batch_dict:
                frustum_features: (B, C, D, H_image, W_image), Image frustum features
                lidar_to_cam: (B, 4, 4), LiDAR to camera frame transformation
                cam_to_img: (B, 3, 4), Camera projection matrix
                image_shape: (B, 2), Image shape [H, W]
        Returns:
            batch_dict:
                voxel_features: (B, C, Z, Y, X), Image voxel features
        """
        # Generate sampling grid for frustum volume
        grid = self.grid_generator(lidar_to_cam=batch_dict["trans_lidar_to_cam"],
                                   cam_to_img=batch_dict["trans_cam_to_img"], # Lidar-X*Y*Z  [pixel-uv, depth] [W, H, D]
                                   image_shape=batch_dict["image_shape"],
                                   bda=batch_dict['lidar_aug_matrix'])  # (B, X_D, Y_W, Z_H, 3) [2, 160, 160, 16, 3]

        # # DEBUG code
        # print("debug code!")
        # batch_dict["features"][0][0, 0, :, :] = 0
        # batch_dict["features"][0][0, 0, 50:70, 220:260] = 1
        # batch_dict["features"][1][0, 0, :, :] = 0
        # batch_dict["features"][1][0, 0, 25:35, 110:130] = 1
        # batch_dict["features"][2][0, 0, :, :] = 0
        # batch_dict["features"][2][0, 0, 25:35, 110:130] = 1
        # batch_dict["features"][3][0, 0, :, :] = 0
        # batch_dict["features"][3][0, 0, 25:35, 110:130] = 1

        source_key = "frustum_features" if self.use_depth else "features"
        source_features = batch_dict[source_key]

        if self.fuse_mode == 'CONCAT':
            reduced = self.sample_and_project_concat(grid, *source_features)
            voxel_features = self.finish_concat_features(reduced)
        else:
            voxel_features = []
            for feature in source_features:
                if not self.use_depth:
                    feature = feature.unsqueeze(2)
                voxel_features.append(
                    self.sampler(input_features=feature, grid=grid).permute(0, 1, 4, 3, 2)
                )

        if self.use_depth:
            batch_dict.pop('frustum_features')

        if self.fuse_mode == 'ADD':
            voxel_features = torch.stack(voxel_features) # [4,1,64,31,320,320]
            voxel_features = torch.sum(voxel_features, dim=0) # [1,64,31,320,320]
        elif self.fuse_mode != 'CONCAT':
            raise NotImplementedError
        
        # print("debug")
        # from matplotlib import pyplot as plt
        # for idx, feat in enumerate(voxel_features):
        #     heatmap = feat[0][0].sum(dim=0)
        #     plt.imshow(heatmap.cpu().detach().numpy())
        #     plt.savefig(f'{idx}.png')


        batch_dict["voxel_features"] = voxel_features
        return batch_dict
