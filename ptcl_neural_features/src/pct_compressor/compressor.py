import torch.nn as nn
import torch
import torch.nn.functional as F
from pct_compressor.utilities import PointNetSetAbstraction, FeatureEnhencementModule
from pytorch3d.loss import chamfer_distance
from pct_compressor.hyper_encoder import HyperEncoder
from pct_compressor.hyper_decoder import HyperDecoder
from pct_compressor.bitEstimator import BitEstimator
import math
from torchinfo import summary

class Compressor(nn.Module):
    def __init__(self, normal_channel=False, bottleneck_size=256, use_hyperprior=True, recon_points=2048):
        super(Compressor, self).__init__()
        self.bottleneck_size = bottleneck_size
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=3 + 3,
                                          mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3,
                                          mlp=[128, 128, 256], group_all=False)
        self.feature_enhence1 = FeatureEnhencementModule(channels=256)
        self.feature_enhence2 = FeatureEnhencementModule(channels=256)
        self.use_hyper = use_hyperprior
        self.recon_points = recon_points
        if use_hyperprior:
            self.he = HyperEncoder(bottleneck_size)
            self.hd = HyperDecoder(bottleneck_size // 32)
            self.bitEstimator_z = BitEstimator(bottleneck_size // 32)
        else:
            self.bitEstimator = BitEstimator(bottleneck_size)

    def forward(self, xyz, global_step=None):
        B, N, C = xyz.shape

        l0_feature = xyz
        l0_xyz = xyz

        l0_xyz = l0_xyz.permute(0, 2, 1)
        l0_feature = l0_feature.permute(0, 2, 1)
        l1_xyz, l1_feature = self.sa1(l0_xyz, l0_feature)
        l2_xyz, l2_feature = self.sa2(l1_xyz, l1_feature)
        x = self.feature_enhence1(l2_xyz.permute(0, 2, 1), l2_feature)

        x = F.adaptive_max_pool1d(x, 1).view(B, -1)
        if self.training:
            x_noise = torch.nn.init.uniform_(torch.zeros_like(x), -0.5, 0.5)
            compressed_x = x + x_noise
        else:
            compressed_x = torch.round(x)
        return compressed_x


model = Compressor()
summary(model,(1,2048,3))