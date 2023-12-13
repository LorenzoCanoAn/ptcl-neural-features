import torch
import torch.nn as nn
from torchinfo import summary
from pct_compressor.compressor import Compressor


class DistanceEstimatorCat(nn.Module):
    def __init__(self, feature_size=256):
        super(self.__class__, self).__init__()
        self.compressor = Compressor(bottleneck_size=feature_size)
        self.dist_measuring = nn.Sequential(
            nn.Linear(int(feature_size*2), int(feature_size)),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(int(feature_size), 2),
        )

    def forward(self, x, y):
        z = self.dist_measuring(torch.cat((x,y),-1))
        return z

