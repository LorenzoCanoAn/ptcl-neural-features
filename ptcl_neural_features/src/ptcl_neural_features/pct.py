import torch
import torch.nn as nn
from torchinfo import summary
from time import time


class InputEmbedding(nn.Module):
    def __init__(self, input_size, output_size):
        super(type(self), self).__init__()
        self.conv = nn.Conv1d(input_size, output_size, 1)
        self.bn = nn.BatchNorm1d(output_size)

    def forward(self, x):
        """
        x.shape = [B,C,N]
            B: batch size
            C: Length of points, for example, if point is only xyz coords, C=3
            N: Number of points in pointcloud
        """
        return self.bn(self.conv(x))


class TransformerMLP(nn.Module):
    def __init__(self, embedding_size, dropout=0.1):
        super(type(self), self).__init__()
        self.conv1 = nn.Conv1d(embedding_size, embedding_size, 1)
        self.conv2 = nn.Conv1d(embedding_size, embedding_size, 1)
        self.bn1 = nn.BatchNorm1d(embedding_size)
        self.bn2 = nn.BatchNorm1d(embedding_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        x.shape = [B,C,N]
            B: batch size
            C: Length of points, for example, if point is only xyz coords, C=3
            N: Number of points in pointcloud
        """
        x = self.dropout(self.relu(self.bn1(self.conv1(x))))
        x = self.dropout(self.relu(self.bn2(self.conv2(x))))
        return x


class TransformerLayer(nn.Module):
    def __init__(self, embedding_size, n_heads, attention_dropout=0.1, mlp_dropout=0.1):
        super(type(self), self).__init__()
        self.multiheadattention = nn.MultiheadAttention(
            embedding_size, n_heads, attention_dropout
        )
        self.mlp = TransformerMLP(embedding_size, dropout=mlp_dropout)

    def forward(self, x):
        x_, _ = self.multiheadattention(x, x, x, need_weights=False)
        x_ = torch.permute(x_, (0, 2, 1))
        x_ = self.mlp(x_)
        x_ = torch.permute(x_, (0, 2, 1))
        return x + x_


class PCT(nn.Module):
    def __init__(
        self,
        size_of_pcl,
        point_size,
        embedding_size,
        n_heads,
        n_layers,
        hidden_dim=1024,
        output_size=128,
    ):
        super(type(self), self).__init__()
        self.input_embedding = InputEmbedding(point_size, embedding_size)
        self.transformerlayers = nn.Sequential()
        for i in range(n_layers):
            self.transformerlayers.add_module(
                f"layer{i}", TransformerLayer(embedding_size, n_heads)
            )
        self.conv1 = nn.Conv1d(embedding_size, 1, 1)
        self.fc1 = nn.Linear(size_of_pcl, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        """Expected size of x is [B,N,P]:
        B: batch size
        N: size of pcl
        P: size of point
        """
        x = torch.permute(x, (0, 2, 1))
        x = self.input_embedding(x)
        x = torch.permute(x, (0, 2, 1))
        x = self.transformerlayers(x)
        x = torch.permute(x, (0, 2, 1))
        x = self.conv1(x)
        x = torch.permute(x, (0, 2, 1))
        x = torch.squeeze(x, dim=-1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x
    
