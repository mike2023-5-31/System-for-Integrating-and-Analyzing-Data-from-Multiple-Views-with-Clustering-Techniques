# Description: 自编码器模型
import torch
import torch.nn as nn
import torch.nn.functional as F
from .gcn_layers import GraphConvolution

class ViewSpecificAutoencoder(nn.Module):
    """
    特定视图的自编码器
    """
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(ViewSpecificAutoencoder, self).__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

class MaskedAutoencoder(nn.Module):
    """
    带掩码的自编码器，包含特征重建和掩码预测
    """
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super(MaskedAutoencoder, self).__init__()
        
        # 编码器 (多层)
        encoder_layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            encoder_layers.append(nn.Linear(dims[i], dims[i+1]))
            encoder_layers.append(nn.ReLU())
        
        encoder_layers.append(nn.Linear(dims[-1], latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # 解码器 (多层)
        decoder_layers = []
        dims = [latent_dim] + hidden_dims[::-1] + [input_dim]
        for i in range(len(dims) - 1):
            decoder_layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                decoder_layers.append(nn.ReLU())
            
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, x, mask=None):
        # 如果提供了掩码，应用掩码
        if mask is not None:
            x_masked = x * mask
        else:
            x_masked = x
            
        # 编码
        z = self.encoder(x_masked)
        
        # 解码/重建
        x_recon = self.decoder(z)
        
        return x_recon, z
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)

class MultiViewAutoencoder(nn.Module):
    """多视图自编码器模型"""
    
    def __init__(self, input_dim, hidden_dims, latent_dim, dropout=0.5):
        """
        初始化多视图自编码器
        
        参数:
            input_dim: 输入特征维度
            hidden_dims: 隐藏层维度列表
            latent_dim: 潜在空间维度
            dropout: Dropout概率
        """
        super(MultiViewAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.dropout = dropout
        
        # 构建编码器
        encoder_layers = []
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            encoder_layers.append(GraphConvolution(dims[i], dims[i+1]))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(dropout))
        
        # 最后一层到潜在空间
        encoder_layers.append(GraphConvolution(dims[-1], latent_dim))
        
        self.encoder = nn.ModuleList(encoder_layers)
        
        # 构建解码器 (反向结构)
        decoder_layers = []
        dims = [latent_dim] + hidden_dims[::-1] + [input_dim]
        
        for i in range(len(dims) - 1):
            decoder_layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:  # 除了最后一层
                decoder_layers.append(nn.ReLU())
                decoder_layers.append(nn.Dropout(dropout))
                
        self.decoder = nn.ModuleList(decoder_layers)
        
    def encode(self, x, adj):
        """编码过程"""
        h = x
        for i, layer in enumerate(self.encoder):
            if isinstance(layer, GraphConvolution):
                h = layer(h, adj)
            else:
                h = layer(h)
        return h
    
    def decode(self, z):
        """解码过程"""
        h = z
        for i, layer in enumerate(self.decoder):
            h = layer(h)
            if i < len(self.decoder) - 1 and isinstance(self.decoder[i+1], nn.ReLU):
                h = F.relu(h)
        return h
    
    def forward(self, x, adj):
        """前向传播"""
        z = self.encode(x, adj)
        x_recon = self.decode(z)
        return z, x_recon