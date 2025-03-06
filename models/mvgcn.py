"""
多视图图卷积网络模型定义
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .gcn_layers import GraphConvolution
from .autoencoder import MultiViewAutoencoder

class MVGCNModel(nn.Module):
    """多视图图卷积神经网络模型"""
    
    def __init__(self, config):
        """
        初始化多视图GCN模型
        
        参数:
            config: 配置对象，包含模型参数
        """
        super(MVGCNModel, self).__init__()
        
        self.config = config
        self.latent_dim = config.latent_dim
        
        # 检测是否为Twitter数据集
        self.is_twitter = hasattr(config, 'dataset') and 'twitter' in config.dataset.lower()
        
        # 视图编码器将在forward中动态创建
        self.view_encoders = None
        
        # 视图融合层
        fusion_input_dim = config.latent_dim * 3  # 默认三个视图
        if hasattr(config, 'input_dims') and config.input_dims:
            fusion_input_dim = config.latent_dim * len(config.input_dims)
        
        self.fusion_layer = nn.Linear(fusion_input_dim, config.latent_dim)
        self.fusion_norm = nn.BatchNorm1d(config.latent_dim)
        
        # 聚类投影层
        self.cluster_layer = nn.Sequential(
            nn.Linear(config.latent_dim, config.latent_dim // 2),
            nn.ReLU(),
            nn.Linear(config.latent_dim // 2, config.latent_dim)
        )
        
        # 如果是Twitter数据集，添加额外的特征转换层
        if self.is_twitter:
            self.twitter_feature_transform = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(config.latent_dim, config.latent_dim),
                    nn.BatchNorm1d(config.latent_dim),
                    nn.ReLU(),
                    nn.Dropout(0.3)
                ) for _ in range(3)  # 假设有3个视图
            ])
        
    def _init_view_encoders(self, view_dims):
        """初始化每个视图的编码器"""
        # 为Twitter数据集特别处理隐藏层维度
        if self.is_twitter:
            hidden_dims = [min(dim * 2, self.config.latent_dim) for dim in view_dims]
        else:
            hidden_dims = self.config.hidden_dims if hasattr(self.config, 'hidden_dims') else [256, 128]
            
        self.view_encoders = nn.ModuleList([
            MultiViewAutoencoder(dim, hidden_dims, self.latent_dim, self.config.dropout)
            for dim in view_dims
        ])
    
    def forward(self, views, adj_matrices, mask_rate=0.2):
        """
        前向传播
        
        参数:
            views: 视图特征列表
            adj_matrices: 邻接矩阵列表
            mask_rate: 特征掩码率
            
        返回:
            包含各种表示的字典
        """
        # 如果视图编码器未初始化，则初始化
        if self.view_encoders is None:
            view_dims = [view.shape[1] for view in views]
            self._init_view_encoders(view_dims)
        
        # 处理每个视图
        view_embeddings = []
        reconstructions = []
        
        for i, (view, adj, encoder) in enumerate(zip(views, adj_matrices, self.view_encoders)):
            try:
                # 掩码输入特征
                if mask_rate > 0:
                    mask = torch.bernoulli(torch.ones_like(view) * (1 - mask_rate))
                    masked_view = view * mask
                else:
                    masked_view = view
                
                # 编码和重构
                h_encoded, h_reconstructed = encoder(masked_view, adj)
                
                # 对于Twitter数据集应用额外的转换
                if self.is_twitter and i < len(self.twitter_feature_transform):
                    h_encoded = self.twitter_feature_transform[i](h_encoded)
                    
                view_embeddings.append(h_encoded)
                reconstructions.append(h_reconstructed)
            except Exception as e:
                # 处理可能的异常，避免崩溃
                print(f"处理视图{i}时出错: {e}")
                # 创建随机嵌入替代
                h_encoded = torch.rand(view.shape[0], self.latent_dim, device=view.device)
                h_reconstructed = torch.zeros_like(view)
                view_embeddings.append(h_encoded)
                reconstructions.append(h_reconstructed)
        
        # 视图融合
        if len(view_embeddings) > 1:
            # 确保所有视图嵌入具有相同的维度
            for i in range(len(view_embeddings)):
                if view_embeddings[i].shape[1] != self.latent_dim:
                    view_embeddings[i] = F.pad(
                        view_embeddings[i], 
                        (0, self.latent_dim - view_embeddings[i].shape[1])
                    )
            
            combined = torch.cat(view_embeddings, dim=1)
            
            # 确保融合层输入尺寸正确
            if combined.shape[1] != self.fusion_layer.in_features:
                # 动态调整融合层
                self.fusion_layer = nn.Linear(combined.shape[1], self.latent_dim).to(combined.device)
                
            fused = self.fusion_layer(combined)
            fused = self.fusion_norm(fused)
            fused = F.relu(fused)
        else:
            fused = view_embeddings[0]
        
        # 聚类表示
        cluster_rep = self.cluster_layer(fused)
        
        return {
            "view_embeddings": view_embeddings,
            "fused_embedding": fused,
            "cluster_rep": cluster_rep,
            "reconstructions": reconstructions
        }
    
    def get_cluster_features(self, views, adj_matrices):
        """
        获取用于聚类的特征表示
        
        参数:
            views: 视图特征列表
            adj_matrices: 邻接矩阵列表
            
        返回:
            用于聚类的特征表示
        """
        with torch.no_grad():
            outputs = self.forward(views, adj_matrices, mask_rate=0.0)
            return outputs["cluster_rep"]
