"""
Twitter数据集适配器
用于预处理Twitter数据集以适应多视图GCN模型架构
"""

import torch
import numpy as np
from scipy.sparse import csr_matrix
import torch.nn.functional as F

def preprocess_twitter_views(data_views, adj_matrices, max_dim=16, normalize=True):
    """
    对Twitter数据集视图进行特殊处理以适应模型架构
    
    参数:
        data_views: 原始数据视图列表
        adj_matrices: 邻接矩阵列表
        max_dim: 每个视图的最大特征维度
        normalize: 是否对处理后的特征进行归一化
        
    返回:
        处理后的数据视图和邻接矩阵
    """
    processed_views = []
    processed_adj = []
    
    # 如果没有提供数据视图，从邻接矩阵生成
    if len(data_views) == 0 and len(adj_matrices) > 0:
        # 使用邻接矩阵作为特征
        data_views = [adj for adj in adj_matrices]
    
    # 确保所有视图具有相同且合适的维度
    for i, (view, adj) in enumerate(zip(data_views, adj_matrices)):
        # 检查并处理数据类型
        if not isinstance(view, torch.Tensor):
            view = torch.tensor(view, dtype=torch.float32)
        if not isinstance(adj, torch.Tensor):
            adj = torch.tensor(adj, dtype=torch.float32)
            
        # 使用SVD降维 - 添加异常处理
        try:
            # 对于非常稀疏的矩阵，可能需要先转换
            if torch.sum(torch.abs(view)) < 1e-8:
                # 如果矩阵几乎全为零，创建随机特征
                reduced_view = torch.rand(view.shape[0], max_dim)
            else:
                # 对于正常矩阵执行SVD
                U, S, V = torch.svd(view)
                # 限制维度
                dim = min(max_dim, min(U.shape[1], S.shape[0]))
                reduced_view = torch.mm(U[:, :dim], torch.diag(S[:dim]))
        except Exception as e:
            print(f"SVD处理视图{i}时出错: {e}")
            # 备选方案：使用随机投影或PCA
            reduced_view = torch.rand(view.shape[0], max_dim)
        
        # 处理邻接矩阵 - 归一化并确保自连接
        try:
            # 确保对角线元素为1（自连接）
            adj_with_self_loops = adj.clone()
            adj_with_self_loops.fill_diagonal_(1.0)
            
            # 对称归一化
            if normalize:
                # D^(-1/2) * A * D^(-1/2) 归一化
                row_sum = adj_with_self_loops.sum(dim=1, keepdim=True)
                row_sum[row_sum == 0] = 1.0  # 避免除零
                d_inv_sqrt = torch.pow(row_sum, -0.5)
                norm_adj = adj_with_self_loops * d_inv_sqrt * d_inv_sqrt.transpose(-1, -2)
            else:
                norm_adj = adj_with_self_loops
        except Exception as e:
            print(f"处理视图{i}的邻接矩阵时出错: {e}")
            # 备选方案：使用单位矩阵
            norm_adj = torch.eye(adj.shape[0])
        
        # 对特征进行归一化
        if normalize:
            reduced_view = F.normalize(reduced_view, p=2, dim=1)
        
        processed_views.append(reduced_view)
        processed_adj.append(norm_adj)
    
    return processed_views, processed_adj

def extract_features_from_adjacency(adj_matrices, max_views=3, feature_dim=16):
    """
    从邻接矩阵中提取节点特征
    
    参数:
        adj_matrices: 邻接矩阵列表
        max_views: 最大视图数量
        feature_dim: 每个视图的特征维度
        
    返回:
        处理后的数据视图列表
    """
    features = []
    
    # 选择部分邻接矩阵作为视图
    selected_adj = adj_matrices[:max_views]
    
    for adj in selected_adj:
        if not isinstance(adj, torch.Tensor):
            adj = torch.tensor(adj, dtype=torch.float32)
        
        # 从邻接矩阵中提取特征
        # 1. 度特征
        degree = adj.sum(dim=1, keepdim=True)
        
        # 2. 中心性特征: 使用PageRank近似
        pagerank = torch.ones(adj.shape[0], 1)
        for _ in range(5):  # 简化的PageRank迭代
            pagerank = 0.85 * torch.sparse.mm(adj, pagerank) + 0.15
        
        # 3. 聚类系数近似
        common_neighbors = torch.mm(adj, adj)
        possible_connections = degree * (degree - 1)
        possible_connections[possible_connections == 0] = 1  # 避免除零
        clustering = common_neighbors.diag().unsqueeze(1) / possible_connections
        
        # 4. 随机投影特征
        random_proj = torch.randn(3, feature_dim - 3)
        node_features = torch.cat([degree, pagerank, clustering], dim=1)
        extra_features = torch.mm(node_features, random_proj)
        
        # 组合所有特征
        combined_features = torch.cat([node_features, extra_features], dim=1)
        
        # 归一化
        combined_features = F.normalize(combined_features, p=2, dim=1)
        
        features.append(combined_features)
    
    return features

class TwitterDataAdapter:
    """Twitter数据集适配器类"""
    
    def __init__(self, config):
        """
        初始化Twitter数据集适配器
        
        参数:
            config: 配置对象
        """
        self.max_dim = config.latent_dim // 2 if hasattr(config, 'latent_dim') else 16
        self.normalize = True
        self.use_svd = True
        self.add_self_loops = True
    
    def __call__(self, data_views, adj_matrices):
        """
        调用适配器处理数据
        
        参数:
            data_views: 数据视图列表
            adj_matrices: 邻接矩阵列表
            
        返回:
            处理后的数据视图和邻接矩阵
        """
        # 如果只提供了邻接矩阵，从中提取特征
        if len(data_views) == 0 or data_views[0] is None:
            data_views = extract_features_from_adjacency(adj_matrices, feature_dim=self.max_dim)
        
        return preprocess_twitter_views(
            data_views, 
            adj_matrices, 
            max_dim=self.max_dim,
            normalize=self.normalize
        )
