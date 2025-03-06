"""
多视图GCN模型包
提供各种模型组件和架构
"""

# 导入主模型类并导出
from .mvgcn import MVGCNModel

# 导出其他可能需要的模型组件
from .autoencoder import MultiViewAutoencoder
from .contrastive import ContrastiveLoss
from .gcn_layers import GraphConvolution

# 版本信息
__version__ = '0.1.0'
