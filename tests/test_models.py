"""模型单元测试"""

import unittest
import torch
import numpy as np
from config import Config
from models import MVGCNModel

class TestModels(unittest.TestCase):
    """模型相关的单元测试"""    
    def setUp(self):
        """测试前准备"""        
        # 创建测试配置
        self.config = Config()
        self.config.latent_dim = 64
        self.config.hidden_dims = [128, 64]
        self.config.n_clusters = 5
        self.config.device = "cpu"
        
        # 创建测试数据
        self.batch_size = 32
        self.feature_dim = 100
        self.n_views = 3
        
        # 创建随机视图数据
        self.data_views = [
            torch.rand(self.batch_size, self.feature_dim) 
            for _ in range(self.n_views)
        ]
        
        # 创建随机邻接矩阵
        self.adj_matrices = [
            torch.rand(self.batch_size, self.batch_size) 
            for _ in range(self.n_views)
        ]
        
    def test_model_creation(self):
        """测试模型创建"""        
        model = MVGCNModel(self.config)
        self.assertIsNotNone(model)
        
    def test_forward_pass(self):
        """测试模型前向传播"""        
        model = MVGCNModel(self.config)
        
        # 前向传播
        outputs = model(self.data_views, self.adj_matrices)
        
        # 检查输出是否正确
        self.assertIn("fused_embedding", outputs)
        self.assertIn("cluster_rep", outputs)
        
        # 检查形状
        self.assertEqual(outputs["fused_embedding"].shape, 
                      (self.batch_size, self.config.latent_dim))
        
    def test_get_cluster_features(self):
        """测试获取聚类特征"""        
        model = MVGCNModel(self.config)
        
        # 获取聚类特征
        cluster_features = model.get_cluster_features(self.data_views, self.adj_matrices)
        
        # 检查形状
        self.assertEqual(cluster_features.shape, 
                      (self.batch_size, self.config.latent_dim))

if __name__ == "__main__":
    unittest.main()
