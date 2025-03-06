"""数据集测试模块"""

import unittest
import os
import numpy as np
import torch
from data.dataset_factory import DatasetFactory

class TestDatasets(unittest.TestCase):
    """数据集单元测试"""    
    def test_example_dataset(self):
        """测试示例数据集"""        
        data_views, adj_matrices, labels = DatasetFactory.create_dataset("example_dataset")
        
        # 检查数据集是否加载成功
        self.assertIsNotNone(data_views)
        self.assertIsNotNone(adj_matrices)
        self.assertIsNotNone(labels)
        
        # 检查数据类型
        if isinstance(data_views[0], torch.Tensor):
            data_views = [v.numpy() for v in data_views]
            
        # 检查视图数量
        self.assertGreater(len(data_views), 0)
        
        # 检查样本数量一致
        n_samples = data_views[0].shape[0]
        self.assertEqual(len(labels), n_samples)
        
        # 检查邻接矩阵尺寸
        if isinstance(adj_matrices[0], torch.Tensor):
            adj_matrices = [a.numpy() for a in adj_matrices]
            
        self.assertEqual(adj_matrices[0].shape, (n_samples, n_samples))
        
    def test_twitter_football(self):
        """测试Twitter足球数据集"""        
        # 跳过测试如果数据集不存在
        if not os.path.exists("data/twitter_football.mat"):
            self.skipTest("Twitter足球数据集不存在，跳过测试")
            
        data_views, adj_matrices, labels = DatasetFactory.create_dataset(
            "twitter_football", download_if_missing=False)
        
        # 检查数据集是否加载成功
        self.assertIsNotNone(data_views)
        
        # 检查视图数量
        self.assertGreaterEqual(len(data_views), 5)

if __name__ == "__main__":
    unittest.main()
