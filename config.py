# Description: 配置文件
import torch

# 配置
class Config:
    def __init__(self):
        # 数据集配置
        self.dataset = 'example_dataset'  # 更改为数据集名称
        
        # 模型参数
        self.hidden_dims = [512, 256]     # 自编码器隐藏层维度
        self.latent_dim = 128            # 潜在空间维度
        self.projection_dim = 64         # 投影空间维度
        self.gcn_hidden = 256            # GCN隐藏层维度
        self.dropout = 0.5               # Dropout率
        
        # 掩码参数
        self.mask_rate = 0.2             # 特征掩码率
        self.adaptive_mask = True        # 是否使用自适应掩码
        
        # 训练参数
        self.lr = 0.001                  # 学习率
        self.weight_decay = 5e-4         # 权重衰减
        self.epochs = 200                # 训练轮次
        self.batch_size = 256            # 批量大小(单视图数据)
        self.eval_interval = 5           # 评估间隔
        
        # 对比学习参数
        self.temperature = 0.5           # 温度参数
        self.contrastive_weight = 1.0    # 对比损失权重
        
        # 聚类参数
        self.n_clusters = 5              # 聚类数(仅当标签不可用时使用)
        
        # 设备配置
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 随机种子
        self.seed = 42
        
        # 可视化
        self.visualize = True
        
        # 更新有效数据集列表，添加新支持的数据集
        self.valid_datasets = [
        'example_dataset', 'handwritten', 'reuters', 'coil20', 'mnist',
        '3sources', 'segment', 'movielists',
        # 添加Twitter数据集，包括新增的三个
        'twitter_football', 'twitter_olympics', 'twitter_politics-uk', 
        'twitter_politics-ie', 'twitter_rugby'
        ]
        
        # 验证数据集
        if self.dataset not in self.valid_datasets:
            print(f"警告: 数据集 '{self.dataset}' 未识别。可用的数据集有: {', '.join(self.valid_datasets)}")
            print("将使用默认数据集: example_dataset")
            self.dataset = 'example_dataset'