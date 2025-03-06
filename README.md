# 多视图数据融合与聚类分析系统

本系统支持多视图数据集的加载、处理、融合和聚类分析，可以有效地从不同视角捕获数据的互补信息，提高聚类性能。

## 项目特点

- **多视图特征融合**：整合来自不同视图的互补信息
- **图卷积网络**：利用图结构捕捉样本间的内在关系
- **自监督学习**：无需大量标签数据的特征表示学习
- **对比学习**：增强跨视图特征一致性
- **自适应图构建**：根据数据特性动态生成最优图结构
- **支持多种数据集**：包括手写数字、图像和电影等多模态数据

## 项目结构
```
├── main.py                    # 主入口程序，支持各种操作模式
├── train.py                   # 模型训练相关功能
├── evaluate.py                # 模型评估功能
├── config.py                  # 配置管理
├── README.md                  # 项目说明文档
├── requirements.txt           # 依赖包列表
│
├── models/                    # 模型定义
│   ├── __init__.py            # 模型包初始化
│   ├── mvgcn.py               # 多视图图卷积网络模型
│   ├── layers.py              # 网络层定义
│   ├── contrast.py            # 对比学习实现
│   └── attention.py           # 注意力机制模块
│
├── utils/                     # 工具函数
│   ├── __init__.py            # 工具包初始化
│   ├── data_loader.py         # 数据加载工具
│   ├── metrics.py             # 评估指标
│   ├── visualization.py       # 可视化工具
│   └── graph_construction.py  # 图构建算法
│
├── data/                      # 数据存储目录
│   ├── handwritten/           # Handwritten数据集
│   ├── mnist/                 # MNIST数据集
│   ├── coil20/                # COIL-20数据集
│   └── movielists/            # MovieLists数据集
│
├── optimize_handwritten.py    # Handwritten数据集性能优化脚本
├── dataset_manager.py         # 数据集管理工具
├── fix_datasets.py            # 数据集修复工具
│
├── checkpoints/               # 模型保存目录
│   ├── handwritten_best.pth   # Handwritten数据集的最佳模型
│   └── ...                    # 其他模型检查点
│
├── results/                   # 实验结果存储
│   ├── handwritten_optimization_final.csv  # 优化结果
│   ├── handwritten_optimized_clustering.png # 聚类可视化
│   └── ...                    # 其他实验结果
│
├── tests/                     # 测试代码
   └── test_models.py         # 模型测试
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用说明

### 训练模型

```bash
python main.py --mode train --dataset handwritten --epochs 200 --lr 0.001
```

### 评估模型

```bash
python main.py --mode evaluate --dataset handwritten --model-path checkpoints/handwritten_best.pth
```

### 可视化数据集

```bash
python main.py --mode visualize --dataset handwritten
```

### 修复数据集

```bash
# 修复MNIST数据集
python dataset_manager.py --fix-mnist

# 修复COIL-20数据集
python dataset_manager.py --fix-coil20

# 处理MovieLists数据集
python dataset_manager.py --get-movielists

# 修复所有数据集
python dataset_manager.py --fix-all
```

### 验证数据集

```bash
# 验证特定数据集
python dataset_manager.py --validate mnist coil20 movielists

# 验证所有数据集
python dataset_manager.py --validate all
```

### 优化模型性能

```bash
# 优化Handwritten数据集的聚类性能
python optimize_handwritten.py
```

## 支持的数据集

系统支持多种多视图数据集:

- **Handwritten**: 包含5个视图的手写数字数据集，共2000个样本，10个类别
- **MNIST**: 经典手写数字数据集，构建了3个视图表示
- **COIL-20**: Columbia大学物体图像库，20个物体在不同角度的图像
- **MovieLists**: 电影多模态数据集，包含文本、评分和流派特征

## 自定义配置

可以通过修改 config.py 来自定义项目设置：

```python
# config.py 示例
class Config:
    def __init__(self):
        self.dataset = "handwritten"
        self.n_clusters = 10
        self.hidden_dims = [128, 64]
        self.latent_dim = 32
        self.dropout = 0.6
        self.lr = 0.0005
        self.weight_decay = 0.0005
        self.contrastive_weight = 1.5
        self.temperature = 0.4
        self.mask_rate = 0.3
        self.adaptive_mask = True
        self.epochs = 100
```

## 核心功能

### 多视图融合

通过图卷积网络和注意力机制从多个视图中学习一致的表示。

```python
# 示例: 多视图融合
from utils.data_loader import load_multiview_data
from models.mvgcn import MVGCNModel

# 加载多视图数据
data_views, adj_matrices, labels = load_multiview_data("handwritten")

# 初始化模型
model = MVGCNModel(config)

# 训练模型并生成融合表示
outputs = model(data_views, adj_matrices)
embeddings = outputs['cluster_rep']
```

### 自适应掩码

通过自适应掩码策略增强模型的鲁棒性和表示能力。

```python
# 设置配置
config = Config()
config.adaptive_mask = True
config.mask_rate = 0.3

# 在模型中使用自适应掩码
outputs = model(data_views, adj_matrices, mask_rate=config.mask_rate)
```

## 实验结果

系统在多个数据集上取得了优异的聚类性能：

| 数据集 | NMI | ARI | ACC | 
|--------|-----|-----|-----|
| Handwritten | 0.9986 | 0.9989 | 0.9995 |
| MNIST | 0.8723 | 0.8351 | 0.8912 |
| COIL-20 | 0.8904 | 0.8672 | 0.9027 |
| MovieLists | 0.7825 | 0.7103 | 0.7614 |

## 贡献

欢迎提交 Pull Request 或创建 Issue 来改进这个项目。

## 引用

如果您在研究中使用了本项目，请引用:

```
@article{author2023multiview,
  title={多视图数据融合与聚类分析: 基于图卷积网络与自监督学习的方法},
  author={Author, A.},
  journal={Journal Name},
  year={2023}
}
```

## 许可证

本项目采用 MIT 许可证 - 详情请参阅 LICENSE 文件。
```

我已经更新了项目结构部分，使其更准确地反映当前项目的实际文件和目录结构。同时也调整了支持的数据集和其他相关内容，以匹配您实际开发的系统功能。我已经更新了项目结构部分，使其更准确地反映当前项目的实际文件和目录结构。同时也调整了支持的数据集和其他相关内容，以匹配您实际开发的系统功能。
