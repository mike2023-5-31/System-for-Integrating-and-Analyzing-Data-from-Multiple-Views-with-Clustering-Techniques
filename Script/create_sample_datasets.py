"""
创建样本数据集 - 生成样本数据集用于测试和开发
"""

import os
import numpy as np
import scipy.io as sio
import torch
from sklearn.datasets import make_blobs, make_moons, make_circles
import matplotlib.pyplot as plt
from utils.data_loader import enhanced_adaptive_graph_construction

def create_sample_dataset(dataset_name, n_samples=500, n_clusters=5, n_features=None, data_dir='data'):
    """
    创建并保存样本数据集
    
    参数:
        dataset_name: 数据集名称
        n_samples: 样本数量
        n_clusters: 聚类数量
        n_features: 特征维度列表，如果为None则自动生成
        data_dir: 保存目录
    """
    print(f"创建样本数据集: {dataset_name}...")
    os.makedirs(data_dir, exist_ok=True)
    
    if n_features is None:
        n_features = [20, 30, 25]  # 默认特征维度
    
    # 为每个视图生成数据
    views = []
    centers = []
    
    # 生成聚类中心
    for dim in n_features:
        # 在高维空间生成分离良好的聚类中心
        center = np.random.randn(n_clusters, dim) * 10
        centers.append(center)
    
    # 生成标签 (确保类别平衡)
    n_per_class = n_samples // n_clusters
    remainder = n_samples % n_clusters
    
    labels = []
    for i in range(n_clusters):
        count = n_per_class + (1 if i < remainder else 0)
        labels.extend([i] * count)
    
    labels = np.array(labels)
    
    # 为每个视图生成数据
    for view_idx, (center, dim) in enumerate(zip(centers, n_features)):
        # 基于聚类中心生成数据
        view_data = np.zeros((n_samples, dim))
        
        for i in range(n_clusters):
            # 获取当前类别的索引
            class_indices = np.where(labels == i)[0]
            # 为每个类别生成数据点
            class_data = np.random.randn(len(class_indices), dim) * 1.5 + center[i]
            # 填充到对应位置
            view_data[class_indices] = class_data
        
        views.append(view_data)
    
    # 创建MAT文件格式数据
    data_dict = {}
    for i, view in enumerate(views):
        data_dict[f'view{i+1}'] = view
    
    data_dict['Y'] = labels.reshape(-1, 1)
    
    # 保存MAT文件
    mat_file = os.path.join(data_dir, f'{dataset_name}.mat')
    sio.savemat(mat_file, data_dict)
    
    # 可视化第一个视图的二维投影
    if n_features[0] >= 2:
        plt.figure(figsize=(10, 8))
        for i in range(n_clusters):
            plt.scatter(views[0][labels == i, 0], views[0][labels == i, 1], 
                       label=f'Cluster {i}', alpha=0.7)
        plt.title(f'{dataset_name} 样本数据 - 视图1 (二维投影)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(data_dir, f'{dataset_name}_visualization.png'))
    
    print(f"样本数据集创建完成: {mat_file}")
    print(f"数据集信息:")
    for i, view in enumerate(views):
        print(f"  视图 {i+1}: {view.shape}")
    print(f"  标签: {len(np.unique(labels))} 类，{n_samples} 样本")
    
    return views, labels

def create_sample_handwritten_dataset(data_dir='data', n_samples=2000):
    """创建手写数字样本数据集"""
    return create_sample_dataset(
        'handwritten', 
        n_samples=n_samples, 
        n_clusters=10, 
        n_features=[76, 64, 240, 47, 6], 
        data_dir=data_dir
    )

def create_sample_reuters_dataset(data_dir='data', n_samples=800):
    """创建路透社多语言文档样本数据集"""
    return create_sample_dataset(
        'reuters', 
        n_samples=n_samples, 
        n_clusters=6, 
        n_features=[500, 500, 500, 500, 500], 
        data_dir=data_dir
    )

def create_sample_coil20_dataset(data_dir='data', n_samples=1440):
    """创建COIL-20物体样本数据集"""
    return create_sample_dataset(
        'coil20', 
        n_samples=n_samples, 
        n_clusters=20, 
        n_features=[1024, 3304, 6750], 
        data_dir=data_dir
    )

def create_sample_mnist_dataset(data_dir='data', n_samples=10000):
    """创建MNIST手写数字样本数据集"""
    return create_sample_dataset(
        'mnist', 
        n_samples=n_samples, 
        n_clusters=10, 
        n_features=[784, 1024, 256], 
        data_dir=data_dir
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="创建样本数据集")
    parser.add_argument('--datasets', nargs='+', 
                      default=['handwritten', 'reuters', 'coil20', 'mnist'],
                      help='要创建的数据集')
    parser.add_argument('--dir', type=str, default='data', help='保存目录')
    
    args = parser.parse_args()
    
    # 创建请求的样本数据集
    for dataset in args.datasets:
        if dataset == 'handwritten':
            create_sample_handwritten_dataset(args.dir)
        elif dataset == 'reuters':
            create_sample_reuters_dataset(args.dir)
        elif dataset == 'coil20':
            create_sample_coil20_dataset(args.dir)
        elif dataset == 'mnist':
            create_sample_mnist_dataset(args.dir)
        else:
            print(f"未知数据集: {dataset}")
    
    print("所有样本数据集创建完成!")
