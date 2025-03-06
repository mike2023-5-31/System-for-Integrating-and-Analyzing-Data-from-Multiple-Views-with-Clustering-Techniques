"""
数据集处理示例脚本
这个脚本用于下载和处理特定的多视图数据集，方便单独测试数据集
"""
import os
import argparse
import torch
import numpy as np
from utils.data_loader import (
    download_dataset, 
    preprocess_movielists, 
    preprocess_3sources, 
    preprocess_twitter_dataset,
    preprocess_segment_dataset,
    load_multiview_data
)
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def process_dataset(dataset_name, visualize=True):
    """
    处理特定数据集
    
    参数:
        dataset_name: 数据集名称
        visualize: 是否可视化数据
    """
    print(f"处理数据集: {dataset_name}")
    
    # 确保数据目录存在
    os.makedirs('data', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    
    # 1. 下载数据集
    print(f"步骤 1: 下载 {dataset_name} 数据集")
    success = download_dataset(dataset_name)
    if not success:
        print(f"下载 {dataset_name} 失败")
        return False
    
    # 2. 根据数据集类型进行处理
    print(f"步骤 2: 预处理 {dataset_name} 数据集")
    try:
        if dataset_name == 'movielists':
            preprocess_movielists('data')
        elif dataset_name == '3sources':
            preprocess_3sources('data')
        elif dataset_name.startswith('twitter_'):
            twitter_name = dataset_name.split('_')[1]
            preprocess_twitter_dataset(twitter_name, 'data')
        elif dataset_name == 'segment':
            preprocess_segment_dataset('data')
        else:
            print(f"没有针对 {dataset_name} 的特定处理方法")
    except Exception as e:
        print(f"预处理 {dataset_name} 出错: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 3. 加载处理后的数据集以验证
    print(f"步骤 3: 验证 {dataset_name} 数据集")
    try:
        views, adj_matrices, labels = load_multiview_data(dataset_name)
        n_samples = views[0].shape[0]
        n_views = len(views)
        n_features = [v.shape[1] for v in views]
        
        print(f"数据集信息:")
        print(f"  样本数: {n_samples}")
        print(f"  视图数: {n_views}")
        for i, dim in enumerate(n_features):
            print(f"  视图 {i+1} 维度: {dim}")
        
        if labels is not None:
            n_clusters = len(torch.unique(labels))
            print(f"  聚类数: {n_clusters}")
            
            # 显示每个类别的样本数
            unique_labels, counts = np.unique(labels.cpu().numpy(), return_counts=True)
            for label, count in zip(unique_labels, counts):
                print(f"    类别 {label}: {count} 样本")
        
        # 4. 可视化数据集
        if visualize:
            print(f"步骤 4: 可视化 {dataset_name} 数据集")
            visualize_dataset(dataset_name, views, labels)
        
        return True
    
    except Exception as e:
        print(f"验证 {dataset_name} 出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def visualize_dataset(dataset_name, views, labels):
    """可视化数据集的二维投影"""
    os.makedirs('visualizations', exist_ok=True)
    
    # 对每个视图进行可视化
    for i, view in enumerate(views):
        # 转换为NumPy数组
        if isinstance(view, torch.Tensor):
            view_data = view.cpu().numpy()
        else:
            view_data = view
        
        # 使用PCA降维到2D
        if view_data.shape[1] > 2:
            pca = PCA(n_components=2)
            view_2d = pca.fit_transform(view_data)
            explained_var = pca.explained_variance_ratio_.sum()
            title = f"{dataset_name} - 视图 {i+1} (PCA解释方差: {explained_var:.2%})"
        else:
            view_2d = view_data
            title = f"{dataset_name} - 视图 {i+1}"
        
        # 可视化
        plt.figure(figsize=(10, 8))
        
        if labels is not None:
            # 获取标签
            if isinstance(labels, torch.Tensor):
                labels_np = labels.cpu().numpy()
            else:
                labels_np = labels
            
            # 获取唯一标签
            unique_labels = np.unique(labels_np)
            
            # 为每个类别使用不同的颜色
            colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
            
            for j, label in enumerate(unique_labels):
                mask = labels_np == label
                plt.scatter(view_2d[mask, 0], view_2d[mask, 1], 
                           color=colors[j], label=f'Class {label}', alpha=0.7)
            plt.legend()
        else:
            plt.scatter(view_2d[:, 0], view_2d[:, 1], alpha=0.7)
        
        plt.title(title)
        plt.tight_layout()
        plt.savefig(f"visualizations/{dataset_name}_view{i+1}.png")
        plt.close()
    
    print(f"可视化保存到 visualizations/{dataset_name}_view*.png")

def main():
    parser = argparse.ArgumentParser(description='处理多视图数据集')
    parser.add_argument('--dataset', type=str, required=True,
                       help='要处理的数据集名称')
    parser.add_argument('--no-viz', dest='visualize', action='store_false',
                       help='禁用可视化')
    args = parser.parse_args()
    
    process_dataset(args.dataset, args.visualize)

if __name__ == "__main__":
    main()
