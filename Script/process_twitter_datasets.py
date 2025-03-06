# Description: 处理Twitter多视图数据集的脚本

"""
处理Twitter多视图数据集的脚本
可以处理单个数据集或所有可用数据集
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
import warnings
from scipy import sparse
from sklearn.preprocessing import StandardScaler
import json
from pathlib import Path

from utils.data_loader import download_dataset, preprocess_twitter_dataset, load_multiview_data

# Twitter数据集列表 - 只包含实际可用的数据集
TWITTER_DATASETS = [
    'twitter_football',    # Twitter football 数据集
    'twitter_olympics',    # Twitter olympics 数据集
    'twitter_rugby',       # Twitter rugby 数据集
    'twitter_politics-ie', # Twitter 爱尔兰政治数据集
]


def get_available_twitter_datasets(data_dir='data', use_all_configured=False):
    """
    获取当前环境中可用的Twitter数据集
    
    参数:
        data_dir: 数据目录
        use_all_configured: 是否返回配置的所有数据集，而不只是已处理的数据集
        
    返回:
        list: 可用的Twitter数据集列表
    """
    # 如果要求使用配置的数据集，直接返回TWITTER_DATASETS
    if use_all_configured:
        return TWITTER_DATASETS
    
    # 首先检查是否存在JSON数据文件
    json_path = os.path.join(data_dir, 'available_twitter_datasets.json')
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                datasets = [f"twitter_{d}" for d in data.get('available_datasets', [])]
                if datasets:
                    return datasets
        except Exception:
            pass  # 如果加载失败，继续尝试其他方法
    
    # 检查是否有MAT文件
    mat_files = [f for f in os.listdir(data_dir) 
                if f.startswith('twitter_') and f.endswith('.mat')]
    datasets = [os.path.splitext(f)[0] for f in mat_files]
    
    # 自动生成JSON配置以便将来使用
    if not os.path.exists(json_path) and mat_files:
        try:
            # 从mat文件名提取数据集基本名称
            base_datasets = [f.replace('twitter_', '').replace('.mat', '') for f in mat_files]
            with open(json_path, 'w') as f:
                json.dump({"available_datasets": base_datasets}, f, indent=2)
            print(f"自动创建可用数据集配置: {json_path}")
        except Exception as e:
            print(f"警告: 无法创建数据集配置: {e}")
    
    # 如果没有找到MAT文件，返回默认列表
    if not datasets:
        return TWITTER_DATASETS
        
    return datasets

def process_twitter_dataset(dataset_name, output_dir, visualize=True):
    """处理指定的Twitter数据集"""
    if not dataset_name.startswith('twitter_'):
        dataset_name = f'twitter_{dataset_name}'
    
    # 检查数据集是否在已知可用数据集列表中
    if dataset_name not in TWITTER_DATASETS:
        print(f"警告: {dataset_name} 不在已知可用数据集列表中，尝试处理但可能会失败")
    
    # 处理特殊情况: 提取数据集名称
    twitter_name = dataset_name.split('_', 1)[1]  # 去掉'twitter_'前缀
    
    # 处理名称中的连字符，例如politics-ie，转换为不带连字符的目录名
    dir_name = twitter_name.replace('-', '')
    
    print(f"处理Twitter数据集: {dataset_name} (子数据集: {twitter_name})")
    print(f"目录名称: {dir_name}")
    
    # 1. 确保目录存在
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    
    # 2. 下载Twitter总数据集并解压
    print("步骤1: 下载Twitter多视图数据集")
    success = download_dataset('multiview_twitter', output_dir)
    if not success:
        print("下载Twitter数据集失败")
        return False
    
    # 3. 处理特定子数据集
    print(f"步骤2: 处理子数据集 {dataset_name}")
    
    try:
        # 对于带连字符的名称，使用不带连字符的目录名进行处理
        preprocess_twitter_dataset(dir_name, output_dir)
        
        # 对于politics-ie这样的名称，需创建正确的mat文件
        if '-' in twitter_name:
            src_path = os.path.join(output_dir, f'twitter_{dir_name}.mat')
            dst_path = os.path.join(output_dir, f'twitter_{twitter_name}.mat')
            
            if os.path.exists(src_path) and not os.path.exists(dst_path):
                import shutil
                shutil.copy2(src_path, dst_path)
                print(f"创建数据集文件: {dst_path}")
    
    except Exception as e:
        print(f"处理 {dataset_name} 失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. 加载处理后的数据集验证
    print(f"步骤3: 验证 {dataset_name} 数据集")
    try:
        views, adj_matrices, labels = load_multiview_data(dataset_name)
        
        n_samples = views[0].shape[0]
        n_views = len(views)
        n_features = [v.shape[1] for v in views]
        
        print(f"\n数据集 {dataset_name} 信息:")
        print(f"  样本数: {n_samples}")
        print(f"  视图数: {n_views}")
        for i, dim in enumerate(n_features):
            print(f"  视图 {i+1} 维度: {dim}")
        
        # 标签统计
        if labels is not None:
            if isinstance(labels, torch.Tensor):
                labels_np = labels.cpu().numpy()
            else:
                labels_np = labels
                
            n_clusters = len(np.unique(labels_np))
            print(f"  聚类数: {n_clusters}")
            
            # 统计各类别样本数量
            unique_labels, counts = np.unique(labels_np, return_counts=True)
            for label, count in zip(unique_labels, counts):
                print(f"    类别 {int(label)}: {count} 样本")
        
        # 5. 可视化
        if visualize:
            print(f"步骤4: 可视化 {dataset_name} 数据集")
            visualize_twitter_dataset(dataset_name, views, labels_np)
        
        return True
        
    except Exception as e:
        print(f"验证 {dataset_name} 数据集失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def preprocess_view_for_visualization(view_np):
    """预处理视图数据以便可视化，处理异常值和稀疏性"""
    # 检查并替换NaN和无穷值
    if np.isnan(view_np).any() or np.isinf(view_np).any():
        print("  警告：发现NaN或无穷值，将被替换为0")
        view_np = np.nan_to_num(view_np, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 检查数据是否全为零
    if np.all(view_np == 0):
        print("  警告：数据全为零，添加微小噪声以便可视化")
        view_np = view_np + np.random.normal(0, 1e-6, view_np.shape)
        return view_np
    
    # 对于非常稀疏的数据，考虑使用更适合的预处理
    sparsity = 1.0 - (np.count_nonzero(view_np) / view_np.size)
    if sparsity > 0.95:  # 如果稀疏度高于95%
        print(f"  警告：数据非常稀疏 (稀疏度: {sparsity:.2%})，使用特殊处理")
        
        # 检查是否为对称矩阵（邻接矩阵特性）
        is_symmetric = np.allclose(view_np, view_np.T)
        
        if is_symmetric and view_np.shape[0] == view_np.shape[1]:
            # 可能是邻接矩阵，使用度量作为特征
            print("  检测到邻接矩阵结构，提取网络特征")
            
            # 计算简单的网络特征：入度、出度、介数中心性近似
            in_degree = np.sum(view_np, axis=0).reshape(-1, 1)  # 入度
            out_degree = np.sum(view_np, axis=1).reshape(-1, 1)  # 出度
            
            # 结合这些特征
            network_features = np.hstack([in_degree, out_degree])
            
            # 添加额外特征: 每个节点的2阶邻域大小
            second_order = np.dot(view_np, view_np)
            second_order_size = np.sum(second_order > 0, axis=1).reshape(-1, 1)
            network_features = np.hstack([network_features, second_order_size])
            
            # 标准化特征
            scaler = StandardScaler()
            network_features = scaler.fit_transform(network_features)
            
            return network_features
    
    # 标准化数据以改善降维效果
    try:
        scaler = StandardScaler()
        view_np = scaler.fit_transform(view_np)
    except:
        print("  警告：无法标准化数据，使用原始数据")
    
    return view_np

def visualize_twitter_dataset(dataset_name, views, labels):
    """为Twitter数据集创建可视化
    
    参数:
        dataset_name: 数据集名称
        views: 视图数据列表
        labels: 标签数组
    """
    # 创建保存目录
    os.makedirs('visualizations', exist_ok=True)
    
    # 1. 可视化每个视图
    for i, view in enumerate(views):
        print(f"  处理视图 {i+1}...")
        
        # 转换为numpy数组
        if isinstance(view, torch.Tensor):
            view_np = view.cpu().numpy()
        else:
            view_np = view
        
        # 特殊处理：预处理数据以处理异常值和稀疏性
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # 预处理数据
                view_np = preprocess_view_for_visualization(view_np)
                
                # 检查数据是否有有效方差
                if np.var(view_np) < 1e-10:
                    print(f"  警告：视图 {i+1} 方差接近零，跳过可视化")
                    continue
                
                # 检查PCA降维的必要性
                if view_np.shape[1] > 50:
                    try:
                        print(f"  使用PCA将维度从{view_np.shape[1]}降至50")
                        pca = PCA(n_components=50, svd_solver='randomized', random_state=42)
                        view_np = pca.fit_transform(view_np)
                    except Exception as e:
                        print(f"  PCA失败: {e}，尝试使用截断SVD")
                        from sklearn.decomposition import TruncatedSVD
                        svd = TruncatedSVD(n_components=min(50, view_np.shape[1]-1), random_state=42)
                        view_np = svd.fit_transform(view_np)
                
                # 计算合适的perplexity值，避免过小或过大
                n_samples = view_np.shape[0]
                perplexity = min(30, max(5, n_samples // 5))
                
                # 使用t-SNE降维到2D，添加容错措施
                print(f"  应用t-SNE降维 (perplexity={perplexity})...")
                tsne = TSNE(n_components=2, perplexity=perplexity, 
                          learning_rate='auto', init='pca', random_state=42)
                view_2d = tsne.fit_transform(view_np)
                
                # 绘制散点图
                plt.figure(figsize=(10, 8))
                
                # 如果有标签，按类别着色
                if labels is not None:
                    unique_labels = np.unique(labels)
                    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
                    
                    for j, label in enumerate(unique_labels):
                        if label == 0 and len(unique_labels) > 1:  # 跳过未分类标签
                            continue
                            
                        mask = labels == label
                        plt.scatter(
                            view_2d[mask, 0], 
                            view_2d[mask, 1],
                            c=[colors[j]],
                            label=f'Class {int(label)}',
                            alpha=0.7,
                            s=50  # 增大点的大小
                        )
                    
                    # 优化图例位置和显示
                    if len(unique_labels) > 10:
                        # 如果标签太多，调整图例
                        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
                    else:
                        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                else:
                    plt.scatter(view_2d[:, 0], view_2d[:, 1], alpha=0.7, s=50)
                
                # 添加标题和保存
                plt.title(f'{dataset_name} - View {i+1} t-SNE Visualization')
                plt.tight_layout()
                plt.savefig(f'visualizations/{dataset_name}_view{i+1}_tsne.png', dpi=300)
                plt.close()
                
                print(f"  视图 {i+1} 可视化已保存")
        
        except Exception as e:
            print(f"  视图 {i+1} 可视化失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 2. 可视化类别分布
    try:
        if labels is not None:
            plt.figure(figsize=(12, 6))
            unique_labels = np.unique(labels)
            
            # 跳过未分类标签
            if 0 in unique_labels and len(unique_labels) > 1:
                mask = unique_labels > 0
                unique_labels = unique_labels[mask]
                
            counts = [np.sum(labels == label) for label in unique_labels]
            
            # 绘制类别分布条形图
            colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
            plt.bar(
                [f'Class {int(l)}' for l in unique_labels], 
                counts, 
                color=colors
            )
            plt.title(f'{dataset_name} - Class Distribution')
            plt.xlabel('Class')
            plt.ylabel('Number of Samples')
            plt.xticks(rotation=45)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'visualizations/{dataset_name}_class_distribution.png', dpi=300)
            plt.close()
            
            print(f"类别分布可视化已保存")
            
            # 3. 特别添加：热力图可视化第一个视图的相关性
            if views and views[0].shape[0] <= 500:  # 限制大小避免内存问题
                plt.figure(figsize=(10, 8))
                view_np = views[0].cpu().numpy() if isinstance(views[0], torch.Tensor) else views[0]
                
                # 检查是否为邻接矩阵（正方形矩阵）
                if view_np.shape[0] == view_np.shape[1]:
                    # 使用带社区结构的热力图
                    print("创建网络热力图...")
                    # 按标签排序
                    if labels is not None:
                        # 获取排序索引
                        sorted_indices = np.argsort(labels)
                        sorted_matrix = view_np[sorted_indices][:, sorted_indices]  # 修正了中文逗号为英文逗号
                        sorted_labels = labels[sorted_indices]
                        
                        # 计算分隔线位置（类别边界）
                        dividers = []
                        prev_label = -1
                        for i, label in enumerate(sorted_labels):
                            if label != prev_label:
                                dividers.append(i)
                                prev_label = label
                        
                        plt.imshow(sorted_matrix, cmap='viridis')
                        plt.colorbar(label='Connection Strength')
                        
                        # 添加分隔线指示不同的社区
                        for d in dividers[1:]:  # 跳过第一个
                            plt.axhline(y=d-0.5, color='red', linestyle='-', alpha=0.3)
                            plt.axvline(x=d-0.5, color='red', linestyle='-', alpha=0.3)
                        
                        plt.title(f'{dataset_name} - Network Structure (Sorted by Community)')
                    else:
                        plt.imshow(view_np, cmap='viridis')
                        plt.colorbar(label='Connection Strength')
                        plt.title(f'{dataset_name} - Network Structure')
                        
                    plt.tight_layout()
                    plt.savefig(f'visualizations/{dataset_name}_network_heatmap.png', dpi=300)
                    plt.close()
    except Exception as e:
        print(f"额外可视化生成失败: {e}")
    
    print(f"可视化结果已保存到 visualizations/{dataset_name}_*.png")

def process_all_twitter_datasets(output_dir, use_all_configured=False):
    """处理所有Twitter数据集
    
    参数:
        output_dir: 输出目录
        use_all_configured: 是否使用所有配置的数据集，即使没有对应的MAT文件
    """
    # 使用动态获取的数据集列表或配置的所有数据集
    datasets = get_available_twitter_datasets(output_dir, use_all_configured=use_all_configured)
    results = {}
    
    for dataset in datasets:
        print(f"\n{'='*50}")
        print(f"正在处理: {dataset}")
        print(f"{'='*50}")
        
        success = process_twitter_dataset(dataset, output_dir)
        results[dataset] = "成功" if success else "失败"
    
    # 打印处理结果摘要
    print("\n处理结果摘要:")
    print("-" * 30)
    for dataset, result in results.items():
        print(f"{dataset.ljust(20)}: {result}")

def list_available_datasets(data_dir='data', show_all=False):
    """列出所有可用的Twitter数据集
    
    参数:
        data_dir: 数据目录
        show_all: 是否显示所有配置的数据集，而不只是已处理的数据集
    """
    # 获取已处理的数据集
    processed_datasets = get_available_twitter_datasets(data_dir)
    
    # 获取所有配置的数据集
    all_datasets = TWITTER_DATASETS
    
    # 根据show_all决定显示哪些数据集
    datasets = all_datasets if show_all else processed_datasets
    
    print("\n可用的Twitter多视图数据集:")
    print("="*40)
    
    if not datasets:
        print("未找到任何可用的Twitter数据集")
        return
        
    for i, dataset in enumerate(datasets, 1):
        base_name = dataset.split('_', 1)[1] if '_' in dataset else dataset
        is_processed = os.path.exists(os.path.join(data_dir, f"{dataset}.mat"))
        status = "可用" if is_processed else "需要处理"
        print(f"{i}. {dataset.ljust(20)} [{status}]")
    
    print("\n使用方法示例:")
    print(f"  python {sys.argv[0]} --dataset twitter_football")
    print(f"  python {sys.argv[0]} --dataset all")
    print(f"  python {sys.argv[0]} --dataset all --process-all  # 处理所有配置的数据集")

def main():
    """主函数，处理命令行参数并启动数据集处理"""
    parser = argparse.ArgumentParser(description='处理Twitter多视图数据集')
    parser.add_argument('--dataset', type=str, default='all',
                        help='指定要处理的数据集名称，例如twitter_football，或使用all处理所有可用数据集')
    parser.add_argument('--output-dir', type=str, default='data',
                        help='输出目录，默认为data')
    parser.add_argument('--no-viz', dest='visualize', action='store_false',
                      help='禁用可视化')
    parser.add_argument('--list', action='store_true', 
                      help='列出所有可用的数据集')
    parser.add_argument('--list-all', action='store_true',
                      help='列出所有配置的数据集，包括未处理的数据集')
    parser.add_argument('--process-all', action='store_true',
                      help='处理所有配置的数据集，包括未处理的数据集')
    parser.add_argument('--force', action='store_true',
                      help='强制处理指定数据集，即使它不在已知可用列表中')
    args = parser.parse_args()
    
    # 列出可用数据集
    if args.list:
        list_available_datasets(args.output_dir)
        return
    
    # 列出所有配置的数据集
    if args.list_all:
        list_available_datasets(args.output_dir, show_all=True)
        return
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 处理指定数据集或所有数据集
    if args.dataset.lower() == 'all':
        process_all_twitter_datasets(args.output_dir, use_all_configured=args.process_all)
    else:
        dataset = args.dataset
        # 确保数据集名称规范化
        if not dataset.startswith('twitter_'):
            dataset = f"twitter_{dataset}"
        
        # 获取可用数据集列表
        if args.process_all or args.force:
            available_datasets = TWITTER_DATASETS if args.process_all else [dataset]
        else:
            available_datasets = get_available_twitter_datasets(args.output_dir)
            
        if dataset not in available_datasets and not args.force:
            print(f"未知的数据集: {dataset}")
            print("可用的数据集:")
            for ds in available_datasets:
                print(f"  {ds}")
            print("\n使用 --force 参数可以尝试强制处理此数据集")
            sys.exit(1)
            
        success = process_twitter_dataset(dataset, args.output_dir, args.visualize)
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
