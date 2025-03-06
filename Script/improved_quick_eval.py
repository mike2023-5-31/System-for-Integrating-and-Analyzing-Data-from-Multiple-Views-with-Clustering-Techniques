import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.manifold import TSNE

# 添加导入路径
sys.path.append('.')

from config import Config
from models.mvgcn import MVGCNModel
from utils.data_loader import load_multiview_data

def cluster_accuracy(y_true, y_pred):
    """计算聚类准确率"""
    from scipy.optimize import linear_sum_assignment
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    
    # 计算混淆矩阵
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    
    # 匈牙利算法求解最佳匹配
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() / y_pred.size

def visualize_results(embeddings, true_labels, pred_labels, dataset_name):
    """可视化结果"""
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # 使用t-SNE降维
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # 绘图
    plt.figure(figsize=(12, 5))
    
    # 真实标签可视化
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=true_labels, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter, label='True Classes')
    plt.title(f'{dataset_name} - True Labels')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    
    # 预测标签可视化
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=pred_labels, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter, label='Predicted Classes')
    plt.title(f'{dataset_name} - Predicted Labels')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    
    plt.tight_layout()
    plt.savefig(f'results/{dataset_name}_visualization.png', dpi=300)
    plt.close()
    
    print(f"可视化结果已保存至 results/{dataset_name}_visualization.png")

def evaluate_dataset(dataset_name, visualize=False):
    """评估单个数据集"""
    print(f"\n===== 评估数据集: {dataset_name} =====")
    
    try:
        # 配置
        config = Config()
        config.dataset = dataset_name
        
        # 加载数据
        print("加载数据...")
        try:
            data_views, adj_matrices, labels = load_multiview_data(dataset_name)
            if labels is None:
                print("警告: 数据集没有标签")
                return {
                    'dataset': dataset_name,
                    'nmi': 0,
                    'ari': 0,
                    'acc': 0,
                    'status': 'no_labels'
                }
            
            # 设置模型参数
            if dataset_name.startswith('twitter_'):
                config.hidden_dims = [32, 16]
                config.latent_dim = 8
                config.gcn_hidden = 16
            else:
                config.hidden_dims = [512, 256]
                config.latent_dim = 128
                config.gcn_hidden = 256
            
            config.input_dims = [v.shape[1] for v in data_views]
            config.n_clusters = len(np.unique(labels))
            
            print(f"数据集信息: {len(data_views)}个视图, {data_views[0].shape[0]}个样本, {config.n_clusters}个聚类")
            
            # 设备
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # 创建模型
            print("初始化模型...")
            model = MVGCNModel(config).to(device)
            
            # 加载模型权重
            model_path = f"checkpoints/{dataset_name}_best_model.pth"
            if os.path.exists(model_path):
                print(f"加载模型: {model_path}")
                try:
                    checkpoint = torch.load(model_path, map_location=device)
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    print("模型加载成功")
                except Exception as e:
                    print(f"模型加载失败: {e}")
                    return {
                        'dataset': dataset_name,
                        'nmi': 0,
                        'ari': 0,
                        'acc': 0,
                        'status': 'model_load_error'
                    }
            else:
                print(f"模型文件不存在: {model_path}")
                return {
                    'dataset': dataset_name,
                    'nmi': 0,
                    'ari': 0,
                    'acc': 0,
                    'status': 'model_not_found'
                }
            
            # 将数据移到设备
            data_views = [v.to(device) for v in data_views]
            adj_matrices = [a.to(device) for a in adj_matrices]
            
            # 评估
            print("执行评估...")
            model.eval()
            with torch.no_grad():
                # 尝试直接使用模型的集群特征提取方法
                try:
                    cluster_features = model.get_cluster_features(data_views, adj_matrices)
                    embeddings = cluster_features.cpu().numpy()
                except Exception as e1:
                    print(f"无法使用get_cluster_features: {e1}")
                    try:
                        # 尝试使用前向传播
                        outputs = model(data_views, adj_matrices, mask_rate=0.0)
                        
                        # 检查所有可能的键
                        if hasattr(outputs, 'keys'):
                            print(f"可用输出键: {list(outputs.keys())}")
                            
                            # 尝试所有可能的键
                            for key in ['fused', 'cluster_rep', 'z', 'latent_z', 
                                       'embedding', 'representations', 'final_embedding']:
                                if key in outputs and torch.is_tensor(outputs[key]):
                                    embeddings = outputs[key].cpu().numpy()
                                    print(f"使用'{key}'作为嵌入")
                                    break
                            else:
                                # 如果没有找到预定义的键，尝试使用第一个张量键
                                for key, value in outputs.items():
                                    if torch.is_tensor(value) and value.dim() == 2:
                                        embeddings = value.cpu().numpy()
                                        print(f"使用'{key}'作为嵌入")
                                        break
                                else:
                                    raise KeyError("找不到有效的嵌入输出")
                        else:
                            # 如果outputs不是字典，尝试使用它本身
                            embeddings = outputs.cpu().numpy() if torch.is_tensor(outputs) else np.array(outputs)
                    except Exception as e2:
                        print(f"前向传播方法也失败: {e2}")
                        return {
                            'dataset': dataset_name,
                            'nmi': 0,
                            'ari': 0,
                            'acc': 0,
                            'status': 'feature_extraction_failed'
                        }
                
                # 聚类
                print("执行聚类...")
                kmeans = KMeans(n_clusters=config.n_clusters, random_state=42)
                cluster_assignments = kmeans.fit_predict(embeddings)
                
                # 确保标签和预测结果都是一维数组
                true_labels = labels.flatten() if isinstance(labels, np.ndarray) else labels
                if isinstance(true_labels, torch.Tensor):
                    true_labels = true_labels.cpu().numpy().flatten()
                
                # 计算指标
                nmi = normalized_mutual_info_score(true_labels, cluster_assignments)
                ari = adjusted_rand_score(true_labels, cluster_assignments)
                acc = cluster_accuracy(true_labels, cluster_assignments)
                
                print(f"评估结果: NMI={nmi:.4f}, ARI={ari:.4f}, ACC={acc:.4f}")
                
                # 可视化
                if visualize:
                    try:
                        visualize_results(embeddings, true_labels, cluster_assignments, dataset_name)
                    except Exception as viz_err:
                        print(f"可视化生成失败: {viz_err}")
                
                return {
                    'dataset': dataset_name,
                    'nmi': nmi,
                    'ari': ari,
                    'acc': acc,
                    'status': 'success'
                }
                
        except Exception as data_err:
            print(f"数据加载失败: {data_err}")
            return {
                'dataset': dataset_name,
                'nmi': 0,
                'ari': 0,
                'acc': 0,
                'status': 'data_load_error'
            }
            
    except Exception as general_err:
        import traceback
        print(f"评估过程出错: {general_err}")
        traceback.print_exc()
        return {
            'dataset': dataset_name,
            'nmi': 0,
            'ari': 0,
            'acc': 0,
            'status': 'evaluation_error'
        }

def main():
    # 所有支持的数据集
    all_datasets = [
        # 标准数据集
        'example_dataset',
        'handwritten',
        'reuters',
        'coil20',
        'mnist',
        '3sources',
        'segment',
        'movielists',
        # Twitter数据集
        'twitter_football', 
        'twitter_olympics',
        'twitter_politics-uk', 
        'twitter_politics-ie',
        'twitter_rugby'
    ]
    
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='快速评估多视图融合模型')
    parser.add_argument('--datasets', nargs='+', default=all_datasets, 
                        help='要评估的数据集列表，默认为所有数据集')
    parser.add_argument('--visualize', action='store_true', help='是否生成可视化结果')
    parser.add_argument('--output', type=str, default='results/evaluation_summary.csv', 
                       help='结果输出文件')
    
    args = parser.parse_args()
    results = []
    
    # 确保结果目录存在
    os.makedirs('results', exist_ok=True)
    
    # 评估每个数据集
    for dataset in args.datasets:
        result = evaluate_dataset(dataset, args.visualize)
        results.append(result)
    
    # 创建结果表格
    df = pd.DataFrame(results)
    print("\n========== 评估结果汇总 ==========")
    print(df[['dataset', 'nmi', 'ari', 'acc', 'status']].to_string(index=False))
    print("===================================\n")
    
    # 保存结果
    df.to_csv(args.output, index=False)
    print(f"结果已保存至 {args.output}")
    
    # 生成条形图
    successful_results = df[df['status'] == 'success']
    if len(successful_results) > 0:
        plt.figure(figsize=(12, 6))
        
        # 设置柱状图宽度和位置
        x = np.arange(len(successful_results))
        width = 0.25
        
        # 绘制三种指标的柱状图
        plt.bar(x - width, successful_results['nmi'], width, label='NMI')
        plt.bar(x, successful_results['ari'], width, label='ARI')
        plt.bar(x + width, successful_results['acc'], width, label='ACC')
        
        plt.xlabel('Datasets')
        plt.ylabel('Score')
        plt.title('Clustering Performance Comparison')
        plt.xticks(x, successful_results['dataset'], rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig('results/performance_comparison.png', dpi=300)
        print("性能比较图已保存至 results/performance_comparison.png")

if __name__ == "__main__":
    main()