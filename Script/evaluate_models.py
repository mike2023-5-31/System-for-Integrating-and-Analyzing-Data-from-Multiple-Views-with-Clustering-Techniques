import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import argparse
import logging
import seaborn as sns
from datetime import datetime

# 导入项目所需模块
from utils.data_loader import load_multiview_data
from models.mvgcn import MVGCNModel
from config import Config

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

def cluster_accuracy(y_true, y_pred):
    """
    计算聚类结果的准确率
    """
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import confusion_matrix
    
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    
    # 确保标签从0开始
    if y_true.min() != 0:
        y_true = y_true - y_true.min()
    if y_pred.min() != 0:
        y_pred = y_pred - y_pred.min()
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 使用匈牙利算法寻找最优匹配
    row_ind, col_ind = linear_sum_assignment(-cm)
    
    # 计算准确率
    acc = cm[row_ind, col_ind].sum() / y_true.shape[0]
    return acc

def parse_args():
    parser = argparse.ArgumentParser(description='多视图模型评估工具')
    parser.add_argument('--datasets', nargs='+', default=['3sources', 'twitter_politics-uk'],
                        help='要评估的数据集列表')
    parser.add_argument('--output', type=str, default='results/evaluation_results.csv',
                        help='评估结果输出CSV文件')
    parser.add_argument('--visualize', action='store_true', help='生成可视化结果')
    parser.add_argument('--verbose', action='store_true', help='显示详细输出')
    return parser.parse_args()

def load_model_for_dataset(dataset_name):
    """
    根据数据集名加载相应模型
    """
    config = Config()
    config.dataset = dataset_name
    
    # 根据数据集设置合适的模型参数
    if dataset_name == '3sources':
        config.hidden_dims = [512, 256]
        config.latent_dim = 128
        config.gcn_hidden = 256
    elif dataset_name.startswith('twitter_'):
        config.hidden_dims = [32, 16]
        config.latent_dim = 8
        config.gcn_hidden = 16
    elif dataset_name == 'segment':
        config.hidden_dims = [256, 128]
        config.latent_dim = 64
        config.gcn_hidden = 128
    else:
        # 默认参数
        config.hidden_dims = [256, 128]
        config.latent_dim = 64
        config.gcn_hidden = 128
    
    # 加载数据集以获取维度
    try:
        data_views, adj_matrices, labels = load_multiview_data(dataset_name)
        if data_views is None or len(data_views) == 0:
            logger.error(f"无法加载数据集: {dataset_name}")
            return None, None, None, None
            
        # 设置输入维度和聚类数
        config.input_dims = [view.shape[1] for view in data_views]
        config.n_clusters = len(np.unique(labels))
        
        # 创建设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化模型
        model = MVGCNModel(config).to(device)
        
        # 确定模型路径
        model_path = f"checkpoints/{dataset_name}_best_model.pth"
        
        if os.path.exists(model_path):
            # 加载模型权重
            logger.info(f"加载模型: {model_path}")
            try:
                checkpoint = torch.load(model_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                logger.info(f"成功加载模型")
                
                # 移动数据到设备
                data_views = [v.to(device) for v in data_views]
                adj_matrices = [a.to(device) for a in adj_matrices]
                
                return model, data_views, adj_matrices, labels
                
            except Exception as e:
                logger.error(f"加载模型失败: {e}")
                return None, data_views, adj_matrices, labels
        else:
            logger.error(f"模型文件不存在: {model_path}")
            return None, data_views, adj_matrices, labels
            
    except Exception as e:
        logger.error(f"处理数据集时出错: {e}")
        return None, None, None, None

def evaluate_dataset(dataset_name, visualize=False):
    """评估指定数据集上的模型性能"""
    logger.info(f"开始评估数据集: {dataset_name}")
    
    # 加载模型和数据
    model, data_views, adj_matrices, labels = load_model_for_dataset(dataset_name)
    
    if model is None:
        logger.error(f"无法评估数据集 {dataset_name}，模型加载失败")
        return {
            'dataset': dataset_name,
            'nmi': 0,
            'ari': 0,
            'acc': 0,
            'status': 'failed',
            'error': 'model_loading_failed'
        }
    
    if data_views is None or adj_matrices is None:
        logger.error(f"无法评估数据集 {dataset_name}，数据加载失败")
        return {
            'dataset': dataset_name,
            'nmi': 0,
            'ari': 0,
            'acc': 0,
            'status': 'failed',
            'error': 'data_loading_failed'
        }
    
    # 创建结果目录
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    # 获取设备
    device = next(model.parameters()).device
    
    # 评估模型
    model.eval()
    with torch.no_grad():
        try:
            outputs = model(data_views, adj_matrices, mask_rate=0.0)
            
            # 获取嵌入
            embeddings = outputs['fused']
            
            # 获取聚类结果
            if 'cluster_logits' in outputs:
                cluster_preds = torch.argmax(outputs['cluster_logits'], dim=1).cpu().numpy()
            else:
                # 如果没有聚类头输出，使用K-means进行聚类
                from sklearn.cluster import KMeans
                n_clusters = len(np.unique(labels))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_preds = kmeans.fit_predict(embeddings.cpu().numpy())
            
            # 转换标签格式
            true_labels = labels
            if isinstance(true_labels, torch.Tensor):
                true_labels = true_labels.cpu().numpy()
            
            # 计算评估指标
            nmi = normalized_mutual_info_score(true_labels, cluster_preds)
            ari = adjusted_rand_score(true_labels, cluster_preds)
            acc = cluster_accuracy(true_labels, cluster_preds)
            
            logger.info(f"数据集 {dataset_name} 评估结果:")
            logger.info(f"NMI: {nmi:.4f}")
            logger.info(f"ARI: {ari:.4f}")
            logger.info(f"ACC: {acc:.4f}")
            
            # 生成可视化结果
            if visualize:
                logger.info(f"生成可视化结果...")
                
                # 生成t-SNE可视化
                try:
                    tsne = TSNE(n_components=2, random_state=42)
                    embeddings_2d = tsne.fit_transform(embeddings.cpu().numpy())
                    
                    plt.figure(figsize=(15, 7))
                    
                    # 真实标签可视化
                    plt.subplot(1, 2, 1)
                    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                 c=true_labels, cmap='tab10', alpha=0.7)
                    plt.colorbar(scatter, label='True Classes')
                    plt.title(f'{dataset_name} - True Labels')
                    plt.xlabel('t-SNE Dimension 1')
                    plt.ylabel('t-SNE Dimension 2')
                    
                    # 预测标签可视化
                    plt.subplot(1, 2, 2)
                    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                 c=cluster_preds, cmap='tab10', alpha=0.7)
                    plt.colorbar(scatter, label='Predicted Classes')
                    plt.title(f'{dataset_name} - Predicted Labels')
                    plt.xlabel('t-SNE Dimension 1')
                    plt.ylabel('t-SNE Dimension 2')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(results_dir, f'{dataset_name}_tsne.png'))
                    plt.close()
                    
                    # 生成混淆矩阵
                    plt.figure(figsize=(8, 6))
                    from sklearn.metrics import confusion_matrix
                    cm = confusion_matrix(true_labels, cluster_preds)
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                    plt.title(f'{dataset_name} - Confusion Matrix')
                    plt.xlabel('Predicted Labels')
                    plt.ylabel('True Labels')
                    plt.tight_layout()
                    plt.savefig(os.path.join(results_dir, f'{dataset_name}_confusion_matrix.png'))
                    plt.close()
                    
                    logger.info(f"可视化结果已保存至 {results_dir} 目录")
                    
                except Exception as e:
                    logger.error(f"生成可视化时出错: {e}")
            
            # 返回结果
            return {
                'dataset': dataset_name,
                'nmi': nmi,
                'ari': ari,
                'acc': acc,
                'status': 'success',
                'error': None
            }
            
        except Exception as e:
            logger.error(f"评估过程出错: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'dataset': dataset_name,
                'nmi': 0,
                'ari': 0,
                'acc': 0,
                'status': 'failed',
                'error': str(e)
            }

def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    logger.info(f"评估以下数据集: {', '.join(args.datasets)}")
    
    # 存储所有评估结果
    all_results = []
    
    # 评估每个数据集
    for dataset in args.datasets:
        result = evaluate_dataset(dataset, visualize=args.visualize)
        all_results.append(result)
    
    # 创建结果DataFrame
    results_df = pd.DataFrame(all_results)
    
    # 保存结果到CSV
    results_df.to_csv(args.output, index=False)
    logger.info(f"评估结果已保存到 {args.output}")
    
    # 打印表格总结
    print("\n========== 评估结果汇总 ==========")
    summary_df = results_df[['dataset', 'nmi', 'ari', 'acc', 'status']]
    print(summary_df.to_string(index=False))
    print("===================================\n")
    
    # 生成性能比较图
    try:
        success_results = results_df[results_df['status'] == 'success']
        if len(success_results) > 1:
            plt.figure(figsize=(10, 6))
            
            # 获取数据
            datasets = success_results['dataset']
            nmi_values = success_results['nmi']
            ari_values = success_results['ari']
            acc_values = success_results['acc']
            
            # 设置x轴位置
            x = np.arange(len(datasets))
            width = 0.25
            
            # 绘制柱状图
            plt.bar(x - width, nmi_values, width, label='NMI')
            plt.bar(x, ari_values, width, label='ARI')
            plt.bar(x + width, acc_values, width, label='ACC')
            
            # 添加标签和图例
            plt.xlabel('Datasets')
            plt.ylabel('Score')
            plt.title('Performance Comparison Across Datasets')
            plt.xticks(x, datasets)
            plt.legend()
            
            # 保存图表
            comparison_path = os.path.join('results', 'performance_comparison.png')
            plt.tight_layout()
            plt.savefig(comparison_path)
            plt.close()
            logger.info(f"性能比较图已保存到 {comparison_path}")
    except Exception as e:
        logger.error(f"生成比较图时出错: {e}")

if __name__ == "__main__":
    main()
