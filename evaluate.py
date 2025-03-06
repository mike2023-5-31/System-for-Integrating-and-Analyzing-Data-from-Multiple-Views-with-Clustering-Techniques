"""评估模块 - 用于评估多视图聚类模型的性能"""
# 标准库导入
import os
import logging
import time
import numpy as np

# 第三方库导入
import torch  # PyTorch库
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics.cluster import contingency_matrix
from scipy.optimize import linear_sum_assignment

# 本地模块导入
from utils.data_loader import load_multiview_data
from models import MVGCNModel
from config import Config
from utils.visualization import visualize_clusters, plot_confusion_matrix

# 在文件顶部使用绝对导入
import torch as pytorch_torch  # 使用不同名称避免冲突

def setup_logger(config):
    """配置日志记录器"""
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_file = f'logs/evaluate_{config.dataset}_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

def cluster_accuracy(y_true, y_pred):
    """
    计算聚类准确率
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    # 计算混淆矩阵
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    
    return w[row_ind, col_ind].sum() / y_pred.size

def ensemble_clustering(features_list, n_clusters, seed=42):
    """通过集成多个特征表示提高聚类性能"""
    predictions = []
    
    # 对每个特征表示进行聚类
    for features in features_list:
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
        pred = kmeans.fit_predict(features)
        predictions.append(pred)
    
    # 如果只有一个特征表示，直接返回结果
    if len(predictions) == 1:
        return predictions[0]
    
    # 构建共现矩阵
    n_samples = len(predictions[0])
    co_matrix = np.zeros((n_samples, n_samples))
    
    # 统计样本对在不同聚类中一起出现的次数
    for pred in predictions:
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                if pred[i] == pred[j]:
                    co_matrix[i, j] += 1
                    co_matrix[j, i] += 1
    
    # 对共现矩阵进行谱聚类
    from sklearn.cluster import SpectralClustering
    spectral = SpectralClustering(n_clusters=n_clusters, random_state=seed, 
                                affinity='precomputed')
    final_pred = spectral.fit_predict(co_matrix)
    
    return final_pred

def evaluate_model(config, model_path=None, save_name=None):
    """评估模型性能
    
    参数:
        config: 配置对象
        model_path: 模型路径（可选）
        save_name: 模型的自定义名称（可选）
    """
    global torch  # 添加全局声明
    logger = setup_logger(config)
    logger.info(f"Evaluating model on dataset: {config.dataset}")
    
    # 使用重命名的导入
    pytorch_torch.manual_seed(config.seed)
    # 设置随机种子
    torch.manual_seed(config.seed)  # 使用全局导入
    np.random.seed(config.seed)
    
    # 加载数据
    data_views, adj_matrices, labels = load_multiview_data(config.dataset)
    
    # 将数据移到设备上
    device = torch.device(config.device)
    data_views = [v.to(device) for v in data_views]
    adj_matrices = [a.to(device) for a in adj_matrices]
    if labels is not None:
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()  # 先转换为NumPy数组
        labels = labels.astype(np.int64)  # 然后再转换类型
    
    # 获取维度信息
    input_dims = [v.shape[1] for v in data_views]
    n_clusters = len(np.unique(labels)) if labels is not None else config.n_clusters

    config.input_dims = input_dims

    # 创建或加载模型
    model = MVGCNModel(config).to(device)
    
    # 如果提供了自定义保存名称，则使用它来构建模型路径
    if model_path is None and save_name is not None:
        model_path = f"checkpoints/{save_name}_best_model.pth"
    elif model_path is None:
        model_path = f"checkpoints/{config.dataset}_best_model.pth"
    
    # 加载模型权重 - 处理不同PyTorch版本的兼容性问题
    if model_path:
        logger.info(f"Loading model from {model_path}")
        try:
            # 尝试直接加载
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            logger.info("Model loaded with non-strict matching (ignoring unexpected keys)")
        except TypeError as e:
            logger.warning(f"Standard loading failed: {e}")
            try:
                # 尝试指定weights_only参数
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
            except Exception as e2:
                try:
                    # 尝试使用safe_globals上下文(PyTorch新版本特性)
                    logger.warning(f"Second attempt failed: {e2}")
                    import torch.serialization
                    safe_list = ['numpy.core.multiarray.scalar']
                    with torch.serialization.safe_globals(safe_list):
                        checkpoint = torch.load(model_path, map_location=device)
                    model.load_state_dict(checkpoint['model_state_dict'])
                except Exception as e3:
                    logger.error(f"All loading methods failed. Last error: {e3}")
                    logger.warning("Continuing with untrained model.")
    else:
        logger.warning("No model path provided. Using untrained model.")
    
    # 评估
    model.eval()
    with torch.no_grad():
        cluster_features = model.get_cluster_features(data_views, adj_matrices)
        cluster_features = cluster_features.cpu().numpy()
        
        # K-means聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=config.seed)
        cluster_assignments = kmeans.fit_predict(cluster_features)
        
        # 评估指标
        if labels is not None:
            nmi = normalized_mutual_info_score(labels, cluster_assignments)
            ari = adjusted_rand_score(labels, cluster_assignments)
            acc = cluster_accuracy(labels, cluster_assignments)
            
            logger.info(f"Evaluation results:")
            logger.info(f"NMI: {nmi:.4f}")
            logger.info(f"ARI: {ari:.4f}")
            logger.info(f"Accuracy: {acc:.4f}")
            
            # 可视化
            if config.visualize:
                logger.info("Generating visualizations...")
                visualize_clusters(cluster_features, labels, cluster_assignments, 
                                  save_path=f"results/{config.dataset}_clusters.png")
                
                conf_mat = contingency_matrix(labels, cluster_assignments)
                plot_confusion_matrix(conf_mat, 
                                     save_path=f"results/{config.dataset}_confusion_matrix.png")
                
                logger.info(f"Visualizations saved to results/{config.dataset}_*.png")
                
            return nmi, ari, acc
        else:
            logger.info("No labels provided. Cannot compute evaluation metrics.")
            return None, None, None

def evaluate_clustering(config, model_path=None, return_embeddings=False, save_name=None):
    """评估聚类性能并可选返回嵌入向量
    
    参数:
        config: 配置对象
        model_path: 模型路径
        return_embeddings: 是否返回嵌入向量
        save_name: 模型的自定义名称（可选）
        
    返回:
        如果return_embeddings=True: (embeddings, labels)
        否则: None
    """
    import torch
    import numpy as np
    from sklearn.cluster import KMeans
    from utils.data_loader import load_multiview_data
    
    # 本地导入load_model函数，不使用 from train import load_model
    from train import load_model
    
    # 加载数据
    data_views, adj_matrices, labels = load_multiview_data(config.dataset)
    device = torch.device(config.device)
    
    # 将数据转换为张量
    data_views = [torch.tensor(v, dtype=torch.float32).to(device) for v in data_views]
    adj_matrices = [a.to(device) for a in adj_matrices]
    
    # 如果提供了自定义保存名称，则使用它来构建模型路径
    if model_path is None and save_name is not None:
        model_path = f"checkpoints/{save_name}_best_model.pth"
    elif model_path is None:
        model_path = f"checkpoints/{config.dataset}_best_model.pth"
    
    # 加载模型
    model = load_model(config, model_path)
    model.eval()
    
    # 获取嵌入向量
    with torch.no_grad():
        # 安全获取嵌入
        try:
            embeddings = model.get_cluster_features(data_views, adj_matrices).cpu().numpy()
        except Exception as e:
            print(f"获取嵌入失败: {e}")
            # 尝试直接使用forward方法
            outputs = model(data_views, adj_matrices)
            embeddings = outputs['cluster_rep'].cpu().numpy()
    
    # 如果需要，直接返回嵌入和标签
    if return_embeddings:
        return embeddings, labels
    
    return None

def main():
    """主函数"""
    import torch  # 在主函数中也添加导入
    import os
    
    config = Config()
    
    # 确保结果目录存在
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # 评估模型
    model_path = f"checkpoints/{config.dataset}_best_model.pth"
    if (os.path.exists(model_path)):
        evaluate_model(config, model_path)
    else:
        print(f"No trained model found at {model_path}. Please train the model first.")

if __name__ == "__main__":
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    main()