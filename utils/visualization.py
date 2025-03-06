# 实现可视化工具
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
import os

def visualize_clusters(features, true_labels, pred_labels, save_path=None, fig_size=(16, 8)):
    """
    使用t-SNE可视化聚类结果
    
    参数:
    features: 特征矩阵
    true_labels: 真实标签
    pred_labels: 预测的聚类标签
    save_path: 保存图像的路径
    fig_size: 图像大小
    """
    # 使用t-SNE进行降维
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(features)
    
    # 创建图形
    plt.figure(figsize=fig_size)
    
    # 绘制真实标签
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=true_labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter)
    plt.title('True Labels')
    
    # 绘制预测标签
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=pred_labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter)
    plt.title('Predicted Clusters')
    
    plt.tight_layout()
    
    # 保存或显示
    if save_path:
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def plot_confusion_matrix(conf_mat, save_path=None, fig_size=(10, 8)):
    """
    绘制混淆矩阵热图
    
    参数:
    conf_mat: 混淆矩阵
    save_path: 保存图像的路径
    fig_size: 图像大小
    """
    plt.figure(figsize=fig_size)
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    
    # 保存或显示
    if save_path:
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def plot_training_curves(train_losses, val_metrics, save_path=None, fig_size=(12, 8)):
    """绘制训练曲线"""
    plt.figure(figsize=fig_size)
    
    # 绘制训练损失
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b-', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss')
    
    # 绘制验证指标
    plt.subplot(1, 2, 2)
    for metric_name, metric_values in val_metrics.items():
        plt.plot(metric_values, label=metric_name.upper())
    
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.title('Validation Metrics')
    
    # 保存或显示
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()