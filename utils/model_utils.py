"""
模型实用工具 - 提供加载、保存和管理模型的通用功能
"""

import os
import torch
import logging
from models import MVGCNModel
from utils.data_loader import load_multiview_data

def safe_model_loading(model_path, device='cpu', weights_only=False):
    """
    安全地加载PyTorch模型，处理不同PyTorch版本的兼容性
    
    参数:
        model_path: 模型文件路径
        device: 设备（'cpu'或'cuda'）
        weights_only: 是否只加载权重
    
    返回:
        加载的检查点数据
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 尝试多种加载方法
    methods = [
        # 方法1：标准加载
        lambda: torch.load(model_path, map_location=device),
        
        # 方法2：指定weights_only参数
        lambda: torch.load(model_path, map_location=device, weights_only=weights_only),
        
        # 方法3：使用pickle模块加载
        lambda: torch.load(model_path, map_location=device, pickle_module=__import__('pickle'))
    ]
    
    last_error = None
    for method in methods:
        try:
            return method()
        except Exception as e:
            last_error = e
            continue
    
    # 所有方法都失败
    raise RuntimeError(f"所有加载方法都失败。最后的错误: {last_error}")

def create_model_for_dataset(config):
    """
    为指定数据集创建模型实例
    
    参数:
        config: 配置对象
    
    返回:
        创建的模型实例
    """
    # 加载数据以获取输入维度
    data_views, _, _ = load_multiview_data(config.dataset)
    input_dims = [v.shape[1] for v in data_views]
    
    # 创建模型
    device = torch.device(config.device)
    model = MVGCNModel(
        input_dims=input_dims,
        hidden_dim=config.hidden_dims[0],
        latent_dim=config.latent_dim,
        gcn_hidden=config.gcn_hidden,
        projection_dim=config.projection_dim,
        dropout=config.dropout
    ).to(device)
    
    return model

def load_model_with_retry(config, model_path=None, custom_name=None):
    """
    加载模型，处理多种异常情况并进行重试
    
    参数:
        config: 配置对象
        model_path: 模型路径（可选）
        custom_name: 自定义名称（可选）
    
    返回:
        加载的模型
    """
    logger = logging.getLogger()
    
    # 处理模型路径
    if model_path is None:
        if custom_name:
            model_path = f"checkpoints/{custom_name}_best_model.pth"
        else:
            model_path = f"checkpoints/{config.dataset}_best_model.pth"
    
    # 创建模型实例
    model = create_model_for_dataset(config)
    
    if not os.path.exists(model_path):
        logger.warning(f"模型文件不存在: {model_path}，使用未训练的模型")
        return model
    
    # 尝试加载模型
    try:
        # 使用安全加载函数
        checkpoint = safe_model_loading(model_path, device=config.device)
        
        # 处理可能的不同保存格式
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # 假设是直接保存的state_dict
            model.load_state_dict(checkpoint)
            
        logger.info(f"成功从 {model_path} 加载模型")
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        logger.warning("使用未训练的模型继续")
    
    return model

def get_model_embeddings(model, data_views, adj_matrices):
    """
    从模型获取嵌入向量
    
    参数:
        model: 模型实例
        data_views: 数据视图列表
        adj_matrices: 邻接矩阵列表
    
    返回:
        嵌入向量
    """
    model.eval()
    with torch.no_grad():
        try:
            # 尝试使用专门的获取嵌入向量方法
            embeddings = model.get_cluster_features(data_views, adj_matrices)
        except (AttributeError, Exception) as e:
            # 如果失败，尝试通过前向传播获取
            outputs = model(data_views, adj_matrices)
            embeddings = outputs['cluster_rep']
    
    return embeddings
