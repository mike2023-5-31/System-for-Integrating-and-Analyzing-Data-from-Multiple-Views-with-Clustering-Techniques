# 修复train.py中的代码重复问题，并增加训练日志和模型保存功能
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, accuracy_score
import logging
import time
import torch.nn.functional as F

from utils.self_supervision import MultiViewSelfSupervision
from utils.data_loader import load_multiview_data
from data.preprocessing import feature_masking, adaptive_feature_masking, dynamic_adaptive_masking
from models import MVGCNModel
from config import Config

# 引入Twitter数据集适配器
from utils.twitter_adapter import TwitterDataAdapter, preprocess_twitter_views

def setup_logger(config):
    """配置日志记录器"""
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_file = f'logs/train_{config.dataset}_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

def train_model(config, save_name=None):
    """训练多视图GCN模型
    
    参数:
        config: 配置对象
        save_name: 保存模型的自定义名称（可选）
    """
    logger = setup_logger(config)
    logger.info(f"Starting training with config: {config.__dict__}")
    
    # 设置随机种子
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # 加载数据
    logger.info(f"Loading dataset: {config.dataset}")
    data_views, adj_matrices, labels = load_multiview_data(config.dataset)
    
    # 检测并处理Twitter数据集
    is_twitter_dataset = 'twitter' in config.dataset.lower()
    if is_twitter_dataset:
        logger.info("检测到Twitter数据集，应用特殊预处理...")
        twitter_adapter = TwitterDataAdapter(config)
        data_views, adj_matrices = twitter_adapter(data_views, adj_matrices)
        logger.info(f"Twitter数据处理完成，调整后维度: {[v.shape for v in data_views]}")
    else:
        # 检查邻接矩阵形状
        for i, (view, adj) in enumerate(zip(data_views, adj_matrices)):
            if adj.shape[0] != view.shape[0] or adj.shape[1] != view.shape[0]:
                logger.warning(f"视图{i}的邻接矩阵形状不匹配: {adj.shape} vs {view.shape[0]}x{view.shape[0]}")
                # 创建正确形状的邻接矩阵
                new_adj = torch.eye(view.shape[0])
                min_size = min(view.shape[0], adj.shape[0], adj.shape[1])
                if min_size > 0:
                    new_adj[:min_size, :min_size] = adj[:min_size, :min_size]
                adj_matrices[i] = new_adj
                logger.info(f"已修复视图{i}的邻接矩阵形状为 {new_adj.shape}")
    
    # 将数据移到设备上
    device = torch.device(config.device)
    data_views = [v.to(device) for v in data_views]
    adj_matrices = [a.to(device) for a in adj_matrices]
    if labels is not None:
        labels = torch.tensor(labels).long().to(device)
    
    # 获取维度信息
    input_dims = [v.shape[1] for v in data_views]
    n_samples = data_views[0].shape[0]
    n_clusters = len(torch.unique(labels)) if labels is not None else config.n_clusters
    
    logger.info(f"Dataset info: {n_samples} samples, {len(data_views)} views, {n_clusters} clusters")
    
    # 将参数合并到配置对象中
    config.input_dims = input_dims
    model = MVGCNModel(config).to(device)
    
    # 在train.py中添加
    if torch.cuda.device_count() > 1:
        logger.info(f"使用 {torch.cuda.device_count()} 个GPU进行训练")
        model = nn.DataParallel(model)
    
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    # 定义优化器后添加以下代码：
    from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts

    # 学习率调度器 - 根据验证性能动态调整学习率
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='max',          # 因为我们监控NMI，越高越好
        factor=0.5,          # 学习率减半
        patience=5,          # 5个周期无改善则减小学习率
        verbose=True,
        threshold=0.001      # 仅当改善超过0.1%时才算有效改善
    )
    
    # 创建保存目录
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    
    # 如果提供了自定义保存名称，则使用它
    model_save_name = save_name if save_name else config.dataset
    
    # 训练循环
    patience = 10
    best_nmi = 0
    counter = 0
    logger.info("Starting training...")
    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad()
        
        # 生成掩码
        if config.adaptive_mask:
            # 使用动态自适应掩码
            masks = []
            for view in data_views:
                masked_view, mask = dynamic_adaptive_masking(
                    view, 
                    epoch=epoch,
                    total_epochs=config.epochs,
                    initial_mask_rate=config.mask_rate * 1.5,
                    final_mask_rate=config.mask_rate * 0.5
                )
                masks.append(mask.to(device))
        else:
            # 使用随机掩码
            masks = [torch.FloatTensor(v.shape).uniform_() > config.mask_rate for v in data_views]
            masks = [m.to(device) for m in masks]
        
        # 前向传播 - 添加异常处理
        try:
            # 修改模型调用方式，使用正确的参数名称
            outputs = model(data_views, adj_matrices, mask_rate=config.mask_rate)  # 使用关键字参数传递
            
            # 计算重构损失
            recon_loss = 0
            for i, (recon, original) in enumerate(zip(outputs['reconstructions'], data_views)):
                recon_loss += nn.MSELoss()(recon, original)
            
            # 计算自监督任务损失
            self_supervised_loss = compute_self_supervised_loss(outputs, data_views)

            # 计算对比损失
            contrastive_loss = compute_contrastive_loss(
                outputs['fused_embedding'], 
                outputs['view_embeddings'], 
                config.temperature
            )

            # 总损失 = 重构损失 + 对比损失 + 自监督任务损失
            total_loss = recon_loss + config.contrastive_weight * contrastive_loss + self_supervised_loss
            
            # 反向传播和优化
            total_loss.backward()
            # 添加梯度裁剪以提高稳定性
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        except RuntimeError as e:
            # 处理可能的运行时错误
            if "out of memory" in str(e):
                logger.error(f"内存不足错误: {e}")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                continue
            elif "mat1 and mat2 shapes cannot be multiplied" in str(e):
                logger.error(f"矩阵形状不兼容: {e}")
                # 跳过当前迭代
                continue
            else:
                logger.error(f"训练过程中出现错误: {e}")
                raise e
        
        # 评估模型性能（每隔一定epoch）
        if (epoch + 1) % config.eval_interval == 0 or epoch == config.epochs - 1:
            model.eval()
            with torch.no_grad():
                cluster_features = model.get_cluster_features(data_views, adj_matrices)
                cluster_features = cluster_features.cpu().numpy()
                
                # K-means聚类
                kmeans = KMeans(n_clusters=n_clusters, random_state=config.seed)
                cluster_assignments = kmeans.fit_predict(cluster_features)
                
                # 评估指标
                if labels is not None:
                    nmi = normalized_mutual_info_score(labels.cpu().numpy(), cluster_assignments)
                    ari = adjusted_rand_score(labels.cpu().numpy(), cluster_assignments)
                    
                    # 根据性能调整学习率
                    scheduler.step(nmi)
                    
                    # 记录当前学习率
                    current_lr = optimizer.param_groups[0]['lr']
                    logger.info(f"Current learning rate: {current_lr}")
                    
                    if nmi > best_nmi + 0.001:  # 添加最小改善阈值
                        best_nmi = nmi
                        counter = 0
                        
                        # 只保存必要组件，减少跨版本兼容性问题
                        save_dict = {
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'nmi': float(nmi),  # 确保使用Python原生类型
                            'ari': float(ari)   # 确保使用Python原生类型
                        }
                        
                        # 使用自定义保存名称
                        torch.save(save_dict, f"checkpoints/{model_save_name}_best_model.pth")
                        logger.info(f"Saved best model with NMI: {nmi:.4f}")
                    else:
                        counter += 1
                        
                    if counter >= patience:
                        logger.info(f"早停: {patience}个周期内性能未提升")
                        break
                else:
                    logger.info(f"Epoch {epoch+1}/{config.epochs}, Loss: {total_loss.item():.4f}")
    
    logger.info("Training completed!")
    return model

def compute_self_supervised_loss(outputs, data_views):
    """计算额外的自监督任务损失"""
    # 1. 特征聚类一致性损失
    cluster_features = outputs['cluster_rep']
    # 使用K-means计算聚类中心
    with torch.no_grad():
        cluster_centers = torch.tensor(KMeans(n_clusters=5).fit(
            cluster_features.detach().cpu().numpy()
        ).cluster_centers_, device=cluster_features.device)
    
    # 计算样本到聚类中心的距离
    dist = torch.cdist(cluster_features, cluster_centers, p=2)
    # 获取最近的聚类中心
    _, cluster_idx = torch.min(dist, dim=1)
    # 计算聚类一致性损失
    cluster_centers_assigned = cluster_centers[cluster_idx]
    clustering_loss = F.mse_loss(cluster_features, cluster_centers_assigned)
    
    # 2. 视图重建一致性损失
    recon_consistency_loss = 0
    for i in range(len(data_views) - 1):
        for j in range(i+1, len(data_views)):
            # 确保维度匹配
            if outputs['reconstructions'][i].shape[1] == outputs['reconstructions'][j].shape[1]:
                recon_consistency_loss += F.mse_loss(
                    outputs['reconstructions'][i], 
                    outputs['reconstructions'][j]
                )
    
    return clustering_loss + 0.1 * recon_consistency_loss  # 权重可调

# 在MultiViewAutoencoder类中添加自监督模块
class MultiViewAutoencoder(nn.Module):
    """多视图自编码器模型"""
    
    def __init__(self, input_dims, hidden_dims, latent_dim, dropout=0.5):
        super(MultiViewAutoencoder, self).__init__()
        
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.n_views = len(input_dims)
        
        # 为每个视图创建编码器和解码器
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        for i, dim in enumerate(input_dims):
            # 编码器 (多层)
            layers = []
            input_size = dim
            
            for hdim in hidden_dims:
                layers.append(nn.Linear(input_size, hdim))
                layers.append(nn.BatchNorm1d(hdim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                input_size = hdim
                
            # 最后一层映射到潜在空间
            layers.append(nn.Linear(hidden_dims[-1], latent_dim))
            self.encoders.append(nn.Sequential(*layers))
            
            # 解码器 (反向层次结构)
            decoder_layers = []
            decoder_dims = list(reversed([*hidden_dims, dim]))
            
            input_size = latent_dim
            for j, hdim in enumerate(decoder_dims[:-1]):
                decoder_layers.append(nn.Linear(input_size, hdim))
                decoder_layers.append(nn.BatchNorm1d(hdim))
                decoder_layers.append(nn.ReLU())
                if j < len(decoder_dims) - 2:  # 不在最后一层添加dropout
                    decoder_layers.append(nn.Dropout(dropout))
                input_size = hdim
                
            decoder_layers.append(nn.Linear(decoder_dims[-2], decoder_dims[-1]))
            self.decoders.append(nn.Sequential(*decoder_layers))
        
        # 视图融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(latent_dim * self.n_views, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 投影头 (用于对比学习)
        self.projection_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.BatchNorm1d(latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, latent_dim // 4)
        )
        
        # 添加自监督学习模块
        self.self_supervision = MultiViewSelfSupervision(
            input_dims=input_dims,
            hidden_dim=hidden_dims[0],
            proj_dim=latent_dim // 2,
            task_weights={'reconstruction': 1.0, 'clustering': 0.5, 'rotation': 0.3},
            fusion_type='attention'
        )
        
    def forward(self, views, masks=None):
        """前向传播
        
        参数:
            views: 视图列表
            masks: 特征掩码列表 (可选)
            
        返回:
            包含各种输出的字典
        """
        batch_size = views[0].size(0)
        device = views[0].device
        
        # 如果没有提供掩码，创建默认掩码(全1)
        if masks is None:
            masks = [torch.ones_like(v) for v in views]
        
        # 应用掩码并获取各视图的嵌入
        view_embeddings = []
        masked_views = []
        
        for i, (view, mask, encoder) in enumerate(zip(views, masks, self.encoders)):
            # 应用掩码
            masked_view = view * mask
            masked_views.append(masked_view)
            
            # 编码
            embedding = encoder(masked_view)
            view_embeddings.append(embedding)
        
        # 融合视图嵌入
        concat_embedding = torch.cat(view_embeddings, dim=1)
        fused_embedding = self.fusion_layer(concat_embedding)
        
        # 投影用于对比学习
        projection = self.projection_head(fused_embedding)
        
        # 重构各视图
        reconstructions = []
        for i, (embedding, decoder) in enumerate(zip(view_embeddings, self.decoders)):
            reconstruction = decoder(embedding)
            reconstructions.append(reconstruction)
        
        # 返回结果
        return {
            'fused_embedding': fused_embedding,
            'view_embeddings': view_embeddings,
            'reconstructions': reconstructions,
            'projection': projection,
            'masked_views': masked_views
        }
    
    def get_cluster_features(self, views, masks=None):
        """获取用于聚类的特征表示"""
        outputs = self.forward(views, masks)
        return outputs['fused_embedding']

# 在train函数中添加自监督学习
def train(model, optimizer, scheduler, data_views, adj_matrices, labels, device, config):
    # 现有代码...
    
    # 设置自监督模块的聚类数
    model.self_supervision.set_n_clusters(config.n_clusters)
    
    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(data_views, adj_matrices, mask_rate=config.mask_rate)
        
        # 计算重构损失
        recon_loss = compute_reconstruction_loss(outputs['reconstructions'], data_views)
        
        # 计算对比损失
        contrast_loss = compute_contrastive_loss(outputs['fused_embedding'], 
                                               outputs['view_embeddings'], 
                                               config.temperature)
        
        # 计算自监督损失
        self_sup_loss, loss_components = model.self_supervision.compute_self_supervised_loss(
            data_views, cluster_centers=outputs.get('cluster_centers', None))
        
        # 总损失 = 重构损失 + 对比损失 + 自监督损失
        loss = recon_loss + config.contrastive_weight * contrast_loss + 0.5 * self_sup_loss
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # 日志记录
        # ...

def compute_reconstruction_loss(reconstructions, originals):
    """计算重构损失
    
    参数:
        reconstructions: 重构后的视图列表
        originals: 原始视图列表
        
    返回:
        重构损失 (MSE)
    """
    loss = 0
    for recon, orig in zip(reconstructions, originals):
        loss += F.mse_loss(recon, orig)
    return loss / len(reconstructions)  # 平均每个视图的损失


def compute_contrastive_loss(fused_embedding, view_embeddings, temperature=0.5):
    """计算对比损失
    
    参数:
        fused_embedding: 融合后的嵌入，形状 [batch_size, dim]
        view_embeddings: 各视图的嵌入列表，每个形状 [batch_size, dim]
        temperature: 温度参数
        
    返回:
        对比损失
    """
    device = fused_embedding.device
    batch_size = fused_embedding.shape[0]
    
    # 归一化嵌入
    fused_embedding = F.normalize(fused_embedding, dim=1)
    norm_view_embeddings = [F.normalize(ve, dim=1) for ve in view_embeddings]
    
    # 1. 计算融合嵌入与各视图嵌入之间的对比损失
    fusion_contrast_loss = 0
    for view_emb in norm_view_embeddings:
        # 计算相似度矩阵
        sim_matrix = torch.mm(fused_embedding, view_emb.t()) / temperature
        
        # 对角线上的元素是正样本对
        labels = torch.arange(batch_size, device=device)
        
        # 计算InfoNCE损失 (双向)
        loss_fusion_to_view = F.cross_entropy(sim_matrix, labels)
        loss_view_to_fusion = F.cross_entropy(sim_matrix.t(), labels)
        
        fusion_contrast_loss += (loss_fusion_to_view + loss_view_to_fusion) / 2
    
    # 2. 计算各视图嵌入之间的对比损失
    view_contrast_loss = 0
    n_views = len(norm_view_embeddings)
    
    for i in range(n_views):
        for j in range(i+1, n_views):
            # 计算相似度矩阵
            sim_matrix = torch.mm(norm_view_embeddings[i], norm_view_embeddings[j].t()) / temperature
            
            # 对角线上的元素是正样本对
            labels = torch.arange(batch_size, device=device)
            
            # 计算InfoNCE损失 (双向)
            loss_i_to_j = F.cross_entropy(sim_matrix, labels)
            loss_j_to_i = F.cross_entropy(sim_matrix.t(), labels)
            
            view_contrast_loss += (loss_i_to_j + loss_j_to_i) / 2
    
    # 计算平均值以平衡两种对比损失
    if n_views > 1:
        view_contrast_loss = view_contrast_loss / (n_views * (n_views - 1) / 2)
    
    # 总对比损失 = 视图间对比损失 + 视图-融合对比损失
    return fusion_contrast_loss / n_views + view_contrast_loss

def load_model(config, model_path=None):
    """加载预训练模型

    参数:
        config: 配置对象
        model_path: 模型路径（可选）

    返回:
        加载的模型
    """
    # 获取维度信息
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
    
    if model_path and os.path.exists(model_path):
        try:
            # 尝试直接加载
            checkpoint = torch.load(model_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"从 {model_path} 加载模型成功")
        except Exception as e:
            print(f"加载模型失败: {e}，使用未训练的模型")
    else:
        print(f"模型文件 {model_path} 不存在，使用未训练的模型")
    
    return model
