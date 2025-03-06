import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SelfSupervisedModule(nn.Module):
    """自监督学习模块，集成多种自监督学习任务"""
    
    def __init__(self, input_dim, hidden_dim=256, proj_dim=128, 
                 task_weights={'reconstruction': 1.0, 'clustering': 0.5, 'rotation': 0.3}):
        """
        参数:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            proj_dim: 投影维度 (用于对比学习)
            task_weights: 不同自监督任务的权重
        """
        super(SelfSupervisedModule, self).__init__()
        
        self.task_weights = task_weights
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.proj_dim = proj_dim
        
        # 特征编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU()
        )
        
        # 特征解码器 (用于重构任务)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # 旋转预测头 (用于旋转预测任务)
        if 'rotation' in self.task_weights and self.task_weights['rotation'] > 0:
            self.rotation_head = nn.Sequential(
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.ReLU(),
                nn.Linear(hidden_dim // 4, 4)  # 预测4种旋转角度: 0, 90, 180, 270度
            )
        
        # 投影头 (用于对比学习和聚类任务)
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, proj_dim)
        )
        
        # 聚类头 (用于聚类任务)
        if 'clustering' in self.task_weights and self.task_weights['clustering'] > 0:
            self.clustering_head = nn.Sequential(
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.ReLU()
                # 最终聚类层会动态设置，基于数据集的聚类数量
            )
            self.n_clusters = None  # 会在训练时设置
        
    def set_n_clusters(self, n_clusters):
        """设置聚类头的聚类数量"""
        if 'clustering' in self.task_weights and self.task_weights['clustering'] > 0:
            # 动态创建聚类层
            self.clustering_layer = nn.Linear(self.hidden_dim // 4, n_clusters)
            self.n_clusters = n_clusters
    
    def encode(self, x):
        """编码输入特征"""
        return self.encoder(x)
    
    def project(self, h):
        """投影特征用于对比学习"""
        return F.normalize(self.projector(h), dim=1)
    
    def forward(self, x, return_all=False):
        """前向传播
        
        参数:
            x: 输入特征
            return_all: 是否返回所有中间特征
            
        返回:
            如果return_all=True, 返回一个字典包含所有特征
            否则，返回编码后的特征h
        """
        h = self.encoder(x)
        z = self.projector(h)
        
        if return_all:
            output = {'h': h, 'z': z}
            
            # 重构任务
            if 'reconstruction' in self.task_weights and self.task_weights['reconstruction'] > 0:
                x_hat = self.decoder(h)
                output['x_hat'] = x_hat
            
            # 聚类任务
            if 'clustering' in self.task_weights and self.task_weights['clustering'] > 0 and self.n_clusters is not None:
                cluster_logits = self.clustering_layer(self.clustering_head(h))
                output['cluster_logits'] = cluster_logits
            
            return output
        else:
            return h
    
    def compute_rotation_labels_and_features(self, x, device):
        """计算旋转标签和特征
        
        参数:
            x: 输入特征 [batch_size, feature_dim]
            device: 设备
            
        返回:
            旋转后的特征和对应的标签
        """
        if 'rotation' not in self.task_weights or self.task_weights['rotation'] <= 0:
            return None, None
            
        batch_size = x.size(0)
        # 创建4种旋转角度的输入
        rotated_x = []
        rotation_labels = []
        
        # 检查输入维度是否可以reshape为图像
        feature_dim = x.size(1)
        # 假设特征可以近似reshape为方形图像
        side_length = int(np.sqrt(feature_dim))
        
        if side_length * side_length == feature_dim:
            # 可以reshape为方形图像
            reshaped_x = x.view(batch_size, 1, side_length, side_length)
            
            # 0度旋转 (原始)
            rotated_x.append(x)
            rotation_labels.extend([0] * batch_size)
            
            # 90度旋转
            rot90 = torch.rot90(reshaped_x, k=1, dims=[2, 3]).reshape(batch_size, -1)
            rotated_x.append(rot90)
            rotation_labels.extend([1] * batch_size)
            
            # 180度旋转
            rot180 = torch.rot90(reshaped_x, k=2, dims=[2, 3]).reshape(batch_size, -1)
            rotated_x.append(rot180)
            rotation_labels.extend([2] * batch_size)
            
            # 270度旋转
            rot270 = torch.rot90(reshaped_x, k=3, dims=[2, 3]).reshape(batch_size, -1)
            rotated_x.append(rot270)
            rotation_labels.extend([3] * batch_size)
            
        else:
            # 不能直接reshape为方形图像，使用特征重排
            # 0度旋转 (原始)
            rotated_x.append(x)
            rotation_labels.extend([0] * batch_size)
            
            # 模拟90度旋转，重排特征
            rot90 = x.flip(1)
            rotated_x.append(rot90)
            rotation_labels.extend([1] * batch_size)
            
            # 模拟180度旋转
            rot180 = torch.roll(x, shifts=feature_dim//2, dims=1)
            rotated_x.append(rot180)
            rotation_labels.extend([2] * batch_size)
            
            # 模拟270度旋转
            rot270 = torch.roll(rot90, shifts=feature_dim//2, dims=1)
            rotated_x.append(rot270)
            rotation_labels.extend([3] * batch_size)
        
        rotated_x = torch.cat(rotated_x, dim=0)
        rotation_labels = torch.tensor(rotation_labels, device=device)
        
        return rotated_x, rotation_labels
    
    def compute_self_supervised_loss(self, x, cluster_centers=None):
        """计算自监督学习损失
        
        参数:
            x: 输入特征
            cluster_centers: 聚类中心（可选）
            
        返回:
            损失值和损失组件字典
        """
        device = x.device
        batch_size = x.size(0)
        
        # 获取所有特征表示
        outputs = self.forward(x, return_all=True)
        h = outputs['h']
        z = outputs['z']
        
        losses = {}
        
        # 重构损失
        if 'reconstruction' in self.task_weights and self.task_weights['reconstruction'] > 0:
            x_hat = outputs['x_hat']
            recon_loss = F.mse_loss(x_hat, x)
            losses['reconstruction'] = recon_loss
        
        # 旋转预测损失
        if 'rotation' in self.task_weights and self.task_weights['rotation'] > 0:
            rotated_x, rotation_labels = self.compute_rotation_labels_and_features(x, device)
            if rotated_x is not None:
                rotated_h = self.encoder(rotated_x)
                rotation_preds = self.rotation_head(rotated_h)
                rotation_loss = F.cross_entropy(rotation_preds, rotation_labels)
                losses['rotation'] = rotation_loss
        
        # 聚类损失
        if 'clustering' in self.task_weights and self.task_weights['clustering'] > 0 and self.n_clusters is not None:
            if 'cluster_logits' in outputs:
                # 如果提供了聚类中心，计算与聚类中心的KL散度
                if cluster_centers is not None:
                    # 计算每个样本到聚类中心的软分配
                    z_norm = F.normalize(z, dim=1)
                    centers_norm = F.normalize(cluster_centers, dim=1)
                    
                    # 计算余弦相似度
                    sim = torch.mm(z_norm, centers_norm.t())
                    sim = (sim + 1) / 2  # 从[-1,1]缩放到[0,1]
                    
                    # 计算软分配 (使用温度参数调整分布)
                    temperature = 0.1
                    q = (sim / temperature).softmax(dim=1)
                    
                    # 计算聚类预测的分布
                    cluster_preds = outputs['cluster_logits'].softmax(dim=1)
                    
                    # KL散度损失
                    clustering_loss = F.kl_div(
                        cluster_preds.log(),
                        q.detach(),
                        reduction='batchmean'
                    )
                    losses['clustering'] = clustering_loss
                else:
                    # 如果没有聚类中心，计算自监督聚类损失
                    # 使用z的内积作为软标签
                    z_norm = F.normalize(z, dim=1)
                    sim_matrix = torch.mm(z_norm, z_norm.t())
                    
                    # 移除对角线
                    mask = torch.ones_like(sim_matrix) - torch.eye(batch_size, device=device)
                    sim_matrix = sim_matrix * mask
                    
                    # 找出每个样本最相似的k个样本作为伪标签
                    k = min(10, batch_size - 1)
                    _, topk_indices = torch.topk(sim_matrix, k=k, dim=1)
                    
                    # 计算伪标签
                    pseudo_labels = torch.zeros(batch_size, batch_size, device=device)
                    for i in range(batch_size):
                        pseudo_labels[i, topk_indices[i]] = 1.0 / k
                    
                    # 计算聚类预测
                    cluster_logits = outputs['cluster_logits']
                    cluster_probs = F.softmax(cluster_logits, dim=1)
                    
                    # 计算聚类损失
                    clustering_loss = F.kl_div(
                        torch.mm(cluster_probs, cluster_probs.t()).log(),
                        pseudo_labels,
                        reduction='batchmean'
                    )
                    losses['clustering'] = clustering_loss
        
        # 计算总损失
        total_loss = sum(self.task_weights[task] * loss 
                          for task, loss in losses.items() 
                          if task in self.task_weights)
        
        return total_loss, losses

class MultiViewSelfSupervision(nn.Module):
    """多视图自监督学习模块，为每个视图实现自监督任务"""
    
    def __init__(self, input_dims, hidden_dim=256, proj_dim=128, 
                 task_weights={'reconstruction': 1.0, 'clustering': 0.5, 'rotation': 0.3},
                 fusion_type='concatenation'):
        """
        参数:
            input_dims: 每个视图的输入维度列表
            hidden_dim: 隐藏层维度
            proj_dim: 投影维度
            task_weights: 不同任务的权重
            fusion_type: 视图融合类型，可选'concatenation','attention','weighted_sum'
        """
        super(MultiViewSelfSupervision, self).__init__()
        
        self.n_views = len(input_dims)
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.proj_dim = proj_dim
        self.task_weights = task_weights
        self.fusion_type = fusion_type
        
        # 为每个视图创建自监督模块
        self.view_modules = nn.ModuleList([
            SelfSupervisedModule(
                input_dim=dim,
                hidden_dim=hidden_dim,
                proj_dim=proj_dim,
                task_weights=task_weights
            ) for dim in input_dims
        ])
        
        # 视图融合模块
        if fusion_type == 'attention':
            # 基于注意力的视图融合
            self.attention_weights = nn.Parameter(torch.ones(self.n_views) / self.n_views)
            self.attention_projection = nn.Linear(hidden_dim // 2, 1)
        elif fusion_type == 'weighted_sum':
            # 可学习权重的加权和融合
            self.view_weights = nn.Parameter(torch.ones(self.n_views) / self.n_views)
        elif fusion_type == 'concatenation':
            # 视图拼接后的映射
            self.fusion_mapping = nn.Linear(self.n_views * (hidden_dim // 2), hidden_dim // 2)
        
        # 共享的对比学习投影头
        self.contrastive_projector = nn.Sequential(
            nn.Linear(hidden_dim // 2, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )
        
    def set_n_clusters(self, n_clusters):
        """为所有视图模块设置聚类数量"""
        for module in self.view_modules:
            module.set_n_clusters(n_clusters)
    
    def fuse_representations(self, view_features):
        """融合不同视图的特征表示"""
        if self.fusion_type == 'attention':
            # 注意力机制融合
            attention_scores = []
            for i, h in enumerate(view_features):
                score = self.attention_projection(h)
                attention_scores.append(score)
            
            attention_scores = torch.cat(attention_scores, dim=1)
            attention_weights = F.softmax(attention_scores, dim=1)
            
            # 加权求和
            fused_feature = 0
            for i, h in enumerate(view_features):
                fused_feature += h * attention_weights[:, i:i+1]
            
        elif self.fusion_type == 'weighted_sum':
            # 使用可学习权重的加权和
            view_weights = F.softmax(self.view_weights, dim=0)
            
            fused_feature = 0
            for i, h in enumerate(view_features):
                fused_feature += h * view_weights[i]
                
        elif self.fusion_type == 'concatenation':
            # 拼接后映射
            fused_feature = torch.cat(view_features, dim=1)
            fused_feature = self.fusion_mapping(fused_feature)
            
        else:
            # 默认使用平均融合
            fused_feature = torch.stack(view_features, dim=0).mean(dim=0)
            
        return fused_feature
    
    def forward_single_view(self, x, view_idx, return_all=False):
        """处理单个视图的前向传播"""
        return self.view_modules[view_idx](x, return_all=return_all)
    
    def forward(self, views, return_all=False):
        """多视图前向传播
        
        参数:
            views: 不同视图的特征列表
            return_all: 是否返回所有中间表示
            
        返回:
            如果return_all=True，返回包含所有视图特征和融合特征的字典
            否则返回融合后的特征
        """
        # 处理每个视图
        view_features = []
        view_outputs = []
        
        for i, (view, module) in enumerate(zip(views, self.view_modules)):
            if return_all:
                output = module(view, return_all=True)
                view_features.append(output['h'])
                view_outputs.append(output)
            else:
                h = module(view, return_all=False)
                view_features.append(h)
        
        # 融合视图特征
        fused_feature = self.fuse_representations(view_features)
        
        # 计算对比学习的投影
        fused_projection = F.normalize(self.contrastive_projector(fused_feature), dim=1)
        
        if return_all:
            return {
                'view_features': view_features,
                'view_outputs': view_outputs,
                'fused_feature': fused_feature,
                'fused_projection': fused_projection
            }
        else:
            return fused_feature
    
    def compute_contrastive_loss(self, views, temperature=0.5):
        """计算多视图对比损失
        
        参数:
            views: 不同视图的特征列表
            temperature: 温度参数
            
        返回:
            对比损失
        """
        outputs = self.forward(views, return_all=True)
        
        batch_size = views[0].size(0)
        device = views[0].device
        
        # 获取每个视图的特征和融合特征的投影
        view_projections = []
        for i, view_output in enumerate(outputs['view_outputs']):
            h = view_output['h']
            z = self.view_modules[i].project(h)
            view_projections.append(z)
        
        fused_projection = outputs['fused_projection']
        
        # 计算多视图对比损失
        total_contrastive_loss = 0
        
        # 1. 不同视图之间的对比损失
        for i in range(self.n_views):
            for j in range(i+1, self.n_views):
                z_i = view_projections[i]
                z_j = view_projections[j]
                
                # 计算相似度矩阵
                sim_matrix = torch.mm(z_i, z_j.t()) / temperature
                
                # 对角线上的元素是正样本对
                positive_samples = torch.arange(batch_size, device=device)
                
                # InfoNCE loss
                loss_i_to_j = F.cross_entropy(sim_matrix, positive_samples)
                loss_j_to_i = F.cross_entropy(sim_matrix.t(), positive_samples)
                
                # 添加到总损失
                total_contrastive_loss += (loss_i_to_j + loss_j_to_i) / 2
        
        # 2. 各视图与融合表示之间的对比损失
        for i in range(self.n_views):
            z_i = view_projections[i]
            
            # 计算与融合表示的相似度
            sim_matrix = torch.mm(z_i, fused_projection.t()) / temperature
            
            # 对角线上的元素是正样本对
            positive_samples = torch.arange(batch_size, device=device)
            
            # InfoNCE loss
            loss_i_to_fused = F.cross_entropy(sim_matrix, positive_samples)
            loss_fused_to_i = F.cross_entropy(sim_matrix.t(), positive_samples)
            
            # 添加到总损失
            total_contrastive_loss += (loss_i_to_fused + loss_fused_to_i) / 2
        
        # 视角数量归一化
        contrastive_factor = self.n_views * (self.n_views - 1) / 2 + self.n_views
        contrastive_loss = total_contrastive_loss / contrastive_factor
        
        return contrastive_loss
    
    def compute_self_supervised_loss(self, views, cluster_centers=None):
        """计算自监督损失
        
        参数:
            views: 不同视图的特征列表
            cluster_centers: 可选的聚类中心
            
        返回:
            总损失和损失组件字典
        """
        # 单视图自监督损失
        view_losses = []
        loss_components = {}
        
        for i, (view, module) in enumerate(zip(views, self.view_modules)):
            view_loss, view_loss_components = module.compute_self_supervised_loss(
                view, cluster_centers)
            view_losses.append(view_loss)
            
            # 收集每个视图的损失组件
            for task, loss in view_loss_components.items():
                if f'view{i+1}_{task}' not in loss_components:
                    loss_components[f'view{i+1}_{task}'] = loss
        
        # 计算多视图对比损失
        contrastive_loss = self.compute_contrastive_loss(views)
        loss_components['contrastive'] = contrastive_loss
        
        # 融合所有损失
        total_loss = sum(view_losses) / len(view_losses) + contrastive_loss
        
        return total_loss, loss_components