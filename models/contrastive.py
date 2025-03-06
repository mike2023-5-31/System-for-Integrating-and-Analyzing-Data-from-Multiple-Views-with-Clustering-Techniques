# Description: 对比学习模块
import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLearningModule(nn.Module):
    """
    跨视图对比学习模块
    """
    def __init__(self, feature_dim, projection_dim, temperature=0.5):
        super(ContrastiveLearningModule, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, projection_dim)
        )
        self.temperature = temperature
        
    def forward(self, features_view1, features_view2):
        # 将特征投影到单位球上
        z1 = F.normalize(self.projection(features_view1), dim=1)
        z2 = F.normalize(self.projection(features_view2), dim=1)
        
        # 计算余弦相似度
        similarity_matrix = torch.matmul(z1, z2.T) / self.temperature
        
        # 计算对比损失
        labels = torch.arange(similarity_matrix.size(0)).to(similarity_matrix.device)
        loss = F.cross_entropy(similarity_matrix, labels) + F.cross_entropy(similarity_matrix.T, labels)
        return loss / 2.0
    
class MultiViewContrastiveLearning(nn.Module):
    """
    支持多视图的对比学习
    """
    def __init__(self, feature_dim, projection_dim, temperature=0.5):
        super(MultiViewContrastiveLearning, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, projection_dim)
        )
        self.temperature = temperature
        
    def forward(self, features_list):
        """
        多视图之间进行对比学习
        
        参数:
        features_list: 多个视图的特征列表
        
        返回:
        loss: 总的对比损失
        """
        n_views = len(features_list)
        total_loss = 0
        
        # 对每对视图进行对比学习
        for i in range(n_views):
            for j in range(i+1, n_views):
                z_i = F.normalize(self.projection(features_list[i]), dim=1)
                z_j = F.normalize(self.projection(features_list[j]), dim=1)
                
                # 计算余弦相似度
                similarity = torch.matmul(z_i, z_j.T) / self.temperature
                
                # 计算对比损失
                labels = torch.arange(similarity.size(0)).to(similarity.device)
                loss_ij = F.cross_entropy(similarity, labels)
                loss_ji = F.cross_entropy(similarity.T, labels)
                
                total_loss += (loss_ij + loss_ji) / 2
                
        # 平均多对视图的损失
        return total_loss / (n_views * (n_views - 1) / 2)
    
    def project(self, features):
        """将特征投影到对比学习空间"""
        return F.normalize(self.projection(features), dim=1)

class ContrastiveLoss(nn.Module):
    """对比损失函数"""
    
    def __init__(self, temperature=0.5, base_temperature=0.07):
        """
        初始化对比损失
        
        参数:
            temperature: 温度参数，控制相似度分布的平滑度
            base_temperature: 基准温度，用于归一化
        """
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        
    def forward(self, features, mask=None):
        """
        计算对比损失
        
        参数:
            features: 特征向量 [batch_size, feature_dim]
            mask: 正样本对掩码 [batch_size, batch_size]，可选
            
        返回:
            对比损失
        """
        batch_size = features.shape[0]
        
        # 特征归一化
        features = F.normalize(features, dim=1)
        
        # 计算相似度矩阵
        sim_matrix = torch.matmul(features, features.T) / self.temperature
        
        # 对角线元素是自身相似度，排除
        sim_matrix.fill_diagonal_(-float('inf'))
        
        # 如果没有提供掩码，则使用对角线掩码
        if mask is None:
            mask = torch.eye(batch_size, device=features.device)
        
        # 计算正样本对和负样本对
        pos_sim = sim_matrix * mask
        neg_sim = sim_matrix * (1 - mask)
        
        # 对每个样本，计算正样本对的平均相似度
        pos_sim = torch.sum(torch.exp(pos_sim), dim=1)
        neg_sim = torch.sum(torch.exp(neg_sim), dim=1)
        
        # 计算InfoNCE损失
        loss = -torch.log(pos_sim / (pos_sim + neg_sim))
        loss = torch.mean(loss)
        
        return loss