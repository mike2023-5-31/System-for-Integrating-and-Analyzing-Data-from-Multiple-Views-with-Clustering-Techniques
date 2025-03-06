"""
图卷积网络层定义
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    """
    简单的GCN层实现
    """
    def __init__(self, in_features, out_features, bias=True, activation=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.activation = activation
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
        
    def reset_parameters(self):
        """初始化参数"""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
            
    def forward(self, x, adj):
        """
        前向传播
        
        参数:
            x: 节点特征矩阵 [batch_size, in_features]
            adj: 邻接矩阵 [batch_size, batch_size]
            
        返回:
            输出特征
        """
        support = torch.mm(x, self.weight)
        output = torch.mm(adj, support)
        
        if self.bias is not None:
            output += self.bias
            
        if self.activation is not None:
            output = self.activation(output)
            
        return output
