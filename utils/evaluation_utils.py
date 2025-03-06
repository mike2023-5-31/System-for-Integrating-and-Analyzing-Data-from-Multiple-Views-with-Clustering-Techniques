"""评估指标工具"""

import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment

def cluster_accuracy(y_true, y_pred):
    """计算聚类准确率
    
    参数:
        y_true: 真实标签
        y_pred: 预测标签
        
    返回:
        准确率
    """
    contingency_matrix = np.zeros((np.max(y_true) + 1, np.max(y_pred) + 1))
    for i in range(len(y_true)):
        contingency_matrix[y_true[i], y_pred[i]] += 1
        
    # 使用匈牙利算法找到最佳匹配
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
    
    # 计算准确率
    accuracy = contingency_matrix[row_ind, col_ind].sum() / len(y_true)
    return accuracy
