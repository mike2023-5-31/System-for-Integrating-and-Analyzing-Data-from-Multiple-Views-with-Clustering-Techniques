"""
数据集辅助工具，用于检测和管理数据集
"""

import os
import json
from pathlib import Path

def detect_available_twitter_subdatasets(twitter_data_dir='data/multiview_twitter/multiview_data_20130124'):
    """
    检测实际可用的Twitter子数据集
    
    参数:
        twitter_data_dir: Twitter数据集解压后的目录
        
    返回:
        list: 可用的Twitter子数据集列表
    """
    if not os.path.exists(twitter_data_dir):
        return []
    
    # 查找子目录，每个子目录对应一个子数据集
    subdatasets = []
    for item in os.listdir(twitter_data_dir):
        item_path = os.path.join(twitter_data_dir, item)
        # 检查是否为目录且包含必要的数据文件
        if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, 'communities.dat')):
            # 处理特殊情况
            if item == 'politicsie':
                subdatasets.append('politics-ie')
            elif item == 'politicsuk':
                subdatasets.append('politics-uk')
            else:
                subdatasets.append(item)
    
    # 将子数据集名称转换为完整数据集名称
    twitter_datasets = [f'twitter_{subdataset}' for subdataset in subdatasets]
    
    # 创建可用数据集的JSON配置文件
    output_dir = os.path.dirname(os.path.dirname(twitter_data_dir))  # 通常是'data'目录
    json_path = os.path.join(output_dir, 'available_twitter_datasets.json')
    
    try:
        with open(json_path, 'w') as f:
            json.dump({"available_datasets": subdatasets}, f, indent=2)
        print(f"已更新可用数据集配置: {json_path}")
    except Exception as e:
        print(f"警告: 无法创建数据集配置: {e}")
    
    return twitter_datasets

def update_twitter_datasets_config():
    """
    更新Twitter数据集配置文件
    使用方法：在下载并解压数据集后调用此函数
    """
    available_datasets = detect_available_twitter_subdatasets()
    if available_datasets:
        print(f"检测到以下可用的Twitter数据集:")
        for i, dataset in enumerate(available_datasets, 1):
            print(f"  {i}. {dataset}")
    else:
        print("未检测到任何Twitter数据集，请确保数据集已正确下载和解压")
    
    return available_datasets

if __name__ == "__main__":
    # 直接运行此脚本将更新Twitter数据集配置
    update_twitter_datasets_config()
