"""
检查Twitter政治相关数据集的可用性
"""

import os
import sys
import argparse
from utils.dataset_utils import detect_available_twitter_subdatasets
from utils.data_loader import download_dataset

def check_politics_datasets():
    """检查政治相关数据集是否可用"""
    # 确保数据目录存在
    os.makedirs('data', exist_ok=True)
    
    # 下载Twitter数据集
    print("正在下载Twitter多视图数据集...")
    success = download_dataset('multiview_twitter', 'data')
    if not success:
        print("下载Twitter数据集失败")
        return False
    
    # 检测可用的子数据集
    print("检测可用的Twitter子数据集...")
    twitter_data_dir = 'data/multiview_twitter/multiview_data_20130124'
    
    if not os.path.exists(twitter_data_dir):
        print(f"目录 {twitter_data_dir} 不存在，可能数据集结构发生了变化")
        return False
    
    # 手动检查政治相关子目录
    politics_dirs = ["politicsie", "politicsuk"]
    found_datasets = []
    
    for politics_dir in politics_dirs:
        dir_path = os.path.join(twitter_data_dir, politics_dir)
        if os.path.isdir(dir_path):
            # 检查是否包含必要的数据文件
            if os.path.exists(os.path.join(dir_path, 'communities.dat')):
                found_datasets.append(politics_dir)
                print(f"找到政治子数据集: {politics_dir}")
            else:
                print(f"找到子目录 {politics_dir}，但缺少必要的数据文件")
        else:
            print(f"未找到子目录: {politics_dir}")
    
    # 输出结果
    if found_datasets:
        print("\n可用的Twitter政治数据集:")
        for i, dataset in enumerate(found_datasets, 1):
            hyphenated_name = dataset
            if dataset == "politicsie":
                hyphenated_name = "politics-ie"
            elif dataset == "politicsuk":
                hyphenated_name = "politics-uk"
            
            print(f"  {i}. twitter_{hyphenated_name}")
        
        # 更新TWITTER_DATASETS的建议
        print("\n建议更新TWITTER_DATASETS列表为:")
        print("TWITTER_DATASETS = [")
        print("    'twitter_football',    # Twitter football 数据集")
        print("    'twitter_olympics',    # Twitter olympics 数据集")
        print("    'twitter_rugby',       # Twitter rugby 数据集")
        
        for dataset in found_datasets:
            hyphenated_name = dataset
            if dataset == "politicsie":
                hyphenated_name = "politics-ie"
            elif dataset == "politicsuk":
                hyphenated_name = "politics-uk"
            
            print(f"    'twitter_{hyphenated_name}', # Twitter {hyphenated_name} 数据集")
        
        print("]")
    else:
        print("\n未找到任何政治相关数据集")
    
    return bool(found_datasets)

if __name__ == "__main__":
    check_politics_datasets()
