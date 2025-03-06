"""
更新数据集配置的辅助脚本
"""

import os
import sys
import argparse
from utils.dataset_utils import update_twitter_datasets_config
from utils.data_loader import download_dataset

def main():
    parser = argparse.ArgumentParser(description='更新数据集配置')
    parser.add_argument('--download', action='store_true', 
                      help='是否下载Twitter数据集')
    parser.add_argument('--output-dir', type=str, default='data',
                      help='输出目录')
    args = parser.parse_args()
    
    if args.download:
        print("下载Twitter多视图数据集...")
        success = download_dataset('multiview_twitter', args.output_dir)
        if not success:
            print("下载失败")
            return
    
    print("检测并更新可用数据集配置...")
    datasets = update_twitter_datasets_config()
    
    print("\n可以在代码中更新以下常量:")
    print("TWITTER_DATASETS = [")
    for dataset in datasets:
        print(f"    '{dataset}',")
    print("]")

if __name__ == "__main__":
    main()
