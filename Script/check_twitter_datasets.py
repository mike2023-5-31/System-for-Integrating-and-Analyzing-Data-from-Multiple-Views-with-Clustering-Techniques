#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
验证Twitter多视图数据集的可用性
检查ZIP文件中存在的子数据集并报告结果
"""

import os
import sys
import zipfile
import argparse
import json
from pathlib import Path

def find_available_twitter_datasets(zip_path=None, data_dir='data', deep_search=False):
    """
    检查指定的ZIP文件或目录中可用的Twitter子数据集
    
    参数:
        zip_path: Twitter多视图数据集ZIP文件的路径
        data_dir: 数据目录(默认为'data')
        deep_search: 是否进行深度搜索，查找可能的子目录
        
    返回:
        list: 可用的Twitter子数据集名称列表
    """
    # 如果没有提供ZIP路径，尝试在数据目录中查找
    if zip_path is None:
        zip_path = os.path.join(data_dir, 'multiview_twitter.zip')
        
    # 检查ZIP文件是否存在
    if not os.path.exists(zip_path):
        print(f"错误：找不到Twitter数据集ZIP文件: {zip_path}")
        return []
    
    # 查找子数据集目录
    available_datasets = []
    try:
        # 打开ZIP文件
        with zipfile.ZipFile(zip_path, 'r') as zip_file:
            # 获取所有文件路径
            all_files = zip_file.namelist()
            
            # 在文件列表中查找不同的子数据集目录
            subdirs = set()
            data_dir_prefix = "multiview_data_20130124/"
            
            for file_path in all_files:
                # 查找以data_dir_prefix开头的路径
                if data_dir_prefix in file_path:
                    # 提取子目录名称
                    parts = file_path.split(data_dir_prefix)
                    if len(parts) > 1:
                        subpath = parts[1]
                        if '/' in subpath:
                            subdir = subpath.split('/')[0]
                            if subdir and not subdir.startswith('.'):
                                subdirs.add(subdir)
            
            # 验证子数据集是否包含必要的文件
            for subdir in subdirs:
                # 检查必要的文件是否存在
                ids_file = f"{data_dir_prefix}{subdir}/{subdir}.ids"
                comm_file = f"{data_dir_prefix}{subdir}/{subdir}.communities"
                
                # 提供的数据集应该至少有IDs文件和communities文件
                if ids_file in all_files and comm_file in all_files:
                    available_datasets.append(subdir)
            
            # 如果启用深度搜索，记录所有文件夹结构
            if deep_search:
                print("\n===== ZIP文件目录结构 =====")
                all_dirs = set()
                for file_path in all_files:
                    parts = file_path.split('/')
                    # 收集所有层级的目录
                    for i in range(1, len(parts)):
                        all_dirs.add('/'.join(parts[:i]))
                
                # 打印目录结构
                for directory in sorted(all_dirs):
                    print(f"  {directory}")
                print("========================")
    
    except Exception as e:
        print(f"检查ZIP文件时出错: {e}")
        return []
    
    return available_datasets

def check_extracted_datasets(data_dir='data', deep_search=False):
    """检查已提取的数据集目录中可用的子数据集"""
    multiview_dir = os.path.join(data_dir, 'multiview_twitter')
    
    if not os.path.exists(multiview_dir):
        print(f"错误：找不到已提取的Twitter数据集目录: {multiview_dir}")
        return []
    
    # 寻找multiview_data目录
    data_dirs = []
    for item in os.listdir(multiview_dir):
        if item.startswith("multiview_data") and os.path.isdir(os.path.join(multiview_dir, item)):
            data_dirs.append(os.path.join(multiview_dir, item))
    
    # 如果启用深度搜索，显示完整的目录结构
    if deep_search:
        print("\n===== 提取目录结构 =====")
        for data_dir in data_dirs:
            print(f"目录: {data_dir}")
            for root, dirs, files in os.walk(data_dir):
                level = root.replace(data_dir, '').count(os.sep)
                indent = ' ' * 4 * level
                print(f"{indent}{os.path.basename(root)}/")
                for f in files:
                    if f.endswith('.ids') or f.endswith('.communities'):
                        print(f"{indent}    {f}")
        print("========================")
    
    # 查找所有可用的子数据集
    available_datasets = []
    for data_dir in data_dirs:
        for item in os.listdir(data_dir):
            subdir_path = os.path.join(data_dir, item)
            if os.path.isdir(subdir_path):
                # 检查是否存在必要文件
                ids_file = os.path.join(subdir_path, f"{item}.ids")
                comm_file = os.path.join(subdir_path, f"{item}.communities")
                
                if os.path.exists(ids_file) and os.path.exists(comm_file):
                    available_datasets.append(item)
    
    return available_datasets

def save_dataset_info(datasets, output_file='available_twitter_datasets.json'):
    """保存可用数据集信息到JSON文件"""
    try:
        with open(output_file, 'w') as f:
            json.dump({"available_datasets": datasets}, f, indent=2)
        print(f"可用数据集信息已保存到 {output_file}")
    except Exception as e:
        print(f"保存数据集信息时出错: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='检查Twitter多视图数据集的可用性')
    parser.add_argument('--zip', type=str, help='Twitter数据集ZIP文件路径')
    parser.add_argument('--data-dir', type=str, default='data', help='数据目录')
    parser.add_argument('--output', type=str, help='输出JSON文件路径')
    parser.add_argument('--deep-search', action='store_true',
                      help='进行深度搜索，显示文件结构')
    parser.add_argument('--update-config', action='store_true',
                      help='自动更新process_twitter_datasets.py中的TWITTER_DATASETS列表')
    args = parser.parse_args()
    
    print("检查可用的Twitter子数据集...")
    
    # 首先检查已提取的目录
    extracted_datasets = check_extracted_datasets(args.data_dir, args.deep_search)
    if extracted_datasets:
        print("\n已提取目录中找到的可用子数据集:")
        for i, dataset in enumerate(sorted(extracted_datasets), 1):
            print(f"  {i}. twitter_{dataset}")
        
        # 保存数据集信息
        if args.output:
            save_dataset_info(extracted_datasets, args.output)
            
        # 为process_twitter_datasets.py生成代码片段
        print("\n用于更新process_twitter_datasets.py的代码片段:")
        print("TWITTER_DATASETS = [")
        for dataset in sorted(extracted_datasets):
            print(f"    'twitter_{dataset}',")
        print("]")
        
        # 如果需要更新配置文件
        if args.update_config:
            try:
                # 更新process_twitter_datasets.py文件
                update_datasets_config(extracted_datasets)
            except Exception as e:
                print(f"更新配置文件失败: {e}")
        
        return
    
    # 如果没有找到已提取的数据集，检查ZIP文件
    zip_datasets = find_available_twitter_datasets(args.zip, args.data_dir, args.deep_search)
    
    if zip_datasets:
        print("\nZIP文件中找到的可用子数据集:")
        for i, dataset in enumerate(sorted(zip_datasets), 1):
            print(f"  {i}. twitter_{dataset}")
        
        # 保存数据集信息
        if args.output:
            save_dataset_info(zip_datasets, args.output)
            
        # 为process_twitter_datasets.py生成代码片段
        print("\n用于更新process_twitter_datasets.py的代码片段:")
        print("TWITTER_DATASETS = [")
        for dataset in sorted(zip_datasets):
            print(f"    'twitter_{dataset}',")
        print("]")
        
        # 如果需要更新配置文件
        if args.update_config:
            try:
                # 更新process_twitter_datasets.py文件
                update_datasets_config(zip_datasets)
            except Exception as e:
                print(f"更新配置文件失败: {e}")
    else:
        print("未找到任何可用的Twitter子数据集")

def update_datasets_config(datasets, config_file='process_twitter_datasets.py'):
    """更新process_twitter_datasets.py中的TWITTER_DATASETS列表"""
    if not datasets:
        print("没有找到可用数据集，不更新配置")
        return
    
    try:
        # 读取文件内容
        with open(config_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 构建新的TWITTER_DATASETS列表
        datasets_code = "TWITTER_DATASETS = [\n"
        for dataset in sorted(datasets):
            name = dataset if dataset.startswith('twitter_') else f'twitter_{dataset}'
            datasets_code += f"    '{name}',    # Twitter {dataset} 数据集\n"
        datasets_code += "]\n"
        
        # 使用正则表达式替换TWITTER_DATASETS定义部分
        import re
        pattern = r"TWITTER_DATASETS\s*=\s*\[[\s\S]*?\]"
        new_content = re.sub(pattern, datasets_code, content)
        
        # 写回文件
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"\n已更新 {config_file} 中的TWITTER_DATASETS列表")
        
    except Exception as e:
        print(f"更新配置文件时出错: {e}")
        raise

if __name__ == "__main__":
    main()
