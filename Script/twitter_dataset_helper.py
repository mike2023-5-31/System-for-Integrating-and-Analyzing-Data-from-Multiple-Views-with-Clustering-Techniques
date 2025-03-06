"""
Twitter数据集辅助工具 - 帮助用户检查和处理Twitter数据集
"""

import os
import sys
import argparse
import shutil
from pathlib import Path

def check_dataset(dataset_name, data_dir='data'):
    """检查数据集状态"""
    print(f"检查数据集: {dataset_name}")
    
    # 检查处理后的MAT文件是否存在
    mat_path = os.path.join(data_dir, f"{dataset_name}.mat")
    if os.path.exists(mat_path):
        print(f"✓ 找到数据集文件: {mat_path}")
        print(f"  文件大小: {os.path.getsize(mat_path)/1024:.1f} KB")
        return True
    else:
        print(f"✗ 未找到数据集文件: {mat_path}")
        
        # 如果是politics-ie类型的名称，检查不带连字符的版本
        if '-' in dataset_name:
            alt_name = dataset_name.replace('-', '')
            alt_path = os.path.join(data_dir, f"{alt_name}.mat")
            if os.path.exists(alt_path):
                print(f"! 找到替代文件: {alt_path}")
                print(f"  可以复制为正确的名称")
                return False
        
        # 检查原始目录是否存在
        twitter_name = dataset_name.split('_', 1)[1] if '_' in dataset_name else dataset_name
        dir_name = twitter_name.replace('-', '')
        twitter_dir = os.path.join(data_dir, 'multiview_twitter', 'multiview_data_20130124', dir_name)
        
        if os.path.exists(twitter_dir):
            print(f"✓ 找到原始数据目录: {twitter_dir}")
            print(f"  数据集可以处理")
            return False
        else:
            print(f"✗ 未找到原始数据目录: {twitter_dir}")
            print(f"  请确认数据集名称正确且已下载")
            return False

def fix_dataset(dataset_name, data_dir='data'):
    """修复数据集问题"""
    if not dataset_name.startswith('twitter_'):
        dataset_name = f'twitter_{dataset_name}'
    
    print(f"尝试修复数据集: {dataset_name}")
    
    # 提取不带twitter_前缀的名称
    base_name = dataset_name.split('_', 1)[1] if '_' in dataset_name else dataset_name
    
    # 对于带连字符的名称，检查不带连字符的版本
    if '-' in base_name:
        dir_name = base_name.replace('-', '')
        src_path = os.path.join(data_dir, f'twitter_{dir_name}.mat')
        dst_path = os.path.join(data_dir, f'{dataset_name}.mat')
        
        if os.path.exists(src_path) and not os.path.exists(dst_path):
            print(f"复制文件 {src_path} 到 {dst_path}")
            shutil.copy2(src_path, dst_path)
            print("✓ 数据集文件已修复")
            return True
    
    # 尝试重新处理数据集
    print("尝试重新处理数据集...")
    try:
        # 确保Twitter数据集已下载
        from utils.data_loader import download_dataset, preprocess_twitter_dataset
        
        # 下载Twitter总数据集
        success = download_dataset('multiview_twitter', data_dir)
        if not success:
            print("✗ 下载Twitter数据集失败")
            return False
        
        # 处理特定子数据集
        try:
            if '-' in base_name:
                # 使用不带连字符的名称进行处理
                dir_name = base_name.replace('-', '')
                preprocess_twitter_dataset(dir_name, data_dir)
                
                # 创建正确名称的文件
                src_path = os.path.join(data_dir, f'twitter_{dir_name}.mat')
                dst_path = os.path.join(data_dir, f'{dataset_name}.mat')
                
                if os.path.exists(src_path) and not os.path.exists(dst_path):
                    shutil.copy2(src_path, dst_path)
            else:
                preprocess_twitter_dataset(base_name, data_dir)
            
            print("✓ 数据集处理成功")
            return True
            
        except Exception as e:
            print(f"✗ 处理数据集失败: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except ImportError:
        print("✗ 导入所需模块失败")
        return False

def find_twitter_dataset_files(base_name, data_dir='data'):
    """
    灵活查找Twitter数据集文件，处理各种命名变体
    
    参数:
        base_name: 子数据集名称（例如'politics-ie'）
        data_dir: 数据目录
        
    返回:
        ids_file: 找到的IDs文件路径
        comm_file: 找到的communities文件路径
    """
    # 处理目录名（移除连字符）
    dir_name = base_name.replace('-', '')
    dataset_dir = os.path.join(data_dir, 'multiview_twitter', 'multiview_data_20130124', dir_name)
    
    if not os.path.exists(dataset_dir):
        print(f"无法找到数据集目录: {dataset_dir}")
        # 尝试列出父目录内容，帮助调试
        parent_dir = os.path.dirname(dataset_dir)
        if os.path.exists(parent_dir):
            print(f"父目录内容: {os.listdir(parent_dir)}")
        return None, None
        
    print(f"找到数据集目录: {dataset_dir}")
    print(f"目录内容: {os.listdir(dataset_dir)}")
    
    # 尝试不同可能的文件名格式
    ids_variants = [
        f"{base_name}.ids",       # 原始格式 (politics-ie.ids)
        f"{dir_name}.ids",        # 无连字符 (politicsie.ids)
        f"{dir_name}.IDs",        # 大写IDs (politicsie.IDs)
        "IDs.txt",                # 通用名称 (IDs.txt)
        "ids.txt",                # 小写通用名称 (ids.txt)
        f"{base_name}_ids.txt",   # 下划线格式 (politics-ie_ids.txt)
        f"{dir_name}_ids.txt",    # 无连字符下划线格式 (politicsie_ids.txt)
    ]
    
    # 查找IDs文件
    ids_file = None
    for variant in ids_variants:
        test_path = os.path.join(dataset_dir, variant)
        if os.path.exists(test_path):
            ids_file = test_path
            print(f"找到IDs文件: {variant}")
            break
            
    # 查找communities文件
    comm_file = os.path.join(dataset_dir, "communities.dat")
    if not os.path.exists(comm_file):
        # 尝试其他可能的命名
        for variant in ["communities.txt", "Communities.dat", "Communities.txt"]:
            test_path = os.path.join(dataset_dir, variant)
            if os.path.exists(test_path):
                comm_file = test_path
                print(f"找到Communities文件: {variant}")
                break
    else:
        print("找到Communities文件: communities.dat")
    
    return ids_file, comm_file

def main():
    parser = argparse.ArgumentParser(description='Twitter数据集辅助工具')
    parser.add_argument('--dataset', type=str, default='twitter_politics-ie',
                       help='要检查或修复的数据集名称')
    parser.add_argument('--fix', action='store_true', 
                       help='尝试修复数据集问题')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='数据目录')
    args = parser.parse_args()
    
    # 确保数据集名称格式化
    dataset_name = args.dataset
    if not dataset_name.startswith('twitter_') and not args.dataset == 'all':
        dataset_name = f'twitter_{dataset_name}'
    
    # 检查或修复指定的数据集
    if args.dataset == 'all':
        datasets = ['twitter_football', 'twitter_olympics', 'twitter_rugby', 'twitter_politics-ie', 'twitter_politics-uk']
        results = {}
        
        print("检查所有Twitter数据集:")
        for dataset in datasets:
            print(f"\n--- {dataset} ---")
            status = check_dataset(dataset, args.data_dir)
            results[dataset] = status
            
            if args.fix and not status:
                print(f"\n尝试修复 {dataset}...")
                fix_status = fix_dataset(dataset, args.data_dir)
                results[dataset] = fix_status
        
        print("\n数据集检查结果:")
        for dataset, status in results.items():
            print(f"{dataset}: {'可用' if status else '不可用'}")
    else:
        status = check_dataset(dataset_name, args.data_dir)
        
        if args.fix and not status:
            fix_dataset(dataset_name, args.data_dir)

if __name__ == "__main__":
    main()
