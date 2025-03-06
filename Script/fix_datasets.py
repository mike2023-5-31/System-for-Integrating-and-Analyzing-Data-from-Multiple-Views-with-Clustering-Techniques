"""
修复Twitter政治数据集处理问题
此脚本专门处理politics-ie数据集
"""

import os
import sys
import shutil
from pathlib import Path

def fix_politics_dataset(output_dir='data'):
    """修复政治数据集的处理"""
    twitter_dir = os.path.join(output_dir, 'multiview_twitter', 'multiview_data_20130124')
    
    if not os.path.exists(twitter_dir):
        print(f"错误: Twitter数据目录 {twitter_dir} 不存在")
        print("请先下载并解压数据集")
        return False
    
    # 列出目录内容进行检查
    print(f"Twitter数据目录内容: {os.listdir(twitter_dir)}")
    
    # 检查politicsie目录是否存在
    politicsie_dir = os.path.join(twitter_dir, 'politicsie')
    if not os.path.exists(politicsie_dir):
        print(f"错误: politicsie数据集目录 {politicsie_dir} 不存在")
        
        # 尝试搜索可能的替代目录
        for item in os.listdir(twitter_dir):
            if 'politic' in item.lower() and 'ie' in item.lower():
                print(f"找到可能的替代目录: {item}")
                politicsie_dir = os.path.join(twitter_dir, item)
                break
    
    # 如果找到目录，进行处理
    if os.path.exists(politicsie_dir):
        print(f"使用目录: {politicsie_dir}")
        
        # 检查必要的文件
        communities_path = os.path.join(politicsie_dir, 'communities.dat')
        if not os.path.exists(communities_path):
            print(f"错误: 社区文件 {communities_path} 不存在")
            return False
        
        # 手动创建twitter_politics-ie.mat文件
        from utils.data_loader import preprocess_twitter_dataset
        try:
            print("手动创建politics-ie数据集...")
            # 使用没有连字符的名称来匹配目录名
            preprocess_twitter_dataset('politicsie', output_dir)
            
            # 检查是否成功创建
            mat_path = os.path.join(output_dir, 'twitter_politicsie.mat')
            if os.path.exists(mat_path):
                # 创建正确名称的符号链接或复制文件
                target_path = os.path.join(output_dir, 'twitter_politics-ie.mat')
                shutil.copy2(mat_path, target_path)
                print(f"创建数据集文件: {target_path}")
                return True
            else:
                print(f"错误: 未能创建数据集文件 {mat_path}")
                return False
        except Exception as e:
            print(f"处理失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print("无法找到politics-ie数据集目录")
        return False

if __name__ == "__main__":
    success = fix_politics_dataset()
    if success:
        print("\n成功修复politics-ie数据集")
        print("现在可以使用以下命令运行模型:")
        print("python main.py --dataset twitter_politics-ie --lr 0.001 --epochs 200")
    else:
        print("\n修复失败，请尝试手动处理数据集")
