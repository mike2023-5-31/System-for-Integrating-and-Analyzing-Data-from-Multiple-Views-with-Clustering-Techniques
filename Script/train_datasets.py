"""
批量训练和评估多视图数据集的脚本
此脚本允许按顺序处理多个数据集，自动下载和处理数据
"""

import os
import argparse
import time
import pandas as pd
from config import Config
from train import train_model
from evaluate import evaluate_model
from utils.data_loader import download_dataset, check_dataset_path, load_multiview_data

# 数据集列表 - 分组为可直接下载和需生成的数据集
DOWNLOADABLE_DATASETS = [
    'movielists',         # MovieLists数据集
    '3sources',           # 3Sources数据集
    'segment',            # Image Segmentation数据集
    'twitter_football',   # Twitter Football数据集
    'twitter_olympics',   # Twitter Olympics数据集
    'twitter_politics-uk', # Twitter UK Politics数据集
    'twitter_politics-ie', # Twitter Ireland Politics数据集
    'twitter_rugby'       # Twitter Rugby数据集
]

SAMPLE_DATASETS = [
    'handwritten',        # 手写数字数据集
    'reuters',            # 路透社多语言文档数据集
    'coil20',             # COIL-20物体数据集
    'mnist'               # MNIST手写数字数据集
]

def parse_args():
    parser = argparse.ArgumentParser(description='批量训练多视图数据集')
    parser.add_argument('--datasets', nargs='+', default=DOWNLOADABLE_DATASETS,
                       help='要处理的数据集列表')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮次')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--mask_rate', type=float, default=0.2, help='掩码率')
    parser.add_argument('--adaptive_mask', action='store_true', help='使用自适应掩码')
    parser.add_argument('--latent_dim', type=int, default=128, help='潜在特征维度')
    parser.add_argument('--generate_samples', action='store_true', help='生成示例数据集')
    parser.add_argument('--force_retrain', action='store_true', help='强制重新训练')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[256, 128], 
                       help='自编码器隐藏层维度列表')
    return parser.parse_args()

def train_dataset(dataset_name, args):
    """训练单个数据集"""
    print(f"\n{'='*20} 处理数据集: {dataset_name} {'='*20}")
    
    # 检查数据集文件是否存在，如果不存在则下载
    if not check_dataset_path(dataset_name):
        print(f"数据集 {dataset_name} 不存在，准备下载...")
        
        if dataset_name in DOWNLOADABLE_DATASETS:
            success = download_dataset(dataset_name)
            if not success:
                print(f"下载 {dataset_name} 失败，跳过")
                return None
        elif args.generate_samples and dataset_name in SAMPLE_DATASETS:
            print(f"生成 {dataset_name} 示例数据集")
            from create_sample_datasets import create_sample_dataset
            
            if dataset_name == 'handwritten':
                create_sample_dataset(dataset_name, n_clusters=10, n_features=[76, 64, 240, 47, 6])
            elif dataset_name == 'reuters':
                create_sample_dataset(dataset_name, n_clusters=6, n_features=[500, 500, 500, 500, 500])
            elif dataset_name == 'coil20':
                create_sample_dataset(dataset_name, n_clusters=20, n_features=[1024, 3304, 6750])
            elif dataset_name == 'mnist':
                create_sample_dataset(dataset_name, n_clusters=10, n_features=[784, 1024, 256])
            else:
                create_sample_dataset(dataset_name, n_clusters=5)
        else:
            print(f"数据集 {dataset_name} 不可下载且不在示例列表中，跳过")
            return None
    
    # 检查现有模型
    model_path = f"checkpoints/{dataset_name}_best_model.pth"
    if os.path.exists(model_path) and not args.force_retrain:
        print(f"模型 {model_path} 已存在，跳过训练，直接评估")
        eval_only = True
    else:
        print(f"准备训练模型: {dataset_name}")
        eval_only = False
    
    # 创建配置
    config = Config()
    config.dataset = dataset_name
    config.epochs = args.epochs
    config.lr = args.lr
    config.mask_rate = args.mask_rate
    config.adaptive_mask = args.adaptive_mask
    config.latent_dim = args.latent_dim
    config.hidden_dims = args.hidden_dims
    
    try:
        # 加载数据 - 验证数据集是否可用
        print(f"加载数据集: {dataset_name}")
        data_views, adj_matrices, labels = load_multiview_data(dataset_name)
        n_samples = data_views[0].shape[0]
        n_views = len(data_views)
        print(f"数据集信息: {n_samples} 个样本, {n_views} 个视图")
        
        # 如果有标签，设置聚类数
        if labels is not None:
            n_clusters = len(torch.unique(labels))
            config.n_clusters = n_clusters
            print(f"识别到 {n_clusters} 个聚类")
        
        # 训练和评估
        start_time = time.time()
        
        if not eval_only:
            print(f"训练模型...")
            train_model(config)
            training_time = time.time() - start_time
            print(f"训练完成，用时: {training_time:.2f} 秒")
        
        # 评估模型
        print(f"评估模型...")
        evaluation_start = time.time()
        metrics = evaluate_model(config, model_path)
        evaluation_time = time.time() - evaluation_start
        
        # 创建结果记录
        results = {
            'dataset': dataset_name,
            'samples': n_samples,
            'views': n_views,
            'clusters': config.n_clusters,
            'nmi': metrics[0] if metrics else None,
            'ari': metrics[1] if metrics else None,
            'acc': metrics[2] if metrics else None,
            'train_time': training_time if not eval_only else 0,
            'eval_time': evaluation_time
        }
        
        print(f"处理 {dataset_name} 完成!")
        if metrics:
            print(f"性能指标: NMI={metrics[0]:.4f}, ARI={metrics[1]:.4f}, ACC={metrics[2]:.4f}")
        
        return results
    
    except Exception as e:
        print(f"处理数据集 {dataset_name} 时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'dataset': dataset_name,
            'error': str(e),
            'status': 'failed'
        }

def main():
    args = parse_args()
    
    # 确保目录存在
    os.makedirs('data', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    print(f"准备处理 {len(args.datasets)} 个数据集:")
    for i, dataset in enumerate(args.datasets):
        print(f"  {i+1}. {dataset}")
    
    # 存储结果
    all_results = []
    
    # 依次处理每个数据集
    for i, dataset in enumerate(args.datasets):
        print(f"\n[{i+1}/{len(args.datasets)}] 处理数据集: {dataset}")
        result = train_dataset(dataset, args)
        if result:
            all_results.append(result)
    
    # 保存结果摘要
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_file = f"results/batch_training_results_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        results_df.to_csv(results_file, index=False)
        print(f"\n所有结果已保存到 {results_file}")
        
        # 打印结果表格
        print("\n训练与评估结果摘要:")
        print(results_df.to_string())
    else:
        print("\n没有成功完成的数据集")

if __name__ == "__main__":
    import torch
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    main()
