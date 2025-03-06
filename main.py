import torch
import argparse
import os
from utils.data_loader import load_multiview_data, download_dataset, check_dataset_path
from train import train_model
from evaluate import evaluate_model
from config import Config

def parse_args():
    parser = argparse.ArgumentParser(description='多视图数据融合与聚类分析')
    parser.add_argument('--dataset', type=str, default='example_dataset', help='数据集名称')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--epochs', type=int, default=200, help='训练轮次')
    parser.add_argument('--mask_rate', type=float, default=0.2, help='掩码率')
    parser.add_argument('--adaptive_mask', action='store_true', help='是否使用自适应掩码')
    parser.add_argument('--evaluate', action='store_true', help='是否只进行评估')
    parser.add_argument('--latent_dim', type=int, default=128, help='潜在特征维度')
    parser.add_argument('--gcn_hidden', type=int, default=256, help='GCN隐藏层维度')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout率')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    # 添加新参数
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[512, 256], 
                       help='自编码器隐藏层维度列表，例如: --hidden_dims 768 384')
    parser.add_argument('--n_clusters', type=int, default=5, help='聚类数量')
    parser.add_argument('--download', action='store_true', help='强制下载并处理数据集')
    parser.add_argument('--generate_samples', action='store_true', help='生成示例数据集用于测试')
    
    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 确保必要的目录存在
    os.makedirs('data', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # 创建配置对象
    config = Config()
    
    # 使用config中的valid_datasets列表验证数据集
    if args.dataset not in config.valid_datasets:
        print(f"警告: 数据集 '{args.dataset}' 未识别。可用的数据集有: {', '.join(config.valid_datasets)}")
        print(f"将使用默认数据集: example_dataset")
        args.dataset = 'example_dataset'
    
    # 检查数据集文件是否存在，如果不存在或强制下载则尝试下载
    if not check_dataset_path(args.dataset) or args.download:
        print(f"准备下载和处理数据集 '{args.dataset}'...")
        if args.dataset in ['movielists', '3sources', 'segment'] or args.dataset.startswith('twitter_'):
            success = download_dataset(args.dataset)
            if not success:
                if args.generate_samples:
                    print(f"下载失败，生成示例数据集替代真实数据集")
                    from create_sample_datasets import create_sample_dataset
                    if args.dataset == 'handwritten':
                        create_sample_dataset('handwritten', n_clusters=10, n_features=[76, 64, 240, 47, 6])
                    elif args.dataset == 'reuters':
                        create_sample_dataset('reuters', n_clusters=6, n_features=[500, 500, 500, 500, 500])
                    elif args.dataset == 'coil20':
                        create_sample_dataset('coil20', n_clusters=20, n_features=[1024, 3304, 6750])
                    elif args.dataset == 'mnist':
                        create_sample_dataset('mnist', n_clusters=10, n_features=[784, 1024, 256])
                    else:
                        create_sample_dataset(args.dataset, n_clusters=5)
                else:
                    print(f"数据集 '{args.dataset}' 下载失败，改用示例数据集。")
                    args.dataset = 'example_dataset'
        else:
            if args.generate_samples:
                print(f"生成 {args.dataset} 示例数据集...")
                from create_sample_datasets import create_sample_dataset
                create_sample_dataset(args.dataset, n_clusters=5)
            else:
                print(f"数据集 '{args.dataset}' 不支持自动下载。使用示例数据集。")
                args.dataset = 'example_dataset'
        
    # 更新配置
    config.dataset = args.dataset
    config.lr = args.lr
    config.epochs = args.epochs
    config.mask_rate = args.mask_rate
    config.adaptive_mask = args.adaptive_mask
    config.latent_dim = args.latent_dim
    config.gcn_hidden = args.gcn_hidden
    config.dropout = args.dropout
    config.seed = args.seed
    
    # 添加新参数
    config.hidden_dims = args.hidden_dims
    config.n_clusters = args.n_clusters
    
    print("\n多视图数据融合与聚类分析系统")
    print("=" * 50)
    print(f"数据集: {config.dataset}")
    print(f"学习率: {config.lr}")
    print(f"训练轮次: {config.epochs}")
    print(f"设备: {config.device}")
    
    # 先加载数据
    data_views, adj_matrices, labels = load_multiview_data(args.dataset)
    if data_views is None:
        print(f"无法加载数据集 {args.dataset}，程序退出")
        return
    
    # 再打印数据相关信息
    print(f"数据集视图数: {len(data_views)}")
    print(f"样本数量: {data_views[0].shape[0]}")
    print("=" * 50)
    
    # 确保目录存在
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # 根据命令决定是训练还是评估
    if args.evaluate:
        print("\n开始模型评估...")
        model_path = f"checkpoints/{config.dataset}_best_model.pth"
        if os.path.exists(model_path):
            metrics = evaluate_model(config, model_path)
            print(f"\n评估结果: NMI={metrics[0]:.4f}, ARI={metrics[1]:.4f}, ACC={metrics[2]:.4f}")
        else:
            print(f"错误: 模型文件 {model_path} 不存在。请先训练模型。")
    else:
        print("\n开始模型训练...")
        model = train_model(config)
        print("\n训练完成! 现在进行最终评估...")
        
        # 训练后评估
        model_path = f"checkpoints/{config.dataset}_best_model.pth"
        if os.path.exists(model_path):
            metrics = evaluate_model(config, model_path)
            print(f"\n最终评估结果: NMI={metrics[0]:.4f}, ARI={metrics[1]:.4f}, ACC={metrics[2]:.4f}")

if __name__ == "__main__":
    main()