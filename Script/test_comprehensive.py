"""
多视图数据融合与聚类分析系统 - 综合测试与验证框架
该脚本执行系统在多个数据集上的全面测试，评估性能指标并生成详细报告
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# 导入项目模块
from config import Config
from train import train_model
from evaluate import evaluate_model, evaluate_clustering
from utils.data_loader import load_multiview_data

# 配置测试参数
class TestConfig:
    def __init__(self):
        # 定义已知可用的数据集列表
        self.available_datasets = [
            'example_dataset',  # 内置的示例数据集
            '3sources',         # 文本多视图数据集
            'segment',          # 图像分割数据集
            'twitter_football', # Twitter社交媒体数据集
            'twitter_olympics', # Twitter社交媒体数据集
            'twitter_politics-uk', # Twitter社交媒体数据集
            'twitter_politics-ie', # Twitter社交媒体数据集
            'twitter_rugby',    # Twitter社交媒体数据集
            'movielists'        # 电影列表多视图数据集
        ]
        
        # 测试数据集列表 - 包含各种类型的多视图数据集
        # 注意：我们包括了所有数据集，但会在运行时检查可用性
        self.datasets = [
            # 标准基准数据集(可能不可用)
            'handwritten', 'reuters', 'coil20', 'mnist', 
            # 文本数据集(可用)
            '3sources', 'segment', 'bbc', 'bbcsport',
            # 社交媒体数据集(可用)
            'twitter_football', 'twitter_olympics', 'twitter_politics-uk', 
            # 多模态数据集(可用)
            'movielists'
        ]
        
        # 模型参数配置
        self.model_configs = [
            # 基本配置
            {'latent_dim': 64, 'hidden_dims': [256, 128], 'mask_rate': 0.2, 'dropout': 0.5},
            # 深度网络配置
            {'latent_dim': 128, 'hidden_dims': [512, 256], 'mask_rate': 0.2, 'dropout': 0.5},
            # 轻量级配置
            {'latent_dim': 32, 'hidden_dims': [128, 64], 'mask_rate': 0.1, 'dropout': 0.3},
        ]
        
        # 是否重新训练模型
        self.retrain = False
        
        # 是否尝试下载缺失的数据集
        self.try_download = True
        
        # 评估指标
        self.metrics = ['nmi', 'ari', 'acc']
        
        # 可视化设置
        self.create_visualizations = True
        
        # 输出目录
        self.output_dir = 'test_results'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 测试ID (使用时间戳)
        self.test_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 检查环境和可用数据集
        self._validate_datasets()
    
    def _validate_datasets(self):
        """验证配置中的数据集是否可用"""
        from utils.data_loader import check_dataset_availability
        
        validated_datasets = []
        
        for dataset in self.datasets:
            # 如果数据集是已知可用的或通过验证
            if dataset in self.available_datasets or check_dataset_availability(dataset):
                validated_datasets.append(dataset)
        
        # 更新数据集列表，确保至少包含example_dataset
        if not validated_datasets:
            validated_datasets = ['example_dataset']
        
        self.datasets = validated_datasets

def create_model_name(dataset, config_idx):
    """创建模型名称标识符"""
    return f"{dataset}_config{config_idx}"

def run_tests(test_config):
    """运行完整测试套件"""
    results = []
    all_embeddings = {}
    all_labels = {}
    
    print(f"===== 开始综合测试 ({test_config.test_id}) =====")
    print(f"测试数据集: {', '.join(test_config.datasets)}")
    print(f"配置数量: {len(test_config.model_configs)}")
    
    # 创建结果目录
    test_dir = os.path.join(test_config.output_dir, test_config.test_id)
    os.makedirs(test_dir, exist_ok=True)
    
    # 保存测试配置
    with open(os.path.join(test_dir, 'test_config.txt'), 'w') as f:
        f.write(f"Test ID: {test_config.test_id}\n")
        f.write(f"Datasets: {test_config.datasets}\n")
        f.write("Model Configurations:\n")
        for i, cfg in enumerate(test_config.model_configs):
            f.write(f"  Config {i}: {cfg}\n")
    
    # 导入模块 - 避免循环导入
    from train import train_model, load_model
    from evaluate import evaluate_model
    from utils.data_loader import download_dataset, check_dataset_path
    
    # 遍历每个数据集
    for dataset_idx, dataset in enumerate(test_config.datasets):
        print(f"\n[{dataset_idx+1}/{len(test_config.datasets)}] 测试数据集: {dataset}")
        
        # 检查数据集是否可用 (避免错误中断测试)
        try:
            # 如果开启了下载选项，尝试下载数据集
            if test_config.try_download and not check_dataset_path(dataset):
                print(f"  数据集 {dataset} 不存在，尝试下载...")
                download_success = download_dataset(dataset)
                if not download_success:
                    print(f"  无法下载数据集 {dataset}，跳过")
                    continue
            
            # 尝试加载数据集
            data_views, adj_matrices, labels = load_multiview_data(dataset)
            print(f"  成功加载数据集 {dataset}，样本数: {len(labels)}, 视图数: {len(data_views)}")
        except Exception as e:
            print(f"  无法加载数据集 {dataset}: {str(e)}")
            continue
        
        # 遍历每个模型配置
        for config_idx, model_config in enumerate(test_config.model_configs):
            model_name = create_model_name(dataset, config_idx)
            print(f"  测试配置 {config_idx+1}/{len(test_config.model_configs)}: {model_name}")
            
            # 创建配置对象
            config = Config()
            config.dataset = dataset
            
            # 应用模型配置
            for key, value in model_config.items():
                setattr(config, key, value)
            
            # 检查模型是否已经训练过
            model_path = f"checkpoints/{model_name}_best_model.pth"
            should_train = test_config.retrain or not os.path.exists(model_path)
            
            # 训练模型或加载已有模型
            train_time = 0
            if should_train:
                print(f"  训练模型: {model_name}")
                start_time = time.time()
                try:
                    # 训练模型并保存为自定义名称
                    model = train_model(config, save_name=model_name)
                    train_time = time.time() - start_time
                    print(f"  训练完成，用时: {train_time:.2f}秒")
                except Exception as train_error:
                    print(f"  训练失败: {str(train_error)}")
                    continue
            else:
                print(f"  使用已有模型: {model_path}")
            
            try:
                # 评估模型
                print(f"  评估模型性能...")
                nmi, ari, acc = evaluate_model(config, model_path, save_name=model_name)
                
                # 获取嵌入向量用于可视化
                if test_config.create_visualizations:
                    try:
                        # 直接调用evaluate_model中的代码获取嵌入向量
                        model = load_model(config, model_path)
                        model.eval()
                        data_views, adj_matrices, true_labels = load_multiview_data(config.dataset)
                        
                        # 转换数据
                        data_views = [torch.tensor(v, dtype=torch.float32).to(config.device) for v in data_views]
                        adj_matrices = [a.to(config.device) for a in adj_matrices]
                        
                        # 获取嵌入向量
                        with torch.no_grad():
                            cluster_features = model.get_cluster_features(data_views, adj_matrices)
                            embeddings = cluster_features.cpu().numpy()
                        
                        all_embeddings[model_name] = embeddings
                        all_labels[model_name] = true_labels
                    except Exception as viz_error:
                        print(f"  获取可视化嵌入向量失败: {str(viz_error)}")
                
                # 记录结果
                result = {
                    'dataset': dataset,
                    'config_idx': config_idx,
                    'model_name': model_name,
                    'latent_dim': config.latent_dim,
                    'hidden_dims': str(config.hidden_dims),
                    'mask_rate': config.mask_rate,
                    'dropout': config.dropout,
                    'nmi': nmi,
                    'ari': ari,
                    'acc': acc,
                    'train_time': train_time,
                    'n_samples': len(labels),
                    'n_views': len(data_views),
                    'n_clusters': len(np.unique(labels))
                }
                results.append(result)
                
                # 保存当前进度
                df = pd.DataFrame(results)
                df.to_csv(os.path.join(test_dir, 'test_results.csv'), index=False)
                
                print(f"  性能: NMI={nmi:.4f}, ARI={ari:.4f}, ACC={acc:.4f}")
                
            except Exception as e:
                print(f"  评估模型失败: {str(e)}")
                import traceback
                traceback.print_exc()  # 打印详细错误信息以便调试
    
    # 测试完成，返回结果
    return results, all_embeddings, all_labels, test_dir

def generate_report(results, embeddings, labels, test_dir):
    """生成综合测试报告"""
    print("\n===== 生成测试报告 =====")
    
    # 创建结果数据框
    df = pd.DataFrame(results)
    
    # 如果没有结果，直接返回
    if len(df) == 0:
        print("没有可用的测试结果")
        return
    
    # 1. 生成总体性能报告
    print("生成总体性能报告...")
    summary = df.groupby('dataset').agg({
        'nmi': ['mean', 'std', 'max'],
        'ari': ['mean', 'std', 'max'],
        'acc': ['mean', 'std', 'max']
    }).round(4)
    
    summary.to_csv(os.path.join(test_dir, 'performance_summary.csv'))
    
    # 2. 找到每个数据集的最佳配置
    print("确定每个数据集的最佳配置...")
    best_configs = pd.DataFrame(columns=['dataset', 'best_config_idx', 'nmi', 'ari', 'acc'])
    
    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset]
        # 基于ACC选择最佳配置
        best_row = dataset_df.loc[dataset_df['acc'].idxmax()]
        best_configs = pd.concat([best_configs, pd.DataFrame({
            'dataset': [dataset],
            'best_config_idx': [best_row['config_idx']],
            'nmi': [best_row['nmi']],
            'ari': [best_row['ari']],
            'acc': [best_row['acc']]
        })], ignore_index=True)
    
    best_configs.to_csv(os.path.join(test_dir, 'best_configs.csv'), index=False)
    
    # 3. 生成可视化
    print("生成性能可视化...")
    
    # 3.1 条形图比较不同数据集的性能
    plt.figure(figsize=(15, 8))
    datasets = df['dataset'].unique()
    x = np.arange(len(datasets))
    width = 0.25
    
    metrics = ['nmi', 'ari', 'acc']
    colors = ['#2196F3', '#4CAF50', '#FFC107']
    
    for i, metric in enumerate(metrics):
        values = [best_configs[best_configs['dataset'] == d][metric].values[0] for d in datasets]
        plt.bar(x + (i-1)*width, values, width, label=metric.upper(), color=colors[i])
    
    plt.xlabel('Dataset')
    plt.ylabel('Score')
    plt.title('Best Performance by Dataset')
    plt.xticks(x, datasets, rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(test_dir, 'performance_by_dataset.png'))
    
    # 3.2 散点图比较不同配置参数对性能的影响
    plt.figure(figsize=(18, 6))
    
    # 潜在维度与性能的关系
    plt.subplot(1, 3, 1)
    sns.scatterplot(data=df, x='latent_dim', y='acc', hue='dataset', s=60, alpha=0.7)
    plt.title('Latent Dimension vs. Accuracy')
    plt.grid(linestyle='--', alpha=0.6)
    
    # 掩码率与性能的关系
    plt.subplot(1, 3, 2)
    sns.scatterplot(data=df, x='mask_rate', y='acc', hue='dataset', s=60, alpha=0.7)
    plt.title('Mask Rate vs. Accuracy')
    plt.grid(linestyle='--', alpha=0.6)
    
    # Dropout率与性能的关系
    plt.subplot(1, 3, 3)
    sns.scatterplot(data=df, x='dropout', y='acc', hue='dataset', s=60, alpha=0.7)
    plt.title('Dropout vs. Accuracy')
    plt.grid(linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(test_dir, 'parameter_impact.png'))
    
    # 4. 嵌入可视化 (对每个最佳配置)
    if embeddings:
        print("生成嵌入可视化...")
        for dataset in best_configs['dataset'].values:
            best_config_idx = best_configs[best_configs['dataset'] == dataset]['best_config_idx'].values[0]
            model_name = create_model_name(dataset, best_config_idx)
            
            if model_name in embeddings:
                emb = embeddings[model_name]
                true_labels = labels[model_name]
                
                # 使用t-SNE可视化
                plt.figure(figsize=(10, 8))
                
                # 如果特征维度很高，先使用PCA降维
                if emb.shape[1] > 50:
                    emb = PCA(n_components=50).fit_transform(emb)
                
                # t-SNE降维到2D
                tsne_result = TSNE(n_components=2, perplexity=min(30, max(5, len(emb)//10))).fit_transform(emb)
                
                # 绘制散点图
                unique_labels = np.unique(true_labels)
                colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
                
                for i, label in enumerate(unique_labels):
                    plt.scatter(tsne_result[true_labels == label, 0],
                               tsne_result[true_labels == label, 1],
                               c=[colors[i]],
                               label=f'Class {label}',
                               alpha=0.7)
                
                plt.title(f't-SNE Visualization: {dataset} (Best Config)')
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                plt.savefig(os.path.join(test_dir, f'{dataset}_tsne.png'))
    
    # 5. 生成HTML报告
    print("生成HTML报告...")
    generate_html_report(df, best_configs, test_dir)
    
    print(f"报告生成完成，保存在: {test_dir}")

def generate_html_report(results_df, best_configs_df, test_dir):
    """生成HTML格式的综合测试报告"""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>多视图数据融合与聚类分析系统 - 测试报告</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                color: #333;
                line-height: 1.6;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
            }}
            h1, h2, h3 {{
                color: #2c3e50;
            }}
            .header {{
                background-color: #34495e;
                color: white;
                padding: 20px;
                text-align: center;
                border-radius: 5px;
                margin-bottom: 30px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 30px;
            }}
            th, td {{
                padding: 12px 15px;
                border-bottom: 1px solid #ddd;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            tr:hover {{
                background-color: #f5f5f5;
            }}
            .metric-high {{
                color: #2ecc71;
                font-weight: bold;
            }}
            .metric-medium {{
                color: #3498db;
            }}
            .metric-low {{
                color: #e74c3c;
            }}
            .visualization {{
                margin: 30px 0;
                text-align: center;
            }}
            img {{
                max-width: 100%;
                border: 1px solid #ddd;
                border-radius: 5px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }}
            .summary {{
                background-color: #ecf0f1;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 30px;
            }}
            .footer {{
                text-align: center;
                margin-top: 50px;
                padding: 20px;
                color: #7f8c8d;
                font-size: 14px;
                border-top: 1px solid #eee;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>多视图数据融合与聚类分析系统</h1>
                <h2>综合测试报告</h2>
                <p>测试ID: {os.path.basename(test_dir)} | 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <h2>测试摘要</h2>
            <div class="summary">
                <p>数据集数量: {len(results_df['dataset'].unique())}</p>
                <p>配置数量: {len(results_df['config_idx'].unique())}</p>
                <p>总测试样例: {len(results_df)}</p>
            </div>
            
            <h2>性能概览</h2>
            <table>
                <tr>
                    <th>数据集</th>
                    <th>最佳配置</th>
                    <th>NMI</th>
                    <th>ARI</th>
                    <th>ACC</th>
                </tr>
    """
    
    # 添加每个数据集的性能数据
    for _, row in best_configs_df.iterrows():
        html_content += f"""
        <tr>
            <td>{row['dataset']}</td>
            <td>配置 {int(row['best_config_idx'])}</td>
            <td class="{'metric-high' if row['nmi'] > 0.7 else 'metric-medium' if row['nmi'] > 0.5 else 'metric-low'}">{row['nmi']:.4f}</td>
            <td class="{'metric-high' if row['ari'] > 0.7 else 'metric-medium' if row['ari'] > 0.5 else 'metric-low'}">{row['ari']:.4f}</td>
            <td class="{'metric-high' if row['acc'] > 0.7 else 'metric-medium' if row['acc'] > 0.5 else 'metric-low'}">{row['acc']:.4f}</td>
        </tr>
        """
    
    html_content += """
            </table>
            
            <h2>可视化结果</h2>
            
            <div class="visualization">
                <h3>不同数据集的最佳性能</h3>
                <img src="performance_by_dataset.png" alt="Performance by Dataset">
            </div>
            
            <div class="visualization">
                <h3>参数对性能的影响</h3>
                <img src="parameter_impact.png" alt="Parameter Impact">
            </div>
    """
    
    # 添加t-SNE可视化（如果有）
    tsne_images = [f for f in os.listdir(test_dir) if f.endswith('_tsne.png')]
    if tsne_images:
        html_content += """
            <h2>嵌入空间可视化</h2>
        """
        
        for img in tsne_images:
            dataset = img.split('_tsne.png')[0]
            html_content += f"""
            <div class="visualization">
                <h3>{dataset} 嵌入空间t-SNE可视化</h3>
                <img src="{img}" alt="{dataset} t-SNE">
            </div>
            """
    
    # 详细测试结果表
    html_content += """
            <h2>详细测试结果</h2>
            <table>
                <tr>
                    <th>数据集</th>
                    <th>配置</th>
                    <th>潜在维度</th>
                    <th>掩码率</th>
                    <th>Dropout</th>
                    <th>NMI</th>
                    <th>ARI</th>
                    <th>ACC</th>
                    <th>训练时间(s)</th>
                </tr>
    """
    
    for _, row in results_df.iterrows():
        html_content += f"""
        <tr>
            <td>{row['dataset']}</td>
            <td>{int(row['config_idx'])}</td>
            <td>{row['latent_dim']}</td>
            <td>{row['mask_rate']}</td>
            <td>{row['dropout']}</td>
            <td>{row['nmi']:.4f}</td>
            <td>{row['ari']:.4f}</td>
            <td>{row['acc']:.4f}</td>
            <td>{row['train_time']:.2f}</td>
        </tr>
        """
    
    html_content += """
            </table>
            
            <div class="footer">
                <p>多视图数据融合与聚类分析系统 - 自动生成测试报告</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # 保存HTML文件
    with open(os.path.join(test_dir, 'test_report.html'), 'w', encoding='utf-8') as f:
        f.write(html_content)

def main():
    """主函数"""
    # 创建测试配置
    test_config = TestConfig()
    
    # 根据命令行参数修改配置
    import argparse
    parser = argparse.ArgumentParser(description="多视图数据融合与聚类分析系统-综合测试")
    parser.add_argument('--datasets', nargs='+', help='要测试的数据集列表')
    parser.add_argument('--retrain', action='store_true', help='是否重新训练模型')
    parser.add_argument('--no-viz', dest='viz', action='store_false', help='禁用可视化生成')
    parser.add_argument('--light', action='store_true', help='使用轻量级测试配置')
    
    args = parser.parse_args()
    
    if args.datasets:
        test_config.datasets = args.datasets
    if args.retrain:
        test_config.retrain = True
    if not args.viz:
        test_config.create_visualizations = False
    if args.light:
        # 轻量级测试配置
        test_config.datasets = test_config.datasets[:3]  # 只测试前3个数据集
        test_config.model_configs = [test_config.model_configs[0]]  # 只使用第一个配置
    
    # 运行测试
    results, embeddings, labels, test_dir = run_tests(test_config)
    
    # 生成报告
    generate_report(results, embeddings, labels, test_dir)

if __name__ == "__main__":
    main()