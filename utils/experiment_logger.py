# 在CSV文件中记录实验结果
def log_experiment_results(config, metrics, save_path="results/experiments.csv"):
    """记录实验结果到CSV文件中"""
    import os
    import csv
    import time
    
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 写入标题行
    if not os.path.exists(save_path):
        with open(save_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Date', 'Dataset', 'LR', 'Epochs', 'MaskRate', 
                            'LatentDim', 'NMI', 'ARI', 'ACC'])
    
    # 添加结果行
    with open(save_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            time.strftime("%Y-%m-%d %H:%M"),
            config.dataset,
            config.lr,
            config.epochs,
            config.mask_rate,
            config.latent_dim,
            f"{metrics[0]:.4f}",
            f"{metrics[1]:.4f}",
            f"{metrics[2]:.4f}"
        ])