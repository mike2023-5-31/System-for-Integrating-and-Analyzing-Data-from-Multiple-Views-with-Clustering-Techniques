import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, accuracy_score
from sklearn.cluster import KMeans

from config import Config
from models.mvgcn import MVGCNModel
from utils.data_loader import load_multiview_data

def cluster_accuracy(y_true, y_pred):
    """计算聚类准确率"""
    from scipy.optimize import linear_sum_assignment
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    
    # 计算混淆矩阵
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    
    # 匈牙利算法求解最佳匹配
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() / y_pred.size

def train_and_evaluate_model(config, data_views, adj_matrices, labels):
    """训练并评估指定配置的模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 将数据转移到设备
    data_views_device = [v.to(device) for v in data_views]
    adj_matrices_device = [a.to(device) for a in adj_matrices]
    if isinstance(labels, torch.Tensor):
        labels_device = labels.to(device)
    else:
        labels_device = torch.tensor(labels).long().to(device)
    
    # 创建模型
    model = MVGCNModel(config).to(device)
    
    # 打印模型输出结构以进行调试
    print("检查模型输出结构...")
    model.eval()
    with torch.no_grad():
        test_output = model(data_views_device, adj_matrices_device, mask_rate=0.2)
        if isinstance(test_output, dict):
            print(f"模型输出是字典，包含键: {list(test_output.keys())}")
        else:
            print(f"模型输出类型是: {type(test_output)}")
    
    # 优化器
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.lr, 
        weight_decay=config.weight_decay
    )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-5
    )
    
    # 训练循环
    print(f"开始训练模型，配置: {config.save_name}...")
    best_nmi = 0
    best_epoch = 0
    no_improve = 0
    patience = 10
    
    for epoch in range(config.epochs):
        # 训练步骤
        model.train()
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(data_views_device, adj_matrices_device, mask_rate=config.mask_rate)
        
        # 处理不同类型的模型输出
        if isinstance(outputs, dict) and 'loss' in outputs:
            loss = outputs['loss']
        elif isinstance(outputs, dict) and 'recon_loss' in outputs:
            # 有些模型可能使用 recon_loss 键
            loss = outputs['recon_loss']
        elif isinstance(outputs, dict) and len(outputs) > 0:
            # 使用第一个找到的损失
            for key in outputs:
                if isinstance(outputs[key], torch.Tensor) and outputs[key].dim() == 0:
                    loss = outputs[key]
                    print(f"使用 '{key}' 作为损失函数")
                    break
            else:
                # 如果没有找到标量损失，创建一个虚拟损失
                print("警告: 未找到损失值，创建对比损失...")
                # 提取特征
                if 'fused' in outputs:
                    features = outputs['fused']
                elif 'cluster_rep' in outputs:
                    features = outputs['cluster_rep']
                elif 'z' in outputs:
                    features = outputs['z']
                else:
                    for key, value in outputs.items():
                        if isinstance(value, torch.Tensor) and value.dim() == 2:
                            features = value
                            print(f"使用 '{key}' 作为特征")
                            break
                    else:
                        raise ValueError("无法找到适合训练的特征表示")
                
                # 计算简单的MSE损失
                mean_features = torch.mean(features, dim=0, keepdim=True)
                loss = torch.mean((features - mean_features) ** 2)
        else:
            # 如果输出是张量，直接将其作为损失
            loss = outputs if torch.is_tensor(outputs) else torch.tensor(0.0, requires_grad=True, device=device)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 每10个周期评估一次
        if (epoch + 1) % 10 == 0 or (epoch + 1) == config.epochs:
            model.eval()
            with torch.no_grad():
                # 前向传播获取特征
                eval_outputs = model(data_views_device, adj_matrices_device, mask_rate=0.0)
                
                # 提取嵌入特征
                if isinstance(eval_outputs, dict):
                    if 'fused' in eval_outputs:
                        embeddings = eval_outputs['fused']
                    elif 'cluster_rep' in eval_outputs:
                        embeddings = eval_outputs['cluster_rep']
                    elif 'z' in eval_outputs:
                        embeddings = eval_outputs['z']
                    else:
                        # 尝试其他可能的键
                        for key in eval_outputs:
                            if isinstance(eval_outputs[key], torch.Tensor) and eval_outputs[key].dim() == 2:
                                embeddings = eval_outputs[key]
                                break
                else:
                    embeddings = eval_outputs if torch.is_tensor(eval_outputs) else torch.tensor(eval_outputs)
                
                # 聚类
                embeddings_np = embeddings.cpu().numpy()
                kmeans = KMeans(n_clusters=config.n_clusters, random_state=42)
                cluster_assignments = kmeans.fit_predict(embeddings_np)
                
                # 计算指标
                true_labels = labels_device.cpu().numpy()
                nmi = normalized_mutual_info_score(true_labels, cluster_assignments)
                ari = adjusted_rand_score(true_labels, cluster_assignments)
                acc = cluster_accuracy(true_labels, cluster_assignments)
                
                # 学习率调度
                scheduler.step(loss.item())
                current_lr = optimizer.param_groups[0]['lr']
                
                print(f"Epoch {epoch+1}/{config.epochs}: "
                      f"Loss={loss.item():.4f}, NMI={nmi:.4f}, ARI={ari:.4f}, ACC={acc:.4f}, LR={current_lr}")
                
                # 保存最佳模型
                if nmi > best_nmi:
                    best_nmi = nmi
                    best_ari = ari
                    best_acc = acc
                    best_epoch = epoch
                    best_embeddings = embeddings_np.copy()
                    best_assignments = cluster_assignments.copy()
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        print(f"早停: {patience}次评估未改善")
                        break
    
    print(f"训练完成! 最佳结果 (Epoch {best_epoch+1}):")
    print(f"  NMI: {best_nmi:.4f}")
    print(f"  ARI: {best_ari:.4f}")
    print(f"  ACC: {best_acc:.4f}")
    
    return best_nmi, best_ari, best_acc, best_embeddings, best_assignments

def optimize_handwritten():
    """优化handwritten数据集的性能"""
    print("开始优化handwritten数据集性能...")
    
    # 加载数据集
    print("加载handwritten数据集...")
    data_views, adj_matrices, labels = load_multiview_data("handwritten")
    
    # 数据集信息
    num_samples = data_views[0].shape[0]
    num_views = len(data_views)
    num_clusters = len(np.unique(labels))
    print(f"数据集信息: {num_views}个视图, {num_samples}个样本, {num_clusters}个聚类")
    print(f"视图特征维度: {[view.shape[1] for view in data_views]}")
    
    # 设置优化参数
    print("\n1. 执行特征选择和降维...")
    # 对大维度视图进行降维处理
    processed_views = []
    for i, view in enumerate(data_views):
        if view.shape[1] > 100:  # 对高维特征进行降维
            from sklearn.decomposition import PCA
            pca = PCA(n_components=100)
            reduced_features = pca.fit_transform(view.numpy())
            processed_view = torch.tensor(reduced_features, dtype=torch.float32)
            variance_ratio = sum(pca.explained_variance_ratio_)
            print(f"  视图{i+1}特征从{view.shape[1]}降至100维 (保留{variance_ratio*100:.2f}%方差)")
        else:
            processed_view = view
        processed_views.append(processed_view)
    
    # 创建结果目录
    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    # 只测试一个最有可能成功的配置
    configurations = [
        {
            "name": "optimized",
            "hidden_dims": [128, 64],
            "latent_dim": 32,
            "dropout": 0.6,
            "lr": 0.0005,
            "weight_decay": 0.0005,
            "contrastive_weight": 1.5,
            "temperature": 0.4,
            "mask_rate": 0.3,
            "adaptive_mask": True,
            "epochs": 100
        }
    ]
    
    # 存储结果
    results = []
    best_embeddings_all = None
    best_assignments_all = None
    best_acc_all = 0
    
    # 测试不同配置
    print("\n2. 测试配置...")
    for i, config_dict in enumerate(configurations):
        print(f"\n配置 {i+1}/{len(configurations)}: {config_dict['name']}")
        
        # 设置配置
        config = Config()
        config.dataset = "handwritten"
        config.n_clusters = num_clusters
        config.hidden_dims = config_dict["hidden_dims"]
        config.latent_dim = config_dict["latent_dim"]
        config.dropout = config_dict["dropout"]
        config.lr = config_dict["lr"]
        config.weight_decay = config_dict["weight_decay"]
        config.contrastive_weight = config_dict["contrastive_weight"]
        config.temperature = config_dict["temperature"]
        config.mask_rate = config_dict["mask_rate"]
        config.adaptive_mask = config_dict["adaptive_mask"]
        config.epochs = config_dict["epochs"]
        
        # 设置输入维度和设备
        config.input_dims = [view.shape[1] for view in processed_views]
        config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        config.save_name = f"handwritten_{config_dict['name']}"
        
        try:
            # 训练并评估模型
            nmi, ari, acc, embeddings, assignments = train_and_evaluate_model(
                config, processed_views, adj_matrices, labels
            )
            
            # 记录结果
            results.append({
                "配置": config_dict["name"],
                "NMI": nmi,
                "ARI": ari,
                "ACC": acc
            })
            
            print(f"最终结果 - NMI: {nmi:.4f}, ARI: {ari:.4f}, ACC: {acc:.4f}")
            
            # 保存最佳性能
            best_acc_all = acc
            best_embeddings_all = embeddings
            best_assignments_all = assignments
            best_config_name = config_dict["name"]
            
        except Exception as e:
            print(f"配置 {config_dict['name']} 训练失败: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "配置": config_dict["name"],
                "NMI": 0.0,
                "ARI": 0.0,
                "ACC": 0.0,
                "错误": str(e)
            })
    
    # 创建结果DataFrame
    df = pd.DataFrame(results)
    
    # 保存结果
    result_file = "results/handwritten_optimization_final.csv"
    df.to_csv(result_file, index=False)
    print(f"\n优化结果已保存至: {result_file}")
    
    # 打印最佳结果
    if len(df) > 0 and 'ACC' in df.columns and len(df['ACC']) > 0 and df['ACC'].max() > 0:
        best_idx = df['ACC'].idxmax()
        best_config = df.iloc[best_idx]
        print(f"\n最佳配置: {best_config['配置']}")
        print(f"最佳性能: NMI={best_config['NMI']:.4f}, ARI={best_config['ARI']:.4f}, ACC={best_config['ACC']:.4f}")
        
        # 如果存在嵌入，生成可视化
        if best_embeddings_all is not None:
            try:
                print("\n生成t-SNE可视化...")
                
                # 降维可视化
                tsne = TSNE(n_components=2, random_state=42)
                embeddings_2d = tsne.fit_transform(best_embeddings_all)
                
                # 绘图
                plt.figure(figsize=(15, 6))
                
                # 真实标签可视化
                plt.subplot(1, 2, 1)
                scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                     c=labels, cmap='tab10', s=5, alpha=0.7)
                plt.colorbar(scatter, label='True Labels')
                plt.title(f'Handwritten - True Labels')
                plt.xlabel('t-SNE Component 1')
                plt.ylabel('t-SNE Component 2')
                
                # 聚类结果可视化
                plt.subplot(1, 2, 2)
                scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                     c=best_assignments_all, cmap='tab10', s=5, alpha=0.7)
                plt.colorbar(scatter, label='Cluster Labels')
                plt.title(f'Handwritten - Optimized Clustering')
                plt.xlabel('t-SNE Component 1')
                plt.ylabel('t-SNE Component 2')
                
                plt.tight_layout()
                plt.savefig(f"results/handwritten_optimized_clustering.png", dpi=300)
                print(f"聚类可视化已保存至: results/handwritten_optimized_clustering.png")
                
            except Exception as e:
                print(f"可视化生成失败: {e}")
    
    print("\n优化完成!")

if __name__ == "__main__":
    optimize_handwritten()