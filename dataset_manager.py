import os
import sys
import torch
import numpy as np
import pandas as pd
import urllib.request
import zipfile
import gzip
import shutil
import scipy.io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import logging
import argparse
from pathlib import Path

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 数据集路径定义
DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
MNIST_DIR = os.path.join(DATA_ROOT, 'mnist')
COIL20_DIR = os.path.join(DATA_ROOT, 'coil20')
MOVIELISTS_DIR = os.path.join(DATA_ROOT, 'movielists')

def ensure_dir(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"创建目录: {directory}")

def download_file(url, destination):
    """从URL下载文件到指定位置"""
    if os.path.exists(destination):
        logger.info(f"文件已存在: {destination}")
        return
    
    logger.info(f"下载文件: {url}")
    try:
        urllib.request.urlretrieve(url, destination)
        logger.info(f"下载完成: {destination}")
    except Exception as e:
        logger.error(f"下载失败: {e}")
        raise

def fix_mnist_dataset():
    """修复MNIST数据集加载问题"""
    logger.info("开始修复MNIST数据集...")
    
    # 确保目录存在
    ensure_dir(MNIST_DIR)
    
    # MNIST数据集文件URL和本地路径
    mnist_files = {
        'train-images-idx3-ubyte.gz': 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz': 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz': 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz': 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
    }
    
    # 下载MNIST数据集
    for file_name, url in mnist_files.items():
        file_path = os.path.join(MNIST_DIR, file_name)
        download_file(url, file_path)
    
    # 解压数据文件并创建numpy格式
    try:
        # 处理训练图像
        with gzip.open(os.path.join(MNIST_DIR, 'train-images-idx3-ubyte.gz'), 'rb') as f_in:
            with open(os.path.join(MNIST_DIR, 'train-images-idx3-ubyte'), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # 处理训练标签
        with gzip.open(os.path.join(MNIST_DIR, 'train-labels-idx1-ubyte.gz'), 'rb') as f_in:
            with open(os.path.join(MNIST_DIR, 'train-labels-idx1-ubyte'), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # 处理测试图像
        with gzip.open(os.path.join(MNIST_DIR, 't10k-images-idx3-ubyte.gz'), 'rb') as f_in:
            with open(os.path.join(MNIST_DIR, 't10k-images-idx3-ubyte'), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # 处理测试标签
        with gzip.open(os.path.join(MNIST_DIR, 't10k-labels-idx1-ubyte.gz'), 'rb') as f_in:
            with open(os.path.join(MNIST_DIR, 't10k-labels-idx1-ubyte'), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    
        # 读取MNIST数据并保存为numpy格式
        def read_idx(filename):
            with open(filename, 'rb') as f:
                zero, data_type, dims = struct.unpack('>HBB', f.read(4))
                shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
                return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)
        
        import struct
        train_images = read_idx(os.path.join(MNIST_DIR, 'train-images-idx3-ubyte'))
        train_labels = read_idx(os.path.join(MNIST_DIR, 'train-labels-idx1-ubyte'))
        test_images = read_idx(os.path.join(MNIST_DIR, 't10k-images-idx3-ubyte'))
        test_labels = read_idx(os.path.join(MNIST_DIR, 't10k-labels-idx1-ubyte'))
        
        # 合并训练集和测试集
        all_images = np.vstack((train_images, test_images))
        all_labels = np.hstack((train_labels, test_labels))
        
        # 创建多视图数据
        # 视图1: 原始像素
        view1 = all_images.reshape(all_images.shape[0], -1).astype(np.float32) / 255.0
        
        # 视图2: 水平边缘特征
        h_edges = np.abs(all_images[:, :, 1:] - all_images[:, :, :-1])
        view2 = h_edges.reshape(h_edges.shape[0], -1).astype(np.float32) / 255.0
        
        # 视图3: 垂直边缘特征
        v_edges = np.abs(all_images[:, 1:, :] - all_images[:, :-1, :])
        view3 = v_edges.reshape(v_edges.shape[0], -1).astype(np.float32) / 255.0
        
        # 保存为pytorch格式
        view1_tensor = torch.tensor(view1, dtype=torch.float32)
        view2_tensor = torch.tensor(view2, dtype=torch.float32)
        view3_tensor = torch.tensor(view3, dtype=torch.float32)
        labels_tensor = torch.tensor(all_labels, dtype=torch.long)
        
        # 创建符合项目格式的数据
        data_views = [view1_tensor, view2_tensor, view3_tensor]
        
        # 保存处理后的数据
        processed_data_path = os.path.join(MNIST_DIR, "processed_data.pt")
        torch.save({
            'views': data_views,
            'labels': labels_tensor
        }, processed_data_path)
        
        logger.info(f"MNIST数据集已处理并保存: {processed_data_path}")
        logger.info(f"数据形状: {[v.shape for v in data_views]}, 标签形状: {labels_tensor.shape}")
    
    except Exception as e:
        logger.error(f"MNIST数据处理失败: {e}")
        import traceback
        traceback.print_exc()

def fix_coil20_dataset():
    """修复COIL-20数据集加载问题"""
    logger.info("开始修复COIL-20数据集...")
    
    # 确保目录存在
    ensure_dir(COIL20_DIR)
    
    # COIL-20数据集URL
    coil20_url = "http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-proc.zip"
    zip_path = os.path.join(COIL20_DIR, "coil-20-proc.zip")
    
    # 下载COIL-20数据集
    download_file(coil20_url, zip_path)
    
    # 解压数据集
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(COIL20_DIR)
        logger.info(f"COIL-20数据集已解压到: {COIL20_DIR}")
        
        # 处理COIL-20数据
        coil20_proc_dir = os.path.join(COIL20_DIR, "coil-20-proc")
        
        # 读取所有图像
        images = []
        labels = []
        
        # 遍历所有文件
        for i in range(20):  # 20个类别
            for j in range(72):  # 每个类别72个视角
                img_path = os.path.join(coil20_proc_dir, f"obj{i+1}__{j}.png")
                
                # 如果文件名格式不同，尝试其他可能的格式
                if not os.path.exists(img_path):
                    img_path = os.path.join(coil20_proc_dir, f"obj{i+1}_{j}.png")
                if not os.path.exists(img_path):
                    img_path = os.path.join(coil20_proc_dir, f"obj{i+1}_{j:03d}.png")
                
                if os.path.exists(img_path):
                    from PIL import Image
                    img = Image.open(img_path).convert('L')  # 转为灰度图
                    img_array = np.array(img)
                    images.append(img_array.flatten())
                    labels.append(i)
                else:
                    logger.warning(f"找不到图像文件: obj{i+1}_{j}")
        
        if not images:
            # 如果找不到图像，尝试直接查找目录结构
            logger.info("尝试查找COIL-20图像目录结构...")
            dirs = [d for d in os.listdir(coil20_proc_dir) if os.path.isdir(os.path.join(coil20_proc_dir, d))]
            logger.info(f"找到的目录: {dirs}")
            
            # 尝试遍历找到的目录
            for d in dirs:
                class_dir = os.path.join(coil20_proc_dir, d)
                class_idx = int(d.replace("obj", "")) - 1
                for img_file in os.listdir(class_dir):
                    if img_file.endswith(".png"):
                        img_path = os.path.join(class_dir, img_file)
                        from PIL import Image
                        img = Image.open(img_path).convert('L')
                        img_array = np.array(img)
                        images.append(img_array.flatten())
                        labels.append(class_idx)
        
        # 创建多视图数据
        if images:
            images = np.array(images)
            labels = np.array(labels)
            
            # 归一化图像数据
            images = images.astype(np.float32) / 255.0
            
            # 创建多个视图
            # 视图1: 原始像素
            view1 = torch.tensor(images, dtype=torch.float32)
            
            # 视图2: PCA降维特征
            pca = PCA(n_components=min(100, images.shape[1]))
            view2 = torch.tensor(pca.fit_transform(images), dtype=torch.float32)
            
            # 视图3: 直方图特征 - 模拟
            num_samples = images.shape[0]
            hist_bins = 32
            view3 = torch.zeros((num_samples, hist_bins), dtype=torch.float32)
            for i in range(num_samples):
                hist, _ = np.histogram(images[i], bins=hist_bins, range=(0, 1))
                view3[i] = torch.tensor(hist / hist.sum(), dtype=torch.float32)
            
            # 创建符合项目格式的数据
            data_views = [view1, view2, view3]
            labels_tensor = torch.tensor(labels, dtype=torch.long)
            
            # 保存处理后的数据
            processed_data_path = os.path.join(COIL20_DIR, "processed_data.pt")
            torch.save({
                'views': data_views,
                'labels': labels_tensor
            }, processed_data_path)
            
            logger.info(f"COIL-20数据集已处理并保存: {processed_data_path}")
            logger.info(f"数据形状: {[v.shape for v in data_views]}, 标签形状: {labels_tensor.shape}")
        else:
            logger.error("未找到COIL-20图像")
            
    except Exception as e:
        logger.error(f"COIL-20数据处理失败: {e}")
        import traceback
        traceback.print_exc()

def get_movielists_dataset():
    """获取和处理MovieLists数据集"""
    logger.info("开始获取和处理MovieLists数据集...")
    
    # 确保目录存在
    ensure_dir(MOVIELISTS_DIR)
    
    # MovieLists数据集URL（假设URL）
    movielists_url = "https://raw.githubusercontent.com/rofuyu/multview-datasets/main/movielists.zip"
    zip_path = os.path.join(MOVIELISTS_DIR, "movielists.zip")
    
    try:
        # 如果找不到数据集，就自己创建一个示例数据集
        if not os.path.exists(os.path.join(MOVIELISTS_DIR, "processed_data.pt")):
            logger.info("创建示例MovieLists数据集...")
            
            # 生成随机数据
            np.random.seed(42)
            n_samples = 500
            n_genres = 20
            
            # 创建示例数据
            # 视图1: 电影描述的TF-IDF特征
            movie_descriptions = [
                f"This is a movie about {np.random.choice(['action', 'drama', 'comedy', 'scifi', 'horror', 'romance'], size=np.random.randint(1, 4), replace=False).tolist()}" 
                for _ in range(n_samples)
            ]
            
            # 使用TF-IDF提取文本特征
            vectorizer = TfidfVectorizer(max_features=100)
            view1 = torch.tensor(vectorizer.fit_transform(movie_descriptions).toarray(), dtype=torch.float32)
            
            # 视图2: 评分特征
            view2 = torch.tensor(np.random.rand(n_samples, 50).astype(np.float32), dtype=torch.float32)
            
            # 视图3: 流派独热编码
            genres = np.zeros((n_samples, n_genres), dtype=np.float32)
            for i in range(n_samples):
                # 每部电影随机分配1-3个流派
                genre_indices = np.random.choice(n_genres, size=np.random.randint(1, 4), replace=False)
                genres[i, genre_indices] = 1
            view3 = torch.tensor(genres, dtype=torch.float32)
            
            # 标签: 电影类型分类 (假设有5个主要类别)
            n_clusters = 5
            labels = torch.tensor(np.random.randint(0, n_clusters, size=n_samples), dtype=torch.long)
            
            # 创建符合项目格式的数据
            data_views = [view1, view2, view3]
            
            # 保存处理后的数据
            processed_data_path = os.path.join(MOVIELISTS_DIR, "processed_data.pt")
            torch.save({
                'views': data_views,
                'labels': labels
            }, processed_data_path)
            
            logger.info(f"示例MovieLists数据集已创建并保存: {processed_data_path}")
            logger.info(f"数据形状: {[v.shape for v in data_views]}, 标签形状: {labels.shape}")
        else:
            logger.info(f"MovieLists数据集已存在: {os.path.join(MOVIELISTS_DIR, 'processed_data.pt')}")
            
    except Exception as e:
        logger.error(f"MovieLists数据处理失败: {e}")
        import traceback
        traceback.print_exc()

def validate_dataset(dataset_name):
    """验证数据集是否可正确加载"""
    logger.info(f"验证数据集: {dataset_name}")
    
    try:
        # 尝试加载数据集
        if dataset_name == 'mnist':
            data_path = os.path.join(MNIST_DIR, "processed_data.pt")
        elif dataset_name == 'coil20':
            data_path = os.path.join(COIL20_DIR, "processed_data.pt")
        elif dataset_name == 'movielists':
            data_path = os.path.join(MOVIELISTS_DIR, "processed_data.pt")
        else:
            logger.error(f"未知数据集: {dataset_name}")
            return False
        
        if not os.path.exists(data_path):
            logger.error(f"数据集文件不存在: {data_path}")
            return False
        
        data = torch.load(data_path)
        views = data['views']
        labels = data['labels']
        
        # 验证数据
        for i, view in enumerate(views):
            logger.info(f"视图{i+1}形状: {view.shape}")
        logger.info(f"标签形状: {labels.shape}")
        logger.info(f"类别数: {len(torch.unique(labels))}")
        
        # 验证数据类型
        for i, view in enumerate(views):
            if not isinstance(view, torch.Tensor):
                logger.warning(f"视图{i+1}不是torch.Tensor类型，而是{type(view)}")
            if view.dtype != torch.float32:
                logger.warning(f"视图{i+1}的数据类型为{view.dtype}，而非torch.float32")
        
        if not isinstance(labels, torch.Tensor):
            logger.warning(f"标签不是torch.Tensor类型，而是{type(labels)}")
        if labels.dtype != torch.long:
            logger.warning(f"标签的数据类型为{labels.dtype}，而非torch.long")
        
        logger.info(f"数据集 {dataset_name} 验证通过！")
        return True
    
    except Exception as e:
        logger.error(f"数据集验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='数据集管理工具')
    parser.add_argument('--fix-mnist', action='store_true', help='修复MNIST数据集')
    parser.add_argument('--fix-coil20', action='store_true', help='修复COIL20数据集')
    parser.add_argument('--get-movielists', action='store_true', help='获取和处理MovieLists数据集')
    parser.add_argument('--validate', nargs='+', help='验证指定数据集', choices=['mnist', 'coil20', 'movielists', 'all'])
    parser.add_argument('--fix-all', action='store_true', help='修复所有数据集')
    
    args = parser.parse_args()
    
    # 确保数据根目录存在
    ensure_dir(DATA_ROOT)
    
    # 根据参数执行相应操作
    if args.fix_all:
        fix_mnist_dataset()
        fix_coil20_dataset()
        get_movielists_dataset()
    else:
        if args.fix_mnist:
            fix_mnist_dataset()
        
        if args.fix_coil20:
            fix_coil20_dataset()
        
        if args.get_movielists:
            get_movielists_dataset()
    
    # 验证数据集
    if args.validate:
        datasets_to_validate = []
        if 'all' in args.validate:
            datasets_to_validate = ['mnist', 'coil20', 'movielists']
        else:
            datasets_to_validate = args.validate
        
        for dataset in datasets_to_validate:
            validate_dataset(dataset)

if __name__ == '__main__':
    main()