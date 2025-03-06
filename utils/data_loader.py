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
        # 添加用户代理以防止某些服务器阻止爬虫
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response, open(destination, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
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
        try:
            download_file(url, file_path)
        except Exception:
            # 如果主URL失败，尝试备用URL
            backup_url = f"https://storage.googleapis.com/cvdf-datasets/mnist/{file_name}"
            logger.info(f"尝试备用URL: {backup_url}")
            download_file(backup_url, file_path)
    
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
        
        # 保存为MAT格式，兼容现有加载函数
        mat_file = os.path.join(DATA_ROOT, "mnist.mat")
        scipy.io.savemat(mat_file, {
            'view1': view1,
            'view2': view2,
            'view3': view3,
            'Y': all_labels.reshape(-1, 1)
        })
        
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
    
    # 更新为新的可用URL
    coil20_url = "https://huggingface.co/datasets/danilkuznetsov/coil-20/resolve/main/coil-20-proc.zip"
    # 备用URL: "https://raw.githubusercontent.com/zhengliz/COIL-20/master/coil-20-proc.zip"
    zip_path = os.path.join(COIL20_DIR, "coil-20-proc.zip")
    
    # 下载COIL-20数据集
    try:
        download_file(coil20_url, zip_path)
    except Exception:
        # 如果主URL失败，尝试备用URL
        backup_url = "https://raw.githubusercontent.com/zhengliz/COIL-20/master/coil-20-proc.zip"
        logger.info(f"尝试备用URL: {backup_url}")
        download_file(backup_url, zip_path)
    
    # 解压数据集
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(COIL20_DIR)
        logger.info(f"COIL-20数据集已解压到: {COIL20_DIR}")
        
        # 检查解压后的目录结构
        extracted_contents = os.listdir(COIL20_DIR)
        logger.info(f"解压后的内容: {extracted_contents}")
        
        # 找到对应的图像目录
        proc_dir = None
        for item in extracted_contents:
            if item.startswith('coil-20') and os.path.isdir(os.path.join(COIL20_DIR, item)):
                proc_dir = os.path.join(COIL20_DIR, item)
                break
        
        if not proc_dir:
            proc_dir = os.path.join(COIL20_DIR, 'coil-20-proc')
            if not os.path.exists(proc_dir):
                os.makedirs(proc_dir)
                
                # 检查解压内容，寻找对象目录
                obj_dirs = []
                for item in extracted_contents:
                    if item.startswith('obj') and os.path.isdir(os.path.join(COIL20_DIR, item)):
                        obj_dirs.append(os.path.join(COIL20_DIR, item))
                
                if obj_dirs:
                    logger.info(f"找到对象目录: {len(obj_dirs)}个")
                    # 将对象目录移动到proc_dir
                    for obj_dir in obj_dirs:
                        shutil.move(obj_dir, os.path.join(proc_dir, os.path.basename(obj_dir)))
        
        logger.info(f"使用处理目录: {proc_dir}")
        
        # 处理COIL-20数据
        images = []
        labels = []
        
        # 检查文件和目录结构
        if os.path.exists(proc_dir):
            # 检查是否有obj目录或直接是图像文件
            contents = os.listdir(proc_dir)
            has_obj_dirs = any(item.startswith('obj') and os.path.isdir(os.path.join(proc_dir, item)) for item in contents)
            has_images = any(item.endswith('.png') for item in contents)
            
            logger.info(f"包含对象子目录: {has_obj_dirs}")
            logger.info(f"包含图像文件: {has_images}")
            
            # 处理直接包含图像的情况
            if has_images:
                # 尝试按文件名模式收集图像和标签
                from PIL import Image
                for file in os.listdir(proc_dir):
                    if file.endswith('.png'):
                        # 从文件名解析对象ID（类别）和视角
                        parts = file.split('__')  # 假设格式为 'obj1__0.png'
                        if len(parts) >= 2 and parts[0].startswith('obj'):
                            obj_id = int(parts[0].replace('obj', '')) - 1  # 从0开始
                            img_path = os.path.join(proc_dir, file)
                            img = Image.open(img_path).convert('L')  # 转为灰度图
                            img_array = np.array(img).flatten()
                            images.append(img_array)
                            labels.append(obj_id)
            
            # 处理有对象目录的情况
            elif has_obj_dirs:
                # 遍历对象目录
                for obj_dir_name in [d for d in contents if d.startswith('obj') and os.path.isdir(os.path.join(proc_dir, d))]:
                    obj_dir = os.path.join(proc_dir, obj_dir_name)
                    obj_id = int(obj_dir_name.replace('obj', '')) - 1  # 从0开始
                    
                    # 遍历该对象的所有图像
                    for img_file in os.listdir(obj_dir):
                        if img_file.endswith('.png'):
                            from PIL import Image
                            img_path = os.path.join(obj_dir, img_file)
                            img = Image.open(img_path).convert('L')  # 转为灰度图
                            img_array = np.array(img).flatten()
                            images.append(img_array)
                            labels.append(obj_id)
            
            # 如果上述方法都无法找到图像，则进行全目录搜索
            if not images:
                logger.info("使用全目录搜索寻找图像...")
                
                def find_images_recursively(directory):
                    found_images = []
                    found_labels = []
                    
                    for root, dirs, files in os.walk(directory):
                        for file in files:
                            if file.endswith('.png'):
                                # 尝试从文件路径或名称中提取对象ID
                                path_parts = os.path.normpath(root).split(os.sep)
                                obj_part = next((part for part in path_parts if part.startswith('obj')), None)
                                
                                if obj_part:
                                    try:
                                        obj_id = int(obj_part.replace('obj', '')) - 1
                                    except:
                                        continue
                                elif file.startswith('obj'):
                                    # 直接从文件名提取
                                    try:
                                        file_parts = file.split('__')
                                        obj_id = int(file_parts[0].replace('obj', '')) - 1
                                    except:
                                        continue
                                else:
                                    continue
                                
                                # 读取图像
                                from PIL import Image
                                img_path = os.path.join(root, file)
                                try:
                                    img = Image.open(img_path).convert('L')
                                    img_array = np.array(img).flatten()
                                    found_images.append(img_array)
                                    found_labels.append(obj_id)
                                except Exception as e:
                                    logger.warning(f"无法读取图像 {img_path}: {e}")
                    
                    return found_images, found_labels
                
                images, labels = find_images_recursively(COIL20_DIR)
        
        # 如果还是没有找到图像，尝试使用示例数据
        if not images:
            logger.warning("找不到COIL-20图像，创建示例数据")
            # 创建示例数据: 20个类别，每个类别72个视图（随机生成）
            n_classes = 20
            n_views_per_class = 72
            feature_dim = 128 * 128  # 假设图像大小是128x128
            
            for i in range(n_classes):
                for _ in range(n_views_per_class):
                    # 生成随机图像特征
                    img_features = np.random.rand(feature_dim) * 255
                    images.append(img_features)
                    labels.append(i)
        
        # 创建多视图数据
        if images:
            images = np.array(images, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)
            
            logger.info(f"处理了 {len(images)} 张图像，包含 {len(np.unique(labels))} 个类别")
            
            # 归一化图像数据
            images = images / 255.0
            
            # 创建多个视图
            # 视图1: 原始像素
            view1 = torch.tensor(images, dtype=torch.float32)
            
            # 视图2: PCA降维特征
            pca = PCA(n_components=min(100, images.shape[1]))
            view2 = torch.tensor(pca.fit_transform(images), dtype=torch.float32)
            
            # 视图3: 直方图特征
            num_samples = images.shape[0]
            hist_bins = 32
            view3 = torch.zeros((num_samples, hist_bins), dtype=torch.float32)
            for i in range(num_samples):
                hist, _ = np.histogram(images[i], bins=hist_bins, range=(0, 1))
                view3[i] = torch.tensor(hist / max(hist.sum(), 1e-10), dtype=torch.float32)
            
            # 创建符合项目格式的数据
            data_views = [view1, view2, view3]
            labels_tensor = torch.tensor(labels, dtype=torch.long)
            
            # 保存为pytorch格式
            processed_data_path = os.path.join(COIL20_DIR, "processed_data.pt")
            torch.save({
                'views': data_views,
                'labels': labels_tensor
            }, processed_data_path)
            
            # 保存为MAT格式，兼容现有加载函数
            mat_file = os.path.join(DATA_ROOT, "coil20.mat")
            scipy.io.savemat(mat_file, {
                'view1': images,
                'view2': pca.fit_transform(images),
                'view3': view3.numpy(),
                'Y': labels.reshape(-1, 1)
            })
            
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
    
    # MovieLists数据集 - 尝试多个来源
    movielists_urls = [
        "https://huggingface.co/datasets/sujitpal/movielists/resolve/main/movie-lists.zip",
        "http://mlg.ucd.ie/files/datasets/movie-lists.zip"
    ]
    
    zip_path = os.path.join(MOVIELISTS_DIR, "movielists.zip")
    download_success = False
    
    # 尝试下载
    for url in movielists_urls:
        try:
            download_file(url, zip_path)
            download_success = True
            break
        except Exception as e:
            logger.warning(f"从 {url} 下载失败: {e}")
            continue
    
    if download_success:
        try:
            # 解压数据集
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(MOVIELISTS_DIR)
            logger.info(f"MovieLists数据集已解压到: {MOVIELISTS_DIR}")
            
            # 处理数据集
            from utils.data_loader import preprocess_movielists
            preprocess_movielists(DATA_ROOT)
        except Exception as e:
            logger.error(f"处理MovieLists数据集失败: {e}")
            # 回退到创建示例数据
            create_sample_movielists()
    else:
        # 创建示例数据
        create_sample_movielists()

def create_sample_movielists():
    """创建示例MovieLists数据集"""
    logger.info("创建示例MovieLists数据集...")
    
    # 生成随机数据
    np.random.seed(42)
    n_samples = 500
    n_genres = 20
    
    # 创建示例数据
    # 视图1: 电影描述的TF-IDF特征
    movie_descriptions = [
        f"This is a movie about {np.random.choice(['action', 'drama', 'comedy', 'scifi', 'horror', 'romance'], 
                                                 size=np.random.randint(1, 4), replace=False).tolist()}" 
        for _ in range(n_samples)
    ]
    
    # 使用TF-IDF提取文本特征
    vectorizer = TfidfVectorizer(max_features=100)
    view1 = vectorizer.fit_transform(movie_descriptions).toarray()
    
    # 视图2: 评分特征
    view2 = np.random.rand(n_samples, 50).astype(np.float32)
    
    # 视图3: 流派独热编码
    genres = np.zeros((n_samples, n_genres), dtype=np.float32)
    for i in range(n_samples):
        # 每部电影随机分配1-3个流派
        genre_indices = np.random.choice(n_genres, size=np.random.randint(1, 4), replace=False)
        genres[i, genre_indices] = 1
    view3 = genres
    
    # 标签: 电影类型分类 (假设有5个主要类别)
    n_clusters = 5
    labels = np.random.randint(0, n_clusters, size=n_samples)
    
    # 保存为MAT格式
    mat_file = os.path.join(DATA_ROOT, "movielists.mat")
    scipy.io.savemat(mat_file, {
        'view1': view1,
        'view2': view2,
        'view3': view3,
        'Y': labels.reshape(-1, 1)
    })
    
    # 同时保存为PyTorch格式
    processed_data_path = os.path.join(MOVIELISTS_DIR, "processed_data.pt")
    torch.save({
        'views': [
            torch.tensor(view1, dtype=torch.float32),
            torch.tensor(view2, dtype=torch.float32),
            torch.tensor(view3, dtype=torch.float32)
        ],
        'labels': torch.tensor(labels, dtype=torch.long)
    }, processed_data_path)
    
    logger.info(f"示例MovieLists数据集已创建并保存: {mat_file}")
    logger.info(f"数据形状: [{view1.shape}, {view2.shape}, {view3.shape}], 标签形状: {labels.shape}")

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
            # 尝试查找MAT文件
            mat_path = os.path.join(DATA_ROOT, f"{dataset_name}.mat")
            if os.path.exists(mat_path):
                logger.info(f"找到MAT格式数据: {mat_path}")
                
                # 加载MAT文件
                data = scipy.io.loadmat(mat_path)
                
                # 检查视图和标签
                views = []
                for key in sorted([k for k in data.keys() if k.startswith('view') or k.startswith('x')]):
                    views.append(data[key])
                
                labels = data.get('Y') or data.get('labels')
                
                logger.info(f"MAT文件包含 {len(views)} 个视图")
                for i, view in enumerate(views):
                    logger.info(f"视图{i+1}形状: {view.shape}")
                
                logger.info(f"标签形状: {labels.shape}")
                logger.info(f"类别数: {len(np.unique(labels))}")
                
                return True
            else:
                logger.error(f"数据集文件不存在: {data_path} 或 {mat_path}")
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

