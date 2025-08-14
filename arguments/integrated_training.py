"""
S3Gaussian整合训练配置文件
结合了三个阶段的功能：
1. 静态场景训练 (5000轮)
2. 动态场景训练 (50000轮)，使用哈希编码器替代HexPlane
"""

from arguments.base import BaseConfig
import torch

class RuntimeConfig(BaseConfig):
    # 数据集配置
    stride = 10  # 每10帧选择一帧作为测试集，确保训练集:测试集 = 9:1
    load_dynamic_mask = True  # 加载动态掩码
    white_background = False  # 使用黑色背景
    
    # 训练阶段配置
    coarse_iterations = 5000  # 第一阶段静态场景训练迭代次数
    iterations = 50000  # 第二阶段动态场景训练迭代次数
    
    # 使用哈希编码器替代HexPlane
    encoder_type = 'hash'
    
    # 哈希编码器配置
    hash_config = {
        'n_levels': 16,             # 哈希编码的层数
        'min_resolution': 16,       # 最低分辨率
        'max_resolution': 512,      # 最高分辨率
        'log2_hashmap_size': 15,    # 哈希表大小 (2^15 = 32768)
        'feature_dim': 2,           # 每层特征维度
    }
    
    # 变形网络配置
    no_dx = False                   # 启用变形场，使高斯点能够移动
    static_mlp = True               # 保持静态MLP用于区分静态和动态点
    deformation_hidden_dim = 64
    deformation_layers = 3
    deformation_feature_dim = 32
    deformation_feature_std = 0.01
    deformation_feature_bias = 0.0
    deformation_base_resolution = 16
    deformation_grid_size = [128, 128, 128, 100]
    
    # 学习率配置
    position_lr_init = 0.00016
    position_lr_final = 0.0000016
    position_lr_delay_mult = 0.01
    position_lr_max_steps = 30000
    feature_lr = 0.0025
    opacity_lr = 0.05
    scaling_lr = 0.005
    rotation_lr = 0.001
    percent_dense = 0.01
    
    # 变形网络学习率
    deformation_lr_init = 0.0001
    deformation_lr_final = 0.000001
    deformation_lr_delay_mult = 0.01
    grid_lr_init = 0.001
    grid_lr_final = 0.00001
    
    # 密集化和修剪参数
    densify_from_iter = 500
    densify_until_iter = 15000
    densification_interval = 100
    opacity_threshold_coarse = 0.005
    opacity_threshold_fine_init = 0.005
    opacity_threshold_fine_after = 0.005
    pruning_interval = 1000
    max_points = 1500000
    opacity_reset_interval = 3000
    
    # 正则化权重
    lambda_dssim = 0.2
    densify_grad_threshold = 0.0002
    time_smoothness_weight = 0.01
    l1_time_planes = 0.0001
    plane_tv = 0.0001
    
    # 连续训练配置
    use_first_stage_result = True  # 使用第一阶段训练结果作为第二阶段起点
    
    # 评估配置
    test_iterations = [1000, 3000, 5000, 10000, 20000, 30000, 40000, 50000]
    save_iterations = [5000, 10000, 20000, 30000, 40000, 50000]
    checkpoint_iterations = [5000, 10000, 20000, 30000, 40000, 50000] 