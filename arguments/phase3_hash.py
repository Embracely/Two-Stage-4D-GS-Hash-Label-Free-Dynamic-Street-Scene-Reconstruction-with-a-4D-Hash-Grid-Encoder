"""
Phase 3 configuration: Replace HexPlane with HashEncoder4D
"""

ModelHiddenParams = dict(
    # 启用哈希编码器替代HexPlane
    encoder_type = 'hash',  # 'hash' or 'hexplane'
    
    # 哈希编码器配置
    hash_config = {
        'n_levels': 16,             # 哈希编码的层数
        'min_resolution': 16,       # 最低分辨率
        'max_resolution': 512,      # 最高分辨率
        'log2_hashmap_size': 15,    # 哈希表大小 (2^15 = 32768)
        'feature_dim': 2,           # 每层特征维度
    },
    
    # 保留原有配置
    no_dx = False,                  # 启用变形场，使高斯点能够移动
    static_mlp = True,              # 保持静态MLP用于区分静态和动态点
    
    # 保留kplanes_config用于兼容原有代码和对照实验
    kplanes_config = {
        'grid_dimensions': 2,
        'input_coordinate_dim': 4,  # x,y,z,t
        'output_coordinate_dim': 8, # 从16减小到8，减少内存使用
        'resolution': [24, 24, 24, 8] # 降低分辨率以减少内存使用
    },
    multires = [1, 2, 4],           # 减少多分辨率层次，从[1,2,4,8]变为[1,2,4]
    
    # 启用特征头，用于识别动态/静态区域
    feat_head = True,
    
    # 设置网络宽度和深度
    net_width = 32,                 # 与静态模型保持一致
    defor_depth = 1                 # 与静态模型保持一致
)

ModelParams = dict(
    stride = 1,                     # 每帧都采样以捕获连续运动
    load_dynamic_mask = True,       # 加载动态掩码辅助训练
    white_background = False,       # 黑色背景
    
    # 在细化阶段使用的密集化参数
    densify_from_iter = 500,
    densify_until_iter = 15000,
    densification_interval = 100
) 