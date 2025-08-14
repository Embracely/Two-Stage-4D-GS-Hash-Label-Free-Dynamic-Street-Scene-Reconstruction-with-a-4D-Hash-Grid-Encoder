ModelHiddenParams = dict(
    no_dx = False,  # 启用变形场，使高斯点能够移动
    static_mlp = True,  # 保持静态MLP用于区分静态和动态点
    # 使用hexplane时空场进行位移场编码
    kplanes_config = {
        'grid_dimensions': 2,
        'input_coordinate_dim': 4,  # x,y,z,t
        'output_coordinate_dim': 8,  # 从16减小到8，减少内存使用
        'resolution': [24, 24, 24, 8]  # 降低分辨率以减少内存使用
    },
    multires = [1, 2, 4],  # 减少多分辨率层次，从[1,2,4,8]变为[1,2,4]
    # 启用特征头，用于识别动态/静态区域
    feat_head = True,
    # 设置网络宽度和深度，与phase1保持一致
    net_width = 32,  # 与静态模型保持一致
    defor_depth = 1   # 与静态模型保持一致
)

ModelParams = dict(
    stride = 1,  # 每帧都采样以捕获连续运动
    load_dynamic_mask = True,  # 加载动态掩码辅助训练
    white_background = False,  # 黑色背景
    # 在细化阶段使用的密集化参数
    densify_from_iter = 500,
    densify_until_iter = 15000,
    densification_interval = 100
)

OptimizationParams = dict(
    # 动态场景的优化参数
    position_lr_init = 0.00016,
    position_lr_final = 0.0000016,
    position_lr_delay_mult = 0.01,
    position_lr_max_steps = 30000,
    
    # 变形场的学习率
    deformation_lr_init = 0.0001,  # 增加变形场学习率
    deformation_lr_final = 0.00001,
    deformation_lr_delay_mult = 0.01,
    
    # 网格学习率
    grid_lr_init = 0.001,  # 增加网格学习率
    grid_lr_final = 0.0001,
    
    # 其他参数
    feature_lr = 0.005,
    opacity_lr = 0.05,
    scaling_lr = 0.005,
    rotation_lr = 0.001,
    
    # 批处理大小
    batch_size = 2,  # 从4减小到2，减少内存使用
    
    # 损失权重
    lambda_dssim = 0.2,
    lambda_depth = 0.05,
    
    # 正则化项
    lambda_dx = 0.001,  # 变形场正则化
    dx_reg = True,
    
    # 添加内存优化相关参数
    max_points = 400000,  # 限制最大点数
    
    # 修复点云崩溃问题的关键参数
    pruning_interval = 100,  # 降低修剪频率，避免过度修剪
    opacity_threshold_coarse = 0.005,  # 降低不透明度阈值，避免过度修剪
    opacity_threshold_fine_init = 0.001,  # 降低不透明度阈值，避免过度修剪
    opacity_threshold_fine_after = 0.001,  # 降低不透明度阈值，避免过度修剪
    densify_grad_threshold_coarse = 0.0002,  # 降低梯度阈值，增加点的保留
    densify_grad_threshold_fine_init = 0.0002,  # 降低梯度阈值，增加点的保留
    densify_grad_threshold_after = 0.00005  # 降低梯度阈值，增加点的保留
) 