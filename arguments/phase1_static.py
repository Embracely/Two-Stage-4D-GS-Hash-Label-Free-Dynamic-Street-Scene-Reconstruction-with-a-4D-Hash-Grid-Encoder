ModelHiddenParams = dict(
    no_dx = True,  # 禁用变形场，因为我们只关注静态场景
    static_mlp = True  # 启用静态MLP处理
)

ModelParams = dict(
    stride = 10,  # 每10帧采样一次
    load_dynamic_mask = True,  # 加载动态掩码
    white_background = False,  # 黑色背景
    # 静态渲染阶段，增加预热迭代次数
    densify_from_iter = 500,
    densify_until_iter = 15000,
    densification_interval = 100
)

OptimizationParams = dict(
    # 静态场景的优化参数 - 降低学习率以提高稳定性
    position_lr_init = 0.00008,  # 降低初始学习率
    position_lr_final = 0.0000008,  # 降低最终学习率
    position_lr_delay_mult = 0.02,  # 增加延迟倍数，使学习率下降更快
    position_lr_max_steps = 30000,
    
    # 调整其他学习率
    feature_lr = 0.005,  # 降低特征学习率
    opacity_lr = 0.005,  # 降低不透明度学习率
    scaling_lr = 0.001,  # 降低缩放学习率
    rotation_lr = 0.001,  # 降低旋转学习率
    
    # 调整密集化和剪枝参数
    densify_grad_threshold_fine_init = 0.0002,  # 降低密集化梯度阈值
    opacity_threshold_fine_init = 0.02,  # 降低不透明度阈值
    
    # 增加批量大小以提高稳定性
    batch_size = 4,  # 增加批量大小
    
    # 降低动态目标的损失权重
    lambda_dssim = 0.2,
    lambda_depth = 0.05  # 降低深度损失权重
) 