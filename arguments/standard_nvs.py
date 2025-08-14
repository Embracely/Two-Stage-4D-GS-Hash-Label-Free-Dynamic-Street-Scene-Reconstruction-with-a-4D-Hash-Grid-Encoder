# 新视角合成训练配置文件
ModelParams = dict(
    # 使用论文中描述的数据分割方法：每10帧选择一帧作为测试集
    stride = 10,
    # 加载动态掩码
    load_dynamic_mask = True
)

# 针对新视角合成任务的模型隐藏参数
ModelHiddenParams = dict(
    # 禁用dx正则化，优化新视角合成任务
    no_dx = True
)

OptimizationParams = dict(
    # 正式训练迭代次数
    coarse_iterations = 5000,
    iterations = 50000,
    # 设置初始学习率
    position_lr_init = 0.00016,
    position_lr_final = 0.0000016,
    position_lr_delay_mult = 0.01,
    position_lr_max_steps = 30000,
    # 特征学习率
    feature_lr = 0.0025,
    # 不透明度和缩放学习率
    opacity_lr = 0.05,
    scaling_lr = 0.005,
    # 旋转学习率
    rotation_lr = 0.001,
    # 设置修剪参数，避免灾难性修剪
    # 减小不透明度阈值
    opacity_threshold_coarse = 0.005,
    opacity_threshold_fine_init = 0.005,
    opacity_threshold_fine_after = 0.005,
    # 增加修剪间隔
    pruning_interval = 1000,
    # 增加安全限制，防止过度修剪
    max_points = 1500000,
) 