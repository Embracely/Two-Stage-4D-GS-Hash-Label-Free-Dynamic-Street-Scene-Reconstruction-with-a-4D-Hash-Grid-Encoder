#!/bin/bash

# S3Gaussian整合训练脚本
# 执行完整的训练流程：
# 1. 静态场景训练 (5000轮)
# 2. 动态场景训练 (50000轮)，使用哈希编码器替代HexPlane

# 默认参数
MODEL_PATH=""
SOURCE_PATH=""
MODE="full"  # 可选: test (快速测试), full (完整训练)
ITERATIONS=50000  # 总训练迭代次数

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --model_path)
            MODEL_PATH="$2"
            shift
            shift
            ;;
        --source_path)
            SOURCE_PATH="$2"
            shift
            shift
            ;;
        --mode)
            MODE="$2"
            shift
            shift
            ;;
        --iterations)
            ITERATIONS="$2"
            shift
            shift
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# 检查必要参数
if [ -z "$MODEL_PATH" ] || [ -z "$SOURCE_PATH" ]; then
    echo "使用方法: $0 --model_path <模型路径> --source_path <数据源路径> [--mode <test|full>] [--iterations <迭代次数>]"
    exit 1
fi

# 根据模式设置参数
if [ "$MODE" == "test" ]; then
    COARSE_ITERATIONS=1000
    ITERATIONS=5000
    echo "运行测试模式，第一阶段迭代次数: $COARSE_ITERATIONS，第二阶段迭代次数: $ITERATIONS"
else
    COARSE_ITERATIONS=5000
    echo "运行完整训练模式，第一阶段迭代次数: $COARSE_ITERATIONS，第二阶段迭代次数: $ITERATIONS"
fi

# 配置文件
CONFIG_FILE="arguments/integrated_training.py"

# 创建输出目录
mkdir -p "$MODEL_PATH"
mkdir -p "$MODEL_PATH/eval"

# 打印训练配置
echo "训练配置:"
echo "- 模型路径: $MODEL_PATH"
echo "- 数据源路径: $SOURCE_PATH"
echo "- 训练模式: $MODE"
echo "- 第一阶段迭代次数: $COARSE_ITERATIONS"
echo "- 第二阶段迭代次数: $ITERATIONS"
echo "- 配置文件: $CONFIG_FILE"

# 恢复原始train.py文件（如果有备份）
if [ -f "train.py.backup" ]; then
    echo "发现train.py备份文件，恢复原始文件..."
    mv train.py.backup train.py
fi

# 应用哈希编码器补丁
echo "应用哈希编码器补丁..."
if [ -f "apply_hash_patch.sh" ]; then
    chmod +x apply_hash_patch.sh
    ./apply_hash_patch.sh
else
    echo "错误: 找不到apply_hash_patch.sh脚本，无法应用哈希编码器补丁"
    exit 1
fi

# 构建训练命令
TRAIN_CMD="python train.py"
TRAIN_CMD="$TRAIN_CMD --model_path $MODEL_PATH"
TRAIN_CMD="$TRAIN_CMD --source_path $SOURCE_PATH"
TRAIN_CMD="$TRAIN_CMD --configs $CONFIG_FILE"
TRAIN_CMD="$TRAIN_CMD --iterations $ITERATIONS"
TRAIN_CMD="$TRAIN_CMD --coarse_iterations $COARSE_ITERATIONS"
TRAIN_CMD="$TRAIN_CMD --use_first_stage_result"  # 使用第一阶段结果作为第二阶段起点
TRAIN_CMD="$TRAIN_CMD --encoder_type hash"  # 使用哈希编码器

# 执行训练
echo "执行命令: $TRAIN_CMD"
eval $TRAIN_CMD

# 训练完成后评估
echo "训练完成，开始评估..."
python eval_metrics.py --model_path "$MODEL_PATH" --iteration "$ITERATIONS"

echo "整合训练完成！" 