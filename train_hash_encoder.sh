#!/bin/bash

# 训练脚本：使用哈希编码器替代HexPlane
# 此脚本支持训练场景重建和新视角合成任务，并可以与HexPlane编码器进行对比

# 默认参数
MODEL_PATH=""
SOURCE_PATH=""
TASK="scene_reconstruction"  # 可选: scene_reconstruction, nvs
ENCODER="hash"               # 可选: hash, hexplane
MODE="test"                  # 可选: test (快速测试), full (完整训练)
ITERATIONS=30000             # 训练迭代次数
STAGE="stage1"               # 可选: stage1, stage2

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
        --task)
            TASK="$2"
            shift
            shift
            ;;
        --encoder)
            ENCODER="$2"
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
        --stage)
            STAGE="$2"
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
    echo "使用方法: $0 --model_path <模型路径> --source_path <数据源路径> [--task <任务类型>] [--encoder <编码器类型>] [--mode <模式>] [--iterations <迭代次数>] [--stage <阶段>]"
    exit 1
fi

# 根据模式设置参数
if [ "$MODE" == "test" ]; then
    ITERATIONS=1000
    echo "运行测试模式，迭代次数: $ITERATIONS"
fi

# 配置文件选择
CONFIG_FILE=""
if [ "$TASK" == "scene_reconstruction" ]; then
    if [ "$STAGE" == "stage1" ]; then
        if [ "$ENCODER" == "hash" ]; then
            CONFIG_FILE="arguments/phase3_hash.py"
        else
            CONFIG_FILE="arguments/standard_training.py"
        fi
    else
        if [ "$ENCODER" == "hash" ]; then
            # 使用标准stage2配置，但会在代码中根据encoder_type参数选择编码器
            CONFIG_FILE="arguments/standard_stage2.py"
        else
            CONFIG_FILE="arguments/standard_stage2.py"
        fi
    fi
elif [ "$TASK" == "nvs" ]; then
    if [ "$STAGE" == "stage1" ]; then
        if [ "$ENCODER" == "hash" ]; then
            # 修改NVS配置以使用哈希编码器
            CONFIG_FILE="arguments/phase3_hash.py"
        else
            CONFIG_FILE="arguments/standard_nvs.py"
        fi
    else
        if [ "$ENCODER" == "hash" ]; then
            # 使用标准stage2 NVS配置，但会在代码中根据encoder_type参数选择编码器
            CONFIG_FILE="arguments/standard_stage2_nvs.py"
        else
            CONFIG_FILE="arguments/standard_stage2_nvs.py"
        fi
    fi
else
    echo "错误: 未知任务类型 '$TASK'"
    exit 1
fi

# 打印训练配置
echo "训练配置:"
echo "- 模型路径: $MODEL_PATH"
echo "- 数据源路径: $SOURCE_PATH"
echo "- 任务类型: $TASK"
echo "- 编码器类型: $ENCODER"
echo "- 训练模式: $MODE"
echo "- 迭代次数: $ITERATIONS"
echo "- 阶段: $STAGE"
echo "- 配置文件: $CONFIG_FILE"

# 构建训练命令
TRAIN_CMD="python train.py"
TRAIN_CMD="$TRAIN_CMD --model_path $MODEL_PATH"
TRAIN_CMD="$TRAIN_CMD --source_path $SOURCE_PATH"
TRAIN_CMD="$TRAIN_CMD --configs $CONFIG_FILE"
TRAIN_CMD="$TRAIN_CMD --iterations $ITERATIONS"

# 如果使用哈希编码器，添加encoder_type参数
if [ "$ENCODER" == "hash" ]; then
    TRAIN_CMD="$TRAIN_CMD --encoder_type hash"
fi

# 执行训练
echo "执行命令: $TRAIN_CMD"
eval $TRAIN_CMD

# 训练完成后评估
echo "训练完成，开始评估..."
python eval_metrics.py --model_path "$MODEL_PATH" --iteration "$ITERATIONS" 