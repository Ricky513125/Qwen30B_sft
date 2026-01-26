#!/bin/bash
# DeepSpeed ZeRO-3 多 GPU 训练启动脚本

# 设置环境变量
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0

# 默认参数
CONFIG_FILE="${1:-/mnt/parallel/8B_Qwen3/Gemma/config.json}"
ABLATION_CONFIG="${2:-profile_and_history_and_context}"
NUM_GPUS="${3:-4}"
MAX_GPUS="${4:-4}"

echo "=========================================="
echo "DeepSpeed ZeRO-3 多 GPU 训练"
echo "=========================================="
echo "配置文件: $CONFIG_FILE"
echo "消融配置: $ABLATION_CONFIG"
echo "GPU 数量: $NUM_GPUS"
echo "最大 GPU: $MAX_GPUS"
echo "=========================================="

# 检查 DeepSpeed 是否安装
if ! command -v deepspeed &> /dev/null; then
    echo "错误: DeepSpeed 未安装"
    echo "请运行: pip install deepspeed"
    exit 1
fi

# 使用 deepspeed 启动训练
deepspeed --num_gpus=$NUM_GPUS train_deepspeed.py \
    --config "$CONFIG_FILE" \
    --ablation_config "$ABLATION_CONFIG" \
    --max_gpus "$MAX_GPUS"
