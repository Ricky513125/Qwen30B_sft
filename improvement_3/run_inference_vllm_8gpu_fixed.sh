#!/bin/bash
# 8卡vLLM并行推理启动脚本（修复版）
# 每个GPU运行一个vLLM实例，处理1/8的数据
# 确保每个进程只使用一张GPU

# 配置参数
SCENARIO_PATH="/mnt/parallel/GIDigitalTwinBench/IdealSelf/LovinkDialogue"
CHECKPOINT_DIR="/mnt/parallel/models/Qwen3-30B-A3B-Instruct-2507"
CONFIG_NAME="profile_and_history_and_context"
OUTPUT_DIR="improvement_3/testleaderboards/0121Q30B_Lovink_phc_vllm"
BASE_MODEL_PATH="/mnt/parallel/models/Qwen3-30B-A3B-Instruct-2507"

# 推理参数
USE_PROFILE="--use_profile"
USE_HISTORY="--use_history"
USE_CONTEXT="--use_context"
NUM_SAMPLES=5
MAX_NEW_TOKENS=512
MAX_OUTPUT_LENGTH=512

# vLLM参数
VLLM_TENSOR_PARALLEL_SIZE=1  # 单卡运行
# VLLM_MAX_MODEL_LEN=2048  # 减少到2048以节省内存（Qwen3-30B MoE模型需要更多内存）
VLLM_MAX_MODEL_LEN=1536  # 减少到2048以节省内存（Qwen3-30B MoE模型需要更多内存）

# 数据分片参数
NUM_SHARDS=8

# 创建输出目录
mkdir -p ${OUTPUT_DIR}

# 检查GPU是否可用
echo "检查GPU状态..."
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

# 启动8个进程，每个进程使用一张GPU
for GPU_ID in {0..7}; do
    echo ""
    echo "=========================================="
    echo "启动 GPU ${GPU_ID} 的推理进程..."
    echo "=========================================="
    
    # 检查该GPU是否被占用
    GPU_IN_USE=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | awk -v gpu=${GPU_ID} -F', ' '$1==gpu && $2>1000 {print "yes"}')
    if [ "$GPU_IN_USE" = "yes" ]; then
        echo "警告: GPU ${GPU_ID} 可能已被占用，继续启动..."
    fi
    
    # 在后台运行，每个进程使用不同的GPU和数据分片
    # 关键：使用env命令确保CUDA_VISIBLE_DEVICES在子进程中正确设置
    # 使用nohup确保进程在后台运行
    nohup env CUDA_VISIBLE_DEVICES=${GPU_ID} python improvement_3/inference.py \
        --checkpoint_dir ${CHECKPOINT_DIR} \
        --scenario_path ${SCENARIO_PATH} \
        --config_name ${CONFIG_NAME} \
        ${USE_PROFILE} \
        ${USE_HISTORY} \
        ${USE_CONTEXT} \
        --gpu 0 \
        --num_samples ${NUM_SAMPLES} \
        --output_dir ${OUTPUT_DIR} \
        --base_model_path ${BASE_MODEL_PATH} \
        --max_new_tokens ${MAX_NEW_TOKENS} \
        --max_output_length ${MAX_OUTPUT_LENGTH} \
        --use_vllm \
        --vllm_tensor_parallel_size ${VLLM_TENSOR_PARALLEL_SIZE} \
        --vllm_max_model_len ${VLLM_MAX_MODEL_LEN} \
        --data_shard_id ${GPU_ID} \
        --num_shards ${NUM_SHARDS} \
        > ${OUTPUT_DIR}/gpu_${GPU_ID}.log 2>&1 &
    
    PID=$!
    echo "GPU ${GPU_ID} 进程已启动 (PID: ${PID})"
    
    # 等待几秒，确保进程正确初始化并隔离GPU
    echo "等待进程初始化..."
    sleep 8  # 增加等待时间，确保GPU隔离和vLLM初始化
    
    # 验证进程是否在运行
    if ps -p ${PID} > /dev/null 2>&1; then
        echo "  ✓ GPU ${GPU_ID} 进程运行正常 (PID: ${PID})"
        # 检查GPU使用情况
        echo "  GPU ${GPU_ID} 内存使用:"
        nvidia-smi --id=${GPU_ID} --query-gpu=memory.used,memory.total --format=csv,noheader | awk -F', ' '{printf "    已用: %s / 总计: %s\n", $1, $2}'
    else
        echo "  ✗ GPU ${GPU_ID} 进程启动失败，请检查日志: ${OUTPUT_DIR}/gpu_${GPU_ID}.log"
        echo "  查看错误: tail -20 ${OUTPUT_DIR}/gpu_${GPU_ID}.log"
    fi
done

echo ""
echo "=========================================="
echo "所有8个进程已启动"
echo "=========================================="
echo "使用以下命令查看进度:"
echo "  tail -f ${OUTPUT_DIR}/gpu_*.log"
echo "  watch -n 1 nvidia-smi"
echo "  ps aux | grep inference.py"
echo ""
echo "等待所有进程完成..."

# 等待所有后台进程完成
wait

echo ""
echo "=========================================="
echo "所有进程已完成！"
echo "=========================================="
echo "结果文件保存在: ${OUTPUT_DIR}/test_leaderboard_shard_*.json"
echo ""
echo "合并结果文件:"
echo "  python improvement_3/merge_shards.py --output_dir ${OUTPUT_DIR} --num_shards ${NUM_SHARDS}"
