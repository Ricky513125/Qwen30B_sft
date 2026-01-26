#!/bin/bash
# Gemma 模型自动检测空闲GPU并行推理脚本
# 自动检测空闲GPU，在空闲GPU上分布数据进行推理

# 配置参数
SCENARIO_PATH="/mnt/parallel/GIDigitalTwinBench/IdealSelf/LovinkDialogue"
CHECKPOINT_DIR="Gemma/outputs/0123_Gemma_LovinkDialogue_profile_and_history_and_context_1400"
CONFIG_NAME="profile_and_history_and_context"
OUTPUT_DIR="Gemma/testleaderboards/0125_LovinkDialogue_profile_and_history_and_context_1400"
BASE_MODEL_PATH="/mnt/parallel/models/gemma-3-27b-it"

# 推理参数
USE_PROFILE="--use_profile"
USE_HISTORY="--use_history"
USE_CONTEXT="--use_context"
NUM_SAMPLES=5
MAX_NEW_TOKENS=4096
MAX_OUTPUT_LENGTH=4096

# GPU检测参数
MEMORY_THRESHOLD_MB=1000  # 显存使用阈值（MB），低于此值认为GPU空闲
UTILIZATION_THRESHOLD=10  # GPU利用率阈值（%），低于此值认为GPU空闲
MAX_GPUS=8  # 最多使用的GPU数量

# 创建输出目录
mkdir -p ${OUTPUT_DIR}

# 函数：检测空闲GPU
find_free_gpus() {
    echo "检测空闲GPU..."
    free_gpus=()
    
    # 获取所有GPU信息
    while IFS=',' read -r gpu_id memory_used memory_total utilization; do
        gpu_id=$(echo $gpu_id | xargs)
        memory_used=$(echo $memory_used | xargs)
        memory_total=$(echo $memory_total | xargs)
        utilization=$(echo $utilization | xargs)
        
        memory_free=$((memory_total - memory_used))
        
        # 检查是否空闲
        if [ $memory_free -gt $MEMORY_THRESHOLD_MB ] && [ $utilization -lt $UTILIZATION_THRESHOLD ]; then
            free_gpus+=($gpu_id)
            echo "  GPU ${gpu_id}: 显存 ${memory_free}MB 空闲, 利用率 ${utilization}%"
        fi
    done < <(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits)
    
    # 限制GPU数量
    if [ ${#free_gpus[@]} -gt $MAX_GPUS ]; then
        free_gpus=("${free_gpus[@]:0:$MAX_GPUS}")
    fi
    
    echo "✓ 找到 ${#free_gpus[@]} 个空闲GPU: ${free_gpus[*]}"
    echo "${free_gpus[@]}"
}

# 检测空闲GPU
FREE_GPUS=($(find_free_gpus))

if [ ${#FREE_GPUS[@]} -eq 0 ]; then
    echo "错误: 未找到空闲GPU，退出"
    exit 1
fi

NUM_SHARDS=${#FREE_GPUS[@]}
echo "将使用 ${NUM_SHARDS} 张GPU进行数据并行推理"

# 检查GPU是否可用
echo ""
echo "检查GPU状态..."
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

# 启动进程，每个进程使用一张空闲GPU
declare -a PIDS=()
SHARD_ID=0

for GPU_ID in "${FREE_GPUS[@]}"; do
    echo ""
    echo "=========================================="
    echo "启动 GPU ${GPU_ID} 的推理进程 (Shard ${SHARD_ID}/${NUM_SHARDS})..."
    echo "=========================================="
    
    # 在后台运行，每个进程使用不同的GPU和数据分片
    nohup env CUDA_VISIBLE_DEVICES=${GPU_ID} python Gemma/inference_no_vllm.py \
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
        --data_shard_id ${SHARD_ID} \
        --num_shards ${NUM_SHARDS} \
        > ${OUTPUT_DIR}/gpu_${GPU_ID}.log 2>&1 &
    
    PID=$!
    PIDS+=($PID)
    echo "GPU ${GPU_ID} 进程已启动 (PID: ${PID}, Shard: ${SHARD_ID})"
    
    # 等待几秒，确保进程正确初始化并隔离GPU
    echo "等待进程初始化..."
    sleep 5
    
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
    
    SHARD_ID=$((SHARD_ID + 1))
done

echo ""
echo "=========================================="
echo "所有 ${NUM_SHARDS} 个进程已启动"
echo "=========================================="
echo "使用以下命令查看进度:"
echo "  tail -f ${OUTPUT_DIR}/gpu_*.log"
echo "  watch -n 1 nvidia-smi"
echo "  ps aux | grep inference_no_vllm.py"
echo ""
echo "等待所有进程完成..."

# 等待所有后台进程完成
for PID in "${PIDS[@]}"; do
    wait ${PID}
done

echo ""
echo "=========================================="
echo "所有进程已完成！"
echo "=========================================="
echo "结果文件保存在: ${OUTPUT_DIR}/test_leaderboard_shard_*.json"
echo ""
echo "合并结果文件..."
echo "  python -c \"import json, glob, os; files=sorted(glob.glob('${OUTPUT_DIR}/test_leaderboard_shard_*.json')); data=[]; [data.extend(json.load(open(f))) for f in files]; json.dump(data, open('${OUTPUT_DIR}/test_leaderboard.json','w'), ensure_ascii=False, indent=2)\""
