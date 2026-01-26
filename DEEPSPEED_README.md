# DeepSpeed ZeRO-3 多 GPU 训练使用说明

## 问题诊断

如果遇到 `CUDA out of memory` 错误，即使使用了 DeepSpeed，可能的原因包括：

1. **模型在 DeepSpeed 初始化前加载**：模型会立即占用所有显存
2. **序列长度太大**：4096 的序列长度对于 27B 模型仍然很大
3. **Batch size 太大**：即使使用 ZeRO-3，batch size 也需要保持较小
4. **DeepSpeed 配置不正确**：需要正确配置 CPU offload

## 解决方案

### 1. 使用优化后的配置

代码已经优化：
- ✅ 使用 `HfDeepSpeedConfig` 正确初始化 DeepSpeed
- ✅ 启用 CPU offload（优化器和参数）
- ✅ 启用 activation checkpointing（CPU checkpointing）
- ✅ 降低 dataloader 相关设置

### 2. 降低序列长度（推荐）

在 `config.json` 中降低 `max_length`：

```json
{
  "training": {
    "max_length": 2048,  // 从 4096 降低到 2048
    "batch_size": 1,
    "gradient_accumulation_steps": 4
  }
}
```

### 3. 确保使用正确的启动方式

**重要**：必须使用 `deepspeed` 命令启动训练，而不是直接运行 Python 脚本！

```bash
# 错误的方式（会导致 OOM）
python train_deepspeed.py --config config.json --ablation_config profile_and_history

# 正确的方式
deepspeed --num_gpus=4 Gemma/train_deepspeed.py \
    --config config.json \
    --ablation_config profile_and_history \
    --output_dir Gemma/outputs/0123_Gemma_RealPersonaChat_profile_and_history_and_context_1100


# 如果要在同一台机器上运行多个训练任务，需要指定不同的GPU
# 方法1：使用环境变量 CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 Gemma/train_deepspeed.py \
    --config config_realpersonachat.json \
    --ablation_config profile_and_history_and_context \
    --output_dir Gemma/outputs/0123_Gemma_RealPersonaChat_profile_and_history_and_context_1400

CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num_processes=4 Gemma/train_deepspeed.py \
    --config config.json \
    --ablation_config profile_and_history_and_context \
    --output_dir Gemma/outputs/0123_Gemma_LovinkDialogue_profile_and_history_and_context_1400

accelerate launch --num_processes=4 Gemma/train_deepspeed.py \
    --config config.json \
    --ablation_config profile_and_context \
    --gpu_ids 4,5,6,7 \
    --output_dir Gemma/outputs/0123_Gemma_LovinkDialogue_profile_and_context_1400


# 方法2：使用 --gpu_ids 参数（推荐，更清晰）
accelerate launch --num_processes=4 Gemma/train_deepspeed.py \
    --config config_realpersonachat.json \
    --ablation_config profile_and_history_and_context \
    --gpu_ids 0,1,2,3 \
    --output_dir Gemma/outputs/0123_Gemma_RealPersonaChat_profile_and_history_and_context_1400

accelerate launch --num_processes=4 Gemma/train_deepspeed.py \
    --config config.json \
    --ablation_config profile_and_history_and_context \
    --gpu_ids 4,5,6,7 \
    --output_dir Gemma/outputs/0123_Gemma_LovinkDialogue_profile_and_history_and_context_1400
```

或者创建一个启动脚本：

```bash
#!/bin/bash
# train_deepspeed_launch.sh

deepspeed --num_gpus=4 train_deepspeed.py \
    --config config.json \
    --ablation_config profile_and_history_and_context \
    --max_gpus 4
```

### 4. 环境变量设置

在训练前设置以下环境变量：

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0
```

### 5. 检查 DeepSpeed 是否正确初始化

训练开始时应该看到类似输出：

```
✓ DeepSpeed 配置文件: /path/to/deepspeed_config/ds_config_zero3.json
✓ DeepSpeed 配置已初始化
✓ 使用 4 张 GPU 进行训练
✓ 启用 DeepSpeed ZeRO-3
✓ 优化器和参数将 offload 到 CPU
✓ 激活检查点已启用（CPU checkpointing）
```

## 推荐的配置

对于 Gemma-3-27B 模型，推荐配置：

```json
{
  "training": {
    "batch_size": 1,
    "gradient_accumulation_steps": 4,
    "max_length": 2048,  // 或更小
    "learning_rate": 1e-5,
    "warmup_steps": 100
  }
}
```

## 显存优化技巧

1. **使用 CPU Offload**：已在配置中启用
2. **降低序列长度**：从 4096 降到 2048 或 1024
3. **使用梯度累积**：保持 batch_size=1，增加 gradient_accumulation_steps
4. **启用 Activation Checkpointing**：已在配置中启用
5. **使用 Flash Attention 2**：如果已安装，会自动启用

## 故障排除

### 问题：仍然 OOM

1. 检查是否使用了 `deepspeed` 命令启动
2. 降低 `max_length` 到 1024 或更小
3. 确保 `batch_size=1`
4. 检查 DeepSpeed 配置文件是否正确生成

### 问题：训练速度慢

1. 减少 CPU offload（如果显存足够）
2. 增加 `dataloader_num_workers`
3. 使用 Flash Attention 2

### 问题：DeepSpeed 未初始化

确保：
1. 已安装 DeepSpeed：`pip install deepspeed`
2. 使用 `deepspeed` 命令启动
3. 检查 DeepSpeed 配置文件路径是否正确

## 验证 DeepSpeed 是否工作

训练日志中应该看到 DeepSpeed 相关信息，而不是标准的 PyTorch DDP 信息。
