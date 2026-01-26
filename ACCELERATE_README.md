# 使用 Accelerate 库进行 DeepSpeed 训练

本训练脚本已重构为使用 **Accelerate 库**作为桥梁来使用 DeepSpeed，这是工业界推荐的最佳实践。

## 为什么使用 Accelerate？

1. **简化配置**：不再需要手动处理 `HfDeepSpeedConfig`、`local_rank` 等复杂参数
2. **自动处理**：Trainer 会自动处理 DeepSpeed 的初始化、模型分片等细节
3. **标准化**：使用标准的 `accelerate launch` 命令，避免参数冲突

## 快速开始

### 方法 1：使用 accelerate launch（推荐）

```bash
# 直接指定 GPU 数量
accelerate launch --num_processes=4 Gemma/train_deepspeed.py \
    --config config.json \
    --ablation_config profile_and_history \
    --output_dir Gemma/outputs/your_output_dir
```

### 方法 2：使用 accelerate config（更灵活）

1. **首次配置 Accelerate**：
   ```bash
   accelerate config
   ```
   
   交互式配置选项：
   - 选择 "This machine"
   - 选择 GPU 数量（例如：4）
   - 选择 "DeepSpeed" 作为分布式类型
   - 选择 "ZeRO Stage 3"
   - 选择 CPU offload（推荐：optimizer 和 parameters 都 offload 到 CPU）
   - 其他选项使用默认值

2. **启动训练**：
   ```bash
   accelerate launch Gemma/train_deepspeed.py \
       --config config.json \
       --ablation_config profile_and_history \
       --output_dir Gemma/outputs/your_output_dir
   ```

## DeepSpeed 配置文件

训练脚本会自动创建 DeepSpeed 配置文件，保存在输出目录的 `deepspeed_config/ds_config_zero3.json`。

你也可以提供自己的配置文件：

```bash
accelerate launch Gemma/train_deepspeed.py \
    --config config.json \
    --ablation_config profile_and_history \
    --deepspeed_config /path/to/your/ds_config.json \
    --output_dir Gemma/outputs/your_output_dir
```

## 自动生成的 DeepSpeed 配置

脚本会自动生成包含以下优化的 DeepSpeed ZeRO-3 配置：

- ✅ ZeRO Stage 3（参数分片）
- ✅ CPU Offload（优化器和参数）
- ✅ Activation Checkpointing（CPU checkpointing）
- ✅ BF16 精度（H100 GPU）
- ✅ 自动 batch size 和梯度累积

## 与旧版本的区别

### 旧版本（手动配置）
```bash
deepspeed --num_gpus=4 train_deepspeed.py --config config.json ...
```

### 新版本（使用 Accelerate）
```bash
accelerate launch --num_processes=4 train_deepspeed.py --config config.json ...
```

**主要改进**：
- ❌ 不再需要手动设置 `CUDA_VISIBLE_DEVICES`
- ❌ 不再需要手动处理 `local_rank` 参数
- ❌ 不再需要手动初始化 `HfDeepSpeedConfig`
- ✅ Trainer 自动处理所有 DeepSpeed 相关逻辑
- ✅ 更简洁、更标准的启动方式

## 故障排除

### 问题 1：找不到 accelerate 命令

```bash
pip install accelerate
```

### 问题 2：模型初始化错误

确保使用 `accelerate launch` 启动，而不是直接运行 Python 脚本。

### 问题 3：GPU 内存不足

在 DeepSpeed 配置中已经启用了 CPU offload，如果仍然 OOM，可以：
- 降低 `max_length`（在 config.json 中）
- 降低 `batch_size`（在 config.json 中）
- 增加 `gradient_accumulation_steps`

## 参考资源

- [Accelerate 官方文档](https://huggingface.co/docs/accelerate)
- [DeepSpeed ZeRO 文档](https://www.deepspeed.ai/tutorials/zero/)
- [Hugging Face Trainer + DeepSpeed](https://huggingface.co/docs/transformers/main/en/deepspeed)
