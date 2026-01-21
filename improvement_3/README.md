# Improvement 3 - 三因素消融实验

## 概述

Improvement 3 是一个增强版的消融实验框架，支持对 **Profile**、**History** 和 **Context** 三种因素进行独立的消融实验。相比 Improvement 2，主要增强包括：

1. **完整加载 Profile 和 History**：自动加载所有用户信息
2. **History 缓存机制**：参考 Improvement 2，实现高效的 history 缓存
3. **智能 History 选择**：当 history 过长时，自动选择与当前 context 最相关的条目
4. **Context 消融支持**：新增对当前对话上下文的消融实验

## 主要特性

### 1. 三因素消融实验

支持以下 7 种消融配置：

- `profile_and_history_and_context`: 使用所有因素（完整配置）
- `profile_and_history`: 使用 Profile + History
- `profile_and_context`: 使用 Profile + Context
- `history_and_context`: 使用 History + Context
- `profile_only`: 仅使用 Profile
- `history_only`: 仅使用 History
- `context_only`: 仅使用 Context

### 2. History 缓存功能

- **自动缓存**：首次计算后自动保存到缓存
- **快速加载**：后续直接从缓存加载，显著提升训练速度
- **数据一致性**：自动排除当前 context 中的内容，避免数据泄露
- **批量构建**：支持预先构建所有用户的缓存

### 3. 智能 History 选择

当 history 长度超过限制时，系统会：
- 计算每条 history 与当前 context 的相似度（基于关键词重叠）
- 选择最相关的几条 history
- 保持时间顺序，优先保留相似度高的条目

## 文件结构

```
improvement_3/
├── README.md                      # 本文档
├── config.json                    # LovinkDialogue 数据集配置
├── config_realpersonachat.json    # RealPersonaChat 数据集配置
├── data_loader.py                 # 数据加载模块（集成缓存和智能选择）
├── history_cache.py               # History 缓存模块
├── prompt_builder.py              # Prompt 构建模块（支持三因素消融）
├── trainer.py                      # 训练器模块
├── train.py                       # 基础训练脚本
├── train_with_early_stopping.py   # 带早停的训练脚本
├── train_with_early_stopping_one_user.py  # 单用户训练脚本
├── inference.py                   # 推理脚本
└── history_cache/                 # 缓存目录（自动创建）
    ├── {user_hash}_history.json
    └── ...
```

## 快速开始

### 0. 多GPU训练说明（Qwen3-30B模型）

对于Qwen3-30B这样的大模型，推荐使用8张GPU进行训练和推理：

**训练时：**
- 使用 `--use_multi_gpu` 参数自动启用多GPU训练
- 或使用 `--gpu 0,1,2,3,4,5,6,7` 指定8张GPU
- 模型会自动使用 `device_map="auto"` 进行模型并行，并使用DDP进行数据并行

**推理时：**
- 使用 `--use_multi_gpu` 参数启用多GPU推理
- 模型会自动分布到所有可用GPU上

**注意事项：**
- 确保有足够的GPU内存（每张GPU至少20GB）
- 多GPU训练会自动使用分布式数据并行（DDP）
- 模型权重会自动分片到多个GPU上

### 1. 配置消融实验

编辑 `config.json` 或 `config_realpersonachat.json`，选择消融配置：

```json
{
  "ablation_configs": {
    "profile_and_history_and_context": {
      "use_profile": true,
      "use_history": true,
      "use_context": true,
      "name": "profile_and_history_and_context"
    },
    "profile_only": {
      "use_profile": true,
      "use_history": false,
      "use_context": false,
      "name": "profile_only"
    }
    // ... 其他配置
  }
}
```

### 2. 训练模型

#### 基础训练（不带早停）

**单GPU训练：**
```bash
python ablation/improvement_3/train.py \
    --config ablation/improvement_3/config.json \
    --ablation_config profile_and_history_and_context \
    --gpu 0 \
    --val_ratio 0.1
```

**8张GPU训练（推荐用于Qwen3-30B模型）：**
```bash
python ablation/improvement_3/train.py \
    --config ablation/improvement_3/config.json \
    --ablation_config profile_and_history_and_context \
    --gpu 0,1,2,3,4,5,6,7 \
    --use_multi_gpu \
    --val_ratio 0.1
```

#### 带早停的训练（推荐）

**单GPU训练：**
```bash
python ablation/improvement_3/train_with_early_stopping.py \
    --config ablation/improvement_3/config.json \
    --ablation_config profile_and_history_and_context \
    --gpu 0 \
    --val_ratio 0.1 \
    --max_epochs 20 \
    --early_stopping_patience 3
```

**8张GPU训练（推荐用于Qwen3-30B模型）：**
```bash
python ablation/improvement_3/train_with_early_stopping.py \
    --config ablation/improvement_3/config.json \
    --ablation_config profile_and_history_and_context \
    --gpu 0,1,2,3,4,5,6,7 \
    --use_multi_gpu \
    --val_ratio 0.1 \
    --max_epochs 20 \
    --early_stopping_patience 3 \
    --output_dir /mnt/parallel/checkpoints/Qwen3-30B_profile_and_history_and_context
```

**或者使用自动检测所有GPU：**
```bash
python ablation/improvement_3/train_with_early_stopping.py \
    --config ablation/improvement_3/config.json \
    --ablation_config profile_and_history_and_context \
    --use_multi_gpu \
    --val_ratio 0.1 \
    --max_epochs 20 \
    --early_stopping_patience 3
```

**其他示例：**
```bash
python ablation/improvement_3/train_with_early_stopping.py \
    --config ablation/improvement_3/config.json \
    --ablation_config profile_and_context \
    --gpu 0,1,2,3,4,5,6,7 \
    --use_multi_gpu \
    --val_ratio 0.1 \
    --max_epochs 20 \
    --early_stopping_patience 3 \
    --output_dir /mnt/parallel/checkpoints/0118Lovink/LovinkDialogue_profile_and_context_16
```

### 3. 推理生成

**单GPU推理：**
```bash
# 基线模型
python improvement_3/inference.py \
    --checkpoint_dir /mnt/parallel/models/Qwen3-30B-A3B-Instruct-2507 \
    --scenario_path /mnt/parallel/GIDigitalTwinBench/IdealSelf/LovinkDialogue \
    --config_name profile_only \
    --use_profile \
    --gpu 0 \
    --num_samples 5
```

**8张GPU推理（推荐用于Qwen3-30B模型）：**
```bash
python improvement_3/inference.py \
    --checkpoint_dir /mnt/parallel/models/Qwen3-30B-A3B-Instruct-2507 \
    --scenario_path /mnt/parallel/GIDigitalTwinBench/IdealSelf/LovinkDialogue \
    --config_name profile_and_history_and_context \
    --use_profile \
    --use_history \
    --use_context \
    --use_multi_gpu \
    --num_samples 5
```

--output_dir
python ablation/improvement_3/inference.py \
    --checkpoint_dir /path/to/checkpoint \
    --scenario_path /path/to/scenario \
    --config_name profile_and_history_and_context \
    --use_profile \
    --use_history \
    --use_context \
    --gpu 1 \
    --num_samples 5

python ablation/improvement_3/inference.py \
    --checkpoint_dir /mnt/parallel/checkpoints/0118Lovink/LovinkDialogue_profile_only_16 \
    --scenario_path /data/jiashu.pu/data1/GIDigitalTwinBench/IdealSelf/LovinkDialogue \
    --config_name profile_only \
    --use_profile \
    --gpu 5 \
    --num_samples 5
    --output_dir /data/lingyu.li/parallel-post-train/ablation/outputs/0119_LovinkDialogue_profile_only_1000

python ablation/improvement_3/inference.py \
    --checkpoint_dir /mnt/parallel/checkpoints/0118Lovink/LovinkDialogue_history_only_16 \
    --scenario_path /data/jiashu.pu/data1/GIDigitalTwinBench/IdealSelf/LovinkDialogue \
    --config_name history_only \
    --use_history \
    --gpu 6 \
    --num_samples 5
    --output_dir /data/lingyu.li/parallel-post-train/ablation/outputs/0119_LovinkDialogue_history_only_1000

python ablation/improvement_3/inference.py     --checkpoint_dir /mnt/parallel/checkpoints/LovinkDialogue_ablation_profile_and_history_and_context_earlystop     --scenario_path /data/jiashu.pu/data1/GIDigitalTwinBench/IdealSelf/LovinkDialogue     --config_name profile_and_history_and_context     --use_profile     --use_history     --use_context     --gpu 0     --num_samples 5  --output_dir /data/lingyu.li/parallel-post-train/ablation/outputs/0119_LovinkDialogue_profile_and_history_and_context_1500

python ablation/improvement_3/inference.py     --checkpoint_dir /mnt/parallel/checkpoints/0118Lovink/LovinkDialogue_profile_and_context_16     --scenario_path /data/jiashu.pu/data1/GIDigitalTwinBench/IdealSelf/LovinkDialogue     --config_name profile_and_context     --use_profile     --use_context     --gpu 1     --num_samples 5  --output_dir /data/lingyu.li/parallel-post-train/ablation/outputs/0119_LovinkDialogue_profile_and_and_context_1500

python ablation/improvement_3/inference.py     --checkpoint_dir /mnt/parallel/checkpoints/0118Lovink/LovinkDialogue_profile_and_history_16     --scenario_path /data/jiashu.pu/data1/GIDigitalTwinBench/IdealSelf/LovinkDialogue     --config_name profile_and_history     --use_profile     --use_history     --gpu 2     --num_samples 5  --output_dir /data/lingyu.li/parallel-post-train/ablation/outputs/0119_LovinkDialogue_profile_and_history_1500

python ablation/improvement_3/inference.py     --checkpoint_dir /mnt/parallel/checkpoints/0118Lovink/LovinkDialogue_history_and_context_16     --scenario_path /data/jiashu.pu/data1/GIDigitalTwinBench/IdealSelf/LovinkDialogue     --config_name history_and_context     --use_history     --use_context     --gpu 4     --num_samples 5  --output_dir /data/lingyu.li/parallel-post-train/ablation/outputs/0119_LovinkDialogue_history_and_context_1500

python ablation/improvement_3/inference.py     --checkpoint_dir /mnt/parallel/checkpoints/0118Lovink/LovinkDialogue_profile_only_16     --scenario_path /data/jiashu.pu/data1/GIDigitalTwinBench/IdealSelf/LovinkDialogue     --config_name profile_only     --use_profile      --gpu 3     --num_samples 5  --output_dir /data/lingyu.li/parallel-post-train/ablation/outputs/0119_LovinkDialogue_profile_only_1500

python ablation/improvement_3/inference.py     --checkpoint_dir /mnt/parallel/checkpoints/0118Lovink/LovinkDialogue_history_only_16     --scenario_path /data/jiashu.pu/data1/GIDigitalTwinBench/IdealSelf/LovinkDialogue     --config_name history_only     --use_history     --gpu 6     --num_samples 5  --output_dir /data/lingyu.li/parallel-post-train/ablation/outputs/0119_LovinkDialogue_history_only_1500  
```

## 详细功能说明

### History 缓存机制

#### 缓存文件格式

每个用户的缓存文件保存在 `history_cache/{user_hash}_history.json`：

```json
{
  "user_hash": "bf8cebe2a324916df7d64417427a7856",
  "history": [
    "你很外向吗",
    "详细讲讲花滑",
    "不太感兴趣呢",
    "..."
  ],
  "count": 15
}
```

#### 使用缓存

缓存功能已集成到 `get_user_only_history` 函数中，默认启用：

```python
from data_loader import get_user_only_history

# 自动使用缓存
history = get_user_only_history(
    all_samples,
    user_hash,
    current_sample=current_sample,
    current_context=current_context,  # 用于排除和智能选择
    max_history=15,
    use_cache=True  # 默认启用
)
```

#### 批量构建缓存

在训练开始前，可以预先构建所有用户的缓存：

```python
from data_loader import build_all_user_history_cache, extract_training_samples, load_train_data

# 加载数据
train_data = load_train_data(train_path)
all_samples = extract_training_samples(train_data)

# 批量构建缓存
stats = build_all_user_history_cache(all_samples, max_history=15)
print(f"构建了 {len(stats)} 个用户的缓存")
```

### 智能 History 选择

当 history 长度超过 `max_history` 时，系统会：

1. **提取关键词**：从当前 context 中提取关键词
2. **计算相似度**：使用 Jaccard 相似度计算每条 history 与当前 context 的相似度
3. **选择最相关**：选择相似度最高的 `max_history` 条
4. **保持顺序**：在原始 history 中的顺序基础上，优先保留相似度高的条目

示例：

```python
# 假设有 30 条 history，但 max_history=15
# 系统会自动选择与当前 context 最相关的 15 条
history = get_user_only_history(
    all_samples,
    user_hash,
    current_sample=sample,
    current_context=sample['context'],  # 用于智能选择
    max_history=15,
    use_cache=True
)
```

### 三因素消融

#### Profile（用户画像）

包含用户的个人信息、性格特征等：

```python
user_profile = {
    "name": "HP",
    "age": 25,
    "personality": {...},
    ...
}
```

#### History（历史对话）

用户之前说过的话，用于 few-shot 学习：

```python
history = [
    "你很外向吗",
    "详细讲讲花滑",
    "不太感兴趣呢",
    ...
]
```

#### Context（当前对话上下文）

当前对话的上下文信息：

```python
context = [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！有什么可以帮助你的吗？"},
    ...
]
```

## 配置说明

### 训练配置

```json
{
  "training": {
    "num_epochs": 3,
    "batch_size": 1,
    "gradient_accumulation_steps": 64,
    "learning_rate": 1e-5,
    "max_length": 1024,
    ...
  }
}
```

### 消融配置

每个消融配置包含三个布尔值：

- `use_profile`: 是否使用用户画像
- `use_history`: 是否使用历史对话
- `use_context`: 是否使用当前对话上下文

## 使用示例

### 完整训练流程

```python
from data_loader import (
    load_train_data,
    extract_training_samples,
    build_all_user_history_cache
)
from train_with_early_stopping import AblationTrainerWithEarlyStopping

# 1. 加载数据
train_data = load_train_data("path/to/train.json")
all_samples = extract_training_samples(train_data)

# 2. 批量构建缓存（可选，但推荐）
print("构建 history 缓存...")
stats = build_all_user_history_cache(all_samples, max_history=15)
print(f"✅ 构建完成，共 {len(stats)} 个用户")

# 3. 添加 history 到样本
from train_with_early_stopping import add_history_to_samples
all_samples = add_history_to_samples(all_samples, all_samples)

# 4. 创建训练器
trainer = AblationTrainerWithEarlyStopping(
    model_path="/path/to/model",
    output_dir="/path/to/output",
    config=config,
    use_profile=True,
    use_history=True,
    use_context=True
)

# 5. 开始训练
trainer.train(
    train_samples,
    val_samples,
    max_epochs=20,
    early_stopping_patience=3
)
```

### 推理示例

```python
from inference import process_scenario

process_scenario(
    scenario_path="/path/to/scenario",
    checkpoint_dir="/path/to/checkpoint",
    config_name="profile_and_history_and_context",
    use_profile=True,
    use_history=True,
    use_context=True,
    num_samples=5,
    gpu_id=1
)
```

## 性能优化建议

1. **预先构建缓存**：在训练开始前批量构建所有用户的 history 缓存
2. **合理设置 max_history**：根据模型的最大长度限制，合理设置 history 数量
3. **使用早停机制**：使用 `train_with_early_stopping.py` 避免过拟合
4. **批量处理**：在推理时使用合适的 batch size

## 注意事项

1. **缓存一致性**：如果训练数据更新，建议清除缓存后重新构建
   ```python
   from history_cache import clear_cache
   clear_cache()  # 清除所有缓存
   ```

2. **数据泄露预防**：系统会自动排除当前 context 中的内容，确保训练数据的独立性

3. **Context 消融**：当 `use_context=False` 时，模型将无法看到当前对话上下文，仅基于 profile 和 history 生成回复

4. **History 长度限制**：默认 `max_history=15`，可根据模型的最大长度调整

## 故障排除

### 问题1: 缓存未生效

**原因**: 可能是导入路径问题

**解决**: 确保 `history_cache.py` 在同一目录下，或检查导入路径

### 问题2: History 选择不准确

**原因**: 关键词提取可能不够精确

**解决**: 可以调整 `select_relevant_history` 函数中的相似度计算逻辑

### 问题3: 内存不足

**原因**: History 缓存或模型过大

**解决**: 
- 减少 `max_history` 的值
- 使用更小的模型
- 减少 `max_length` 参数

### 问题4: 训练速度慢

**原因**: 首次运行需要计算所有 history

**解决**: 预先运行 `build_all_user_history_cache` 批量构建缓存

## 与 Improvement 2 的区别

| 特性 | Improvement 2 | Improvement 3 |
|------|---------------|---------------|
| 消融因素 | Profile + History | Profile + History + Context |
| History 缓存 | ✅ | ✅（参考 Improvement 2） |
| 智能选择 | ❌ | ✅（基于相似度） |
| Context 消融 | ❌ | ✅ |
| 配置数量 | 3 种 | 7 种 |

## 更新日志

### v3.0 (当前版本)

- ✅ 添加 Context 消融支持
- ✅ 实现智能 History 选择
- ✅ 集成 History 缓存机制
- ✅ 支持 7 种消融配置
- ✅ 优化数据加载和缓存逻辑

## 贡献

如有问题或建议，请提交 Issue 或 Pull Request。

## 许可证

请参考项目根目录的 LICENSE 文件。
