"""
训练器模块 - DeepSpeed ZeRO-3 多 GPU 版本
支持自动检测空闲 GPU，最多使用 4 张 GPU
"""
import os
import re
import time
import json
import torch
import torch.nn as nn
import subprocess
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    # HfDeepSpeedConfig,
)
# 不再需要手动导入 HfDeepSpeedConfig，Trainer 会自动处理
from typing import List, Dict, Any, Optional, Tuple
import sys
from pathlib import Path

# 添加当前目录到路径，确保能导入 prompt_builder
sys.path.insert(0, str(Path(__file__).parent))
from prompt_builder import build_training_prompt


def get_gpu_memory_usage():
    """获取所有 GPU 的显存使用情况"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu', 
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )
        
        gpu_info = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = line.split(', ')
                if len(parts) >= 4:
                    gpu_id = int(parts[0])
                    memory_used = int(parts[1])
                    memory_total = int(parts[2])
                    utilization = int(parts[3])
                    memory_free = memory_total - memory_used
                    memory_usage_percent = (memory_used / memory_total) * 100
                    
                    gpu_info.append({
                        'id': gpu_id,
                        'memory_used': memory_used,
                        'memory_total': memory_total,
                        'memory_free': memory_free,
                        'memory_usage_percent': memory_usage_percent,
                        'utilization': utilization
                    })
        return gpu_info
    except Exception as e:
        print(f"警告: 无法获取 GPU 信息: {e}")
        return []


def find_free_gpus(max_gpus=4, memory_threshold_mb=1000, utilization_threshold=10):
    """
    自动检测空闲的 GPU
    
    Args:
        max_gpus: 最多使用的 GPU 数量
        memory_threshold_mb: 显存使用阈值（MB），低于此值认为 GPU 空闲
        utilization_threshold: GPU 利用率阈值（%），低于此值认为 GPU 空闲
    
    Returns:
        空闲 GPU ID 列表
    """
    gpu_info = get_gpu_memory_usage()
    
    if not gpu_info:
        print("警告: 无法检测 GPU，将使用默认 GPU")
        return [0] if torch.cuda.is_available() else []
    
    # 筛选空闲 GPU
    free_gpus = []
    for gpu in gpu_info:
        if (gpu['memory_free'] > memory_threshold_mb and 
            gpu['utilization'] < utilization_threshold):
            free_gpus.append(gpu)
    
    # 按显存空闲量排序（从大到小）
    free_gpus.sort(key=lambda x: x['memory_free'], reverse=True)
    
    # 选择最多 max_gpus 个 GPU
    selected_gpus = free_gpus[:max_gpus]
    gpu_ids = [gpu['id'] for gpu in selected_gpus]
    
    if gpu_ids:
        print(f"✓ 检测到 {len(gpu_ids)} 个空闲 GPU: {gpu_ids}")
        for gpu in selected_gpus:
            print(f"  GPU {gpu['id']}: 显存 {gpu['memory_free']}MB 空闲, 利用率 {gpu['utilization']}%")
    else:
        print("⚠ 未检测到空闲 GPU，将使用 GPU 0")
        gpu_ids = [0] if torch.cuda.is_available() else []
    
    return gpu_ids


class AblationDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length=32768, use_profile=True, use_history=True, use_context=True):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_profile = use_profile
        self.use_history = use_history
        self.use_context = use_context

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 1. 初始构建
        messages, target_answer = build_training_prompt(
            context=sample['context'],
            next_question=sample['next_question'],
            user_profile=sample.get('user_profile') if self.use_profile else None,
            task_description=sample.get('task_description'),
            history=sample.get('history') if self.use_history else None,
            use_profile=self.use_profile,
            use_history=self.use_history,
            use_context=self.use_context
        )

        # --- 核心优化：动态裁剪历史以防止截断 ---
        # 如果消息太长，循环删除 messages 中最早的对话轮次（保留 system 提示词）
        # 索引 0 是 system，之后是 user/assistant 交替
        # 为了保持角色交替，需要成对删除（user+assistant）
        max_iterations = 100  # 防止无限循环
        iteration = 0
        while iteration < max_iterations:
            try:
                tokenized_length = len(self.tokenizer.apply_chat_template(messages, tokenize=True))
                if tokenized_length <= (self.max_length - 512):
                    break
            except Exception as e:
                # 如果 apply_chat_template 失败，说明消息格式有问题
                # 尝试重新规范化消息
                from prompt_builder import _normalize_messages
                messages = _normalize_messages(messages)
                # 如果规范化后仍然失败，跳出循环
                try:
                    tokenized_length = len(self.tokenizer.apply_chat_template(messages, tokenize=True))
                    if tokenized_length <= (self.max_length - 512):
                        break
                except:
                    print(f"警告: 无法规范化消息格式，跳过裁剪: {e}")
                    break
            
            if len(messages) > 3:  # system + 至少一对 user/assistant
                # 检查索引 1 是否是 user，索引 2 是否是 assistant
                if messages[1].get('role') == 'user' and len(messages) > 2 and messages[2].get('role') == 'assistant':
                    # 删除一对 user/assistant（保持交替）
                    messages.pop(1)  # 删除 user
                    messages.pop(1)  # 删除 assistant（现在索引 1 的位置）
                else:
                    # 如果格式不对，只删除一条（向后兼容，但可能破坏交替）
                    # 这种情况不应该发生，因为 normalize_messages 已经确保了交替
                    messages.pop(1)
            else:
                break
            iteration += 1

        # 2. 生成 Prompt
        # 确保 messages 以 assistant 结尾（用于预测用户下一句话）
        # 如果最后一条不是 assistant，说明有问题，应该已经被 normalize_messages 处理了
        full_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        
        # 3. 手动添加 "user: " 提示，让模型预测用户会说什么
        # 注意：Gemma 的 chat template 使用 <start_of_turn> 和 <end_of_turn>，但我们希望使用简单的 "user: " 格式
        generation_suffix = "\nuser: "

        # 4. 组合成真正的 Prompt
        full_prompt = full_prompt.strip() + generation_suffix
        # 确保不包含答案，使用 <|im_end|> 作为结束标记（让模型学会在正确位置停止）
        im_end_token = "<|im_end|>"
        full_text = full_prompt + target_answer + im_end_token

        # 3. 编码
        encoded = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].squeeze()
        attention_mask = encoded['attention_mask'].squeeze()

        # --- 核心优化：高精度计算 Prompt 长度 ---
        # 我们不直接 encode(full_prompt)，而是通过寻找 target 的起始 token 来确定
        target_ids = self.tokenizer.encode(target_answer, add_special_tokens=False)
        
        # 寻找分界点：在 input_ids 中找到第一个不属于 prompt 的位置
        # 我们可以先 encode 一个完全没带特殊字符的 prompt
        prompt_ids = self.tokenizer.encode(full_prompt, add_special_tokens=False)
        actual_prompt_len = len(prompt_ids)

        labels = input_ids.clone()
        
        # 屏蔽 Prompt：确保不会越界
        safe_prompt_len = min(actual_prompt_len, self.max_length - 1)
        labels[:safe_prompt_len] = -100
        
        # 屏蔽 Padding
        labels[input_ids == self.tokenizer.pad_token_id] = -100

        # --- 屏蔽特殊 Token (保留 EOS 和 <|im_end|>) ---
        # 获取 <|im_end|> 的 token ID，确保它被包含在损失计算中
        im_end_token = "<|im_end|>"
        im_end_id = None
        try:
            # 尝试获取 <|im_end|> 的 token ID
            im_end_ids = self.tokenizer.encode(im_end_token, add_special_tokens=False)
            if im_end_ids:
                im_end_id = im_end_ids[0]  # 通常 <|im_end|> 是一个单独的 token
                # 调试信息（只在第一次打印）
                if not hasattr(self, '_im_end_logged'):
                    print(f"✓ <|im_end|> token ID: {im_end_id}，将被包含在损失计算中")
                    self._im_end_logged = True
        except Exception as e:
            if not hasattr(self, '_im_end_error_logged'):
                print(f"警告: 无法获取 <|im_end|> token ID: {e}")
                self._im_end_error_logged = True
        
        special_ids = set(self.tokenizer.all_special_ids)
        eos_id = self.tokenizer.eos_token_id
        # 保留 EOS 和 <|im_end|> token，让模型学会在正确位置停止
        tokens_to_keep = {eos_id}
        if im_end_id is not None:
            tokens_to_keep.add(im_end_id)
        
        for tid in special_ids:
            if tid not in tokens_to_keep:
                labels[labels == tid] = -100
        
        # 验证 <|im_end|> 是否在 labels 中（用于调试）
        if im_end_id is not None and (labels == im_end_id).any():
            if not hasattr(self, '_im_end_verified'):
                print(f"✓ 确认: <|im_end|> token (ID: {im_end_id}) 已包含在损失计算中")
                self._im_end_verified = True

        # 4. 最终验证：防止 NaN
        if (labels != -100).sum() == 0:
            # 挽救逻辑：如果全被屏蔽了（说明截断太严重），强行暴露最后 32 个 token 
            # 这种情况通常发生在答案极长或截断刚好切在了答案开头
            labels[-32:] = input_ids[-32:]
            labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


class CustomTrainer(Trainer):
    """带实时日志的自定义训练器"""
    
    def __init__(self, *args, verbose_logging=False, log_file_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.verbose_logging = verbose_logging
        self.log_file_path = log_file_path
        self.log_entry_count = 0
        
        if self.log_file_path:
            os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)
            self.log_file = open(self.log_file_path, 'w', encoding='utf-8')
            self.log_file.write("[\n")

    def __del__(self):
        if hasattr(self, 'log_file') and self.log_file:
            try:
                self.log_file.write("\n]")
                self.log_file.close()
            except: pass

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.loss if hasattr(outputs, 'loss') else None
        
        if loss is None and "labels" in inputs:
            logits = outputs.get("logits")
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = inputs["labels"][..., 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if self.verbose_logging and (self.state.global_step % self.args.logging_steps == 0):
            self._log_details(inputs, outputs, loss.item())

        return (loss, outputs) if return_outputs else loss

    def clean_output_text(self, text: str) -> str:
        # 移除思考过程
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        text = text.replace('<think>', '').replace('</think>', '')
        return text.strip()

    def _log_details(self, inputs, outputs, loss_val):
        """记录训练细节：对比 Target 和模型的预测 (Argmax)"""
        try:
            batch_idx = 0
            ids = inputs['input_ids'][batch_idx]
            lbs = inputs['labels'][batch_idx]
            logits = outputs.get("logits")[batch_idx]
            
            # 解码 Target
            target_ids = [t.item() for t in lbs if t != -100]
            target_text = self.tokenizer.decode(target_ids, skip_special_tokens=True)
            
            # 解码预测 (寻找 label 有效位对应的预测位)
            pred_ids_all = logits.argmax(dim=-1)
            valid_pos = (lbs != -100).nonzero(as_tuple=True)[0]
            pred_ids = [pred_ids_all[p-1].item() for p in valid_pos if p > 0]
            predict_text = self.tokenizer.decode(pred_ids, skip_special_tokens=True)
            
            print(f"\n[Step {self.state.global_step}] Loss: {loss_val:.4f}")
            print(f"🎯 Target: {target_text[:100]}")
            print(f"🤖 Predict: {predict_text[:100]}")

            if hasattr(self, 'log_file'):
                log_data = {
                    "step": self.state.global_step,
                    "loss": loss_val,
                    "target": target_text,
                    "predict": predict_text
                }
                if self.log_entry_count > 0: self.log_file.write(",\n")
                self.log_file.write(json.dumps(log_data, ensure_ascii=False))
                self.log_file.flush()
                self.log_entry_count += 1
        except Exception as e:
            print(f"Log Error: {e}")


class AblationTrainerDeepSpeed:
    """消融实验主控类 - DeepSpeed ZeRO-3 多 GPU 版本（使用 Accelerate 库）"""
    
    def __init__(self, model_path: str, output_dir: str, config: Dict[str, Any], 
                 use_profile: bool = True, use_history: bool = True, use_context: bool = True, 
                 log_file_path: Optional[str] = None, deepspeed_config_path: Optional[str] = None):
        self.model_path = model_path
        self.output_dir = output_dir
        self.config = config
        self.use_profile = use_profile
        self.use_history = use_history
        self.use_context = use_context
        self.log_file_path = log_file_path

        # 1. 创建或使用提供的 DeepSpeed 配置文件
        # 如果提供了配置文件路径，直接使用；否则创建默认配置
        if deepspeed_config_path and os.path.exists(deepspeed_config_path):
            self.deepspeed_config_path = deepspeed_config_path
            print(f"✓ 使用提供的 DeepSpeed 配置文件: {self.deepspeed_config_path}")
        else:
            self.deepspeed_config_path = self._create_deepspeed_config()
            print(f"✓ 创建 DeepSpeed 配置文件: {self.deepspeed_config_path}")
        
        # 2. 加载 Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # 确保 tokenizer 有正确的 padding side
        if not hasattr(self.tokenizer, 'padding_side') or self.tokenizer.padding_side is None:
            self.tokenizer.padding_side = "right"
        print(f"✓ Tokenizer 词汇表大小: {len(self.tokenizer)}")
        
        # 3. 模型将在训练时通过 Trainer 和 DeepSpeed 自动加载
        # Accelerate 和 Trainer 会自动处理所有 DeepSpeed 相关的初始化
        print("✓ 模型将在训练时通过 Accelerate + DeepSpeed 自动加载")

    def _create_deepspeed_config(self) -> str:
        """创建 DeepSpeed ZeRO-3 配置文件"""
        config_dir = Path(self.output_dir) / "deepspeed_config"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "ds_config_zero3.json"
        
        # DeepSpeed ZeRO-3 配置 - 优化显存使用
        deepspeed_config = {
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": "auto",
            "gradient_accumulation_steps": "auto",
            "gradient_clipping": 1.0,
            "zero_optimization": {
                "stage": 3,
                # CPU Offload 优化器和参数以节省显存
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "offload_param": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "overlap_comm": True,
                "contiguous_gradients": True,
                "sub_group_size": 1e9,
                "reduce_bucket_size": "auto",
                "stage3_prefetch_bucket_size": "auto",
                "stage3_param_persistence_threshold": "auto",
                "stage3_max_live_parameters": 1e9,
                "stage3_max_reuse_distance": 1e9,
                "stage3_gather_16bit_weights_on_model_save": True,
                "round_robin_gradients": True
            },
            "bf16": {
                "enabled": True
            },
            "fp16": {
                "enabled": False
            },
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": "auto",
                    "betas": "auto",
                    "eps": "auto",
                    "weight_decay": "auto"
                }
            },
            "scheduler": {
                "type": "WarmupLR",
                "params": {
                    "warmup_min_lr": "auto",
                    "warmup_max_lr": "auto",
                    "warmup_num_steps": "auto"
                }
            },
            "wall_clock_breakdown": False,
            "steps_per_print": 10,
            # 显存优化
            "activation_checkpointing": {
                "partition_activations": True,
                "cpu_checkpointing": True,
                "contiguous_memory_optimization": True,
                "number_checkpoints": None,
                "synchronize_checkpoint_boundary": True,
                "profile": False
            }
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(deepspeed_config, f, indent=2, ensure_ascii=False)
        
        return str(config_path)

    def train(self, train_samples: List[Dict[str, Any]], val_samples: Optional[List[Dict[str, Any]]] = None):
        import torch
        import os
        import transformers.models.gemma3.modeling_gemma3 as gemma3_module
        from transformers import TrainingArguments, AutoConfig, AutoModelForCausalLM
        
        # 尝试导入 no_init_weights，如果失败则使用替代方案
        try:
            from transformers.integrations import no_init_weights
            NO_INIT_WEIGHTS_AVAILABLE = True
        except ImportError:
            try:
                from accelerate import init_empty_weights
                NO_INIT_WEIGHTS_AVAILABLE = False
                print("⚠ transformers.integrations.no_init_weights 不可用，将使用 accelerate.init_empty_weights")
            except ImportError:
                NO_INIT_WEIGHTS_AVAILABLE = False
                print("⚠ no_init_weights 和 init_empty_weights 都不可用，将使用标准加载方式")

        train_config = self.config.get('training', {})
        
        # 1. 序列长度与显存预判
        max_length = train_config.get('max_length', 4096)
        if max_length > 2048:
            print(f"⚠ 序列长度 {max_length} 较大。在 H100 上 ZeRO-3 虽然能跑，但会显著降低吞吐。")
        
        train_dataset = AblationDataset(
            train_samples, self.tokenizer, 
            max_length=max_length,
            use_profile=self.use_profile, use_history=self.use_history, use_context=self.use_context
        )

        # 2. 训练参数优化：针对 27B + H100
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=train_config.get('num_epochs', 3),
            per_device_train_batch_size=train_config.get('batch_size', 1),
            gradient_accumulation_steps=train_config.get('gradient_accumulation_steps', 4),
            learning_rate=train_config.get('learning_rate', 2e-5), # 27B 全量微调建议调低 LR
            logging_steps=train_config.get('logging_steps', 10),
            save_steps=train_config.get('save_steps', 500),
            save_total_limit=train_config.get('save_total_limit', 3),
            deepspeed=self.deepspeed_config_path,
            bf16=True, 
            gradient_checkpointing=True,
            # 针对 H100 的特殊优化
            gradient_checkpointing_kwargs={"use_reentrant": False}, # 避免某些环境下的梯度警告
            warmup_steps=train_config.get('warmup_steps', 100),
            weight_decay=0.01,
            report_to="none",
            dataloader_num_workers=4, # H100 节点 CPU 性能通常较强，可以略微调高
            remove_unused_columns=False,
        )

        # 3. 定义最核心的模型初始化函数 (修复 IndexError 的关键)
        def model_init():
            print(">>> 正在启动 ZeRO-3 兼容性初始化流程...")
            
            # --- 核心修复：对 Gemma 3 类的 _init_weights 进行全局 Patch ---
            # 理由：ZeRO-3 会分片参数，导致某些卡上的本地 weight 为空，
            # 原生代码中 .zero_() 访问 index 0 就会报错。
            
            # 关键修复：需要 patch 父类 PreTrainedModel 的 _init_weights 方法
            # 因为 Gemma3 的 _init_weights 会调用 super()._init_weights(module)
            from transformers.modeling_utils import PreTrainedModel
            
            # 保存原始的父类 _init_weights 方法
            original_parent_init_weights = PreTrainedModel._init_weights
            
            def safe_parent_init_weights(self, module):
                """安全的父类权重初始化，避免在 DeepSpeed ZeRO-3 环境下出错"""
                # 只有当 weight 确实在本地显存中有元素时，才执行初始化
                if isinstance(module, torch.nn.Embedding):
                    # 在 DeepSpeed ZeRO-3 环境下，需要更严格的检查
                    try:
                        # 检查 weight 是否存在且有数据
                        if hasattr(module, 'weight') and module.weight is not None:
                            # 尝试获取 weight 的形状
                            weight_shape = module.weight.shape
                            if len(weight_shape) > 0 and weight_shape[0] > 0:
                                if module.padding_idx is not None:
                                    # 再次检查 padding_idx 是否在有效范围内
                                    if 0 <= module.padding_idx < weight_shape[0]:
                                        module.weight.data[module.padding_idx].zero_()
                    except (IndexError, RuntimeError, AttributeError) as e:
                        # 在 DeepSpeed ZeRO-3 环境下，某些卡上的权重可能为空或分片
                        # 这是正常的，直接跳过
                        pass
                    return  # 跳过该层的默认初始化
                # 对于其他层，调用原始方法
                original_parent_init_weights(self, module)
            
            # Patch 父类方法
            PreTrainedModel._init_weights = safe_parent_init_weights
            print("✓ 已 patch PreTrainedModel._init_weights (父类方法)")
            
            # 同时 patch Gemma3 的 _init_weights 方法（虽然它调用 super，但为了保险）
            def safe_init_weights_patch(self, module):
                # Gemma3 的 _init_weights 会调用 super()._init_weights(module)
                # 我们已经 patch 了父类方法，所以这里直接调用父类方法即可
                safe_parent_init_weights(self, module)

            # 找到目标类并替换方法
            target_classes = []
            if hasattr(gemma3_module, 'Gemma3ForCausalLM'):
                target_classes.append(gemma3_module.Gemma3ForCausalLM)
            if hasattr(gemma3_module, 'Gemma3ForConditionalGeneration'):
                target_classes.append(gemma3_module.Gemma3ForConditionalGeneration)
            if hasattr(gemma3_module, 'Gemma3Model'):
                target_classes.append(gemma3_module.Gemma3Model)
            
            patched_count = 0
            for cls in target_classes:
                if hasattr(cls, "_init_weights"):
                    cls._init_weights = safe_init_weights_patch
                    patched_count += 1
            
            print(f"✓ 已注入安全权重初始化补丁 (Monkey Patch) - 已 patch {patched_count} 个 Gemma3 类")

            # --- 使用 no_init_weights 上下文加载模型 ---
            # 注意：no_init_weights 会禁用权重初始化，直接从 checkpoint 加载
            config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
            
            # 准备模型加载参数
            model_kwargs = {
                "config": config,
                "trust_remote_code": True,
                "torch_dtype": torch.bfloat16,
                "low_cpu_mem_usage": True,
            }

            # 自动探测并开启 Flash Attention 2
            try:
                import flash_attn
                model_kwargs["attn_implementation"] = "flash_attention_2"
                print("✓ 检测到 Flash Attention 2，已开启加速。")
            except ImportError:
                print("⚠ Flash Attention 2 未安装，使用默认注意力实现")
            except Exception as e:
                print(f"⚠ 无法启用 Flash Attention 2: {e}")

            # 根据可用的工具选择加载方式
            if NO_INIT_WEIGHTS_AVAILABLE:
                try:
                    with no_init_weights():
                        model = AutoModelForCausalLM.from_pretrained(
                            self.model_path,
                            **model_kwargs
                        )
                except Exception as e:
                    print(f"⚠ 使用 no_init_weights 加载失败: {e}")
                    print("回退到标准加载方式（已 patch _init_weights）...")
                    model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        **model_kwargs
                    )
            else:
                # 使用标准加载方式，但已经 patch 了 _init_weights，应该能工作
                print("使用标准加载方式（已 patch _init_weights 以处理 ZeRO-3）...")
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    **model_kwargs
                )
            
            return model

        # 4. 创建分布式训练器
        print(">>> 正在构建 Trainer (ZeRO-3 已挂载)...")
        trainer = CustomTrainer(
            model_init=model_init, # 通过 model_init 延迟加载
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer,
            verbose_logging=True,
            log_file_path=self.log_file_path
        )

        print(f"🚀 任务启动: 消融配置 -> Profile={self.use_profile}, History={self.use_history}")
        trainer.train()
        
        # 5. 保存结果（Rank 0 会自动处理）
        trainer.save_model(self.output_dir)
        print(f"✓ 训练圆满完成。模型已导出至: {self.output_dir}")