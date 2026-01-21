"""
消融实验训练脚本（带早停机制）
支持三种配置：
1. profile + history
2. profile only
3. history only
"""
import json
import argparse
import os
import sys
from pathlib import Path
import random
import torch

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_train_data, extract_training_samples, get_user_history_samples, get_user_only_history
from trainer import AblationTrainer, AblationDataset
from transformers import EarlyStoppingCallback, TrainingArguments, Trainer
from typing import List, Dict, Any, Optional
import torch.nn as nn


def split_train_val(samples, val_ratio=0.15, seed=42):
    """划分训练集和验证集（按用户划分，避免数据泄露）"""
    random.seed(seed)
    
    user_samples = {}
    for sample in samples:
        user_hash = sample['user_hash']
        if user_hash not in user_samples:
            user_samples[user_hash] = []
        user_samples[user_hash].append(sample)
    
    train_samples = []
    val_samples = []
    
    for user_hash, user_data in user_samples.items():
        random.shuffle(user_data)
        split_idx = int(len(user_data) * (1 - val_ratio))
        train_samples.extend(user_data[:split_idx])
        val_samples.extend(user_data[split_idx:])
    
    return train_samples, val_samples


def add_history_to_samples(train_samples, all_samples):
    """为每个样本添加历史信息（只包含用户的问题，不包含assistant内容）"""
    samples_with_history = []
    for sample in train_samples:
        user_hash = sample['user_hash']
        # 使用 get_user_only_history 只获取用户的问题文本列表，不包含assistant内容
        # 传入current_sample和current_context，用于排除和智能选择
        history = get_user_only_history(
            all_samples, 
            user_hash,
            current_sample=sample,
            current_context=sample.get('context'),
            max_history=15,
            use_cache=True
        )
        sample['history'] = history
        samples_with_history.append(sample)
    return samples_with_history


class AblationTrainerWithEarlyStopping(AblationTrainer):
    def __init__(self, *args, use_multi_gpu=False, **kwargs):
        super().__init__(*args, use_multi_gpu=use_multi_gpu, **kwargs)
    """带早停功能的训练器，支持context消融"""
    """带早停功能的训练器"""
    
    def train(
        self,
        train_samples: List[Dict[str, Any]],
        val_samples: Optional[List[Dict[str, Any]]] = None,
        max_epochs: int = 10,
        early_stopping_patience: int = 3,
        early_stopping_threshold: float = 0.00001
    ):
        """
        训练模型（带早停）
        
        Args:
            train_samples: 训练样本
            val_samples: 验证样本
            max_epochs: 最大训练轮次
            early_stopping_patience: 早停耐心值（验证loss不下降的轮次）
            early_stopping_threshold: 早停阈值（loss改善的最小值）
        """
        train_config = self.config.get('training', {})
        
        # 创建数据集
        print("创建训练数据集...")
        train_dataset = AblationDataset(
            samples=train_samples,
            tokenizer=self.tokenizer,
            max_length=train_config.get('max_length', 32768),
            use_profile=self.use_profile,
            use_history=self.use_history,
            use_context=self.use_context
        )
        
        val_dataset = None
        if val_samples:
            print("创建验证数据集...")
            val_dataset = AblationDataset(
                samples=val_samples,
                tokenizer=self.tokenizer,
                max_length=train_config.get('max_length', 32768),
                use_profile=self.use_profile,
                use_history=self.use_history,
                use_context=self.use_context
            )
        
        # 数据整理器
        def simple_collate_fn(examples):
            batch = {}
            for key in examples[0].keys():
                if key in ['input_ids', 'attention_mask', 'labels']:
                    batch[key] = torch.stack([ex[key] for ex in examples])
                else:
                    batch[key] = [ex[key] for ex in examples]
            return batch
        
        # 计算每个epoch的步数和评估步数
        steps_per_epoch = len(train_dataset) // (train_config.get('batch_size', 1) * train_config.get('gradient_accumulation_steps', 16))
        eval_steps_value = max(1, steps_per_epoch // 2) if val_dataset else None  # 每个epoch评估2次
        
        # 调整 save_steps 使其是 eval_steps 的整数倍（如果启用了 load_best_model_at_end）
        save_steps_value = train_config.get('save_steps', 500)
        if val_dataset and eval_steps_value and save_steps_value % eval_steps_value != 0:
            # 将 save_steps 调整为 eval_steps 的最近整数倍（向上取整）
            save_steps_value = ((save_steps_value + eval_steps_value - 1) // eval_steps_value) * eval_steps_value
            print(f"调整 save_steps 为 {save_steps_value}（eval_steps={eval_steps_value} 的整数倍）")
        
        # 检查学习率是否过大
        learning_rate = train_config.get('learning_rate', 1e-5)
        if learning_rate > 1e-5:
            print(f"警告: 学习率 {learning_rate} 可能过大，全量微调建议使用 1e-5")
            print(f"如果训练中出现nan/inf，建议降低学习率到 5e-6")
        elif learning_rate < 5e-6:
            print(f"警告: 学习率 {learning_rate} 可能过小，训练可能很慢")
        print(f"使用学习率: {learning_rate}")
        
        # 检测是否使用多GPU
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        use_ddp = num_gpus > 1
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=max_epochs,
            per_device_train_batch_size=train_config.get('batch_size', 1),
            per_device_eval_batch_size=train_config.get('eval_batch_size', 1),
            gradient_accumulation_steps=train_config.get('gradient_accumulation_steps', 16),
            learning_rate=learning_rate,
            weight_decay=train_config.get('weight_decay', 0.01),
            warmup_steps=train_config.get('warmup_steps', 100),
            logging_steps=train_config.get('logging_steps', 10),
            save_steps=save_steps_value,
            eval_steps=eval_steps_value,
            eval_strategy="steps" if val_dataset else "no",
            save_total_limit=train_config.get('save_total_limit', 3),
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=False,
            bf16=True,  # 启用 bf16 提升计算精度和稳定性
            dataloader_pin_memory=False,
            gradient_checkpointing=True,
            optim="adamw_torch",
            max_grad_norm=0.5,  # 更严格的梯度裁剪，防止 NaN（从 1.0 降低到 0.5）
            report_to="none",
            ddp_find_unused_parameters=False,
            ddp_backend="nccl" if use_ddp else None,  # 多GPU时使用NCCL后端
        )
        
        if use_ddp:
            print(f"检测到 {num_gpus} 张GPU，将使用分布式数据并行（DDP）训练")
        
        # 自定义 Trainer（复用父类的损失计算，并添加详细日志）
        class CustomTrainer(Trainer):
            def __init__(self, *args, verbose_logging=False, log_file_path=None, log_dir=None, **kwargs):
                super().__init__(*args, **kwargs)
                self.verbose_logging = verbose_logging
                self.log_file_path = log_file_path
                self.log_dir = log_dir
                self.log_file = None
                self.log_entry_count = 0
                
                # 打开日志文件（如果指定）
                if self.log_file_path:
                    try:
                        os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)
                        self.log_file = open(self.log_file_path, 'w', encoding='utf-8')
                        self.log_file.write("[\n")  # 开始 JSON 数组
                        print(f"训练日志将保存到: {self.log_file_path}")
                    except Exception as e:
                        print(f"警告: 无法打开日志文件 {self.log_file_path}: {e}")
                        self.log_file = None
                
                # 创建输入输出日志文件
                if self.log_dir:
                    try:
                        os.makedirs(self.log_dir, exist_ok=True)
                        self.io_log_file = open(os.path.join(self.log_dir, 'io_logs.jsonl'), 'w', encoding='utf-8')
                        print(f"输入输出日志将保存到: {os.path.join(self.log_dir, 'io_logs.jsonl')}")
                    except Exception as e:
                        print(f"警告: 无法创建输入输出日志文件: {e}")
                        self.io_log_file = None
                else:
                    self.io_log_file = None
            
            def __del__(self):
                """关闭日志文件"""
                if hasattr(self, 'log_file') and self.log_file:
                    try:
                        self.log_file.write("\n]\n")  # 关闭 JSON 数组
                        self.log_file.close()
                    except:
                        pass
                if hasattr(self, 'io_log_file') and self.io_log_file:
                    try:
                        self.io_log_file.close()
                    except:
                        pass
            
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                outputs = model(**inputs)
                logits = outputs.get("logits")
                labels = inputs.get("labels")
                input_ids = inputs.get("input_ids")
                
                # 检查并清理logits中的nan/inf值，确保数值稳定性
                # 使用更内存高效的方式检查（避免创建大的中间张量）
                if logits is not None:
                    # 使用更内存高效的方式：直接检查 logits 的统计信息，而不是创建索引
                    has_nan = False
                    has_inf = False
                    # 只检查 logits 的一小部分（使用切片而不是随机索引，避免创建大的索引张量）
                    # 检查前1000个元素和后1000个元素
                    if logits.numel() > 0:
                        # 检查开头和结尾，避免创建大的中间张量
                        check_size = min(1000, logits.numel() // 2)
                        if logits.numel() > check_size * 2:
                            # 检查开头和结尾
                            head_values = logits.view(-1)[:check_size]
                            tail_values = logits.view(-1)[-check_size:]
                            if torch.isnan(head_values).any() or torch.isnan(tail_values).any():
                                has_nan = True
                            if torch.isinf(head_values).any() or torch.isinf(tail_values).any():
                                has_inf = True
                        else:
                            # 如果 logits 很小，直接检查全部
                            if torch.isnan(logits).any():
                                has_nan = True
                            if torch.isinf(logits).any():
                                has_inf = True
                    
                    # 如果样本中有问题，再检查全部（但使用更高效的方式）
                    if has_nan or has_inf:
                        nan_count = torch.isnan(logits).sum().item()
                        inf_count = torch.isinf(logits).sum().item()
                        
                        if nan_count > 0 or inf_count > 0:
                            print(f"警告: Step {self.state.global_step} logits中有 {nan_count} 个nan, {inf_count} 个inf，正在清理...")
                            # 将nan和inf替换为0（之后会通过softmax处理）
                            logits = torch.where(
                                torch.isnan(logits) | torch.isinf(logits),
                                torch.tensor(0.0, device=logits.device, dtype=logits.dtype),
                                logits
                            )
                            # 限制logits范围，避免数值溢出
                            logits = torch.clamp(logits, min=-50.0, max=50.0)
                    else:
                        # 即使样本中没有问题，也检查范围（使用更高效的方式，避免创建大的中间张量）
                        # 只检查最大值和最小值，而不是全部
                        logits_max = logits.max().item()
                        logits_min = logits.min().item()
                        if abs(logits_max) > 100 or abs(logits_min) > 100:
                            print(f"警告: Step {self.state.global_step} logits范围异常: [{logits_min:.2f}, {logits_max:.2f}]，正在裁剪...")
                            logits = torch.clamp(logits, min=-50.0, max=50.0)
                
                if hasattr(outputs, 'loss') and outputs.loss is not None:
                    loss = outputs.loss
                elif labels is not None:
                    # 确保labels有效
                    valid_labels_count = (labels != -100).sum().item()
                    assert valid_labels_count > 0, f"错误: Step {self.state.global_step} 没有有效的labels！"
                    
                    if valid_labels_count == 0:
                        print(f"错误: Step {self.state.global_step} 没有有效的labels，使用备用损失")
                        loss = torch.tensor(2.0, device=logits.device, requires_grad=True)
                    else:
                        # 使用CrossEntropyLoss
                        loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = labels[..., 1:].contiguous()
                        valid_shift_labels = (shift_labels != -100).sum().item()
                        if valid_shift_labels > 0:
                            # 再次检查shift_logits
                            if torch.isnan(shift_logits).any() or torch.isinf(shift_logits).any():
                                print(f"警告: Step {self.state.global_step} shift_logits中有nan/inf，使用备用损失")
                                loss = torch.tensor(2.0, device=logits.device, requires_grad=True)
                            else:
                                loss = loss_fct(
                                    shift_logits.view(-1, shift_logits.size(-1)),
                                    shift_labels.view(-1)
                                )
                        else:
                            loss = torch.tensor(2.0, device=logits.device, requires_grad=True)
                else:
                    loss = torch.tensor(2.0, device=logits.device, requires_grad=True)
                
                # 检查损失是否异常
                if loss is not None and torch.is_tensor(loss):
                    if loss.dim() > 0:
                        loss = loss.mean()
                    
                    loss_value = loss.item()
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"错误: Step {self.state.global_step} loss为nan/inf，使用备用损失值2.0")
                        loss = torch.tensor(2.0, device=logits.device, requires_grad=True)
                        loss_value = 2.0
                    elif loss_value > 1e6:
                        print(f"警告: Step {self.state.global_step} loss过大 ({loss_value:.2f})，裁剪到100.0")
                        loss = torch.clamp(loss, max=100.0)
                        loss_value = 100.0
                    elif loss_value < 0:
                        print(f"警告: Step {self.state.global_step} loss为负 ({loss_value:.2f})，取绝对值")
                        loss = torch.abs(loss)
                        loss_value = abs(loss_value)
                
                # 记录所有输入和输出（每个step都记录）
                self._log_inputs_outputs(input_ids, labels, logits, loss_value)
                
                # 详细日志输出（每隔一定步数或每个epoch开始时）
                if self.verbose_logging and (self.state.global_step % self.args.logging_steps == 0 or self.state.global_step == 0):
                    self._log_training_details(input_ids, labels, logits, loss, loss_value)
                
                # 定期清理CUDA缓存，避免内存碎片化（每10步清理一次）
                if self.state.global_step % 10 == 0:
                    torch.cuda.empty_cache()
                
                if return_outputs:
                    return loss, outputs
                return loss
            
            def _log_inputs_outputs(self, input_ids, labels, logits, loss_value):
                """记录所有输入和输出到文件"""
                if not self.io_log_file:
                    return
                
                try:
                    import json
                    # 获取 tokenizer
                    tokenizer = getattr(self, 'processing_class', None) or getattr(self, 'tokenizer', None)
                    if tokenizer is None:
                        return
                    
                    batch_size = input_ids.shape[0]
                    
                    # 处理batch中的每个样本
                    for batch_idx in range(batch_size):
                        # 获取输入tokens（完整序列）
                        input_tokens = input_ids[batch_idx].cpu().tolist()
                        # 过滤padding tokens
                        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
                        valid_input_tokens = [t for t in input_tokens if t != pad_token_id and t != 0]
                        
                        # 解码完整输入文本
                        full_input_text = tokenizer.decode(valid_input_tokens, skip_special_tokens=False)
                        
                        # 获取目标tokens（labels中不为-100的部分）
                        target_tokens = labels[batch_idx].cpu().tolist()
                        valid_target_tokens = [t for t in target_tokens if t != -100]
                        # 解码目标文本（ground truth）
                        target_text = tokenizer.decode(valid_target_tokens, skip_special_tokens=False) if len(valid_target_tokens) > 0 else ""
                        
                        # 获取预测tokens（从logits中获取）
                        valid_target_positions = [i for i, t in enumerate(target_tokens) if t != -100]
                        pred_tokens = []
                        if len(valid_target_positions) > 0 and batch_idx < logits.shape[0]:
                            for pos in valid_target_positions:
                                if pos > 0 and pos - 1 < logits[batch_idx].shape[0]:
                                    logit_slice = logits[batch_idx][pos - 1].clone().cpu()
                                    # 排除pad_token_id
                                    if pad_token_id is not None and pad_token_id < logit_slice.shape[0]:
                                        logit_slice[pad_token_id] = float('-inf')
                                    if 0 < logit_slice.shape[0]:
                                        logit_slice[0] = float('-inf')
                                    pred_token = logit_slice.argmax().item()
                                    pred_tokens.append(pred_token)
                        
                        # 解码预测文本
                        pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=False) if len(pred_tokens) > 0 else ""
                        
                        # 构建日志条目
                        log_entry = {
                            "step": self.state.global_step,
                            "epoch": round(self.state.epoch, 4),
                            "batch_idx": batch_idx,
                            "loss": float(loss_value) if isinstance(loss_value, (int, float)) else loss_value,
                            "input_text": full_input_text,
                            "input_tokens": valid_input_tokens,
                            "target_text": target_text,
                            "target_tokens": valid_target_tokens,
                            "predicted_text": pred_text,
                            "predicted_tokens": pred_tokens
                        }
                        
                        # 写入JSONL格式（每行一个JSON对象）
                        self.io_log_file.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
                        self.io_log_file.flush()  # 立即刷新到磁盘
                        
                except Exception as e:
                    # 不中断训练，只打印警告
                    if self.state.global_step % 100 == 0:  # 每100步才打印一次警告，避免刷屏
                        print(f"警告: 记录输入输出时出错: {e}")
            
            def _log_training_details(self, input_ids, labels, logits, loss, loss_value):
                """记录训练详情：输入token、目标token、预测token、loss"""
                try:
                    # 获取 tokenizer（兼容新旧版本）
                    tokenizer = getattr(self, 'processing_class', None) or getattr(self, 'tokenizer', None)
                    if tokenizer is None:
                        print("警告: 无法获取 tokenizer，跳过详细日志")
                        return
                    
                    # 只处理batch中的第一个样本（避免输出过多）
                    batch_idx = 0
                    
                    # 获取输入tokens
                    input_tokens = input_ids[batch_idx].cpu().tolist()
                    # 过滤padding tokens (通常是0或eos_token_id)
                    valid_input_tokens = [t for t in input_tokens if t != tokenizer.pad_token_id and t != 0]
                    
                    # 获取目标tokens（labels中不为-100的部分）
                    # 注意：logits[i] 预测的是 input_ids[i+1]，所以需要对齐
                    target_tokens = labels[batch_idx].cpu().tolist()
                    valid_target_tokens = [t for t in target_tokens if t != -100]
                    
                    # 获取预测tokens（logits中argmax）
                    # logits[i] 预测的是下一个token，所以 logits[i] 对应 labels[i+1]
                    # 注意：logits的形状是 [seq_len, vocab_size]
                    # 我们需要从logits中获取预测，但要排除padding位置
                    
                    # 找到有效目标的位置（labels不为-100的位置）
                    valid_target_positions = [i for i, t in enumerate(target_tokens) if t != -100]
                    
                    valid_pred_tokens = []
                    if len(valid_target_positions) > 0:
                        # 对于每个有效目标位置 i，预测来自 logits[i-1]（如果 i > 0）
                        # 因为 logits[i-1] 预测的是位置 i 的token
                        for pos in valid_target_positions:
                            if pos > 0 and pos - 1 < len(logits[batch_idx]):
                                # 直接从logits获取预测，排除pad_token_id
                                logit_slice = logits[batch_idx][pos - 1].clone()
                                
                                # 将pad_token_id和0的logit设为-inf，避免预测到这些token
                                if tokenizer.pad_token_id is not None and tokenizer.pad_token_id < logit_slice.shape[0]:
                                    logit_slice[tokenizer.pad_token_id] = float('-inf')
                                if 0 < logit_slice.shape[0]:
                                    logit_slice[0] = float('-inf')  # 也排除0
                                
                                # 获取argmax
                                pred_token = logit_slice.argmax().item()
                                
                                # 验证预测token是否有效
                                if pred_token == 0 or (tokenizer.pad_token_id is not None and pred_token == tokenizer.pad_token_id):
                                    # 如果还是预测到无效token，使用第二大的
                                    logit_slice[pred_token] = float('-inf')
                                    if logit_slice.max() > float('-inf'):
                                        pred_token = logit_slice.argmax().item()
                                
                                valid_pred_tokens.append(pred_token)
                            elif pos == 0:
                                # 第一个位置没有对应的预测（因为logits[0]预测的是input_ids[1]）
                                valid_pred_tokens.append(-1)  # 标记为无效
                    else:
                        valid_pred_tokens = []
                    
                    # 解码tokens为文本
                    input_text = tokenizer.decode(valid_input_tokens[:50], skip_special_tokens=True)  # 只显示前50个token
                    target_text = tokenizer.decode(valid_target_tokens[:50], skip_special_tokens=True) if len(valid_target_tokens) > 0 else ""
                    # 过滤无效预测后再解码
                    valid_pred_for_decode = [p for p in valid_pred_tokens[:50] if p != -1]
                    pred_text = tokenizer.decode(valid_pred_for_decode, skip_special_tokens=True) if len(valid_pred_for_decode) > 0 else ""
                    
                    print("\n" + "="*80)
                    print(f"Step {self.state.global_step} | Epoch {self.state.epoch:.2f} | Loss: {loss_value:.4f}")
                    print("-"*80)
                    print(f"输入 Tokens (前20个): {valid_input_tokens[:20]}")
                    print(f"输入文本 (前50 tokens): {input_text[:200]}...")
                    print("-"*80)
                    if len(valid_target_tokens) > 0:
                        print(f"目标 Tokens (前20个): {valid_target_tokens[:20]}")
                        print(f"目标文本 (前50 tokens): {target_text[:200]}...")
                    else:
                        print("目标 Tokens: (无有效目标)")
                    print("-"*80)
                    if len(valid_pred_tokens) > 0:
                        valid_pred_display = [p for p in valid_pred_tokens[:20] if p != -1]
                        print(f"预测 Tokens (前20个有效): {valid_pred_display}")
                        print(f"预测文本 (前50 tokens): {pred_text[:200]}...")
                    else:
                        print("预测 Tokens: (无预测)")
                    print("-"*80)
                    
                    # 计算准确率（在有效目标位置，排除无效预测）
                    if len(valid_target_tokens) > 0 and len(valid_pred_tokens) > 0:
                        # 过滤掉无效预测（-1）
                        valid_pairs = [(t, p) for t, p in zip(valid_target_tokens, valid_pred_tokens) if p != -1]
                        if len(valid_pairs) > 0:
                            correct = sum(1 for t, p in valid_pairs if t == p)
                            accuracy = correct / len(valid_pairs) if len(valid_pairs) > 0 else 0.0
                            print(f"Token准确率: {accuracy:.2%} ({correct}/{len(valid_pairs)})")
                        else:
                            print("Token准确率: 无法计算（无有效预测位置）")
                    else:
                        print("Token准确率: 无法计算（无有效目标或预测）")
                    
                    # 显示一些统计信息
                    print(f"有效目标tokens数: {len(valid_target_tokens)}")
                    print(f"有效预测tokens数: {len([p for p in valid_pred_tokens if p != -1])}")
                    
                    print("="*80 + "\n")
                except Exception as e:
                    print(f"记录训练详情时出错: {e}")
                    import traceback
                    traceback.print_exc()
        
        # 创建早停回调
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=early_stopping_patience,
            early_stopping_threshold=early_stopping_threshold
        )
        
        # 设置日志目录（从output_dir派生）
        log_dir = os.path.join(self.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        io_log_file_path = os.path.join(log_dir, 'io_logs.jsonl')
        print(f"输入输出日志将保存到: {io_log_file_path}")
        
        # 创建 Trainer（启用详细日志）
        trainer = CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=simple_collate_fn,
            tokenizer=self.tokenizer,
            callbacks=[early_stopping] if val_dataset else [],
            verbose_logging=True,  # 启用详细日志，显示每次迭代的输出、目标、token和loss
            log_dir=log_dir,  # 传递日志目录
        )
        
        # 开始训练
        print("开始训练（带早停机制）...")
        print(f"训练样本数: {len(train_dataset)}")
        if val_dataset:
            print(f"验证样本数: {len(val_dataset)}")
        print(f"使用配置: profile={self.use_profile}, history={self.use_history}, context={self.use_context}")
        print(f"最大轮次: {max_epochs}")
        print(f"早停耐心值: {early_stopping_patience}")
        print(f"早停阈值: {early_stopping_threshold}")
        if val_dataset:
            steps_per_epoch = len(train_dataset) // (train_config.get('batch_size', 1) * train_config.get('gradient_accumulation_steps', 16))
            eval_steps = max(1, steps_per_epoch // 2)
            print(f"每个epoch步数: {steps_per_epoch}, 每 {eval_steps} 步评估一次（每个epoch约2次）")
        
        trainer.train()
        
        # 保存最终模型（带错误处理）
        print(f"保存最终模型到 {self.output_dir}")
        try:
            trainer.save_model()
            self.tokenizer.save_pretrained(self.output_dir)
            print("✓ 模型保存成功")
        except (OSError, IOError) as e:
            print(f"警告: 无法保存模型到 {self.output_dir}: {e}")
            # 尝试保存到本地目录
            local_checkpoint_dir = os.path.join(os.path.expanduser("~"), "checkpoints")
            local_output_dir = os.path.join(local_checkpoint_dir, os.path.basename(self.output_dir))
            os.makedirs(local_output_dir, exist_ok=True)
            print(f"尝试保存到本地目录: {local_output_dir}")
            try:
                trainer.save_model(local_output_dir)
                self.tokenizer.save_pretrained(local_output_dir)
                print(f"✓ 模型已保存到本地目录: {local_output_dir}")
                self.output_dir = local_output_dir  # 更新输出目录
            except Exception as e2:
                print(f"错误: 无法保存模型到本地目录: {e2}")
                print("模型可能已保存在训练过程中的checkpoint目录")
        
        print("训练完成！")


def main():
    parser = argparse.ArgumentParser(description='消融实验训练（带早停）')
    parser.add_argument('--config', type=str,
                       default='/data/lingyu.li/parallel-post-train/ablation/config.json',
                       help='配置文件路径')
    parser.add_argument('--ablation_config', type=str, required=True,
                       choices=['profile_and_history_and_context', 'profile_and_history', 'profile_and_context', 
                               'history_and_context', 'profile_only', 'history_only', 'context_only'],
                       help='消融实验配置')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='验证集比例')
    parser.add_argument('--gpu', type=str, default="0",
                       help='使用的GPU编号，可以是单个GPU（如"0"）或多个GPU（如"0,1,2,3,4,5,6,7"），默认：0')
    parser.add_argument('--use_multi_gpu', action='store_true',
                       help='使用多GPU训练（自动检测所有可用GPU）')
    parser.add_argument('--max_epochs', type=int, default=20,
                       help='最大训练轮次（默认：20）')
    parser.add_argument('--early_stopping_patience', type=int, default=3,
                       help='早停耐心值：验证loss不下降的轮次数（默认：3）')
    parser.add_argument('--early_stopping_threshold', type=float, default=0.001,
                       help='早停阈值：loss改善的最小值（默认：0.001）')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='模型输出目录（覆盖配置文件中的设置）')
    parser.add_argument('--log_file', type=str, default=None,
                       help='训练日志文件路径（如果未指定，将自动生成）')
    
    args = parser.parse_args()
    
    # 设置使用的GPU
    if args.use_multi_gpu:
        # 自动使用所有可用GPU
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if num_gpus == 0:
            print("警告: 未检测到GPU，将使用CPU")
            gpu_ids = ""
        else:
            gpu_ids = ",".join([str(i) for i in range(num_gpus)])
            print(f"自动检测到 {num_gpus} 张GPU，将使用所有GPU")
    else:
        # 使用指定的GPU
        gpu_ids = args.gpu
        num_gpus = len(gpu_ids.split(",")) if gpu_ids else 1
    
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
    print(f"=" * 60)
    print(f"GPU 设置:")
    print(f"  GPU IDs: {gpu_ids}")
    print(f"  CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    print(f"  将使用 {num_gpus} 张GPU")
    print(f"=" * 60)
    
    # 验证GPU是否可用
    if torch.cuda.is_available():
        visible_gpu_count = torch.cuda.device_count()
        print(f"CUDA 可用，可见 GPU 数量: {visible_gpu_count}")
        for i in range(visible_gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {gpu_name}, 内存: {gpu_memory:.2f} GB")
    else:
        print("警告: CUDA 不可用，将使用 CPU")
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 获取消融配置
    ablation_config = config['ablation_configs'][args.ablation_config]
    use_profile = ablation_config.get('use_profile', True)
    use_history = ablation_config.get('use_history', True)
    use_context = ablation_config.get('use_context', True)
    config_name = ablation_config['name']
    
    print("=" * 60)
    print(f"消融实验（带早停）: {config_name}")
    print(f"使用配置: profile={use_profile}, history={use_history}, context={use_context}")
    print("=" * 60)
    
    # 加载训练数据
    print("加载训练数据...")
    train_path = config['data']['train_path']
    train_data = load_train_data(train_path)
    
    if not train_data:
        print(f"错误: 无法加载训练数据 from {train_path}")
        return
    
    # 提取训练样本（启用调试模式）
    all_samples = extract_training_samples(train_data, debug=True)
    print(f"提取了 {len(all_samples)} 个训练样本")
    
    # 如果需要使用 history，添加历史信息
    if use_history:
        print("添加历史信息...")
        all_samples = add_history_to_samples(all_samples, all_samples)
    
    # 划分训练集和验证集
    train_samples, val_samples = split_train_val(all_samples, args.val_ratio)
    print(f"训练集: {len(train_samples)} 个样本")
    print(f"验证集: {len(val_samples)} 个样本")
    
    # 获取模型配置（需要在所有地方都可用）
    model_config = config['model']
    
    # 设置输出目录（包含数据集名称）
    if args.output_dir:
        # 如果指定了输出目录，直接使用
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        print(f"使用指定的输出目录: {output_dir}")
    else:
        # 否则使用配置文件中的设置
        checkpoint_dir = model_config['checkpoint_dir']
        # 从数据路径提取数据集名称
        dataset_name = os.path.basename(os.path.dirname(train_path))
        output_dir = os.path.join(checkpoint_dir, f"{dataset_name}_ablation_{config_name}_earlystop")
        
        # 尝试创建目录，如果失败则使用本地目录
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"输出目录: {output_dir}")
        except (OSError, IOError) as e:
            print(f"警告: 无法在 {checkpoint_dir} 创建目录: {e}")
            # 使用本地目录作为备选
            local_checkpoint_dir = os.path.join(os.path.expanduser("~"), "checkpoints")
            output_dir = os.path.join(local_checkpoint_dir, f"{dataset_name}_ablation_{config_name}_earlystop")
            os.makedirs(output_dir, exist_ok=True)
            print(f"使用本地目录: {output_dir}")
    
    # 设置日志文件路径
    if args.log_file:
        log_file_path = args.log_file
    else:
        # 自动生成日志文件路径
        from datetime import datetime
        log_dir = os.path.join(output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = os.path.basename(os.path.dirname(train_path))
        log_file_path = os.path.join(log_dir, f"training_{dataset_name}_{config_name}_{timestamp}.json")
    
    print(f"训练日志将保存到: {log_file_path}")
    
    # 创建训练器
    model_path = model_config['path']
    trainer = AblationTrainerWithEarlyStopping(
        model_path=model_path,
        output_dir=output_dir,
        config=config,
        use_profile=use_profile,
        use_history=use_history,
        use_context=use_context,
        log_file_path=log_file_path,
        use_multi_gpu=args.use_multi_gpu or num_gpus > 1
    )
    
    # 开始训练（带早停）
    trainer.train(
        train_samples, 
        val_samples,
        max_epochs=args.max_epochs,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=args.early_stopping_threshold
    )
    
    print(f"\n训练完成！模型保存在: {output_dir}")


if __name__ == '__main__':
    main()
