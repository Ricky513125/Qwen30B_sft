"""
训练器模块 - 优化版
适配消融实验，支持严格的角色控制与日志监控
"""
import os
import re
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from typing import List, Dict, Any, Optional, Tuple
import sys
from pathlib import Path

# 添加当前目录到路径，确保能导入 prompt_builder
sys.path.insert(0, str(Path(__file__).parent))
from prompt_builder import build_training_prompt


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
        # 索引 0 是 system，1 和 2 是最早的一对 user/assistant
        while len(self.tokenizer.apply_chat_template(messages, tokenize=True)) > (self.max_length - 512):
            if len(messages) > 2:
                messages.pop(1) # 弹出最早的对话
            else:
                break

        # 2. 生成 Prompt (手动添加引导符)
        full_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        generation_suffix = "<|im_start|>user\n"

        # 3. 组合成真正的 Prompt
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


class AblationTrainer:
    """消融实验主控类"""
    
    def __init__(self, model_path: str, output_dir: str, config: Dict[str, Any], 
                 use_profile: bool = True, use_history: bool = True, use_context: bool = True, log_file_path: Optional[str] = None):
        self.model_path = model_path
        self.output_dir = output_dir
        self.config = config
        self.use_profile = use_profile
        self.use_history = use_history
        self.use_context = use_context
        self.log_file_path = log_file_path

        # 1. 加载 Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 2. 加载模型
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            trust_remote_code=True
        ).to(self.device)
        
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()

    def train(self, train_samples: List[Dict[str, Any]], val_samples: Optional[List[Dict[str, Any]]] = None):
        train_config = self.config.get('training', {})
        
        train_dataset = AblationDataset(
            train_samples, self.tokenizer, 
            max_length=train_config.get('max_length', 32768),
            use_profile=self.use_profile, use_history=self.use_history, use_context=self.use_context
        )

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=train_config.get('num_epochs', 3),
            per_device_train_batch_size=train_config.get('batch_size', 1),
            gradient_accumulation_steps=train_config.get('gradient_accumulation_steps', 16),
            learning_rate=train_config.get('learning_rate', 2e-5),
            logging_steps=10,
            save_steps=100,
            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(),
            report_to="none",
            remove_unused_columns=False
        )

        trainer = CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer,
            verbose_logging=True,
            log_file_path=self.log_file_path
        )

        print(f"🚀 开始训练: Profile={self.use_profile}, History={self.use_history}, Context={self.use_context}")
        trainer.train()
        
        # 保存
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)