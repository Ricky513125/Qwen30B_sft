"""
推理脚本：使用微调后的模型生成 test_leaderboard.json 的 continuations（不使用 vLLM）
支持6个消融实验模型（3种配置 × 2个数据集）
"""
import json
import argparse
import os
import sys
import re
import torch
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import logging

# 设置 logger
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))
try:
    from data_loader import load_train_data, extract_training_samples
    from prompt_builder import build_prompt
except ImportError:
    print("错误: 无法导入 data_loader 或 prompt_builder，请确保脚本在正确目录下。")

from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("警告: peft 库未安装，无法加载 LoRA 模型")

try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False
    print("警告: deepspeed 库未安装，无法使用 DeepSpeed ZeRO-3")


def load_test_leaderboard(test_leaderboard_path: str) -> list:
    with open(test_leaderboard_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def get_user_info_from_leaderboard(sample: dict, train_data: list) -> dict:
    """从训练数据中获取用户信息（profile, history等）"""
    user_hash = sample.get('user_hash', '')
    user = sample.get('user', {})
    user_profile = user.get('profile')
    
    user_train_samples = []
    if train_data:
        for item in train_data:
            if item.get('user_hash') == user_hash:
                user_train_samples.append(item)
    
    if not user_profile and user_train_samples:
        first_sample = user_train_samples[0]
        train_user = first_sample.get('user', {})
        user_profile = train_user.get('profile')
    
    return {
        'user_hash': user_hash,
        'user_profile': user_profile,
        'user_train_samples': user_train_samples
    }


def clean_model_output(text: str, max_length: int = 512) -> str:
    """精简优化：只保留第一段核心对话，处理复读和污染。"""
    if not text: return ""

    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<\|im_start\|>.*?\n|<\|im_end\|>|<\|user\|>|<\|assistant\|>', '', text)
    text = text.replace('<think>', '').replace('</think>', '').replace('ROKE', '')

    stop_markers = [
        r'\n（', r'\n\(', r'\n[*]{2,}', r'\n※', r'\n問題[：:]',
        r'\n最终生成', r'\n分析', r'\n建议', r'\n\s*---', r'\n[A-Z]\)', r'RealPersonaChat'
    ]
    
    combined_stop_pattern = '|'.join(stop_markers)
    match = re.search(combined_stop_pattern, text)
    if match:
        text = text[:match.start()]

    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    if not paragraphs: return ""
    
    result = paragraphs[0]
    result = re.sub(r'[（\(][^）\)]*?(感情|态度|注|角色|姿态|笑顔|喜)[^）\)]*?[）\)]', '', result)
    
    for marker in ['回答：', '返答：', '响应：', '答：', '最终生成：']:
        if marker in result:
            result = result.split(marker)[-1]

    return result[:max_length].strip()


def generate_continuations(
    model, tokenizer, messages, num_samples=5, max_new_tokens=512,
    device=None, do_sample=True, temperature=0.7, top_p=0.9,
    repetition_penalty=1.2, no_repeat_ngram_size=4, max_output_length=512,
    is_japanese_task=False, max_input_length=4096, use_batch_generation=True
):
    """使用标准 transformers 生成 continuations（适配 Gemma 模型）"""
    # Gemma 模型的 chat template 使用方式
    # 与训练时保持一致：使用 add_generation_prompt=False，然后手动添加引导符
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False  # 与训练时保持一致
    )
    
    # 手动添加 "user: " 提示（与训练时完全一致）
    # 让模型预测用户会说什么
    generation_suffix = "\nuser: "
    text = text.strip() + generation_suffix

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_input_length)
    if device:
        inputs = {k: v.to(device) for k, v in inputs.items()}

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "num_return_sequences": num_samples if use_batch_generation else 1
    }

    with torch.inference_mode():
        if use_batch_generation and num_samples > 1:
            outputs = model.generate(**inputs, **generation_kwargs)
        else:
            # 单次生成，循环多次
            outputs = []
            for _ in range(num_samples):
                output = model.generate(**inputs, **generation_kwargs)
                outputs.append(output[0])
            outputs = torch.stack(outputs) if len(outputs) > 1 else outputs[0]
    
    results = []
    input_len = inputs["input_ids"].shape[1]
    if isinstance(outputs, torch.Tensor) and outputs.dim() > 1:
        for i in range(len(outputs)):
            gen_text = tokenizer.decode(outputs[i][input_len:], skip_special_tokens=True)
            results.append(clean_model_output(gen_text, max_output_length))
    else:
        gen_text = tokenizer.decode(outputs[input_len:], skip_special_tokens=True)
        results.append(clean_model_output(gen_text, max_output_length))
    
    return results


def check_model_files(model_path: str) -> dict:
    result = {'has_safetensors': False, 'has_pytorch': False, 'safetensors_files': [], 'pytorch_files': []}
    if not os.path.exists(model_path): return result
    for file in os.listdir(model_path):
        if file.endswith('.safetensors'):
            result['has_safetensors'] = True
            result['safetensors_files'].append(file)
        elif file.endswith('.bin') and 'pytorch_model' in file:
            result['has_pytorch'] = True
            result['pytorch_files'].append(file)
    return result


def process_scenario(
    scenario_path, checkpoint_dir, config_name, use_profile, use_history,
    use_context=True, num_samples=5, output_dir=None, gpu_id=0,
    log_file_path=None, base_model_path=None, max_new_tokens=512,
    max_output_length=200, use_multi_gpu=False, use_deepspeed=False,
    data_shard_id=None, num_shards=8
):
    print(f"\n>>> 启动场景处理: {os.path.basename(scenario_path)}")
    
    # 1. 加载数据
    test_leaderboard_path = os.path.join(scenario_path, 'test_leaderboard.json')
    test_data = load_test_leaderboard(test_leaderboard_path)
    
    # 数据分片（如果指定了data_shard_id）
    if data_shard_id is not None:
        shard_size = len(test_data) // num_shards
        start_idx = data_shard_id * shard_size
        if data_shard_id == num_shards - 1:
            # 最后一个分片包含剩余的所有数据
            end_idx = len(test_data)
        else:
            end_idx = start_idx + shard_size
        test_data = test_data[start_idx:end_idx]
        print(f"数据分片 {data_shard_id}/{num_shards}: 处理样本 {start_idx} 到 {end_idx} (共 {len(test_data)} 个样本)")
    
    train_path = os.path.join(scenario_path, 'train.json')
    train_data = load_train_data(train_path) if os.path.exists(train_path) else []
    all_train_samples = extract_training_samples(train_data) if train_data else []

    # 2. 加载 tokenizer（适配 Gemma 模型）
    # Gemma tokenizer 需要 tokenizer.model 文件（SentencePiece模型）
    # 如果checkpoint中没有tokenizer.model，需要从base_model_path加载
    checkpoint_dir = str(checkpoint_dir)
    
    # 检查checkpoint中是否有完整的tokenizer文件
    has_tokenizer_json = os.path.exists(os.path.join(checkpoint_dir, 'tokenizer.json'))
    has_tokenizer_model = os.path.exists(os.path.join(checkpoint_dir, 'tokenizer.model'))
    has_tokenizer_config = os.path.exists(os.path.join(checkpoint_dir, 'tokenizer_config.json'))
    
    # 如果checkpoint中有tokenizer.json和tokenizer_config.json，尝试使用
    # 但如果缺少tokenizer.model，可能需要从base_model_path加载
    if has_tokenizer_json and has_tokenizer_config:
        # 检查是否需要tokenizer.model（通过读取tokenizer_config.json）
        try:
            import json
            with open(os.path.join(checkpoint_dir, 'tokenizer_config.json'), 'r') as f:
                tokenizer_config = json.load(f)
            # 如果config中有vocab_file或model_file字段，检查文件是否存在
            vocab_file = tokenizer_config.get('vocab_file') or tokenizer_config.get('model_file')
            if vocab_file:
                # 如果是相对路径，转换为绝对路径
                if not os.path.isabs(vocab_file):
                    vocab_file = os.path.join(checkpoint_dir, vocab_file)
                if not os.path.exists(vocab_file):
                    # tokenizer.model不存在，需要从base_model_path加载
                    print(f"checkpoint中缺少tokenizer.model文件，从base_model_path加载")
                    if base_model_path and os.path.exists(str(base_model_path)):
                        tokenizer_path = str(base_model_path)
                    else:
                        tokenizer_path = '/mnt/parallel/models/gemma-3-27b-it'
                else:
                    tokenizer_path = str(checkpoint_dir)
            else:
                # 没有vocab_file字段，可能使用tokenizer.json，直接使用checkpoint
                tokenizer_path = str(checkpoint_dir)
        except Exception as e:
            print(f"读取tokenizer_config.json失败: {e}，尝试从base_model_path加载")
            if base_model_path and os.path.exists(str(base_model_path)):
                tokenizer_path = str(base_model_path)
            else:
                tokenizer_path = '/mnt/parallel/models/gemma-3-27b-it'
    else:
        # checkpoint中没有tokenizer文件，从base_model_path加载
        if base_model_path and os.path.exists(str(base_model_path)):
            tokenizer_path = str(base_model_path)
        else:
            tokenizer_path = '/mnt/parallel/models/gemma-3-27b-it'
    
    # 确保tokenizer_path是字符串且存在
    tokenizer_path = str(tokenizer_path)
    if not os.path.exists(tokenizer_path):
        raise ValueError(f"tokenizer路径不存在: {tokenizer_path}")
    
    print(f"使用tokenizer路径: {tokenizer_path}")
    
    # 加载tokenizer
    try:
        # 优先使用快速tokenizer（如果tokenizer.json存在）
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            use_fast=True,
            trust_remote_code=True,
            local_files_only=True
        )
    except Exception as e:
        # 如果快速tokenizer失败，尝试使用慢速tokenizer
        print(f"快速tokenizer加载失败，尝试慢速tokenizer: {e}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                use_fast=False,
                trust_remote_code=True,
                local_files_only=True
            )
        except Exception as e2:
            # 如果都失败，尝试不使用local_files_only
            print(f"慢速tokenizer也失败，尝试允许下载: {e2}")
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                use_fast=False,
                trust_remote_code=True,
                local_files_only=False
            )
    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token
    print("✓ Tokenizer 加载成功")

    # 3. 加载模型（适配 Gemma 模型，从 checkpoint_dir 加载）
    print("加载模型（使用标准 transformers，适配 Gemma）...")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    # Gemma 模型直接从 checkpoint_dir 加载
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir,
        torch_dtype=dtype,
        device_map="auto" if use_multi_gpu else {"": f"cuda:{gpu_id}"},
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        local_files_only=True
    )
    
    model.eval()
    print("✓ 模型加载完成")

    # 4. 推理循环（适配 Gemma 模型，保持与原始 inference.py 一致的数据结构）
    for sample_idx, sample in enumerate(tqdm(test_data, desc="生成中")):
        # test_leaderboard.json的结构：context在task.task_behavior_collections[0].data[0].context
        task = sample.get('task', {})
        collections = task.get('task_behavior_collections', [])
        
        if not collections:
            print(f"警告: 样本 {sample_idx} 缺少 task_behavior_collections，跳过")
            continue
        
        # 获取用户信息
        u_info = get_user_info_from_leaderboard(sample, train_data)
        
        # 获取 history（如果需要）
        history = None
        if use_history and u_info.get('user_train_samples'):
            # 使用 get_user_only_history 获取历史（与训练时一致）
            from data_loader import get_user_only_history
            history = get_user_only_history(
                all_train_samples,
                u_info['user_hash'],
                current_sample=sample,
                current_context=None,  # 将在下面的循环中获取
                max_history=15,
                use_cache=True
            )
        
        # 处理每个collection中的data
        for collection in collections:
            data_items = collection.get('data', [])
            for data_item in data_items:
                if data_item.get('continuations'): 
                    continue  # 跳过已存在的
                
                # 获取 context
                context = data_item.get('context', [])
                
                # 构建 prompt（使用 Gemma 的 build_prompt）
                msgs = build_prompt(
                    context=context if use_context else None,
                    user_profile=u_info['user_profile'] if use_profile else None,
                    history=history,
                    use_profile=use_profile, 
                    use_history=use_history,
                    use_context=use_context
                )
                
                # 判断日语任务
                is_jp = 'RealPersonaChat' in scenario_path or any(
                    re.search(r'[\u3040-\u309f]', t.get('content','')) 
                    for t in (context[-2:] if context else [])
                )
                
                conts = generate_continuations(
                    model=model, 
                    tokenizer=tokenizer, 
                    messages=msgs,
                    num_samples=num_samples, 
                    max_new_tokens=max_new_tokens,
                    device=f"cuda:{gpu_id}" if not use_multi_gpu else None,
                    is_japanese_task=is_jp,
                    max_output_length=max_output_length
                )
                data_item['continuations'] = conts

    # 5. 保存结果
    if not output_dir:
        output_dir = os.path.join("/mnt/parallel/outputs", f"{os.path.basename(scenario_path)}_{config_name}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 如果使用了数据分片，文件名包含分片ID
    if data_shard_id is not None:
        out_file = os.path.join(output_dir, f'test_leaderboard_shard_{data_shard_id}.json')
    else:
        out_file = os.path.join(output_dir, 'test_leaderboard.json')
    
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    print(f"成功保存至: {out_file}")


def main():
    parser = argparse.ArgumentParser(description='推理脚本（不使用 vLLM）')
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--scenario_path', type=str, required=True)
    parser.add_argument('--config_name', type=str, required=True)
    parser.add_argument('--base_model_path', type=str, default="/mnt/parallel/models/gemma-3-27b-it",
                       help='基础模型路径（用于加载tokenizer，如果checkpoint中没有）')
    parser.add_argument('--use_profile', action='store_true')
    parser.add_argument('--use_history', action='store_true')
    parser.add_argument('--use_context', action='store_true', default=False)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--max_output_length', type=int, default=200)
    parser.add_argument('--use_multi_gpu', action='store_true', help='使用多 GPU 分布式加载模型')
    parser.add_argument('--use_deepspeed', action='store_true', help='使用 DeepSpeed ZeRO-3')
    parser.add_argument('--data_shard_id', type=int, default=None, help='数据分片ID（0-7，用于8卡并行，每个进程处理1/8数据）')
    parser.add_argument('--num_shards', type=int, default=8, help='数据分片总数（默认：8）')
    
    args = parser.parse_args()

    process_scenario(
        scenario_path=args.scenario_path,
        checkpoint_dir=args.checkpoint_dir,
        config_name=args.config_name,
        use_profile=args.use_profile,
        use_history=args.use_history,
        use_context=args.use_context,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        gpu_id=args.gpu,
        base_model_path=args.base_model_path,
        max_new_tokens=args.max_new_tokens,
        max_output_length=args.max_output_length,
        use_multi_gpu=args.use_multi_gpu,
        use_deepspeed=args.use_deepspeed,
        data_shard_id=args.data_shard_id,
        num_shards=args.num_shards
    )

if __name__ == '__main__':
    main()
