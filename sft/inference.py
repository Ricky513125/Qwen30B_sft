"""
推理脚本：使用微调后的模型生成 test_leaderboard.json 的 continuations
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
import signal
from contextlib import contextmanager
import logging

# 设置 logger
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))
# 假设这些模块在同一目录下
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

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("警告: vLLM 库未安装，无法使用 vLLM 加速推理")


@contextmanager
def timeout_handler(seconds):
    """超时处理上下文管理器（仅Linux/Unix）"""
    def timeout_signal(signum, frame):
        raise TimeoutError(f"操作超时（{seconds}秒）")
    
    if hasattr(signal, 'SIGALRM'):
        old_handler = signal.signal(signal.SIGALRM, timeout_signal)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        yield


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
    is_japanese_task=False, max_input_length=4096, use_batch_generation=True,
    vllm_engine=None
):
    if vllm_engine is not None:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        text = text.strip() + "<|im_start|>user\n"
        
        if is_japanese_task:
            text = text.rstrip() + "\n回答："
        
        sampling_params = SamplingParams(
            temperature=temperature if do_sample else 0.0,
            top_p=top_p if do_sample else 1.0,
            max_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
        )
        
        prompts = [text] * num_samples
        outputs = vllm_engine.generate(prompts, sampling_params)
        return [clean_model_output(out.outputs[0].text, max_output_length) for out in outputs]

    # 标准 Transformers 模式
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    text = text.strip() + "<|im_start|>user\n"
    if is_japanese_task:
        text = text.rstrip() + "\n回答："

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
        outputs = model.generate(**inputs, **generation_kwargs)
    
    results = []
    input_len = inputs["input_ids"].shape[1]
    for i in range(len(outputs)):
        gen_text = tokenizer.decode(outputs[i][input_len:], skip_special_tokens=True)
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
    use_vllm=False, vllm_tensor_parallel_size=1, vllm_max_model_len=None,
    data_shard_id=None, num_shards=8
):
    print(f"\n>>> 启动场景处理: {os.path.basename(scenario_path)}")
    
    # 1. 加载数据
    test_leaderboard_path = os.path.join(scenario_path, 'test_leaderboard.json')
    test_data = load_test_leaderboard(test_leaderboard_path)
    
    if data_shard_id is not None:
        shard_size = len(test_data) // num_shards
        test_data = test_data[data_shard_id*shard_size : (data_shard_id+1)*shard_size] if data_shard_id < num_shards-1 else test_data[data_shard_id*shard_size:]

    train_path = os.path.join(scenario_path, 'train.json')
    train_data = load_train_data(train_path) if os.path.exists(train_path) else []

    # 2. 模型加载逻辑
    vllm_engine = None
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    if use_vllm and VLLM_AVAILABLE:
        vllm_kwargs = {
            "model": base_model_path,
            "tensor_parallel_size": vllm_tensor_parallel_size,
            "trust_remote_code": True,
            "gpu_memory_utilization": 0.75,
        }
        if vllm_max_model_len:
            vllm_kwargs["max_model_len"] = vllm_max_model_len
        vllm_engine = LLM(**vllm_kwargs)
        model = None
    else:
        # 非 vLLM 加载
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        is_lora = os.path.exists(os.path.join(checkpoint_dir, 'adapter_config.json'))
        
        if is_lora and PEFT_AVAILABLE:
            base = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=dtype, device_map="auto" if use_multi_gpu else f"cuda:{gpu_id}", trust_remote_code=True)
            model = PeftModel.from_pretrained(base, checkpoint_dir)
        else:
            # --- 修正后的普通模型/DeepSpeed加载逻辑 ---
            if use_deepspeed and DEEPSPEED_AVAILABLE:
                print("使用 DeepSpeed ZeRO-3 加载...")
                # 这里简化处理，实际DS推理通常需要ds_config
                model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=dtype, device_map="auto", trust_remote_code=True)
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    base_model_path if not is_lora else checkpoint_dir,
                    torch_dtype=dtype,
                    device_map="auto" if use_multi_gpu else { "": f"cuda:{gpu_id}" },
                    trust_remote_code=True
                )
        if model is not None:
            model.eval()

    # 3. 推理循环
    for sample in tqdm(test_data, desc="生成中"):
        collections = sample.get('task', {}).get('task_behavior_collections', [])
        for coll in collections:
            for item in coll.get('data', []):
                if item.get('continuations'): continue # 跳过已存在的
                
                u_info = get_user_info_from_leaderboard(sample, train_data)
                hist = u_info['user_train_samples'][-3:] if use_history else None
                
                msgs = build_prompt(
                    context=item.get('context', []),
                    user_profile=u_info['user_profile'] if use_profile else None,
                    history=hist,
                    use_profile=use_profile, use_history=use_history
                )
                
                # 判断日语任务
                is_jp = 'RealPersonaChat' in scenario_path or any(re.search(r'[\u3040-\u309f]', t.get('content','')) for t in item.get('context',[])[-2:])
                
                conts = generate_continuations(
                    model=model, tokenizer=tokenizer, messages=msgs,
                    num_samples=num_samples, max_new_tokens=max_new_tokens,
                    device=f"cuda:{gpu_id}" if model else None,
                    is_japanese_task=is_jp, vllm_engine=vllm_engine
                )
                item['continuations'] = conts

    # 4. 保存结果
    if not output_dir:
        output_dir = os.path.join("/mnt/parallel/outputs", f"{os.path.basename(scenario_path)}_{config_name}")
        os.makedirs(output_dir, exist_ok=True)
    
    out_file = os.path.join(output_dir, f'test_leaderboard{"_shard_"+str(data_shard_id) if data_shard_id is not None else ""}.json')
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    print(f"成功保存至: {out_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--scenario_path', type=str, required=True)
    parser.add_argument('--config_name', type=str, required=True)
    parser.add_argument('--base_model_path', type=str, default="/mnt/parallel/models/Qwen3-8B")
    parser.add_argument('--use_profile', action='store_true')
    parser.add_argument('--use_history', action='store_true')
    parser.add_argument('--use_context', action='store_true', default=False)
    parser.add_argument('--use_vllm', action='store_true')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--max_output_length', type=int, default=200)
    parser.add_argument('--vllm_tensor_parallel_size', type=int, default=1)
    parser.add_argument('--vllm_max_model_len', type=int, default=None)
    parser.add_argument('--data_shard_id', type=int, default=None)
    parser.add_argument('--num_shards', type=int, default=8)
    
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
        use_vllm=args.use_vllm,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        vllm_max_model_len=args.vllm_max_model_len,
        data_shard_id=args.data_shard_id,
        num_shards=args.num_shards
    )

if __name__ == '__main__':
    main()