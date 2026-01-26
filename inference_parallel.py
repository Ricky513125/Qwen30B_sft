"""
推理脚本：支持单卡优先，否则数据并行
- 优先尝试单卡部署（无通信）
- 如果单卡OOM，则使用数据并行（将数据分片到多张卡）
"""
import json
import argparse
import os
import sys
import re
import torch
import torch.multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from typing import List, Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_train_data, extract_training_samples, get_user_history_samples
from prompt_builder import build_prompt
from transformers import AutoTokenizer, AutoModelForCausalLM

# 导入原始推理函数
from inference import generate_continuations, load_test_leaderboard, extract_context_from_leaderboard, get_user_info_from_leaderboard


def load_model_single_gpu(checkpoint_dir: str, gpu_id: int, base_model_path: str = None):
    """
    尝试在单张GPU上加载模型
    如果成功返回模型和tokenizer，如果OOM则返回None
    """
    print(f"尝试在 GPU {gpu_id} 上加载模型...")
    
    # 设置目标设备
    target_device = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(gpu_id)
    
    # 加载tokenizer（优先使用checkpoint_dir，因为tokenizer应该和模型一起保存）
    # 检查checkpoint中是否有tokenizer文件
    tokenizer_files = ['tokenizer.json', 'tokenizer_config.json', 'spiece.model', 'vocab.json']
    has_tokenizer = any(os.path.exists(os.path.join(checkpoint_dir, f)) for f in tokenizer_files)
    
    if has_tokenizer:
        tokenizer_path = str(checkpoint_dir)  # 确保是字符串
        print(f"  从checkpoint加载tokenizer: {tokenizer_path}")
    else:
        # 如果checkpoint中没有tokenizer，尝试使用base_model_path
        if base_model_path and os.path.exists(str(base_model_path)):
            tokenizer_path = str(base_model_path)
            print(f"  从基础模型路径加载tokenizer: {tokenizer_path}")
        else:
            # 尝试使用Gemma模型路径
            possible_paths = [
                '/mnt/parallel/models/gemma-3-27b-it',
                '/mnt/parallel/models/gemma-3-27b',
            ]
            tokenizer_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    tokenizer_path = str(path)
                    print(f"  从默认模型路径加载tokenizer: {tokenizer_path}")
                    break
            
            if tokenizer_path is None:
                raise ValueError(f"无法找到tokenizer。请确保checkpoint_dir中有tokenizer文件，或提供--base_model_path")
    
    # 确保tokenizer_path是字符串且存在
    tokenizer_path = str(tokenizer_path)
    if not os.path.exists(tokenizer_path):
        raise ValueError(f"tokenizer路径不存在: {tokenizer_path}")
    
    print(f"  使用tokenizer路径: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        use_fast=False,
        trust_remote_code=True,
        local_files_only=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 尝试加载模型
    try:
        # 首先尝试强制单卡加载
        print("  尝试强制加载到单张GPU...")
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_dir,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map={"": target_device},
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        
        # 验证模型是否真的在单张卡上
        device_set = set()
        for param in model.parameters():
            device_set.add(param.device)
        
        if len(device_set) == 1:
            actual_device = list(device_set)[0]
            if actual_device == target_device:
                print(f"  ✓ 模型成功加载到单张GPU: {target_device}")
                return model, tokenizer
            else:
                print(f"  ⚠ 模型被加载到 {actual_device}，期望 {target_device}")
                # 尝试移动到目标设备
                model = model.to(target_device)
                print(f"  ✓ 模型已移动到目标GPU: {target_device}")
                return model, tokenizer
        else:
            print(f"  ⚠ 模型被分配到多张GPU: {device_set}")
            del model
            torch.cuda.empty_cache()
            return None, tokenizer
            
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "OOM" in str(e) or "CUDA out of memory" in str(e):
            print(f"  ⚠ GPU {gpu_id} 显存不足，无法单卡部署: {e}")
            if 'model' in locals():
                del model
            torch.cuda.empty_cache()
            return None, tokenizer
        else:
            raise e


def process_data_shard(
    shard_id: int,
    num_shards: int,
    test_data: List[Dict],
    checkpoint_dir: str,
    scenario_path: str,
    config_name: str,
    use_profile: bool,
    use_history: bool,
    use_context: bool,
    num_samples: int,
    gpu_id: int,
    output_dir: str,
    train_data: List[Dict],
    all_train_samples: List[Dict],
    max_new_tokens: int,
    max_output_length: int,
):
    """
    处理一个数据分片（数据并行）
    """
    # 设置GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    torch.cuda.set_device(0)  # 在CUDA_VISIBLE_DEVICES设置后，GPU 0就是指定的GPU
    
    print(f"\n[Shard {shard_id}/{num_shards}] 处理数据分片，使用 GPU {gpu_id}")
    
    # 分割数据
    shard_size = len(test_data) // num_shards
    start_idx = shard_id * shard_size
    if shard_id == num_shards - 1:
        # 最后一个分片包含剩余的所有数据
        end_idx = len(test_data)
    else:
        end_idx = (shard_id + 1) * shard_size
    
    shard_data = test_data[start_idx:end_idx]
    print(f"  处理样本 {start_idx} 到 {end_idx} (共 {len(shard_data)} 个样本)")
    
    # 加载模型
    target_device = torch.device('cuda:0')  # 在CUDA_VISIBLE_DEVICES设置后，使用0
    
    # 加载tokenizer
    # 确保checkpoint_dir是字符串
    checkpoint_dir = str(checkpoint_dir)
    
    # 检查checkpoint中是否有tokenizer文件
    tokenizer_files = ['tokenizer.json', 'tokenizer_config.json', 'spiece.model', 'vocab.json']
    has_tokenizer = any(os.path.exists(os.path.join(checkpoint_dir, f)) for f in tokenizer_files)
    
    if has_tokenizer:
        tokenizer_path = checkpoint_dir
        print(f"  从checkpoint加载tokenizer: {tokenizer_path}")
    else:
        # 如果checkpoint中没有tokenizer，尝试从base_model_path加载
        if base_model_path:
            base_model_path = str(base_model_path)
            if os.path.exists(base_model_path):
                tokenizer_path = base_model_path
                print(f"  从基础模型路径加载tokenizer: {tokenizer_path}")
            else:
                raise ValueError(f"base_model_path不存在: {base_model_path}")
        else:
            # 尝试使用Gemma模型路径
            possible_paths = [
                '/mnt/parallel/models/gemma-3-27b-it',
                '/mnt/parallel/models/gemma-3-27b',
            ]
            tokenizer_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    tokenizer_path = path
                    print(f"  从默认模型路径加载tokenizer: {tokenizer_path}")
                    break
            
            if tokenizer_path is None:
                raise ValueError(f"无法找到tokenizer。请确保checkpoint_dir中有tokenizer文件，或提供--base_model_path")
    
    # 确保tokenizer_path是字符串且存在
    tokenizer_path = str(tokenizer_path)
    if not os.path.exists(tokenizer_path):
        raise ValueError(f"tokenizer路径不存在: {tokenizer_path}")
    
    print(f"  使用tokenizer路径: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        use_fast=False,
        trust_remote_code=True,
        local_files_only=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型（使用device_map="auto"让transformers自动分配，可能跨多卡）
    print("  加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",  # 允许跨多卡，因为单卡可能放不下
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    
    model.eval()
    print("  ✓ 模型加载完成")
    
    # 处理数据
    results = []
    for sample in tqdm(shard_data, desc=f"Shard {shard_id}"):
        try:
            # 获取用户信息
            user_info = get_user_info_from_leaderboard(sample, train_data)
            context = extract_context_from_leaderboard(sample)
            
            # 构建prompt
            messages = build_prompt(
                user_profile=user_info.get('user_profile'),
                history=user_info.get('user_train_samples', []) if use_history else [],
                context=context if use_context else None,
                use_profile=use_profile,
                use_history=use_history,
                use_context=use_context
            )
            
            # 生成continuations
            continuations = generate_continuations(
                model=model,
                tokenizer=tokenizer,
                messages=messages,
                num_samples=num_samples,
                max_new_tokens=max_new_tokens,
                device=target_device,
                max_output_length=max_output_length
            )
            
            # 构建结果
            result = {
                'id': sample.get('id', ''),
                'user_hash': sample.get('user_hash', ''),
                'continuations': continuations
            }
            results.append(result)
            
        except Exception as e:
            print(f"  处理样本 {sample.get('id', 'unknown')} 时出错: {e}")
            results.append({
                'id': sample.get('id', ''),
                'user_hash': sample.get('user_hash', ''),
                'continuations': []
            })
    
    # 保存分片结果
    shard_output_file = os.path.join(output_dir, f'shard_{shard_id}.json')
    os.makedirs(output_dir, exist_ok=True)
    with open(shard_output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"  ✓ 分片 {shard_id} 完成，结果保存到: {shard_output_file}")
    return shard_output_file


def process_scenario_parallel(
    scenario_path: str,
    checkpoint_dir: str,
    config_name: str,
    use_profile: bool,
    use_history: bool,
    use_context: bool = True,
    num_samples: int = 5,
    output_dir: str = None,
    gpu_ids: str = None,  # 例如 "0,1,2,3" 或 "4,5,6,7"
    log_file_path: str = None,
    base_model_path: str = None,
    max_new_tokens: int = 4096,
    max_output_length: int = 200
):
    """
    处理场景：优先单卡，否则数据并行
    """
    print(f"\n处理场景: {scenario_path}")
    print(f"模型: {checkpoint_dir}")
    print(f"配置: profile={use_profile}, history={use_history}, context={use_context}")
    
    # 解析GPU IDs
    if gpu_ids:
        gpu_list = [int(x.strip()) for x in gpu_ids.split(',')]
    else:
        # 默认使用所有可见的GPU
        gpu_list = list(range(torch.cuda.device_count()))
    
    print(f"可用GPU: {gpu_list}")
    
    # 加载测试数据
    test_leaderboard_path = os.path.join(scenario_path, 'test_leaderboard.json')
    if not os.path.exists(test_leaderboard_path):
        print(f"警告: {test_leaderboard_path} 不存在，跳过")
        return
    
    test_data = load_test_leaderboard(test_leaderboard_path)
    print(f"测试样本数: {len(test_data)}")
    
    # 加载训练数据
    train_path = os.path.join(scenario_path, 'train.json')
    train_data = load_train_data(train_path) if os.path.exists(train_path) else []
    all_train_samples = extract_training_samples(train_data) if train_data else []
    print(f"训练样本数: {len(all_train_samples)}")
    
    # 设置输出目录
    if output_dir is None:
        dataset_name = os.path.basename(scenario_path)
        output_dir = os.path.join(scenario_path, f"output_{config_name}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 策略1：尝试单卡部署
    print("\n=== 策略1: 尝试单卡部署 ===")
    single_gpu_success = False
    
    for gpu_id in gpu_list:
        model, tokenizer = load_model_single_gpu(checkpoint_dir, gpu_id, base_model_path)
        if model is not None:
            print(f"\n✓ 成功在 GPU {gpu_id} 上单卡部署模型")
            single_gpu_success = True
            
            # 处理所有数据
            target_device = torch.device(f'cuda:{gpu_id}')
            model.eval()
            
            results = []
            for sample in tqdm(test_data, desc="生成continuations"):
                try:
                    user_info = get_user_info_from_leaderboard(sample, train_data)
                    context = extract_context_from_leaderboard(sample)
                    
                    messages = build_prompt(
                        user_profile=user_info.get('user_profile'),
                        history=user_info.get('user_train_samples', []) if use_history else [],
                        context=context if use_context else None,
                        use_profile=use_profile,
                        use_history=use_history,
                        use_context=use_context
                    )
                    
                    continuations = generate_continuations(
                        model=model,
                        tokenizer=tokenizer,
                        messages=messages,
                        num_samples=num_samples,
                        max_new_tokens=max_new_tokens,
                        device=target_device,
                        max_output_length=max_output_length
                    )
                    
                    results.append({
                        'id': sample.get('id', ''),
                        'user_hash': sample.get('user_hash', ''),
                        'continuations': continuations
                    })
                except Exception as e:
                    print(f"处理样本 {sample.get('id', 'unknown')} 时出错: {e}")
                    results.append({
                        'id': sample.get('id', ''),
                        'user_hash': sample.get('user_hash', ''),
                        'continuations': []
                    })
            
            # 保存结果
            output_file = os.path.join(output_dir, 'test_leaderboard.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            print(f"\n✓ 推理完成，结果保存到: {output_file}")
            return
    
    # 策略2：数据并行
    if not single_gpu_success:
        print("\n=== 策略2: 使用数据并行 ===")
        print(f"将数据分片到 {len(gpu_list)} 张GPU")
        
        # 使用多进程处理数据分片
        processes = []
        shard_files = []
        
        for i, gpu_id in enumerate(gpu_list):
            p = mp.Process(
                target=process_data_shard,
                args=(
                    i, len(gpu_list), test_data, checkpoint_dir, scenario_path,
                    config_name, use_profile, use_history, use_context,
                    num_samples, gpu_id, output_dir, train_data, all_train_samples,
                    max_new_tokens, max_output_length, base_model_path
                )
            )
            p.start()
            processes.append(p)
            shard_files.append(os.path.join(output_dir, f'shard_{i}.json'))
        
        # 等待所有进程完成
        for p in processes:
            p.join()
        
        # 合并结果
        print("\n合并分片结果...")
        all_results = []
        for shard_file in shard_files:
            if os.path.exists(shard_file):
                with open(shard_file, 'r', encoding='utf-8') as f:
                    shard_results = json.load(f)
                    all_results.extend(shard_results)
        
        # 按原始顺序排序（如果有id的话）
        if all_results and 'id' in all_results[0]:
            all_results.sort(key=lambda x: x.get('id', ''))
        
        # 保存最终结果
        output_file = os.path.join(output_dir, 'test_leaderboard.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ 数据并行推理完成，结果保存到: {output_file}")
        
        # 清理临时分片文件
        for shard_file in shard_files:
            if os.path.exists(shard_file):
                os.remove(shard_file)


def main():
    parser = argparse.ArgumentParser(description='推理脚本：支持单卡优先，否则数据并行')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='模型检查点目录')
    parser.add_argument('--scenario_path', type=str, required=True,
                       help='场景目录路径（包含test_leaderboard.json）')
    parser.add_argument('--config_name', type=str, required=True,
                       help='配置名称（用于输出目录命名）')
    parser.add_argument('--use_profile', action='store_true',
                       help='是否使用profile')
    parser.add_argument('--use_history', action='store_true',
                       help='是否使用history')
    parser.add_argument('--use_context', action='store_true', default=True,
                       help='是否使用context（默认：True）')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='每个样本生成的continuation数量（默认：5）')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录')
    parser.add_argument('--gpu_ids', type=str, default=None,
                       help='要使用的GPU ID列表，用逗号分隔（例如：0,1,2,3 或 4,5,6,7）。如果未指定，使用所有可见的GPU')
    parser.add_argument('--log_file', type=str, default=None,
                       help='详细日志文件路径')
    parser.add_argument('--base_model_path', type=str, default=None,
                       help='基础模型路径（用于加载tokenizer）')
    parser.add_argument('--max_new_tokens', type=int, default=4096,
                       help='最大生成token数（默认：4096）')
    parser.add_argument('--max_output_length', type=int, default=4096,
                       help='输出文本最大字符数（默认：4096）')
    
    args = parser.parse_args()
    
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    
    process_scenario_parallel(
        scenario_path=args.scenario_path,
        checkpoint_dir=args.checkpoint_dir,
        config_name=args.config_name,
        use_profile=args.use_profile,
        use_history=args.use_history,
        use_context=args.use_context,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        gpu_ids=args.gpu_ids,
        log_file_path=args.log_file,
        base_model_path=args.base_model_path,
        max_new_tokens=args.max_new_tokens,
        max_output_length=args.max_output_length
    )


if __name__ == '__main__':
    main()
