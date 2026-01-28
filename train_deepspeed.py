"""
消融实验训练脚本 - DeepSpeed ZeRO-3 多 GPU 版本（使用 Accelerate 库）
使用 accelerate launch 启动，自动处理多 GPU 和 DeepSpeed 配置
"""
import json
import argparse
import os
import sys
import time
from pathlib import Path
import random
import torch

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_train_data, extract_training_samples, get_user_history_samples, get_user_only_history
# from trainer_deepspeed import AblationTrainerDeepSpeed
from trainer_deepspeed_speedup import AblationTrainerDeepSpeed


def split_train_val(samples, val_ratio=0.1, seed=42):
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


def main():
    parser = argparse.ArgumentParser(description='消融实验训练 - DeepSpeed ZeRO-3 多 GPU')
    parser.add_argument('--config', type=str,
                       default='/mnt/parallel/8B_Qwen3/Gemma/config.json',
                       help='配置文件路径')
    parser.add_argument('--ablation_config', type=str, required=True,
                       choices=['profile_and_history_and_context', 'profile_and_history', 'profile_and_context', 
                               'history_and_context', 'profile_only', 'history_only', 'context_only'],
                       help='消融实验配置')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='验证集比例')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='模型输出目录（覆盖配置文件中的设置）')
    parser.add_argument('--log_file', type=str, default=None,
                       help='训练日志文件路径（如果未指定，将自动生成）')
    parser.add_argument('--deepspeed_config', type=str, default=None,
                       help='DeepSpeed 配置文件路径（如果未指定，将自动创建）')
    parser.add_argument('--gpu_ids', type=str, default=None,
                       help='指定要使用的GPU ID（例如：0,1,2,3 或 4,5,6,7）。如果未指定，Accelerate将使用所有可见的GPU')
    
    args = parser.parse_args()
    
    # 如果指定了GPU ID，设置CUDA_VISIBLE_DEVICES
    if args.gpu_ids:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
        print(f"✓ 设置 CUDA_VISIBLE_DEVICES={args.gpu_ids}")
        print(f"  注意：Accelerate将只看到这些GPU，local_rank将基于这些GPU重新编号")
    
    # 解析配置文件路径：如果是相对路径，则相对于脚本所在目录
    if not os.path.isabs(args.config):
        script_dir = Path(__file__).parent
        args.config = str(script_dir / args.config)
    
    # 验证 GPU 是否可用
    if not torch.cuda.is_available():
        print("错误: CUDA 不可用，无法使用 DeepSpeed 多 GPU 训练")
        return
    
    print(f"CUDA 可用，GPU 数量: {torch.cuda.device_count()}")
    print("使用 Accelerate 库自动处理多 GPU 和 DeepSpeed 配置")
    
    # 检查配置文件是否存在
    if not os.path.exists(args.config):
        print(f"错误: 配置文件不存在: {args.config}")
        return
    
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
    print(f"消融实验: {config_name}")
    print(f"使用配置: profile={use_profile}, history={use_history}, context={use_context}")
    print(f"DeepSpeed ZeRO-3 多 GPU 训练（通过 Accelerate）")
    print("=" * 60)
    
    # 加载训练数据
    print("加载训练数据...")
    train_path = config['data']['train_path']
    train_data = load_train_data(train_path)
    
    if not train_data:
        print(f"错误: 无法加载训练数据 from {train_path}")
        return
    
    # 提取训练样本
    all_samples = extract_training_samples(train_data)
    print(f"提取了 {len(all_samples)} 个训练样本")
    
    # 如果需要使用 history，添加历史信息
    if use_history:
        print("添加历史信息...")
        all_samples = add_history_to_samples(all_samples, all_samples)
    
    # 划分训练集和验证集
    train_samples, val_samples = split_train_val(all_samples, args.val_ratio)
    print(f"训练集: {len(train_samples)} 个样本")
    print(f"验证集: {len(val_samples)} 个样本")
    
    # 获取模型配置
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
        train_path = config['data']['train_path']
        dataset_name = os.path.basename(os.path.dirname(train_path))  # 例如: LovinkDialogue 或 RealPersonaChat
        output_dir = os.path.join(checkpoint_dir, f"{dataset_name}_ablation_{config_name}_deepspeed")
        
        # 尝试创建目录，如果失败则使用本地目录
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"输出目录: {output_dir}")
        except (OSError, IOError) as e:
            print(f"警告: 无法在 {checkpoint_dir} 创建目录: {e}")
            # 使用本地目录作为备选
            local_checkpoint_dir = os.path.join(os.path.expanduser("~"), "checkpoints")
            output_dir = os.path.join(local_checkpoint_dir, f"{dataset_name}_ablation_{config_name}_deepspeed")
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
        train_path = config['data']['train_path']
        dataset_name = os.path.basename(os.path.dirname(train_path))
        log_file_path = os.path.join(log_dir, f"training_{dataset_name}_{config_name}_{timestamp}.json")
    
    print(f"训练日志将保存到: {log_file_path}")
    
    # 创建训练器（Accelerate 会自动处理 GPU 分配）
    model_path = model_config['path']
    trainer = AblationTrainerDeepSpeed(
        model_path=model_path,
        output_dir=output_dir,
        config=config,
        use_profile=use_profile,
        use_history=use_history,
        use_context=use_context,
        log_file_path=log_file_path,
        deepspeed_config_path=args.deepspeed_config
    )
    
    # 开始训练
    trainer.train(train_samples, val_samples)
    
    print(f"\n训练完成！模型保存在: {output_dir}")


if __name__ == '__main__':
    # 检查是否通过 accelerate launch 启动
    import sys
    if 'accelerate' not in ' '.join(sys.argv) and 'ACCELERATE_USE_CPU' not in os.environ:
        print("=" * 60)
        print("⚠ 警告: 建议使用 accelerate launch 命令启动训练！")
        print("=" * 60)
        print("推荐的方式（使用 Accelerate 库）:")
        print(f"  accelerate launch --num_processes={{GPU数量}} {__file__} \\")
        print("    --config config.json --ablation_config profile_and_history")
        print("")
        print("或者先配置 Accelerate:")
        print("  1. 运行: accelerate config")
        print("  2. 选择 DeepSpeed ZeRO-3 配置")
        print("  3. 运行: accelerate launch {__file__} --config config.json --ablation_config profile_and_history")
        print("=" * 60)
        print("继续使用当前方式（可能遇到问题）...")
        print("=" * 60)
        time.sleep(3)
    
    main()
