#!/usr/bin/env python3
"""
合并多个数据分片的结果文件
"""
import json
import argparse
import os
from pathlib import Path


def merge_shards(output_dir: str, num_shards: int = 8, output_filename: str = "test_leaderboard.json"):
    """合并多个分片的结果文件"""
    print(f"合并 {num_shards} 个分片的结果...")
    
    all_data = []
    shard_files = []
    
    # 收集所有分片文件
    for shard_id in range(num_shards):
        shard_file = os.path.join(output_dir, f"test_leaderboard_shard_{shard_id}.json")
        if os.path.exists(shard_file):
            shard_files.append((shard_id, shard_file))
            print(f"  找到分片 {shard_id}: {shard_file}")
        else:
            print(f"  警告: 分片 {shard_id} 文件不存在: {shard_file}")
    
    if not shard_files:
        print("错误: 没有找到任何分片文件")
        return
    
    # 按分片ID排序
    shard_files.sort(key=lambda x: x[0])
    
    # 加载并合并所有分片
    for shard_id, shard_file in shard_files:
        print(f"加载分片 {shard_id}...")
        try:
            with open(shard_file, 'r', encoding='utf-8') as f:
                shard_data = json.load(f)
            
            if isinstance(shard_data, list):
                all_data.extend(shard_data)
                print(f"  分片 {shard_id}: {len(shard_data)} 个样本")
            else:
                all_data.append(shard_data)
                print(f"  分片 {shard_id}: 1 个样本")
        except Exception as e:
            print(f"  错误: 加载分片 {shard_id} 失败: {e}")
            continue
    
    print(f"\n总共合并了 {len(all_data)} 个样本")
    
    # 保存合并后的结果
    output_path = os.path.join(output_dir, output_filename)
    print(f"保存合并结果到: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 合并完成！结果保存在: {output_path}")
    print(f"  总样本数: {len(all_data)}")


def main():
    parser = argparse.ArgumentParser(description='合并多个数据分片的结果文件')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='输出目录（包含所有分片文件）')
    parser.add_argument('--num_shards', type=int, default=8,
                       help='分片总数（默认：8）')
    parser.add_argument('--output_filename', type=str, default='test_leaderboard.json',
                       help='输出文件名（默认：test_leaderboard.json）')
    
    args = parser.parse_args()
    
    merge_shards(args.output_dir, args.num_shards, args.output_filename)


if __name__ == '__main__':
    main()
