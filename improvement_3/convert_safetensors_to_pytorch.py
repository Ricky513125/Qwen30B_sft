"""
将 Safetensors 格式的模型转换为 PyTorch 格式
用于解决 "header too large" 错误
"""
import argparse
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def convert_model(input_path: str, output_path: str = None):
    """
    将 Safetensors 格式的模型转换为 PyTorch 格式
    
    Args:
        input_path: 输入模型路径（Safetensors 格式）
        output_path: 输出模型路径（如果为 None，则在输入路径后加 _pytorch）
    """
    if output_path is None:
        output_path = input_path + "_pytorch"
    
    print(f"开始转换模型...")
    print(f"输入路径: {input_path}")
    print(f"输出路径: {output_path}")
    
    # 检查输入路径是否存在
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"模型路径不存在: {input_path}")
    
    # 加载 tokenizer
    print("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(input_path, trust_remote_code=True)
    
    # 加载模型（使用 safetensors）
    print("加载模型（Safetensors 格式）...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            input_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="auto"  # 如果有多 GPU，自动分配
        )
        print("✓ 模型加载成功")
    except Exception as e:
        print(f"加载模型失败: {e}")
        print("尝试使用 CPU 加载...")
        model = AutoModelForCausalLM.from_pretrained(
            input_path,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        print("✓ 模型加载成功（CPU）")
    
    # 保存为 PyTorch 格式（不使用 safetensors）
    print(f"保存模型为 PyTorch 格式到: {output_path}")
    model.save_pretrained(
        output_path,
        safe_serialization=False,  # 关键：不使用 safetensors
        max_shard_size="10GB"  # 分片大小，避免单个文件过大
    )
    
    # 保存 tokenizer
    tokenizer.save_pretrained(output_path)
    
    print(f"✓ 转换完成！")
    print(f"现在可以使用 PyTorch 格式的模型路径: {output_path}")
    print(f"在 inference.py 中使用 --checkpoint_dir {output_path} 或 --base_model_path {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将 Safetensors 格式转换为 PyTorch 格式")
    parser.add_argument("--input_path", type=str, required=True,
                       help="输入模型路径（Safetensors 格式）")
    parser.add_argument("--output_path", type=str, default=None,
                       help="输出模型路径（PyTorch 格式），如果未指定，则在输入路径后加 _pytorch")
    
    args = parser.parse_args()
    convert_model(args.input_path, args.output_path)
