"""
将 Safetensors 格式的模型转换为 PyTorch 格式
用于解决 "header too large" 错误
使用 accelerate 库绕过 safetensors header 限制
"""
import argparse
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch

def convert_model(input_path: str, output_path: str = None, use_accelerate: bool = True):
    """
    将 Safetensors 格式的模型转换为 PyTorch 格式
    
    Args:
        input_path: 输入模型路径（Safetensors 格式）
        output_path: 输出模型路径（如果为 None，则在输入路径后加 _pytorch）
        use_accelerate: 是否使用 accelerate 库绕过 header 问题（推荐）
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
    tokenizer = AutoTokenizer.from_pretrained(input_path, trust_remote_code=True, local_files_only=True)
    print("✓ Tokenizer 加载成功")
    
    # 选择数据类型
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
            dtype_str = "bfloat16"
        else:
            dtype = torch.float16
            dtype_str = "float16"
    else:
        dtype = torch.float32
        dtype_str = "float32"
    
    print(f"使用数据类型: {dtype_str}")
    
    # 加载模型
    model = None
    if use_accelerate:
        print("使用 accelerate 库加载模型（绕过 safetensors header 限制）...")
        try:
            from accelerate import load_checkpoint_and_dispatch, init_empty_weights
            
            # 加载配置
            config = AutoConfig.from_pretrained(
                input_path,
                trust_remote_code=True,
                local_files_only=True
            )
            
            # 创建空模型
            print("创建空模型结构...")
            with init_empty_weights():
                model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
            
            # 使用 accelerate 分片加载权重
            print("使用 accelerate 分片加载权重...")
            if torch.cuda.is_available():
                num_gpus = torch.cuda.device_count()
                max_memory = {i: "20GiB" for i in range(num_gpus)}
                print(f"检测到 {num_gpus} 个 GPU，将使用多 GPU 加载")
                model = load_checkpoint_and_dispatch(
                    model,
                    input_path,
                    device_map="auto",
                    max_memory=max_memory,
                    dtype=dtype,
                    no_split_module_classes=[]
                )
            else:
                model = load_checkpoint_and_dispatch(
                    model,
                    input_path,
                    device_map="cpu",
                    dtype=dtype,
                    no_split_module_classes=[]
                )
            print("✓ 使用 accelerate 库加载成功")
        except ImportError:
            print("警告: accelerate 库未安装，回退到标准加载方式...")
            use_accelerate = False
        except Exception as e:
            print(f"使用 accelerate 加载失败: {e}")
            print("回退到标准加载方式...")
            use_accelerate = False
    
    if not use_accelerate or model is None:
        print("使用标准方式加载模型...")
        try:
            # 尝试使用 device_map="auto" 和 low_cpu_mem_usage
            if torch.cuda.is_available():
                model = AutoModelForCausalLM.from_pretrained(
                    input_path,
                    torch_dtype=dtype,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    device_map="auto",
                    local_files_only=True
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    input_path,
                    torch_dtype=dtype,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    local_files_only=True
                )
            print("✓ 模型加载成功")
        except Exception as e:
            if "header too large" in str(e) or "SafetensorError" in str(e):
                print(f"错误: safetensors header 太大: {e}")
                print("\n建议:")
                print("1. 确保已安装 accelerate 库: pip install accelerate")
                print("2. 如果已安装，请检查 accelerate 版本: pip install --upgrade accelerate")
                print("3. 或者尝试使用更少的 GPU（如果使用多 GPU）")
                raise RuntimeError("无法加载模型，safetensors header 太大。请使用 accelerate 库或联系模型提供者获取 PyTorch 格式。")
            else:
                raise
    
    # 保存为 PyTorch 格式（不使用 safetensors）
    print(f"\n保存模型为 PyTorch 格式到: {output_path}")
    print("这可能需要一些时间，请耐心等待...")
    
    # 如果模型在多个设备上，先合并到 CPU
    if hasattr(model, 'hf_device_map') and model.hf_device_map:
        print("检测到模型分布在多个设备上，正在合并到 CPU...")
        # 将模型移到 CPU 以便保存
        if torch.cuda.is_available():
            model = model.cpu()
            torch.cuda.empty_cache()
    
    model.save_pretrained(
        output_path,
        safe_serialization=False,  # 关键：不使用 safetensors
        max_shard_size="10GB"  # 分片大小，避免单个文件过大
    )
    
    # 保存 tokenizer
    print("保存 tokenizer...")
    tokenizer.save_pretrained(output_path)
    
    print(f"\n✓ 转换完成！")
    print(f"现在可以使用 PyTorch 格式的模型路径: {output_path}")
    print(f"在 inference.py 中使用 --base_model_path {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将 Safetensors 格式转换为 PyTorch 格式")
    parser.add_argument("--input_path", type=str, required=True,
                       help="输入模型路径（Safetensors 格式）")
    parser.add_argument("--output_path", type=str, default=None,
                       help="输出模型路径（PyTorch 格式），如果未指定，则在输入路径后加 _pytorch")
    parser.add_argument("--use_accelerate", action="store_true", default=True,
                       help="使用 accelerate 库绕过 safetensors header 限制（默认：True）")
    parser.add_argument("--no_accelerate", action="store_false", dest="use_accelerate",
                       help="不使用 accelerate 库（使用标准加载方式）")
    
    args = parser.parse_args()
    convert_model(args.input_path, args.output_path, use_accelerate=args.use_accelerate)
