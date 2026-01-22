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
from data_loader import load_train_data, extract_training_samples, get_user_history_samples
from prompt_builder import build_prompt
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
    
    # 设置信号处理器（仅Linux/Unix）
    if hasattr(signal, 'SIGALRM'):
        old_handler = signal.signal(signal.SIGALRM, timeout_signal)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        # Windows不支持SIGALRM，使用简单的try-except
        # 注意：Windows上无法强制中断，只能依赖模型自身的超时
        yield


def load_test_leaderboard(test_leaderboard_path: str) -> list:
    """加载 test_leaderboard.json"""
    with open(test_leaderboard_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def extract_context_from_leaderboard(sample: dict) -> list:
    """从 leaderboard 样本中提取 context"""
    return sample.get('context', [])


def get_user_info_from_leaderboard(sample: dict, train_data: list) -> dict:
    """从训练数据中获取用户信息（profile, history等）"""
    # 从测试样本中获取user_hash
    user_hash = sample.get('user_hash', '')
    if not user_hash:
        # 如果顶层没有，尝试从user字段获取
        user = sample.get('user', {})
        # user_hash可能在user字段中，或者需要从其他字段推断
    
    # 从测试样本中获取user_profile（IdealSelf场景）
    user_profile = None
    user = sample.get('user', {})
    if user:
        user_profile = user.get('profile')
    
    # 找到该用户的训练数据
    user_train_samples = []
    if train_data:
        for item in train_data:
            if item.get('user_hash') == user_hash:
                user_train_samples.append(item)
    
    # 如果测试样本中没有profile，从训练数据中获取
    if not user_profile and user_train_samples:
        first_sample = user_train_samples[0]
        train_user = first_sample.get('user', {})
        user_profile = train_user.get('profile')
    
    return {
        'user_hash': user_hash,
        'user_profile': user_profile,
        'user_train_samples': user_train_samples
    }

def generate_continuations(
    model,
    tokenizer,
    messages,
    num_samples=5,
    max_new_tokens=512,  # 增加到512，给模型足够空间写完"废话"，后续通过清洗函数截断
    device=None,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.2,  # 增加重复惩罚
    no_repeat_ngram_size=4,  # 增加n-gram大小
    max_output_length=512,  # 输出文本最大字符数
    is_japanese_task=False,  # 是否为日语任务
    max_input_length=4096,  # 输入最大长度，默认2048（可以根据模型调整）
    use_batch_generation=True,  # 是否使用批处理生成（一次生成多个）
    vllm_engine=None,  # vLLM引擎（如果使用vLLM）
):
    # 如果使用vLLM，使用不同的生成方式
    if vllm_engine is not None:
        # vLLM模式：直接使用prompt生成
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        generation_suffix = "<|im_start|>user\n"
        text = text.strip() + generation_suffix
        
        # 如果是日语任务，添加引导词
        if is_japanese_task:
            import random
            japanese_prompts = ["回答：", "返答：", "応答："]
            text = text.rstrip() + "\n" + random.choice(japanese_prompts)
        
        # 使用vLLM生成
        sampling_params = SamplingParams(
            temperature=temperature if do_sample else 0.0,
            top_p=top_p if do_sample else 1.0,
            max_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
        )
        
        # 生成num_samples个结果
        prompts = [text] * num_samples
        outputs = vllm_engine.generate(prompts, sampling_params)
        
        # 提取生成的文本
        results = []
        for output in outputs:
            generated_text = output.outputs[0].text
            cleaned_text = clean_model_output(generated_text, max_length=max_output_length)
            results.append(cleaned_text if cleaned_text else "")
        
        return results if results else [""]
    
    # 标准transformers模式
    # 与训练时保持一致：使用 add_generation_prompt=False，然后手动添加引导符
    # 训练时的逻辑：full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    #                generation_suffix = "<|im_start|>user\n"
    #                full_prompt = full_prompt.strip() + generation_suffix
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False  # 与训练时保持一致
    )
    
    # 手动添加引导符（与训练时完全一致）
    generation_suffix = "<|im_start|>user\n"
    text = text.strip() + generation_suffix
    
    # 如果是日语任务，在末尾添加日语引导词，迫使模型进入日语语境
    if is_japanese_task:
        # 添加日语引导词，帮助模型直接进入日语回答模式
        japanese_prompts = [
            "回答：",
            "返答：",
            "応答：",
        ]
        import random
        text = text.rstrip() + "\n" + random.choice(japanese_prompts)

    # 对于多 GPU 模型或 DeepSpeed 模型，如果 device 为 None，让模型自动处理设备分配
    # 否则将输入放到指定设备
    
    # 获取模型的实际最大长度（从config中）
    model_max_length = None
    if hasattr(model, 'config'):
        # 尝试从config中获取最大长度
        if hasattr(model.config, 'max_position_embeddings'):
            model_max_length = model.config.max_position_embeddings
        elif hasattr(model.config, 'max_seq_len'):
            model_max_length = model.config.max_seq_len
        elif hasattr(model.config, 'model_max_length'):
            model_max_length = model.config.model_max_length
    
    # 如果没有找到，使用默认值或传入的参数
    if model_max_length is None:
        model_max_length = max_input_length * 2  # 默认使用输入长度的2倍，给生成留空间
    
    # 计算实际可用的输入长度（留出空间给生成）
    # 至少保留512 tokens用于生成
    available_input_length = max(model_max_length - max_new_tokens - 100, max_input_length)
    
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=available_input_length  # 使用动态计算的长度
    )
    # 如果指定了 device，将输入移到该设备
    # 对于使用 device_map 或 DeepSpeed 的模型，可以不移动，让模型自动处理
    # 但为了兼容性，如果指定了 device，仍然移动输入
    if device is not None:
        inputs = {k: v.to(device) for k, v in inputs.items()}
    else:
        # 如果没有指定 device，尝试从模型获取第一个参数所在的设备
        try:
            model_device = next(model.parameters()).device
            inputs = {k: v.to(model_device) for k, v in inputs.items()}
        except (StopIteration, AttributeError):
            # 如果无法获取设备，使用默认设备
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # 确保 input_ids 在有效范围内
    model_vocab_size = model.config.vocab_size
    if inputs["input_ids"].max().item() >= model_vocab_size:
        print(f"  警告: 检测到 token ID 超出模型词汇表范围，进行裁剪...")
        inputs["input_ids"] = torch.clamp(inputs["input_ids"], 0, model_vocab_size - 1)

    # 设置生成参数
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "repetition_penalty": repetition_penalty,  # 防止重复，推荐 1.1-1.2
        "no_repeat_ngram_size": no_repeat_ngram_size,  # 防止 n-gram 重复，推荐 3-4
    }
    
    if do_sample:
        generation_kwargs["temperature"] = temperature  # 0.7-0.8 以增强逻辑稳定性
        generation_kwargs["top_p"] = top_p
        generation_kwargs["top_k"] = 50  # 限制候选 token 数量，提高质量
    
    # 如果词汇表不匹配，使用 LogitsProcessor 限制词汇表范围并清理无效值
    if hasattr(model.config, 'vocab_size') and tokenizer.vocab_size < model.config.vocab_size:
        from transformers import LogitsProcessor
        class VocabSizeLimiter(LogitsProcessor):
            def __init__(self, max_vocab_size):
                self.max_vocab_size = max_vocab_size
            
            def __call__(self, input_ids, scores):
                # 将超出范围的 logits 设为负无穷
                scores[:, self.max_vocab_size:] = float('-inf')
                
                # 清理无效值：将 inf 和 nan 替换为负无穷
                scores = torch.where(
                    torch.isnan(scores) | torch.isinf(scores),
                    torch.tensor(float('-inf'), device=scores.device, dtype=scores.dtype),
                    scores
                )
                
                # 确保没有正无穷（会导致 softmax 后概率为 nan）
                scores = torch.clamp(scores, min=-1e10, max=1e10)
                
                return scores
        
        from transformers import LogitsProcessorList
        logits_processor = LogitsProcessorList([VocabSizeLimiter(tokenizer.vocab_size)])
        generation_kwargs["logits_processor"] = logits_processor
    else:
        # 即使词汇表匹配，也添加清理无效值的处理器
        from transformers import LogitsProcessor
        class LogitsCleaner(LogitsProcessor):
            def __call__(self, input_ids, scores):
                # 清理无效值：将 inf 和 nan 替换为负无穷
                scores = torch.where(
                    torch.isnan(scores) | torch.isinf(scores),
                    torch.tensor(float('-inf'), device=scores.device, dtype=scores.dtype),
                    scores
                )
                # 限制 logits 范围，避免数值溢出
                scores = torch.clamp(scores, min=-1e10, max=1e10)
                return scores
        
        # 添加重复惩罚处理器（额外保护）
        class RepetitionPenaltyProcessor(LogitsProcessor):
            """额外的重复惩罚处理器，防止生成重复内容"""
            def __init__(self, penalty=1.1):
                self.penalty = penalty
            
            def __call__(self, input_ids, scores):
                # 对最近生成的 token 应用额外的惩罚
                if len(input_ids[0]) > 0:
                    # 获取最近 10 个 token
                    recent_tokens = input_ids[0][-10:].tolist()
                    for token_id in recent_tokens:
                        if token_id < scores.shape[-1]:
                            scores[0, token_id] = scores[0, token_id] / self.penalty
                return scores
        
        from transformers import LogitsProcessorList
        logits_processor = LogitsProcessorList([
            LogitsCleaner(),
            RepetitionPenaltyProcessor(penalty=1.1)  # 额外的重复惩罚
        ])
        generation_kwargs["logits_processor"] = logits_processor

    try:
        # 注意：transformers的generate方法不支持stop_strings参数
        # 停止逻辑通过eos_token_id和max_new_tokens来控制
        # 如果需要更复杂的停止词逻辑，可以使用StoppingCriteria，但这里暂时不需要
        
        # 添加生成超时保护（通过max_new_tokens限制，避免无限生成）
        # 如果输入太长，减少max_new_tokens
        input_length = inputs["input_ids"].shape[1]
        
        # 获取模型的实际最大长度
        model_max_length = None
        if hasattr(model, 'config'):
            if hasattr(model.config, 'max_position_embeddings'):
                model_max_length = model.config.max_position_embeddings
            elif hasattr(model.config, 'max_seq_len'):
                model_max_length = model.config.max_seq_len
            elif hasattr(model.config, 'model_max_length'):
                model_max_length = model.config.model_max_length
        
        # 如果没有找到，使用保守的默认值（32768是Qwen3-30B的常见值）
        if model_max_length is None:
            model_max_length = 4096
        
        # 动态调整max_new_tokens，确保总长度不超过模型限制
        # 留出100 tokens的缓冲空间
        max_total_length = model_max_length - 100
        if input_length + max_new_tokens > max_total_length:
            adjusted_max_new_tokens = max_total_length - input_length
            if adjusted_max_new_tokens < 50:
                adjusted_max_new_tokens = 50  # 至少生成50个tokens
            generation_kwargs["max_new_tokens"] = adjusted_max_new_tokens
            # 减少不必要的打印（只在调试时打印）
            # if adjusted_max_new_tokens != max_new_tokens:
            #     print(f"    提示: 输入长度 {input_length}, 模型最大长度 {model_max_length}, 调整 max_new_tokens 为 {adjusted_max_new_tokens}")
        
        # 如果使用批处理且需要生成多个样本，使用num_return_sequences
        if use_batch_generation and num_samples > 1:
            # 使用num_return_sequences一次性生成多个序列（更高效）
            generation_kwargs["num_return_sequences"] = num_samples
            outputs = model.generate(
                **inputs,
                **generation_kwargs,
            )
            # 处理多个输出
            all_gens = []
            input_length = inputs["input_ids"].shape[1]
            # outputs的形状是 [num_return_sequences, seq_len]
            if outputs.dim() == 2:
                # 单个输入，多个输出
                for i in range(num_samples):
                    gen = outputs[i][input_length:]
                    all_gens.append(gen)
            else:
                # 多个输入，多个输出
                for i in range(outputs.shape[0]):
                    gen = outputs[i][input_length:]
                    all_gens.append(gen)
        else:
            # 单次生成
            outputs = model.generate(
                **inputs,
                **generation_kwargs,
            )
            gen = outputs[0][inputs["input_ids"].shape[1]:]
            all_gens = [gen]
        
        # 处理所有生成的序列
        tokenizer_vocab_size = tokenizer.vocab_size
        input_length = inputs["input_ids"].shape[1]
        results = []
        
        for gen in all_gens:
            # 确保 gen 是 tensor 并且移到 CPU
            if isinstance(gen, torch.Tensor):
                gen = gen.cpu()
            else:
                gen = torch.tensor(gen, dtype=torch.long)
            
            # 清理无效的 token IDs（超出词汇表范围、NaN、Inf等）
            valid_mask = (gen >= 0) & (gen < tokenizer_vocab_size) & torch.isfinite(gen.float())
            
            if valid_mask.sum() == 0:
                results.append("")
                continue
            
            # 只保留有效的 token IDs
            cleaned_gen = gen[valid_mask]
            
            # 如果清理后没有有效 token，返回空字符串
            if len(cleaned_gen) == 0:
                results.append("")
                continue
            
            # 解码
            try:
                decoded_text = tokenizer.decode(cleaned_gen, skip_special_tokens=True).strip()
        
                if not decoded_text:
                    results.append("")
                    continue
                
                # --- 调用清洗函数 ---
                cleaned_text = clean_model_output(decoded_text, max_length=max_output_length)
                
                # 如果清理后为空或过短，返回原始文本（至少保留一些内容）
                if not cleaned_text or len(cleaned_text.strip()) < 3:
                    if decoded_text and len(decoded_text.strip()) > 0:
                        # 只做最基本的清理：移除明显的垃圾token
                        fallback = decoded_text
                        for pattern in [r'\bVRTX\b', r'\bVERTEX\b', r'<Vertex>', r'\(Vertex\)', r'_VERTEX_',
                                       r'\bPorno\b', r'\bporno\b', r'Viagra\s+Porno']:
                            fallback = re.sub(pattern, '', fallback, flags=re.IGNORECASE)
                        cleaned_text = fallback[:max_output_length].strip() if fallback.strip() else ""
                    else:
                        cleaned_text = ""
                
                results.append(cleaned_text if cleaned_text else "")
            except Exception as decode_error:
                # 如果解码失败，尝试逐个 token 解码
                try:
                    decoded_parts = []
                    for token_id in cleaned_gen.tolist():
                        try:
                            token_text = tokenizer.decode([token_id], skip_special_tokens=True)
                            if token_text:
                                decoded_parts.append(token_text)
                        except:
                            continue
                    decoded_text = "".join(decoded_parts).strip()
                    results.append(decoded_text if decoded_text else "")
                except:
                    results.append("")
        
        return results if results else [""]
    
    except Exception as e:
        # 如果解码失败，返回空字符串而不是抛出异常
        error_msg = str(e)
        if "NoneType" in error_msg or "expected str instance" in error_msg:
            print(f"  警告: 解码时遇到 None 值，返回空字符串")
            return [""]
        else:
            # 其他错误，重新抛出
            raise



def log_inference_details(
    log_file,
    sample_idx: int,
    data_item_idx: int,
    context: list,
    messages: list,
    user_info: dict,
    history_evidence: list,
    continuations: list,
    reference: str = None,
    error: str = None,
    is_first_entry: bool = False
):
    """记录推理详情到日志文件"""
    try:
        # 构建日志条目
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "sample_idx": sample_idx,
            "data_item_idx": data_item_idx,
            "context": context,
            "input_prompt": {
                "messages": messages,
                "message_count": len(messages)
            },
            "user_info": {
                "user_hash": user_info.get('user_hash', ''),
                "has_profile": user_info.get('user_profile') is not None,
                "profile": user_info.get('user_profile'),
                "history_count": len(user_info.get('user_train_samples', []))
            },
            "history_evidence_count": len(history_evidence),
            "history_evidence": [
                {
                    "context": h.get('context', [])[:3] if h.get('context') else [],  # 只记录前3个turn
                    "continuation": h.get('continuation', '')[:100] if h.get('continuation') else ''  # 只记录前100字符
                }
                for h in history_evidence[:3]  # 最多3个历史样本
            ],
            "outputs": {
                "continuations": continuations,
                "count": len(continuations)
            },
            "target": reference if reference else None,
            "error": error if error else None
        }
        
        # 写入日志文件（追加模式）
        if not is_first_entry:
            log_file.write(",\n")  # 如果不是第一条，先写逗号
        log_file.write(json.dumps(log_entry, ensure_ascii=False, indent=2))
        log_file.flush()  # 立即刷新到磁盘
        
    except Exception as e:
        print(f"  警告: 记录日志时出错: {e}")


import re

def check_model_files(model_path: str) -> dict:
    """
    检查模型目录中的文件类型
    
    Returns:
        dict with keys: 'has_safetensors', 'has_pytorch', 'safetensors_files', 'pytorch_files'
    """
    result = {
        'has_safetensors': False,
        'has_pytorch': False,
        'safetensors_files': [],
        'pytorch_files': []
    }
    
    if not os.path.exists(model_path):
        return result
    
    for file in os.listdir(model_path):
        if file.endswith('.safetensors'):
            result['has_safetensors'] = True
            result['safetensors_files'].append(file)
        elif file.endswith('.bin') and 'pytorch_model' in file:
            result['has_pytorch'] = True
            result['pytorch_files'].append(file)
    
    return result




def clean_model_output(text: str, max_length: int = 512) -> str:
    """
    精简优化版：通过“强力截断”逻辑，只保留第一段核心对话。
    适用于解决：模型复读、带括号的心理描述、中文推理污染日语回答等问题。
    """
    if not text:
        return ""

    # 1. 结构化清理：移除思考过程和所有 Chat 模板标签
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # 移除 <|im_start|>, <|im_end|>, <|user|>, <|assistant|> 等及其后的换行
    text = re.sub(r'<\|im_start\|>.*?\n|<\|im_end\|>|<\|user\|>|<\|assistant\|>', '', text)
    text = text.replace('<think>', '').replace('</think>', '').replace('ROKE', '')

    # 2. 定义“停止点”标识符 (Stop Markers)
    # 一旦在换行后匹配到这些模式，说明正文结束，开始输出元数据或垃圾内容
    stop_markers = [
        r'\n（', r'\n\(',      # 换行后的括号（角色/心理说明）
        r'\n[*]{2,}',          # 换行后的连续星号
        r'\n※',                # 换行后的特殊符号
        r'\n問題[：:]',         # 训练题目复读
        r'\n最终生成',          # 中文推理标识
        r'\n分析',              # 中文分析
        r'\n建议',              # 中文建议
        r'\n\s*---',           # 分隔线
        r'\n[A-Z]\)',          # 选择题 A) B) 模式
        r'RealPersonaChat',    # 场景名泄露
    ]
    
    # 执行强力截断：只保留第一个停止位之前的内容
    combined_stop_pattern = '|'.join(stop_markers)
    match = re.search(combined_stop_pattern, text)
    if match:
        text = text[:match.start()]

    # 3. 提取“第一段”有效内容
    # 过滤掉空白行，只取第一行或第一段有意义的文字
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    if not paragraphs:
        return ""
    
    # 拿到第一段话作为核心回复
    result = paragraphs[0]

    # 4. 行内清洗：处理 PersonaChat 特有的行内括号污染
    # 比如移除末尾的 "(笑顔で)" 或 "（嬉しい）"
    result = re.sub(r'[（\(][^）\)]*?(感情|态度|注|角色|姿态|笑顔|喜)[^）\)]*?[）\)]', '', result)
    
    # 5. 语言污染处理 (针对“中文推理 + 日语回答”的情况)
    # 如果第一段话里有“回答：”或“答：”，只取冒号后的内容
    for marker in ['回答：', '返答：', '応答：', '答：', '最终生成：']:
        if marker in result:
            result = result.split(marker)[-1]

    # 6. 极致标点清理 (移除末尾多余的符号)
    result = result.strip().rstrip('※-* \n')
    
    # 7. 兜底逻辑：如果清理后太短而原始文本很长，说明可能误删了，保留原始文本前一段
    if len(result) < 2 and len(text) > 5:
        return text.strip()[:max_length]

    return result[:max_length].strip()


def process_scenario(
    scenario_path: str,
    checkpoint_dir: str,
    config_name: str,
    use_profile: bool,
    use_history: bool,
    use_context: bool = True,
    num_samples: int = 5,
    output_dir: str = None,
    gpu_id: int = 2,
    log_file_path: str = None,
    base_model_path: str = None,
    max_new_tokens: int = 512,  # 默认增加到512
    max_output_length: int = 200,
    use_multi_gpu: bool = False,  # 是否使用多 GPU 分布式加载
    use_deepspeed: bool = False,  # 是否使用 DeepSpeed ZeRO-3
    batch_size: int = 1,  # 批处理大小
    disable_batch_generation: bool = False,  # 是否禁用批处理生成
    use_vllm: bool = False,  # 是否使用vLLM
    vllm_tensor_parallel_size: int = 1,  # vLLM tensor并行大小
    vllm_max_model_len: int = None,  # vLLM最大模型长度
    data_shard_id: int = None,  # 数据分片ID
    num_shards: int = 8  # 数据分片总数
):
    """
    处理单个场景：生成test_leaderboard.json
    
    Args:
        scenario_path: 场景目录路径
        checkpoint_dir: 模型检查点目录
        config_name: 配置名称（用于输出目录命名）
        use_profile: 是否使用profile
        use_history: 是否使用history
        use_context: 是否使用context
        num_samples: 每个样本生成的continuation数量
        output_dir: 输出目录（如果为None，则在scenario_path下创建）
    """
    # 初始化 model_files 默认值
    model_files = {
        "has_pytorch": False,
        "has_safetensors": False,
        "pytorch_files": [],
        "safetensors_files": []
    }
    
    print(f"\n处理场景: {scenario_path}")
    print(f"模型: {checkpoint_dir}")
    print(f"配置: profile={use_profile}, history={use_history}, context={use_context}")
    if base_model_path is None:
        base_model_path = "/mnt/parallel/models/Qwen3-30B-A3B-Instruct-2507"
    # 加载测试数据
    test_leaderboard_path = os.path.join(scenario_path, 'test_leaderboard.json')
    if not os.path.exists(test_leaderboard_path):
        print(f"警告: {test_leaderboard_path} 不存在，跳过")
        return
    
    test_data = load_test_leaderboard(test_leaderboard_path)
    print(f"测试样本数: {len(test_data)}")
    
    # 数据分片（如果指定了data_shard_id）
    if data_shard_id is not None:
        num_shards = num_shards if 'num_shards' in locals() else 8
        shard_size = len(test_data) // num_shards
        start_idx = data_shard_id * shard_size
        if data_shard_id == num_shards - 1:
            # 最后一个分片包含剩余的所有数据
            end_idx = len(test_data)
        else:
            end_idx = start_idx + shard_size
        test_data = test_data[start_idx:end_idx]
        print(f"数据分片 {data_shard_id}/{num_shards}: 处理样本 {start_idx} 到 {end_idx} (共 {len(test_data)} 个样本)")
    
    # 加载训练数据（用于获取用户信息和历史）
    train_path = os.path.join(scenario_path, 'train.json')
    train_data = load_train_data(train_path) if os.path.exists(train_path) else []
    all_train_samples = extract_training_samples(train_data) if train_data else []
    print(f"训练样本数: {len(all_train_samples)}")
    
    # 如果使用vLLM，使用不同的加载方式
    vllm_engine = None
    if use_vllm and VLLM_AVAILABLE:
        print("使用 vLLM 加速推理...")
        # 注意：CUDA_VISIBLE_DEVICES应该在shell脚本中设置（通过CUDA_VISIBLE_DEVICES=${GPU_ID}）
        # 这里不再重复设置，避免覆盖shell中的设置
        # 如果shell中没有设置，才在这里设置
        if 'CUDA_VISIBLE_DEVICES' not in os.environ:
            if gpu_id is not None:
                os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
                print(f"在Python中设置 CUDA_VISIBLE_DEVICES={gpu_id}")
            else:
                print("警告: 未设置CUDA_VISIBLE_DEVICES，将使用所有可见GPU")
        else:
            print(f"使用环境变量 CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
        
        # 验证可见GPU（在导入torch后）
        import torch
        if torch.cuda.is_available():
            visible_gpus = torch.cuda.device_count()
            print(f"可见GPU数量: {visible_gpus}")
            if visible_gpus > 0:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                free_memory = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**3
                print(f"GPU 0: {gpu_name}, 总内存: {gpu_memory:.2f} GB, 可用内存: {free_memory:.2f} GB")
                
                # 检查是否有足够内存
                if free_memory < 20:
                    print(f"警告: GPU可用内存不足 ({free_memory:.2f} GB)，可能需要清理其他进程")
        
        # 加载vLLM引擎
        vllm_kwargs = {
            "model": base_model_path,
            "tensor_parallel_size": vllm_tensor_parallel_size,
            "trust_remote_code": True,
            "dtype": "bfloat16" if torch.cuda.is_bf16_supported() else "float16",
            "gpu_memory_utilization": 0.75,  # 限制GPU内存使用为75%，留出更多缓冲（Qwen3-30B MoE需要更多内存）
            "max_model_len": vllm_max_model_len if vllm_max_model_len else 1024,  # 默认1024，减少内存占用
        }
        
        # 对于Qwen3-30B MoE模型，可能需要更保守的设置
        print(f"vLLM配置: gpu_memory_utilization={vllm_kwargs['gpu_memory_utilization']}, max_model_len={vllm_kwargs['max_model_len']}")
        
        print(f"初始化 vLLM 引擎...")
        try:
            vllm_engine = LLM(**vllm_kwargs)
            print("✓ vLLM 引擎加载成功")
        except RuntimeError as e:
            error_msg = str(e)
            if "out of memory" in error_msg.lower() or "CUDA" in error_msg:
                print(f"vLLM初始化失败（内存不足），尝试降低内存使用...")
                print(f"错误详情: {error_msg[:500]}")
                
                # 尝试更保守的设置
                vllm_kwargs["gpu_memory_utilization"] = 0.60  # 进一步降低到60%
                vllm_kwargs["max_model_len"] = min(vllm_kwargs.get("max_model_len", 1024), 768)  # 进一步减少
                print(f"重试配置: gpu_memory_utilization=0.60, max_model_len={vllm_kwargs['max_model_len']}")
                try:
                    vllm_engine = LLM(**vllm_kwargs)
                    print("✓ vLLM 引擎加载成功（使用保守配置）")
                except RuntimeError as e2:
                    print(f"错误: vLLM初始化仍然失败: {str(e2)[:500]}")
                    print("\n建议解决方案:")
                    print("  1. 检查是否有其他进程占用GPU: nvidia-smi")
                    print("  2. 减少max_model_len到1024或更小")
                    print("  3. 使用量化: 添加 --quantization awq 参数（如果模型支持）")
                    print("  4. 减少gpu_memory_utilization到0.60或更低")
                    print("  5. 确保每个进程只使用一张GPU，检查CUDA_VISIBLE_DEVICES设置")
                    raise RuntimeError(f"vLLM初始化失败，请检查GPU内存和进程隔离: {str(e2)[:200]}")
            else:
                raise
        
        # vLLM模式下，不需要加载tokenizer（vLLM内部会处理）
        # 但为了兼容性，仍然加载tokenizer用于prompt构建
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = None  # vLLM模式下不使用transformers模型
        print("vLLM 模式：跳过 transformers 模型加载")
    elif use_vllm and not VLLM_AVAILABLE:
        print("警告: vLLM 未安装，回退到标准 transformers 模式")
        vllm_engine = None
    
    # 加载模型（如果不是vLLM模式）
    if vllm_engine is None:
        print("加载模型...")
        
        # 检查模型文件类型
        print(f"检查模型文件: {base_model_path}")
        try:
            model_files = check_model_files(base_model_path)
            print(f"  发现 safetensors 文件: {len(model_files['safetensors_files'])} 个")
            print(f"  发现 PyTorch 文件: {len(model_files['pytorch_files'])} 个")
        except Exception as e:
            logger.warning(f"device_map 加载失败: {e}")
            print(f"  警告: 检查模型文件失败，使用默认值: {e}")
        
        # 尝试加载 tokenizer（与训练脚本保持一致）
        tokenizer_json_path = os.path.join(checkpoint_dir, 'tokenizer.json')

        print("加载 tokenizer（必须与模型 checkpoint 完全一致）")

        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            # checkpoint_dir,
            use_fast=False,          # 强烈建议
            trust_remote_code=True,
            local_files_only=True
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print("✓ Tokenizer 加载成功")
        print(f"  vocab_size: {tokenizer.vocab_size}")
        print(f"  pad_token_id: {tokenizer.pad_token_id}")
        print(f"  eos_token_id: {tokenizer.eos_token_id}")

    
    # 验证 tokenizer 配置（仅非vLLM模式）
    if vllm_engine is None:
        vocab_size = len(tokenizer)
        print(f"  Tokenizer 词汇表大小: {vocab_size}")
        print(f"  pad_token_id: {tokenizer.pad_token_id}")
        print(f"  eos_token_id: {tokenizer.eos_token_id}")
        print(f"  bos_token_id: {tokenizer.bos_token_id}")
        
        # 加载模型
        print("  加载模型权重...")
        # 根据是否使用多 GPU 或 DeepSpeed 来决定设备设置
        # DeepSpeed 会自动使用所有 GPU，所以也应该被视为多 GPU 模式
        # 检查 DeepSpeed 是否可用（在函数内部检查，确保正确）
        deepspeed_available = False
        if use_deepspeed:
            try:
                import deepspeed
                deepspeed_available = True
            except ImportError:
                deepspeed_available = False
                print("  警告: DeepSpeed 未安装，将回退到普通加载方式")
        
        if torch.cuda.is_available():
            if use_multi_gpu or (use_deepspeed and deepspeed_available):
                # 使用所有可用的 GPU
                num_gpus = torch.cuda.device_count()
                if use_deepspeed and deepspeed_available:
                    print(f"  使用 DeepSpeed 多 GPU 模式，可用 GPU 数量: {num_gpus}")
                else:
                    print(f"  使用多 GPU 模式，可用 GPU 数量: {num_gpus}")
                print(f"  将使用 GPU: {list(range(num_gpus))}")
                target_device = None  # device_map="auto" 或 DeepSpeed 会自动分配
            else:
            # 使用指定的单个 GPU
            target_device = torch.device(f'cuda:{gpu_id}')
            print(f"  使用单 GPU 模式，目标设备: {target_device}")
        else:
            target_device = torch.device('cpu')
            print(f"  目标设备: {target_device}")
        
        # 检查是否是 LoRA 模型（检查是否存在 adapter_config.json）
        adapter_config_path = os.path.join(checkpoint_dir, 'adapter_config.json')
        is_lora_model = os.path.exists(adapter_config_path)
        
        if is_lora_model and PEFT_AVAILABLE:
        print("  检测到 LoRA 模型，使用 PeftModel 加载...")
        # 首先需要加载基础模型，然后加载 LoRA 适配器
        # 从 adapter_config.json 读取基础模型路径
        try:
            with open(adapter_config_path, 'r', encoding='utf-8') as f:
                adapter_config = json.load(f)
            base_model_name_or_path = adapter_config.get('base_model_name_or_path', None)
            
            # 如果配置中没有基础模型路径，尝试从参数或常见路径推断
            if base_model_name_or_path is None or not os.path.exists(base_model_name_or_path):
                # 优先使用传入的参数
                if base_model_path and os.path.exists(base_model_path):
                    base_model_name_or_path = base_model_path
                else:
                    # 尝试常见的模型路径
                    possible_paths = [
                        '/mnt/parallel/models/Qwen3-30B-A3B-Instruct-2507',
                        '/mnt/parallel/models/Qwen3-8B',
                        '/mnt/parallel/models/Qwen3-4B',
                    ]
                    for path in possible_paths:
                        if os.path.exists(path):
                            base_model_name_or_path = path
                            break
                
                if base_model_name_or_path is None or not os.path.exists(base_model_name_or_path):
                    raise RuntimeError(f"无法找到 LoRA 模型的基础模型路径。请使用 --base_model_path 参数指定。")
            
            print(f"  基础模型路径: {base_model_name_or_path}")
            
            # 加载基础模型
            # 选择数据类型：优先使用 bfloat16（如果 GPU 支持），否则使用 float16
            if torch.cuda.is_available():
                if torch.cuda.is_bf16_supported():
                    lora_dtype = torch.bfloat16
                else:
                    lora_dtype = torch.float16
            else:
                lora_dtype = torch.float32
            
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name_or_path,
                torch_dtype=lora_dtype,
                device_map=None,
                trust_remote_code=True,
                local_files_only=True
            )
            
            # 加载 LoRA 适配器
            # 注意：需要指定 local_files_only=True 以避免将本地路径误认为 HuggingFace Hub repo id
            model = PeftModel.from_pretrained(
                base_model,
                checkpoint_dir,
                torch_dtype=lora_dtype,
                local_files_only=True
            )
            
            # 移动到目标设备
            model = model.to(target_device)
            print("  ✓ LoRA 模型加载成功")
            
        except Exception as e:
            print(f"  加载 LoRA 模型失败: {e}")
            print("  尝试作为普通模型加载...")
            is_lora_model = False
        
        if not is_lora_model:
        # 加载普通模型（非 LoRA）
        try:
            if use_deepspeed and deepspeed_available and torch.cuda.is_available():
                # DeepSpeed 的 tensor_parallel 需要 MPI 或分布式环境
                # 对于单机多 GPU，我们使用 device_map="auto" 更简单可靠
                print("  注意: DeepSpeed tensor_parallel 需要 MPI/分布式环境")
                print("  回退到使用 device_map='auto' 方式加载（更简单可靠）...")
                # 选择数据类型：优先使用 bfloat16（如果 GPU 支持），否则使用 float16
                if torch.cuda.is_bf16_supported():
                    dtype = torch.bfloat16
                    dtype_str = "bfloat16"
                else:
                    dtype = torch.float16
                    dtype_str = "float16"
                print(f"  使用数据类型: {dtype_str}")
                
                # 使用 device_map="auto" 配合 max_memory 和 low_cpu_mem_usage
                # 如果 safetensors header 太大，尝试使用 PyTorch 格式
                max_memory = {i: "20GiB" for i in range(torch.cuda.device_count())}
                
                # 根据可用文件类型选择加载策略
                load_success = False
                last_error = None
                
                # 策略1: 如果有 PyTorch 文件，优先使用（transformers 会自动选择 PyTorch 文件）
                if model_files['has_pytorch']:
                    print("  检测到 PyTorch 格式文件，transformers 将自动使用 PyTorch 格式...")
                    try:
                        model = AutoModelForCausalLM.from_pretrained(
                            base_model_path,
                            torch_dtype=dtype,
                            device_map="auto",
                            max_memory=max_memory,
                            low_cpu_mem_usage=True,
                            trust_remote_code=True,
                            local_files_only=True
                        )
                        print("  ✓ 使用 PyTorch 格式加载成功")
                        if hasattr(model, 'hf_device_map'):
                            print(f"  模型设备分布: {model.hf_device_map}")
                        load_success = True
                    except Exception as e:
                        last_error = e
                        print(f"  PyTorch 格式加载失败: {e}")
                
                # 策略2: 如果 PyTorch 加载失败或只有 safetensors，尝试 safetensors
                if not load_success and model_files['has_safetensors']:
                    print("  尝试使用 safetensors 格式加载...")
                    try:
                        model = AutoModelForCausalLM.from_pretrained(
                            base_model_path,
                            torch_dtype=dtype,
                            device_map="auto",
                            max_memory=max_memory,
                            low_cpu_mem_usage=True,
                            trust_remote_code=True,
                            local_files_only=True
                        )
                        print("  ✓ 使用 safetensors 格式加载成功")
                        if hasattr(model, 'hf_device_map'):
                            print(f"  模型设备分布: {model.hf_device_map}")
                        load_success = True
                    except Exception as e:
                        last_error = e
                        if "header too large" in str(e) or "SafetensorError" in str(e):
                            print(f"  safetensors header 太大: {e}")
                        else:
                            print(f"  safetensors 加载失败: {e}")
                
                # 策略3: 使用 accelerate 库的分片加载（适用于 header too large 问题）
                if not load_success:
                    print("  尝试使用 accelerate 库分片加载...")
                    try:
                        from accelerate import load_checkpoint_and_dispatch, init_empty_weights
                        from transformers import AutoConfig
                        
                        print("  使用 accelerate 库分片加载...")
                        config = AutoConfig.from_pretrained(
                            base_model_path,
                            trust_remote_code=True,
                            local_files_only=True
                        )
                        
                        # 先创建空模型
                        with init_empty_weights():
                            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
                        
                        # 使用 accelerate 分片加载权重
                        # accelerate 会自动选择可用的格式（优先 PyTorch）
                        model = load_checkpoint_and_dispatch(
                            model,
                            base_model_path,
                            device_map="auto",
                            max_memory=max_memory,
                            dtype=dtype,
                            no_split_module_classes=[]  # 让 accelerate 自动决定如何分片
                        )
                        print("  ✓ 使用 accelerate 库加载成功")
                        load_success = True
                    except Exception as e2:
                        last_error = e2
                        print(f"  accelerate 加载也失败: {e2}")
                
                # 如果所有策略都失败，抛出详细错误
                if not load_success:
                    error_msg = f"""
所有加载方式都失败。最后错误: {last_error}

模型文件检查结果:
  - Safetensors 文件: {len(model_files['safetensors_files'])} 个

"""
                    raise RuntimeError(error_msg)
            elif torch.cuda.is_available():
                # 选择数据类型：优先使用 bfloat16（如果 GPU 支持），否则使用 float16
                if torch.cuda.is_bf16_supported():
                    dtype = torch.bfloat16
                    dtype_str = "bfloat16"
                else:
                    dtype = torch.float16
                    dtype_str = "float16"
                print(f"  使用数据类型: {dtype_str}")
                
                if use_multi_gpu:
                    # 多 GPU 模式：使用 device_map="auto" 自动在所有 GPU 上分布模型
                    print("  使用多 GPU 分布式加载...")
                    # 计算每个 GPU 的最大内存（可选，让系统自动分配）
                    max_memory = {i: "20GiB" for i in range(torch.cuda.device_count())}
                    
                    # 根据可用文件类型选择加载策略
                    load_success = False
                    last_error = None
                    
                    # 策略1: 如果有 PyTorch 文件，优先使用（transformers 会自动选择 PyTorch 文件）
                    if model_files['has_pytorch']:
                        print("  检测到 PyTorch 格式文件，transformers 将自动使用 PyTorch 格式...")
                        try:
                            model = AutoModelForCausalLM.from_pretrained(
                                base_model_path,
                                torch_dtype=dtype,
                                device_map="auto",
                                max_memory=max_memory,
                                low_cpu_mem_usage=True,
                                trust_remote_code=True,
                                local_files_only=True
                            )
                            print("  ✓ 使用 PyTorch 格式多 GPU 加载成功")
                            if hasattr(model, 'hf_device_map'):
                                print(f"  模型设备分布: {model.hf_device_map}")
                            load_success = True
                        except Exception as e:
                            last_error = e
                            print(f"  PyTorch 格式加载失败: {e}")
                    
                    # 策略2: 如果 PyTorch 加载失败或只有 safetensors，尝试 safetensors
                    if not load_success and model_files['has_safetensors']:
                        print("  尝试使用 safetensors 格式加载...")
                        try:
                            model = AutoModelForCausalLM.from_pretrained(
                                base_model_path,
                                torch_dtype=dtype,
                                device_map="auto",
                                max_memory=max_memory,
                                low_cpu_mem_usage=True,
                                trust_remote_code=True,
                                local_files_only=True
                            )
                            print("  ✓ 使用 safetensors 格式多 GPU 加载成功")
                            if hasattr(model, 'hf_device_map'):
                                print(f"  模型设备分布: {model.hf_device_map}")
                            load_success = True
                        except Exception as e:
                            last_error = e
                            if "header too large" in str(e) or "SafetensorError" in str(e):
                                print(f"  safetensors header 太大: {e}")
                            else:
                                print(f"  safetensors 加载失败: {e}")
                    
                    # 如果都失败，抛出错误
                    if not load_success:
                        raise RuntimeError(f"多 GPU 加载失败: {last_error}\n请尝试使用单 GPU 模式或转换模型为 PyTorch 格式")
                else:
                    # 单 GPU 模式：对于 MoE 模型，优先使用 device_map="auto" 自动分配设备
                    # 这样可以更好地处理大模型和 MoE 架构
                    load_success = False
                    last_error = None
                    
                    # 策略1: 如果有 PyTorch 文件，优先使用（transformers 会自动选择 PyTorch 文件）
                    if model_files['has_pytorch']:
                        print("  检测到 PyTorch 格式文件，transformers 将自动使用 PyTorch 格式...")
                        try:
                            model = AutoModelForCausalLM.from_pretrained(
                                base_model_path,
                                torch_dtype=dtype,
                                device_map="auto",
                                low_cpu_mem_usage=True,
                                trust_remote_code=True,
                                local_files_only=True
                            )
                            print("  ✓ 使用 PyTorch 格式加载成功")
                            load_success = True
                        except Exception as e:
                            last_error = e
                            print(f"  PyTorch 格式加载失败: {e}")
                    
                    # 策略2: 如果 PyTorch 加载失败或只有 safetensors，尝试 safetensors
                    if not load_success and model_files['has_safetensors']:
                        print("  尝试使用 safetensors 格式加载...")
                        try:
                            model = AutoModelForCausalLM.from_pretrained(
                                base_model_path,
                                torch_dtype=dtype,
                                device_map="auto",
                                low_cpu_mem_usage=True,
                                trust_remote_code=True,
                                local_files_only=True
                            )
                            print("  ✓ 使用 safetensors 格式加载成功")
                            load_success = True
                        except Exception as e:
                            last_error = e
                            if "header too large" in str(e) or "SafetensorError" in str(e):
                                print(f"  safetensors header 太大: {e}")
                            else:
                                print(f"  safetensors 加载失败: {e}")
                    
                    # 策略3: 回退到指定单个设备
                    if not load_success:
                        print(f"  回退到指定设备加载: {target_device}")
                        try:
                            # transformers 会自动选择可用的格式（优先 PyTorch）
                            model = AutoModelForCausalLM.from_pretrained(
                                base_model_path,
                                torch_dtype=dtype,
                                device_map={"": target_device},
                                trust_remote_code=True,
                                local_files_only=True
                            )
                            print("  ✓ 使用指定设备加载成功")
                            load_success = True
                        except Exception as e:
                            last_error = e
                            print(f"  指定设备加载也失败: {e}")
                    
                    # 如果都失败，抛出错误
                    if not load_success:
                        raise RuntimeError(f"单 GPU 加载失败: {last_error}\n请尝试转换模型为 PyTorch 格式")
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    base_model_path,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    local_files_only=True
                )
                model = model.to(target_device)
        except Exception as e:
            print(f"  使用 device_map 加载失败: {e}")
            print("  尝试使用传统方式加载...")
            # 回退到传统方式（MoE 模型可能需要特殊处理）
            # 选择数据类型：优先使用 bfloat16（如果 GPU 支持），否则使用 float16
            if torch.cuda.is_available():
                if torch.cuda.is_bf16_supported():
                    fallback_dtype = torch.bfloat16
                else:
                    fallback_dtype = torch.float16
            else:
                fallback_dtype = torch.float32
                
            try:
                load_success = False
                last_error = None
                
                if use_multi_gpu and torch.cuda.is_available():
                    # 多 GPU 回退：使用 max_memory 限制和 low_cpu_mem_usage
                    max_memory = {i: "20GiB" for i in range(torch.cuda.device_count())}
                    
                    # 优先使用 PyTorch 格式（transformers 会自动选择）
                    if model_files['has_pytorch']:
                        try:
                            model = AutoModelForCausalLM.from_pretrained(
                                base_model_path,
                                torch_dtype=fallback_dtype,
                                device_map="auto",
                                max_memory=max_memory,
                                low_cpu_mem_usage=True,
                                trust_remote_code=True,
                                local_files_only=True
                            )
                            print("  ✓ 使用 PyTorch 格式多 GPU 回退方式加载成功")
                            load_success = True
                        except Exception as e:
                            last_error = e
                            print(f"  PyTorch 格式多 GPU 回退失败: {e}")
                    
                    # 如果 PyTorch 失败，尝试 safetensors
                    if not load_success and model_files['has_safetensors']:
                        try:
                            model = AutoModelForCausalLM.from_pretrained(
                                base_model_path,
                                torch_dtype=fallback_dtype,
                                device_map="auto",
                                max_memory=max_memory,
                                low_cpu_mem_usage=True,
                                trust_remote_code=True,
                                local_files_only=True
                            )
                            print("  ✓ 使用 safetensors 格式多 GPU 回退方式加载成功")
                            load_success = True
                        except Exception as e:
                            last_error = e
                            print(f"  safetensors 格式多 GPU 回退失败: {e}")
                else:
                    # 单 GPU 回退：尝试使用 device_map="auto" 自动分配设备
                    if model_files['has_pytorch']:
                        try:
                            model = AutoModelForCausalLM.from_pretrained(
                                base_model_path,
                                torch_dtype=fallback_dtype,
                                device_map="auto",
                                low_cpu_mem_usage=True,
                                trust_remote_code=True,
                                local_files_only=True
                            )
                            print("  ✓ 使用 PyTorch 格式 device_map='auto' 加载成功")
                            load_success = True
                        except Exception as e:
                            last_error = e
                            print(f"  PyTorch 格式 device_map='auto' 失败: {e}")
                    
                    # 如果 PyTorch 失败，尝试 safetensors
                    if not load_success and model_files['has_safetensors']:
                        try:
                            model = AutoModelForCausalLM.from_pretrained(
                                base_model_path,
                                torch_dtype=fallback_dtype,
                                device_map="auto",
                                low_cpu_mem_usage=True,
                                trust_remote_code=True,
                                local_files_only=True
                            )
                            print("  ✓ 使用 safetensors 格式 device_map='auto' 加载成功")
                            load_success = True
                        except Exception as e:
                            last_error = e
                            print(f"  safetensors 格式 device_map='auto' 失败: {e}")
                
                # 如果都失败，尝试最简单的加载方式
                if not load_success:
                    print("  尝试最简单的加载方式...")
                    # 最简单的加载方式，不指定设备映射
                    # transformers 会自动选择可用的格式（优先 PyTorch）
                    try:
                        model = AutoModelForCausalLM.from_pretrained(
                            base_model_path,
                            torch_dtype=fallback_dtype,
                            trust_remote_code=True,
                            local_files_only=True
                        )
                        # 手动移动到目标设备（如果指定了单 GPU）
                        if target_device is not None:
                            model = model.to(target_device)
                        print("  ✓ 使用简单方式加载成功")
                        load_success = True
                    except Exception as e:
                        last_error = e
                        print(f"  简单方式加载也失败: {e}")
                        raise RuntimeError(f"所有加载方式都失败。最后错误: {last_error}\n请尝试转换模型为 PyTorch 格式")
            except Exception as e2:
                # 如果上面的所有尝试都失败，抛出详细错误
                error_msg = f"""
所有加载方式都失败。错误: {e2}

模型文件检查结果:
  - Safetensors 文件: {len(model_files['safetensors_files'])} 个
  - PyTorch 文件: {len(model_files['pytorch_files'])} 个

"""
                raise RuntimeError(error_msg)
            # 先检查模型状态，再移动到设备
            try:
                # 验证模型权重是否有效
                if hasattr(model, 'lm_head'):
                    weight = model.lm_head.weight
                    if torch.isnan(weight).any() or torch.isinf(weight).any():
                        print("  警告: 检测到模型权重中有 NaN 或 Inf 值")
            except:
                pass
            
            # 将模型移到指定设备
            try:
                model = model.to(target_device)
            except RuntimeError as e:
                if "CUDA" in str(e) or "device-side assert" in str(e):
                    print(f"  CUDA 错误: {e}")
                    print("  尝试清理 CUDA 缓存并重试...")
                    torch.cuda.empty_cache()
                    import time
                    time.sleep(2)
                    model = model.to(target_device)
                else:
                    raise
    
        model.eval()
        # 使用torch.inference_mode()加速推理（PyTorch 1.9+）
        if hasattr(torch, 'inference_mode'):
            # inference_mode比no_grad更快
            pass
        print("  模型权重加载完成")
        
        # 验证模型词汇表大小
        model_vocab_size = model.config.vocab_size
        tokenizer_vocab_size = tokenizer.vocab_size

        if model_vocab_size != tokenizer_vocab_size:
            print(f"警告: tokenizer vocab ({tokenizer_vocab_size}) != model vocab ({model_vocab_size})")
            print("这是正常的，因为 Qwen3 模型的词汇表大小可能大于 tokenizer 的词汇表大小")
            print("模型会自动处理这种情况，只要 tokenizer 的词汇表是模型词汇表的子集即可")
            
            # 验证 tokenizer 的词汇表是否是模型词汇表的子集
            if tokenizer_vocab_size > model_vocab_size:
                raise RuntimeError(
                    f"FATAL: tokenizer vocab ({tokenizer_vocab_size}) > model vocab ({model_vocab_size})"
                )
            else:
                print(f"✓ Tokenizer 词汇表 ({tokenizer_vocab_size}) 是模型词汇表 ({model_vocab_size}) 的子集，可以继续")
        else:
            print("✓ Tokenizer 和模型的词汇表大小匹配")

    
    # 获取任务描述（从第一个测试样本中提取，如果有）
    task_description = ""
    if test_data and 'task' in test_data[0]:
        task_description = test_data[0]['task'].get('description', '')
    
    # 打开日志文件（如果指定）
    log_file = None
    is_first_log_entry = True
    if log_file_path:
        log_file_dir = os.path.dirname(log_file_path)
        if log_file_dir:
            os.makedirs(log_file_dir, exist_ok=True)
        log_file = open(log_file_path, 'w', encoding='utf-8')
        log_file.write("[\n")  # 开始 JSON 数组
        print(f"日志文件: {log_file_path}")
    
    # 处理每个测试样本
    print("生成 continuations...")
    total_items = 0
    generated_count = 0
    error_count = 0
    skipped_count = 0
    
    # 使用tqdm显示进度，并添加位置参数以便手动更新
    pbar = tqdm(test_data, desc="生成进度", total=len(test_data))
    for sample_idx, sample in enumerate(pbar):
        # 更新进度条描述
        pbar.set_description(f"生成进度 (样本 {sample_idx+1}/{len(test_data)})")
        
        # test_leaderboard.json的结构：context在task.task_behavior_collections[0].data[0].context
        task = sample.get('task', {})
        collections = task.get('task_behavior_collections', [])
        
        if not collections:
            print(f"警告: 样本 {sample_idx} 缺少 task_behavior_collections，跳过")
            pbar.update(1)
            continue
        
        # 处理每个collection中的data
        for collection in collections:
            data_items = collection.get('data', [])
            for data_item in data_items:
                total_items += 1
                
                # 检查是否已有continuations（非空）
                existing_conts = data_item.get('continuations', [])
                if existing_conts and len(existing_conts) > 0:
                    skipped_count += 1
                    continue
                
                context = data_item.get('context', [])
                if not context:
                    print(f"警告: 样本 {sample_idx} 缺少context，跳过")
                    skipped_count += 1
                    continue
                
                user_info = get_user_info_from_leaderboard(sample, train_data)
                
                # 获取历史证据（如果需要）
                history_evidence = []
                if use_history and user_info['user_train_samples']:
                    history_evidence = user_info['user_train_samples'][-3:]  # 使用最近3个样本
                
                # 构建prompt
                messages = build_prompt(
                    context=context,
                    user_profile=user_info['user_profile'] if use_profile else None,
                    task_description=task_description,
                    history=history_evidence if use_history else None,
                    use_profile=use_profile,
                    use_history=use_history,
                    use_context=use_context
                )
                
                # 检测是否为日语任务（通过场景路径或任务描述判断）
                is_japanese_task = False
                scenario_name = os.path.basename(scenario_path)
                if 'RealPersonaChat' in scenario_name or 'realpersonachat' in scenario_name.lower():
                    is_japanese_task = True
                elif task_description and ('日本語' in task_description or '日语' in task_description or 'Japanese' in task_description):
                    is_japanese_task = True
                # 也可以通过检查context中是否有日语字符来判断
                if not is_japanese_task and context:
                    for turn in context[-3:]:  # 检查最后3个turn
                        content = turn.get('content', '')
                        if re.search(r'[\u3040-\u309f\u30a0-\u30ff]', content):
                            is_japanese_task = True
                            break
                
                # 生成continuations
                if vllm_engine is not None:
                    # vLLM模式
                    model_device = None  # vLLM自动处理设备
                else:
                    # 标准模式：获取模型设备
                    try:
                        model_device = next(model.parameters()).device
                    except StopIteration:
                        # 如果模型没有参数（不太可能），使用默认设备
                        model_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                continuations = []
                
                # 生成 num_samples 个 continuations（使用批处理加速）
                generation_timeout = 300  # 每个样本最多5分钟
                
                # 尝试一次性生成所有continuations（批处理）
                try:
                    # 添加超时保护（仅Linux/Unix）
                    if hasattr(signal, 'SIGALRM'):
                        with timeout_handler(generation_timeout * num_samples):  # 总超时时间
                            result = generate_continuations(
                                model=model,
                                tokenizer=tokenizer,
                                messages=messages,
                                num_samples=num_samples,  # 一次性生成所有
                                max_new_tokens=max_new_tokens,
                                device=model_device,
                                do_sample=True,
                                temperature=0.7,
                                top_p=0.9,
                                repetition_penalty=1.2,
                                no_repeat_ngram_size=4,
                                max_output_length=max_output_length,
                                is_japanese_task=is_japanese_task,
                                use_batch_generation=not disable_batch_generation,  # 根据参数决定
                                vllm_engine=vllm_engine  # 传递vLLM引擎
                            )
                    else:
                        # Windows系统，不使用超时
                        result = generate_continuations(
                            model=model,
                            tokenizer=tokenizer,
                            messages=messages,
                            num_samples=num_samples,  # 一次性生成所有
                            max_new_tokens=max_new_tokens,
                            device=model_device,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9,
                            repetition_penalty=1.2,
                            no_repeat_ngram_size=4,
                            max_output_length=max_output_length,
                            is_japanese_task=is_japanese_task,
                            use_batch_generation=not disable_batch_generation,  # 根据参数决定
                            vllm_engine=vllm_engine  # 传递vLLM引擎
                        )
                    
                    if result and len(result) > 0:
                        continuations = result[:num_samples]  # 确保不超过需要的数量
                    else:
                        continuations = []
                except (RuntimeError, TimeoutError) as e:
                    error_msg = str(e)
                    # 如果批处理失败（可能是内存不足），回退到逐个生成
                    if "out of memory" in error_msg.lower() or "CUDA" in error_msg or "memory" in error_msg.lower():
                        print(f"  批处理失败（可能是内存不足），回退到逐个生成: {error_msg[:100]}")
                        torch.cuda.empty_cache()
                        # 逐个生成
                        for sample_num in range(num_samples):
                            try:
                                result = generate_continuations(
                                    model=model,
                                    tokenizer=tokenizer,
                                    messages=messages,
                                    num_samples=1,
                                    max_new_tokens=max_new_tokens,
                                    device=model_device,
                                    do_sample=True,
                                    temperature=0.7,
                                    top_p=0.9,
                                    repetition_penalty=1.2,
                                    no_repeat_ngram_size=4,
                                    max_output_length=max_output_length,
                                    is_japanese_task=is_japanese_task,
                                    use_batch_generation=False,
                                    vllm_engine=vllm_engine
                                )
                                if result and len(result) > 0:
                                    continuations.extend(result)
                            except Exception as e2:
                                print(f"  警告: 第 {sample_num+1} 个生成失败: {str(e2)[:100]}")
                                continue
                    else:
                        raise
                
                # 获取参考值（如果有）
                reference = data_item.get('continuation', '') or data_item.get('reference', '')
                
                # 验证生成结果
                if not continuations or len(continuations) == 0:
                    print(f"警告: 样本 {sample_idx} 生成了空的continuations")
                    error_count += 1
                    error_msg = "生成了空的continuations"
                else:
                    # 填充到data_item中
                    data_item['continuations'] = continuations
                    generated_count += 1
                    if generated_count % 50 == 0:  # 减少打印频率以提升速度
                        print(f"已生成 {generated_count} 个样本的continuations")
                    error_msg = None
                
                # 记录日志
                if log_file:
                    try:
                        log_inference_details(
                            log_file=log_file,
                            sample_idx=sample_idx,
                            data_item_idx=total_items - 1,
                            context=context,
                            messages=messages,
                            user_info=user_info,
                            history_evidence=history_evidence,
                            continuations=continuations,
                            reference=reference,
                            error=error_msg,
                            is_first_entry=is_first_log_entry
                        )
                        is_first_log_entry = False
                    except Exception as log_error:
                        print(f"  警告: 记录日志失败: {log_error}")
        
        # 每个样本处理完后，更新进度条
        pbar.update(1)
        
        # 定期清理CUDA缓存，防止内存累积（减少频率以提升速度）
        if sample_idx % 50 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 关闭日志文件
    if log_file:
        log_file.write("\n]")  # 关闭 JSON 数组
        log_file.close()
        print(f"\n✓ 详细日志已保存到: {log_file_path}")
    
    print(f"\n生成统计:")
    print(f"  总data项: {total_items}")
    print(f"  成功生成: {generated_count}")
    print(f"  跳过: {skipped_count}")
    print(f"  错误: {error_count}")
    
    # 保存结果到 /mnt/parallel/lingyu.li
    if output_dir is None:
        # 从场景路径提取数据集名称
        dataset_name = os.path.basename(scenario_path)
        # 保存到 /mnt/parallel/lingyu.li/{数据集名}_{配置名}/
        base_output_dir = "/mnt/parallel/lingyu.li"
        output_dir = os.path.join(base_output_dir, f"{dataset_name}_{config_name}")
        dataset_name_for_fallback = dataset_name  # 保存用于错误处理
    
    # 尝试创建目录，如果失败则使用本地目录
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"输出目录: {output_dir}")
    except (OSError, IOError) as e:
        print(f"警告: 无法在 {output_dir} 创建目录: {e}")
        # 使用本地目录作为备选
        if 'dataset_name_for_fallback' in locals():
            dataset_name = dataset_name_for_fallback
        else:
            dataset_name = os.path.basename(scenario_path)
        local_output_dir = os.path.join(os.path.expanduser("~"), "checkpoints", f"{dataset_name}_{config_name}")
        output_dir = local_output_dir
        os.makedirs(output_dir, exist_ok=True)
        print(f"使用本地目录: {output_dir}")
    
    # 如果使用了数据分片，输出文件名包含分片ID
    if data_shard_id is not None:
        output_path = os.path.join(output_dir, f'test_leaderboard_shard_{data_shard_id}.json')
    else:
        output_path = os.path.join(output_dir, 'test_leaderboard.json')
    
    # 保存结果（只保留test_leaderboard.json，保持原有结构）
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 结果已保存到: {output_path}")
    print(f"  每个样本生成了 {num_samples} 个 continuations")
    print(f"  输出目录: {output_dir}")
    print(f"  注意: 最终交付时只需要保留 test_leaderboard.json 文件")


def main():
    parser = argparse.ArgumentParser(description='使用微调模型生成test_leaderboard.json')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='模型检查点目录')
    parser.add_argument('--scenario_path', type=str, required=True,
                       help='场景目录路径（包含test_leaderboard.json）')
    parser.add_argument('--config_name', type=str, required=True,
                       help='配置名称（用于输出目录命名，如：profile_and_history）')
    parser.add_argument('--use_profile', action='store_true',
                       help='是否使用profile')
    parser.add_argument('--use_history', action='store_true',
                       help='是否使用history')
    parser.add_argument('--use_context', action='store_true', default=True,
                       help='是否使用context（默认：True）')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='每个样本生成的continuation数量（默认：5）')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录（默认：在scenario_path下创建output_{config_name}）')
    parser.add_argument('--gpu', type=int, default=1,
                       help='使用的GPU编号（默认：1）')
    parser.add_argument('--log_file', type=str, default=None,
                       help='详细日志文件路径（JSON格式，记录每次调用的上下文、输入、输出等。如果未指定，将自动生成）')
    parser.add_argument('--base_model_path', type=str, default=None,
                       help='LoRA 模型的基础模型路径（如果未指定，将尝试自动检测）')
    parser.add_argument('--max_new_tokens', type=int, default=512,
                       help='最大生成token数（默认：512，给模型足够空间写完"废话"，后续通过清洗函数截断）')
    parser.add_argument('--max_output_length', type=int, default=512,
                       help='输出文本最大字符数（默认：200，用于后处理截断）')
    parser.add_argument('--use_multi_gpu', action='store_true',
                       help='使用多 GPU 分布式加载模型（适用于大模型，会自动在所有可用 GPU 上分布）')
    parser.add_argument('--use_deepspeed', action='store_true',
                       help='使用 DeepSpeed ZeRO-3 加载模型（适用于超大模型，可以解决 "header too large" 错误）')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='批处理大小（默认：1，如果内存充足可以增加以加速）')
    parser.add_argument('--disable_batch_generation', action='store_true',
                       help='禁用批处理生成（如果遇到内存问题）')
    parser.add_argument('--use_vllm', action='store_true',
                       help='使用 vLLM 加速推理（推荐，速度更快）')
    parser.add_argument('--vllm_tensor_parallel_size', type=int, default=1,
                       help='vLLM tensor并行大小（默认：1，单卡）')
    parser.add_argument('--vllm_max_model_len', type=int, default=None,
                       help='vLLM最大模型长度（默认：自动检测）')
    parser.add_argument('--data_shard_id', type=int, default=None,
                       help='数据分片ID（0-7，用于8卡并行，每个进程处理1/8数据）')
    parser.add_argument('--num_shards', type=int, default=8,
                       help='数据分片总数（默认：8）')
    
    args = parser.parse_args()
    
    # 设置GPU（直接使用指定的 GPU 编号，不使用 CUDA_VISIBLE_DEVICES）
    target_gpu = args.gpu
    print(f"目标 GPU: {target_gpu}")
    
    # 验证 GPU 是否可用
    if torch.cuda.is_available():
        if target_gpu >= torch.cuda.device_count():
            print(f"警告: GPU {target_gpu} 不存在，可用 GPU: {list(range(torch.cuda.device_count()))}")
            print(f"将使用 GPU 0")
            target_gpu = 0
        print(f"CUDA 可用，GPU 数量: {torch.cuda.device_count()}")
        print(f"将使用 GPU: {target_gpu}")
    else:
        print("警告: CUDA 不可用，将使用 CPU")
    
    # 如果未指定日志文件，自动生成一个
    if args.log_file is None:
        # 从 scenario_path 提取数据集名称
        dataset_name = os.path.basename(args.scenario_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.output_dir:
            log_dir = args.output_dir
        else:
            log_dir = os.path.join(os.path.dirname(args.checkpoint_dir), "logs")
        os.makedirs(log_dir, exist_ok=True)
        args.log_file = os.path.join(log_dir, f"inference_{dataset_name}_{args.config_name}_{timestamp}.json")
    
    # 将 GPU 编号和基础模型路径传递给 process_scenario
    process_scenario(
        scenario_path=args.scenario_path,
        checkpoint_dir=args.checkpoint_dir,
        config_name=args.config_name,
        use_profile=args.use_profile,
        use_history=args.use_history,
        use_context=args.use_context,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        gpu_id=target_gpu,
        log_file_path=args.log_file,
        base_model_path=args.base_model_path,
        max_new_tokens=args.max_new_tokens,
        max_output_length=args.max_output_length,
        use_multi_gpu=args.use_multi_gpu,
        use_deepspeed=args.use_deepspeed,
        batch_size=args.batch_size,
        disable_batch_generation=args.disable_batch_generation,
        use_vllm=args.use_vllm,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        vllm_max_model_len=args.vllm_max_model_len,
        data_shard_id=args.data_shard_id,
        num_shards=args.num_shards
    )


if __name__ == '__main__':
    main()
