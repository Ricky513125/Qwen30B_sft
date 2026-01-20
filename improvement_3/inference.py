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
):
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
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
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
        
        outputs = model.generate(
            **inputs,
            **generation_kwargs,
        )
        gen = outputs[0][inputs["input_ids"].shape[1]:]
        
        # 清理生成的 token IDs，确保它们在有效范围内
        tokenizer_vocab_size = tokenizer.vocab_size
        
        # 确保 gen 是 tensor 并且移到 CPU
        if isinstance(gen, torch.Tensor):
            gen = gen.cpu()
        else:
            gen = torch.tensor(gen, dtype=torch.long)
        
        # 清理无效的 token IDs（超出词汇表范围、NaN、Inf等）
        # 将无效值替换为 eos_token_id 或 pad_token_id
        valid_mask = (gen >= 0) & (gen < tokenizer_vocab_size) & torch.isfinite(gen.float())
        
        if valid_mask.sum() == 0:
            # 如果没有有效 token，返回空字符串
            return [""]
        
        # 只保留有效的 token IDs
        cleaned_gen = gen[valid_mask]
        
        # 如果清理后没有有效 token，返回空字符串
        if len(cleaned_gen) == 0:
            return [""]
        
        # 解码
        try:
            decoded_text = tokenizer.decode(cleaned_gen, skip_special_tokens=True).strip()
            # decoded_text = tokenizer.decode(gen, skip_special_tokens=True).strip()
    
            # 调试：打印原始解码文本
            if not decoded_text:
                print(f"  警告: 解码后文本为空")
                return [""]
            
            # --- 调用清洗函数 ---
            cleaned_text = clean_model_output(decoded_text, max_length=max_output_length)
            
            # 调试：如果清理后为空或过短，返回原始文本（至少保留一些内容）
            if not cleaned_text or len(cleaned_text.strip()) < 3:
                if decoded_text and len(decoded_text.strip()) > 0:
                    print(f"  警告: 清理后文本为空或过短，保留原始文本的前{min(max_output_length, len(decoded_text))}字符")
                    # 只做最基本的清理：移除明显的垃圾token
                    fallback = decoded_text
                    for pattern in [r'\bVRTX\b', r'\bVERTEX\b', r'<Vertex>', r'\(Vertex\)', r'_VERTEX_',
                                   r'\bPorno\b', r'\bporno\b', r'Viagra\s+Porno']:
                        fallback = re.sub(pattern, '', fallback, flags=re.IGNORECASE)
                    cleaned_text = fallback[:max_output_length].strip() if fallback.strip() else ""
                else:
                    cleaned_text = ""
            
            return [cleaned_text if cleaned_text else ""]
        except Exception as decode_error:
            # 如果解码仍然失败，尝试逐个 token 解码
            print(f"  警告: 批量解码失败，尝试逐个解码: {decode_error}")
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
                return [decoded_text if decoded_text else ""]
            except:
                return [""]
    
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


# def clean_model_output(text: str, max_length: int = 32768) -> str:
#     """极致清理：移除思考过程、ChatML 标签、角色标识、垃圾token"""
#     if not text:
#         return ""
    
#     # 保存原始文本，以防清理后为空
#     original_text = text
    
#     # 1. 移除 <think>...</think> 及其内部的所有内容
#     text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
#     # 2. 移除残余的或未闭合的标签
#     text = text.replace('<think>', '').replace('</think>', '')
    
#     # 3. 移除垃圾token和模式（VRTX, Vertex, Porno, oplayer, optimizer等）
#     # 注意：只移除明确的垃圾token，不要误删正常内容
#     garbage_patterns = [
#         r'\bVRTX\b', r'\bVERTEX\b', r'<Vertex>', r'\(Vertex\)', r'_VERTEX_',
#         r'\bPorno\b', r'\bporno\b', r'\bPorn\b', r'\bporn\b', r'\bXXX\b', r'\bxxx\b',
#         r'\boplayer\b', r'\boptimizer\b', r'\boyal\b',
#         r'Viagra\s+Porno', r'Porno\s+Porno',
#         r'色情', r'暴力', r'恐怖', r'成人', r'性爱', r'激情', r'情色',
#     ]
#     for pattern in garbage_patterns:
#         text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
#     # 4. 移除模型可能输出的 meta-commentary（例如：好的，我需要分析...）
#     # 这是一个启发式规则：如果回答中包含大段关于"用户"、"对话"、"分析"的中文解释，尝试截断
#     # 注意：这个逻辑可能过于激进，只在明确找到答案时才截断，否则保留原文本
#     if "分析" in text or "建议" in text or "要求" in text:
#         # 如果模型在最后才给答案，答案通常在 [[ ]] 或 "最终生成：" 之后
#         final_ans = re.search(r'\[\[(.*?)\]\]', text)
#         if final_ans and final_ans.group(1).strip():
#             text = final_ans.group(1)
#         elif "最终生成" in text:
#             parts = text.split("最终生成")
#             if len(parts) > 1 and parts[-1].strip():
#                 text = parts[-1]
#             # 如果分割后为空，保留原文本（不做任何修改）

#     # 5. 移除 Chat 模板相关的标记
#     special_patterns = [r'<\|im_start\|>.*?\n', r'<\|im_end\|>', r'<\|user\|>', r'<\|assistant\|>']
#     for pattern in special_patterns:
#         text = re.sub(pattern, '', text)
        
#     # 6. 移除奇怪的重复后缀 (如你结果中的 ROKE)
#     text = text.replace('ROKE', '').strip()
    
#     # 6.5. 移除零宽字符和特殊Unicode控制字符
#     text = re.sub(r'[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff]', '', text)
#     text = re.sub(r'[\u2028\u2029]', '\n', text)
    
#     # 6.6. 优先在第一个换行符处截断（如果后面跟着垃圾内容）
#     first_newline_pos = text.find('\n')
#     if first_newline_pos > 0:
#         # 检查第一个换行符后的内容
#         after_first_line = text[first_newline_pos + 1:].strip()
#         # 垃圾模式：以括号、符号、元数据等开头
#         garbage_start_patterns = [
#             r'^（',  # 以"（"开头（如"（感情・態度）"、"（注：...）"）
#             r'^\(',  # 以"("开头
#             r'^※',  # 以"※"开头
#             r'^\*\*\*\*',  # 以"****"开头
#             r'^\*\*',  # 以"**"开头
#             r'^---',  # 以"---"开头
#             r'^stage',  # 以"stage"开头
#             r'^問題',  # 以"問題"开头
#             r'^（以下',  # 以"（以下"开头
#             r'^（注',  # 以"（注"开头
#             r'^（上記',  # 以"（上記"开头
#             r'^[A-Z]\)',  # 以"A)"开头（选择题）
#         ]
#         for pattern in garbage_start_patterns:
#             if re.match(pattern, after_first_line, re.IGNORECASE):
#                 # 在第一个换行符处截断
#                 text = text[:first_newline_pos].strip()
#                 break
    
#     # 6.7. 检测并截断其他垃圾模式
#     garbage_markers = [
#         r'\n（',  # 换行后跟"（"
#         r'\n\(',  # 换行后跟"("
#         r'\n※',  # 换行后跟"※"
#         r'\n\*\*\*\*',  # 换行后跟"****"
#         r'\n\*\*[：:]',  # 换行后跟"**:"
#         r'\n問題[：:]',  # 换行后跟"問題："
#         r'\nstage',  # 换行后跟"stage"
#         r'\n---',  # 换行后跟"---"
#         r'\n[A-Z]\)\s*[A-Z]\)',  # 换行后跟选择题模式
#     ]
#     first_garbage_pos = len(text)
#     for pattern in garbage_markers:
#         match = re.search(pattern, text, re.IGNORECASE)
#         if match:
#             pos = match.start()
#             if pos < first_garbage_pos:
#                 first_garbage_pos = pos
    
#     if first_garbage_pos < len(text):
#         text = text[:first_garbage_pos].strip()
    
#     # 6.8. 移除末尾的重复垃圾模式
#     trailing_garbage = [
#         r'(\n\*\*\*\*[：:]\s*)+$',
#         r'(\n\*\*[：:]\s*)+$',
#         r'(\n\s*[※\-\*]+\s*)+$',
#     ]
#     for pattern in trailing_garbage:
#         text = re.sub(pattern, '', text)
    
#     # 7. 移除过多的emoji（保留前3个，删除后面的）
#     # 注意：如果文本中没有emoji，这个逻辑不应该影响文本
#     emoji_pattern = r'[😀-🙏🌀-🗿🚀-🛿Ⓜ-🉑]+'
#     emoji_matches = list(re.finditer(emoji_pattern, text))
#     if len(emoji_matches) > 3:
#         # 保留前3个emoji，删除后面的
#         keep_end = emoji_matches[2].end() if len(emoji_matches) > 2 else len(text)
#         # 找到keep_end之后第一个非emoji字符的位置
#         remaining_text = text[keep_end:]
#         # 移除所有emoji
#         remaining_text = re.sub(emoji_pattern, '', remaining_text)
#         text = text[:keep_end] + remaining_text
#     # 如果没有超过3个emoji，不做任何处理
    
#     # 8. 移除重复的标点符号（超过2个连续的标点）
#     text = re.sub(r'([。！？，、；：])\1{2,}', r'\1', text)
#     text = re.sub(r'([.!?,;:])\1{2,}', r'\1', text)
    
#     # 8.5. 在句子结尾处截断（如果后面跟着垃圾内容）
#     sentence_endings = ['。', '！', '？', '.', '!', '?', '～', '〜']
#     for ending in sentence_endings:
#         last_pos = text.rfind(ending)
#         if last_pos > 0:
#             after = text[last_pos + 1:].strip()
#             # 如果后面跟着垃圾模式，截断
#             if after and (
#                 after.startswith(('（', '(', '※', '****', '**', '問題', '---', 'stage')) or
#                 re.match(r'^[A-Z]\)', after)
#             ):
#                 text = text[:last_pos + 1].strip()
#                 break
    
#     # 9. 按句子边界截断（如果太长）
#     text = text.strip()
#     if len(text) > max_length:
#         # 尝试在句子边界截断
#         sentences = re.split(r'([。！？.!?])', text)
#         truncated = ""
#         for i in range(0, len(sentences), 2):
#             if i + 1 < len(sentences):
#                 candidate = truncated + sentences[i] + sentences[i+1]
#             else:
#                 candidate = truncated + sentences[i]
            
#             if len(candidate) <= max_length:
#                 truncated = candidate
#             else:
#                 break
        
#         if truncated and len(truncated) > 0:
#             text = truncated
#         else:
#             # 如果找不到句子边界，直接截断
#             text = text[:max_length].rstrip()
#             # 尝试在最后一个标点处截断
#             last_punct = max(
#                 text.rfind('。'), text.rfind('！'), text.rfind('？'),
#                 text.rfind('.'), text.rfind('!'), text.rfind('?'),
#                 text.rfind('，'), text.rfind(',')
#             )
#             if last_punct > max_length * 0.5:  # 至少保留一半长度
#                 text = text[:last_punct + 1]
#             # 如果截断后为空，至少保留前max_length个字符
#             if not text.strip():
#                 text = text[:max_length].strip()
    
#     # 10. 处理多语言污染（如果目标是日语，过滤掉大段的中文解释）
#     # 针对"中文推理+日语回答"的情况，进行强力截断
#     # 检测是否包含大量中文字符和少量日文字符（可能是中文推理+日语回答）
#     chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
#     japanese_chars = len(re.findall(r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9faf]', text))
#     total_cjk_chars = chinese_chars + japanese_chars
    
#     if total_cjk_chars > 10:
#         chinese_ratio = chinese_chars / total_cjk_chars if total_cjk_chars > 0 else 0
#         japanese_ratio = japanese_chars / total_cjk_chars if total_cjk_chars > 0 else 0
        
#         # 如果中文比例很高（>70%），可能是中文推理，尝试找到日语回答的开始位置
#         if chinese_ratio > 0.7 and japanese_ratio > 0.1:
#             # 尝试找到第一个连续的日语段落（至少10个字符）
#             japanese_pattern = r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9faf]{10,}'
#             japanese_matches = list(re.finditer(japanese_pattern, text))
            
#             if japanese_matches:
#                 # 找到第一个日语段落，保留从该位置开始的内容
#                 first_japanese_start = japanese_matches[0].start()
#                 # 检查前面是否有明显的分隔符（如"回答："、"返答："等）
#                 before_japanese = text[:first_japanese_start]
#                 separators = ['回答：', '返答：', '応答：', '答：', '：', '\n\n', '。\n']
                
#                 # 寻找最后一个分隔符的位置
#                 last_sep_pos = -1
#                 for sep in separators:
#                     pos = before_japanese.rfind(sep)
#                     if pos > last_sep_pos:
#                         last_sep_pos = pos + len(sep)
                
#                 if last_sep_pos > 0:
#                     # 从分隔符后开始保留
#                     text = text[last_sep_pos:].strip()
#                 else:
#                     # 从第一个日语段落开始保留
#                     text = text[first_japanese_start:].strip()
        
#         # 如果文本开头是中文推理模式（包含"分析"、"建议"、"需要"等），尝试截断
#         if text.startswith(('分析', '建议', '需要', '根据', '基于', '我认为', '我觉得')):
#             # 寻找第一个日语段落或明显的回答标记
#             answer_markers = ['回答：', '返答：', '応答：', '答：']
#             for marker in answer_markers:
#                 if marker in text:
#                     text = text.split(marker, 1)[-1].strip()
#                     break
            
#             # 如果没找到标记，寻找第一个连续的日语段落
#             if any(marker in text for marker in answer_markers) == False:
#                 japanese_matches = list(re.finditer(r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9faf]{5,}', text))
#                 if japanese_matches:
#                     text = text[japanese_matches[0].start():].strip()
    
#     # 最终清理
#     text = text.strip()
    
#     # 如果清理后为空或过短，尝试保留原始文本的一部分
#     if not text or len(text) < 5:
#         # 如果原始文本存在，至少保留前max_length个字符
#         if original_text and len(original_text.strip()) > 0:
#             # 只做最基本的清理：移除明显的垃圾token
#             fallback_text = original_text
#             for pattern in [r'\bVRTX\b', r'\bVERTEX\b', r'<Vertex>', r'\(Vertex\)', r'_VERTEX_',
#                            r'\bPorno\b', r'\bporno\b', r'Viagra\s+Porno']:
#                 fallback_text = re.sub(pattern, '', fallback_text, flags=re.IGNORECASE)
#             fallback_text = fallback_text.strip()
#             if fallback_text:
#                 text = fallback_text[:max_length] if len(fallback_text) > max_length else fallback_text
    
#     return text.strip()



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
    use_deepspeed: bool = False  # 是否使用 DeepSpeed ZeRO-3
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
    print(f"\n处理场景: {scenario_path}")
    print(f"模型: {checkpoint_dir}")
    print(f"配置: profile={use_profile}, history={use_history}, context={use_context}")
    base_model_path = "/mnt/parallel/models/Qwen3-30B-A3B-Instruct-2507"
    # 加载测试数据
    test_leaderboard_path = os.path.join(scenario_path, 'test_leaderboard.json')
    if not os.path.exists(test_leaderboard_path):
        print(f"警告: {test_leaderboard_path} 不存在，跳过")
        return
    
    test_data = load_test_leaderboard(test_leaderboard_path)
    print(f"测试样本数: {len(test_data)}")
    
    # 加载训练数据（用于获取用户信息和历史）
    train_path = os.path.join(scenario_path, 'train.json')
    train_data = load_train_data(train_path) if os.path.exists(train_path) else []
    all_train_samples = extract_training_samples(train_data) if train_data else []
    print(f"训练样本数: {len(all_train_samples)}")
    
    # 加载模型
    print("加载模型...")
    
    # 检查模型文件类型
    print(f"检查模型文件: {base_model_path}")
    model_files = check_model_files(base_model_path)
    print(f"  发现 safetensors 文件: {len(model_files['safetensors_files'])} 个")
    print(f"  发现 PyTorch 文件: {len(model_files['pytorch_files'])} 个")
    
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




    
    # 验证 tokenizer 配置
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
  - PyTorch 文件: {len(model_files['pytorch_files'])} 个

可能的原因和解决方案：
1. Safetensors header 太大：这是 Qwen3-30B-A3B 模型的已知问题
   - 解决方案 A: 将 safetensors 转换为 PyTorch 格式
     运行: python -c "from transformers import AutoModel; model = AutoModel.from_pretrained('{base_model_path}', trust_remote_code=True); model.save_pretrained('{base_model_path}_pytorch', safe_serialization=False)"
     然后使用转换后的路径: {base_model_path}_pytorch
   - 解决方案 B: 更新 safetensors 库到最新版本
     pip install --upgrade safetensors transformers accelerate
   - 解决方案 C: 使用更少的 GPU（减少到 4 张或更少）
   - 解决方案 D: 使用单 GPU 模式（不使用 --use_multi_gpu）

2. 内存不足：确保有足够的系统内存（建议至少 64GB）

3. 模型文件损坏：检查模型文件完整性

如果问题持续，请联系模型提供者获取 PyTorch 格式的权重文件。
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

建议的解决方案：
1. 将 safetensors 转换为 PyTorch 格式:
   python -c "from transformers import AutoModel; model = AutoModel.from_pretrained('{base_model_path}', trust_remote_code=True); model.save_pretrained('{base_model_path}_pytorch', safe_serialization=False)"

2. 更新库版本:
   pip install --upgrade safetensors transformers accelerate

3. 使用单 GPU 模式（不使用 --use_multi_gpu）
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
    print("  模型权重加载完成")
    
    # 验证模型词汇表大小
    # model_vocab_size = model.config.vocab_size
    # print(f"  模型词汇表大小: {model_vocab_size}")
    # if vocab_size != model_vocab_size:
    #     print(f"  警告: Tokenizer 和模型的词汇表大小不匹配！")
    
    # print("模型加载完成")

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
    
    for sample_idx, sample in enumerate(tqdm(test_data, desc="生成进度")):
        # test_leaderboard.json的结构：context在task.task_behavior_collections[0].data[0].context
        task = sample.get('task', {})
        collections = task.get('task_behavior_collections', [])
        
        if not collections:
            print(f"警告: 样本 {sample_idx} 缺少 task_behavior_collections，跳过")
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
                # 获取模型设备（对于多 GPU 模型，使用第一个参数所在的设备）
                # 如果模型使用 device_map，输入会自动路由到正确的设备
                try:
                    model_device = next(model.parameters()).device
                except StopIteration:
                    # 如果模型没有参数（不太可能），使用默认设备
                    model_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                continuations = []
                
                # 生成 num_samples 个 continuations
                for _ in range(num_samples):
                    try:
                        result = generate_continuations(
                            model=model,
                            tokenizer=tokenizer,
                            messages=messages,
                            num_samples=1,  # 每次生成1个
                            max_new_tokens=max_new_tokens,  # 使用参数
                            device=model_device,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9,
                            repetition_penalty=1.2,  # 增加重复惩罚
                            no_repeat_ngram_size=4,
                            max_output_length=max_output_length,
                            is_japanese_task=is_japanese_task  # 传递日语任务标识
                        )
                        if result and len(result) > 0:
                            continuations.extend(result)
                    except RuntimeError as e:
                        if "CUDA" in str(e) or "device-side assert" in str(e) or "inf" in str(e) or "nan" in str(e):
                            # 尝试使用 greedy decoding
                            try:
                                result = generate_continuations(
                                    model=model,
                                    tokenizer=tokenizer,
                                    messages=messages,
                                    num_samples=1,
                                    max_new_tokens=max_new_tokens,  # 使用参数
                                    device=model_device,
                                    do_sample=False,  # Greedy decoding
                                    repetition_penalty=1.2,  # 增加重复惩罚
                                    no_repeat_ngram_size=4,
                                    max_output_length=max_output_length,
                                    is_japanese_task=is_japanese_task  # 传递日语任务标识
                                )
                                if result and len(result) > 0:
                                    continuations.extend(result)
                            except Exception as e2:
                                print(f"  警告: 生成失败: {e2}")
                                break
                        else:
                            raise
                    except Exception as e:
                        print(f"  警告: 生成失败: {e}")
                        break
                
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
                    if generated_count % 10 == 0:
                        print(f"已生成 {generated_count} 个样本的continuations")
                    error_msg = None
                
                # 记录日志
                if log_file:
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
        use_deepspeed=args.use_deepspeed
    )


if __name__ == '__main__':
    main()
