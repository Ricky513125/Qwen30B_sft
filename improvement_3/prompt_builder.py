"""
Prompt 构建模块 - 优化版
支持 profile 和 history 的消融实验，并严格区分 Prompt 与 Target
"""
import json
from typing import List, Dict, Any, Optional, Tuple


def build_prompt(
    context: List[Dict[str, str]],
    user_profile: Optional[Dict[str, Any]] = None,
    task_description: Optional[str] = None,
    history: Optional[List[Any]] = None,
    use_profile: bool = True,
    use_history: bool = True,
    use_context: bool = True
) -> List[Dict[str, str]]:
    """
    构建 prompt 消息列表
    """
    messages = []

    # --- 1. 确定当前样本的唯一合法 'user' 标识符 ---
    # 每次函数运行都会创建一个全新的 set
    current_target_name = ""
    if user_profile and isinstance(user_profile, dict):
        # 提取当前样本要模拟的对象名字，例如 "CX"
        current_target_name = str(user_profile.get('name', '')).strip()
    
    # 1. 构建系统指令 (System Message)
    system_parts = [
        "请作为一个理性的观察者，基于给定的上下文生成下一个问题。要求：",
        "1. 语气自然，逻辑连贯",
        "2. 不重复已出现的内容",
        "3. 不使用多余的标点符号",
        "4. 避免无限循环的短语",
        "5. 生成简洁、有意义的回复\n",
        "【重要指令】：",
        "1. 严禁使用 <think> 标签，严禁输出任何思考过程。",
        "2. 必须直接模拟用户的语气和回答内容。",
        "3. 立即开始对话，不要有任何开场白。"
    ]
    
    if task_description:
        system_parts.append(f"任务描述: {task_description}")
    
    # 添加 Profile (消融项1)
    if use_profile and user_profile:
        profile_str = json.dumps(user_profile, ensure_ascii=False, indent=2)
        system_parts.append(f"用户信息:\n{profile_str}")
    
    messages.append({"role": "system", "content": "\n".join(system_parts)})
    
    # 2. 添加历史示例 (消融项2) - 作为独立的user消息，让模型学习用户之前说过什么
    if use_history and history:
        for hist_item in history[:3]:  # 最多3个历史示例
            if isinstance(hist_item, str):
                # 如果是字符串，直接作为user消息
                messages.append({"role": "user", "content": hist_item})
            else:
                # 兼容 data_loader 中提取出的完整 context 格式
                h_next = hist_item.get('next_question', '') or hist_item.get('continuation', '')
                if h_next:
                    # 历史示例都是用户说的话，role应该是user
                    messages.append({"role": "user", "content": h_next})
    
    # 3. 添加当前对话上下文（消融项3）
    # 注意：这里的 context 可能包含原始格式（source字段），需要识别用户说的话
    if use_context and context:
        for turn in context:
            source = str(turn.get('source', '')).strip()
            source_lower = source.lower()
            
            # 判定准则：识别用户说的话
            # - source 是 "user"（通用标识，不区分大小写）
            # - source 是 profile 中的 name（如 "HP", "AH" 等，需要精确匹配，因为可能有大小写）
            is_user = False
            if source_lower == 'user':
                is_user = True
            elif current_target_name and source == current_target_name:
                # 精确匹配 profile 中的 name（保持原始大小写）
                is_user = True
            
            role = 'user' if is_user else 'assistant'
            messages.append({"role": role, "content": turn.get('content', '')})
    return messages


def build_training_prompt(
    context: List[Dict[str, str]],
    next_question: str,
    user_profile: Optional[Dict[str, Any]] = None,
    task_description: Optional[str] = None,
    history: Optional[List[Any]] = None,
    use_profile: bool = True,
    use_history: bool = True,
    use_context: bool = True
) -> Tuple[List[Dict[str, str]], str]:
    """
    构建训练用的 prompt 和目标文本
    
    注意：为了防止标签泄露，返回的 messages 列表只包含到上一个 Assistant 的回复为止。
    next_question 作为独立的 target 返回，在 Dataset 中进行拼接。
    """
    # 获取纯净的上下文消息
    messages = build_prompt(
        context=context,
        user_profile=user_profile,
        task_description=task_description,
        history=history,
        use_profile=use_profile,
        use_history=use_history,
        use_context=use_context
    )
    
    # 返回：(用于构建 Prompt 的消息, 预测的目标文本)
    return messages, next_question.strip()