"""
Prompt 构建模块 - 优化版
支持 profile 和 history 的消融实验，并严格区分 Prompt 与 Target
"""
import json
from typing import List, Dict, Any, Optional, Tuple


def _normalize_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    规范化消息列表，确保严格遵循 user/assistant 交替（system 后必须是 user）
    Gemma-3-27B 的 chat template 要求严格交替：system -> user -> assistant -> user -> assistant -> ...
    
    策略：
    1. 保留 system 消息在第一位
    2. 合并连续的相同角色消息
    3. 确保严格交替：user -> assistant -> user -> assistant -> ...
    4. 如果最后一条是 user，添加一个占位符 assistant 响应
    """
    if not messages:
        return messages
    
    normalized = []
    
    # 保留 system 消息（必须在第一位）
    if messages and messages[0].get('role') == 'system':
        normalized.append(messages[0])
        messages = messages[1:]
    
    if not messages:
        return normalized
    
    # 处理剩余消息，合并连续的相同角色消息
    merged_messages = []
    current_role = None
    current_content = []
    
    for msg in messages:
        role = msg.get('role', '').lower()
        content = msg.get('content', '').strip()
        
        if not content:
            continue
        
        if role == current_role:
            # 合并到当前消息
            current_content.append(content)
        else:
            # 保存前一个消息（如果有）
            if current_role and current_content:
                merged_content = "\n".join(current_content) if len(current_content) > 1 else current_content[0]
                merged_messages.append({"role": current_role, "content": merged_content})
            # 开始新消息
            current_role = role
            current_content = [content]
    
    # 保存最后一个消息
    if current_role and current_content:
        merged_content = "\n".join(current_content) if len(current_content) > 1 else current_content[0]
        merged_messages.append({"role": current_role, "content": merged_content})
    
    # 确保严格交替：system -> user -> assistant -> user -> assistant -> ...
    # 期望的下一个角色（system 后应该是 user）
    expected_role = 'user'
    
    for msg in merged_messages:
        role = msg.get('role', '').lower()
        
        if role == expected_role:
            # 角色正确，直接添加
            normalized.append(msg)
            # 切换期望的角色
            expected_role = 'assistant' if expected_role == 'user' else 'user'
        elif role == 'user' and expected_role == 'assistant':
            # 当前是 user，但期望 assistant，说明缺少一个 assistant 响应
            # 添加一个占位符 assistant 响应
            normalized.append({"role": "assistant", "content": "好的，我明白了。"})
            # 然后添加当前的 user 消息
            normalized.append(msg)
            expected_role = 'assistant'  # 下一个应该是 assistant
        elif role == 'assistant' and expected_role == 'user':
            # 当前是 assistant，但期望 user，说明缺少一个 user 消息
            # 这种情况不应该发生，但为了安全，我们跳过这个 assistant 消息
            # 或者添加一个占位符 user 消息
            normalized.append({"role": "user", "content": "请继续。"})
            normalized.append(msg)
            expected_role = 'user'  # 下一个应该是 user
    
    # 确保最后一条消息是 assistant（如果不是，添加一个占位符）
    if normalized and normalized[-1].get('role') == 'user':
        normalized.append({"role": "assistant", "content": "好的，我明白了。"})
    
    return normalized


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
    确保严格遵循 Gemma-3-27B 的 chat template 要求：system -> user -> assistant -> user -> assistant -> ...
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
    
    # 4. 规范化消息，确保严格遵循 user/assistant 交替
    messages = _normalize_messages(messages)
    
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
    
    注意：
    1. 返回的 messages 列表会通过 _normalize_messages 规范化，确保以 assistant 结尾
    2. 在训练时，会在 assistant 消息后添加 "user: " 提示，然后接 next_question 作为 target
    3. 这样模型学习的是：在看到 assistant 的回复后，预测用户会说什么
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