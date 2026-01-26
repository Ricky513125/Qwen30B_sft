"""
数据加载模块
用于加载 LovinkDialogue 数据集
"""
import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path

# 导入缓存模块
try:
    from .history_cache import save_history, load_history
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from history_cache import save_history, load_history


def load_json_file(file_path: str) -> Any:
    """加载 JSON 文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_train_data(train_path: str) -> List[Dict[str, Any]]:
    """加载训练数据"""
    if not os.path.exists(train_path):
        return []
    return load_json_file(train_path)


def load_test_data(test_path: str) -> List[Dict[str, Any]]:
    """加载测试数据"""
    if not os.path.exists(test_path):
        return []
    return load_json_file(test_path)


def get_user_profile(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """获取用户 profile 信息"""
    user = sample.get('user', {})
    return user.get('profile', None)

def get_user_name(sample: Dict[str, Any]) -> Optional[str]:
    """获取用户 name（从 profile 中）"""
    user = sample.get('user', {})
    profile = user.get('profile', {})
    if isinstance(profile, dict):
        return profile.get('name', None)
    return None

def get_user_name(sample: Dict[str, Any]) -> Optional[str]:
    """获取用户 name（从 profile 中）"""
    user = sample.get('user', {})
    profile = user.get('profile', {})
    if isinstance(profile, dict):
        return profile.get('name', None)
    return None


def get_task_description(sample: Dict[str, Any]) -> str:
    """获取任务描述"""
    task = sample.get('task', {})
    return task.get('description', '')


def extract_context_from_sample(sample: Dict[str, Any]) -> List[Dict[str, str]]:
    """从样本中提取 context"""
    task = sample.get('task', {})
    collections = task.get('task_behavior_collections', [])
    
    if collections and len(collections) > 0:
        collection = collections[0]
        data_items = collection.get('data', [])
        if data_items and len(data_items) > 0:
            return data_items[0].get('context', [])
    
    return []


def extract_continuation_from_sample(sample: Dict[str, Any]) -> str:
    """从样本中提取 continuation"""
    task = sample.get('task', {})
    collections = task.get('task_behavior_collections', [])
    
    if collections and len(collections) > 0:
        collection = collections[0]
        data_items = collection.get('data', [])
        if data_items and len(data_items) > 0:
            return data_items[0].get('continuation', '')
    
    return ''


def extract_training_samples(train_data: List[Dict[str, Any]], debug: bool = False) -> List[Dict[str, Any]]:
    """
    从训练数据中提取训练样本
    目标：将'user'设为目标角色，'user_wxxxx'设为对话者(assistant)
    训练模式：[History...] -> Predict Next 'user' Turn
    """
    samples = []
    target_role_name = "user"  # 我们要预测的那个人的 source 标识符

    if debug:
        print(f"\n开始提取训练样本，总数据项数: {len(train_data)}\n" + "="*50)

    for item_idx, item in enumerate(train_data):
        user_hash = item.get('user_hash', '')
        user_profile = get_user_profile(item)
        # 获取完整的user对象（包含personality），用于人格映射
        user_object = item.get('user', {})
        task_description = get_task_description(item)
        
        # 获取用户的 name（用于识别 context 中的用户消息）
        user_name = None
        if user_profile and isinstance(user_profile, dict):
            user_name = str(user_profile.get('name', '')).strip()
        
        # 提取对话集合
        collections = item.get('task', {}).get('task_behavior_collections', [])
        
        for collection in collections:
            for data_item in collection.get('data', []):
                context = data_item.get('context', [])
                continuation = data_item.get('continuation', '').strip()
                
                # 构建完整对话列表用于切分
                # 注意：我们要预测的是用户说话的内容
                # 用户在数据中的 source 可能是 "user"，也可能是在 profile 里显示的 name
                full_dialogue = []
                for turn in context:
                    source = str(turn.get('source', '')).strip()
                    # 判断是否是用户说的话：
                    # - source 是 "user"（通用标识）
                    # - source 是 profile 中的 name（如 "HP", "AH" 等）
                    is_user = False
                    if source.lower() == 'user':
                        is_user = True
                    elif user_name and source == user_name:
                        is_user = True
                    
                    role = "user" if is_user else "assistant"
                    full_dialogue.append({"role": role, "content": turn.get('content', '')})
                
                # 将 continuation 也加入队列（作为最后一个样本的目标）
                if continuation:
                    full_dialogue.append({"role": "user", "content": continuation})

                # --- 样本切分逻辑 ---
                # 我们寻找每一个 user 回复的位置，将其作为 target，之前的作为 context
                for i in range(len(full_dialogue)):
                    if full_dialogue[i]['role'] == 'user':
                        # 只有当 user 说话时，我们才可能创建一个样本
                        # context 是 0 到 i-1 轮
                        input_context = full_dialogue[:i]
                        target_text = full_dialogue[i]['content']

                        if target_text and len(input_context) > 0:
                            # 确保 context 的最后一轮是 assistant (对话者)
                            # 这样符合 LLM "User -> Assistant -> User" 的交替逻辑
                            if input_context[-1]['role'] == 'assistant':
                                samples.append({
                                    'context': input_context,     # 包含 role 和 content 的列表
                                    'next_question': target_text, # 目标文本
                                    'user_profile': user_profile,  # profile部分（用于向后兼容）
                                    'user_object': user_object,   # 完整的user对象（包含personality，用于人格映射）
                                    'task_description': task_description,
                                    'user_hash': user_hash
                                })
    # --- 新增：保存样本逻辑 ---
    # 定义保存路径（自动处理 ~ 符号）
    save_dir = os.path.expanduser("~/parallel-post-train/ablation/sample_results")
    os.makedirs(save_dir, exist_ok=True) # 如果目录不存在则创建
    
    # 建议保存为 .jsonl 格式，方便大规模数据处理
    save_path = os.path.join(save_dir, "extracted_samples.jsonl")
    
    try:
        with open(save_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        print(f"✅ 成功将 {len(samples)} 个样本保存至: {save_path}")
    except Exception as e:
        print(f"❌ 保存失败: {e}")

        
    if debug:
        print(f"提取完成！有效样本总数: {len(samples)}\n" + "="*50)
    return samples

def extract_test_samples(test_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """提取测试样本，逻辑同上，但保持 context 为原始格式以供模板转换"""
    samples = []

    for item in test_data:
        user_hash = item.get('user_hash', '')
        user_profile = get_user_profile(item)
        task_description = get_task_description(item)
        
        # 获取用户的 name（用于识别 context 中的用户消息）
        user_name = None
        if user_profile and isinstance(user_profile, dict):
            user_name = str(user_profile.get('name', '')).strip()
        
        collections = item.get('task', {}).get('task_behavior_collections', [])
        
        for collection in collections:
            for data_item in collection.get('data', []):
                context = data_item.get('context', [])
                reference = data_item.get('continuation', '')
                
                if context and reference:
                    # 转换格式
                    formatted_context = []
                    for turn in context:
                        source = str(turn.get('source', '')).strip()
                        # 判断是否是用户说的话：
                        # - source 是 "user"（通用标识）
                        # - source 是 profile 中的 name（如 "HP", "AH" 等）
                        is_user = False
                        if source.lower() == 'user':
                            is_user = True
                        elif user_name and source == user_name:
                            is_user = True
                        
                        role = "user" if is_user else "assistant"
                        formatted_context.append({"role": role, "content": turn.get('content', '')})
                    
                    samples.append({
                        'context': formatted_context,
                        'reference': reference,
                        'user_profile': user_profile,
                        'task_description': task_description,
                        'user_hash': user_hash
                    })
    return samples


def get_user_history_samples(
    all_samples: List[Dict[str, Any]],
    user_hash: str,
    current_sample: Optional[Dict[str, Any]] = None,
    max_history: int = 2000
) -> List[Dict[str, Any]]:
    """
    获取用户的历史样本（用于构建 H_{s,u}）
    
    Args:
        all_samples: 所有训练样本
        user_hash: 用户哈希
        current_sample: 当前样本（用于排除）
        max_history: 最大历史样本数
    
    Returns:
        历史样本列表
    """
    history = []
    for sample in all_samples:
        if sample['user_hash'] == user_hash:
            if current_sample is None or sample != current_sample:
                history.append(sample)
                if len(history) >= max_history:
                    break
    return history


def select_relevant_history(history: List[str], current_context: List[Dict[str, str]], max_items: int = 15) -> List[str]:
    """
    从历史中选择与当前context最相关的几条
    
    Args:
        history: 完整的历史列表
        current_context: 当前对话上下文
        max_items: 最大选择数量
    
    Returns:
        选择后的历史列表
    """
    if len(history) <= max_items:
        return history
    
    # 提取当前context中的关键词（从所有turn的content中）
    current_keywords = set()
    for turn in current_context:
        content = turn.get('content', '').strip()
        if content:
            # 简单的关键词提取：分词（按空格和标点）
            words = content.lower().split()
            current_keywords.update(words)
    
    # 计算每条历史与当前context的相似度
    history_scores = []
    for hist_item in history:
        # 提取历史项的关键词
        hist_words = set(hist_item.lower().split())
        # 计算交集（共同关键词）
        common_keywords = current_keywords & hist_words
        # 相似度 = 共同关键词数量 / (当前关键词数量 + 历史关键词数量 - 共同关键词数量)
        # 使用Jaccard相似度
        union_size = len(current_keywords | hist_words)
        similarity = len(common_keywords) / union_size if union_size > 0 else 0.0
        history_scores.append((hist_item, similarity))
    
    # 按相似度排序，选择最相关的
    history_scores.sort(key=lambda x: x[1], reverse=True)
    selected = [item for item, _ in history_scores[:max_items]]
    
    # 保持原始顺序（按在原始history中的位置）
    # 但优先选择相似度高的
    result = []
    for hist_item in history:
        if hist_item in selected:
            result.append(hist_item)
            if len(result) >= max_items:
                break
    
    # 如果还没达到max_items，补充剩余的（按时间顺序）
    if len(result) < max_items:
        for hist_item in history:
            if hist_item not in result:
                result.append(hist_item)
                if len(result) >= max_items:
                    break
    
    return result


def get_user_only_history(
    all_samples: List[Dict[str, Any]], 
    user_hash: str,
    current_sample: Optional[Dict[str, Any]] = None,
    current_context: Optional[List[Dict[str, str]]] = None,
    max_history: int = 15,
    use_cache: bool = True
) -> List[str]:
    """
    获取用户历史发送的问题列表，用于 Few-shot 或 RAG 增强
    
    用户可能在数据中的 source 命名为 "user"，也可能是在 profile 里显示的 name（如 "HP", "AH" 等）
    需要从 context 中识别用户说的话，以及 next_question
    
    Args:
        all_samples: 所有训练样本
        user_hash: 用户哈希
        current_sample: 当前样本（用于排除）
        current_context: 当前context（用于排除和智能选择）
        max_history: 最大历史数量
        use_cache: 是否使用缓存
    
    Returns:
        用户历史对话列表（只包含用户说的话）
    """
    # 尝试从缓存加载
    if use_cache:
        cached_history = load_history(user_hash)
        if cached_history is not None:
            # 如果提供了current_context，需要排除其中的内容并智能选择
            if current_context:
                # 提取当前context中用户说的话
                current_user_contents = set()
                user_profile = None
                user_name = None
                
                # 获取用户名称
                for s in all_samples:
                    if s.get('user_hash') == user_hash:
                        user_profile = s.get('user_profile')
                        if user_profile and isinstance(user_profile, dict):
                            user_name = user_profile.get('name', '').strip()
                        break
                
                # 从current_context中提取用户说的话
                for turn in current_context:
                    if isinstance(turn, dict):
                        # 如果turn是已处理格式（有role字段）
                        if 'role' in turn and turn['role'] == 'user':
                            content = turn.get('content', '').strip()
                            if content:
                                current_user_contents.add(content)
                        # 如果turn是原始格式（有source字段）
                        elif 'source' in turn:
                            source = str(turn.get('source', '')).strip()
                            content = turn.get('content', '').strip()
                            is_user_turn = (source.lower() == 'user') or (user_name and source == user_name)
                            if is_user_turn and content:
                                current_user_contents.add(content)
                
                # 从缓存的历史中排除当前context中的内容
                filtered_history = [h for h in cached_history if h not in current_user_contents]
                
                # 如果长度过长，智能选择最相关的
                if len(filtered_history) > max_history:
                    return select_relevant_history(filtered_history, current_context, max_history)
                else:
                    return filtered_history[:max_history]
            else:
                # 没有current_context，直接返回缓存
                return cached_history[:max_history]
    
    # 缓存未命中或未启用缓存，重新计算
    user_history = []
    user_profile = None
    user_name = None
    
    # 获取用户 profile 名称
    for s in all_samples:
        if s.get('user_hash') == user_hash:
            user_profile = s.get('user_profile')
            if user_profile and isinstance(user_profile, dict):
                user_name = user_profile.get('name', '').strip()
            break
    
    # 提取当前context中用户说的话（用于排除）
    current_user_contents = set()
    if current_context:
        for turn in current_context:
            if isinstance(turn, dict):
                # 已处理格式
                if 'role' in turn and turn['role'] == 'user':
                    content = turn.get('content', '').strip()
                    if content:
                        current_user_contents.add(content)
                # 原始格式
                elif 'source' in turn:
                    source = str(turn.get('source', '')).strip()
                    content = turn.get('content', '').strip()
                    is_user_turn = (source.lower() == 'user') or (user_name and source == user_name)
                    if is_user_turn and content:
                        current_user_contents.add(content)
    
    # 遍历所有样本
    for s in all_samples:
        if s.get('user_hash') != user_hash:
            continue  # 不同用户，跳过
        
        if current_sample is not None and s == current_sample:
            continue  # 跳过当前 sample
        
        context = s.get('context', [])
        if context:
            for turn in context:
                # 处理已处理格式（有role字段）
                if isinstance(turn, dict) and 'role' in turn:
                    if turn['role'] == 'user':
                        content = turn.get('content', '').strip()
                        if content and content not in current_user_contents:
                            user_history.append(content)
                # 处理原始格式（有source字段）
                elif isinstance(turn, dict) and 'source' in turn:
                    source = str(turn.get('source', '')).strip()
                    content = turn.get('content', '').strip()
                    
                    is_user_turn = (source.lower() == 'user') or (user_name and source == user_name)
                    if is_user_turn and content and content not in current_user_contents:
                        user_history.append(content)
        
        # next_question 也算用户说的话（但要排除当前context中的内容）
        q = s.get('next_question', '').strip()
        if q and q not in current_user_contents:
            user_history.append(q)
    
    # 去重并限制数量
    seen = set()
    unique_history = []
    for item in reversed(user_history):
        if item and item not in seen:
            seen.add(item)
            unique_history.append(item)
            if len(unique_history) >= max_history * 2:  # 先收集更多，用于智能选择
                break
    
    result = list(reversed(unique_history))
    
    # 如果长度过长，智能选择最相关的
    if len(result) > max_history and current_context:
        result = select_relevant_history(result, current_context, max_history)
    else:
        result = result[:max_history]
    
    # 保存到缓存（如果启用缓存）
    if use_cache and result:
        save_history(user_hash, result)
    
    return result


def build_all_user_history_cache(all_samples: List[Dict[str, Any]], max_history: int = 15) -> Dict[str, int]:
    """
    批量构建所有用户的历史缓存
    
    Args:
        all_samples: 所有训练样本
        max_history: 每个用户的最大历史数量
    
    Returns:
        统计信息：{user_hash: history_count}
    """
    user_hashes = set()
    for sample in all_samples:
        user_hash = sample.get('user_hash')
        if user_hash:
            user_hashes.add(user_hash)
    
    stats = {}
    print(f"开始构建 {len(user_hashes)} 个用户的历史缓存...")
    
    for idx, user_hash in enumerate(user_hashes, 1):
        # 获取该用户的所有历史（不排除任何样本，因为这是全量构建）
        history = get_user_only_history(
            all_samples,
            user_hash,
            current_sample=None,
            current_context=None,
            max_history=max_history * 2,  # 缓存时保存更多，使用时再智能选择
            use_cache=False  # 先不保存，等计算完再保存
        )
        
        # 保存到缓存
        if history:
            save_history(user_hash, history)
            stats[user_hash] = len(history)
        
        if idx % 100 == 0:
            print(f"  已处理 {idx}/{len(user_hashes)} 个用户...")
    
    print(f"✅ 缓存构建完成！共 {len(stats)} 个用户有历史数据")
    return stats