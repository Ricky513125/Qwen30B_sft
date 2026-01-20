"""
History 缓存模块
用于缓存用户的历史对话，避免重复计算
"""
import json
import os
from typing import List, Optional, Dict, Any
from pathlib import Path


# 缓存目录
CACHE_DIR = os.path.join(os.path.dirname(__file__), "history_cache")


def ensure_cache_dir():
    """确保缓存目录存在"""
    os.makedirs(CACHE_DIR, exist_ok=True)


def get_cache_path(user_hash: str) -> str:
    """获取缓存文件路径"""
    ensure_cache_dir()
    # 使用安全的文件名（移除特殊字符）
    safe_hash = user_hash.replace('/', '_').replace('\\', '_')
    return os.path.join(CACHE_DIR, f"{safe_hash}_history.json")


def save_history(user_hash: str, history: List[str]) -> bool:
    """
    保存用户历史到缓存
    
    Args:
        user_hash: 用户哈希
        history: 历史对话列表（用户说的话）
    
    Returns:
        是否保存成功
    """
    try:
        ensure_cache_dir()
        cache_path = get_cache_path(user_hash)
        
        cache_data = {
            "user_hash": user_hash,
            "history": history,
            "count": len(history)
        }
        
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        
        return True
    except Exception as e:
        print(f"警告: 保存历史缓存失败 (user_hash={user_hash}): {e}")
        return False


def load_history(user_hash: str) -> Optional[List[str]]:
    """
    从缓存加载用户历史
    
    Args:
        user_hash: 用户哈希
    
    Returns:
        历史对话列表，如果不存在则返回 None
    """
    try:
        cache_path = get_cache_path(user_hash)
        
        if not os.path.exists(cache_path):
            return None
        
        with open(cache_path, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        
        # 验证数据格式
        if isinstance(cache_data, dict) and "history" in cache_data:
            return cache_data["history"]
        elif isinstance(cache_data, list):
            # 兼容旧格式（直接是列表）
            return cache_data
        else:
            print(f"警告: 缓存文件格式不正确 (user_hash={user_hash})")
            return None
            
    except Exception as e:
        print(f"警告: 加载历史缓存失败 (user_hash={user_hash}): {e}")
        return None


def clear_cache(user_hash: Optional[str] = None) -> bool:
    """
    清除缓存
    
    Args:
        user_hash: 如果指定，只清除该用户的缓存；否则清除所有缓存
    
    Returns:
        是否清除成功
    """
    try:
        ensure_cache_dir()
        
        if user_hash:
            # 清除指定用户的缓存
            cache_path = get_cache_path(user_hash)
            if os.path.exists(cache_path):
                os.remove(cache_path)
                return True
        else:
            # 清除所有缓存
            for file in os.listdir(CACHE_DIR):
                if file.endswith('_history.json'):
                    os.remove(os.path.join(CACHE_DIR, file))
            return True
            
    except Exception as e:
        print(f"警告: 清除缓存失败: {e}")
        return False


def get_cache_info() -> Dict[str, Any]:
    """
    获取缓存信息
    
    Returns:
        缓存统计信息
    """
    try:
        ensure_cache_dir()
        
        cache_files = [f for f in os.listdir(CACHE_DIR) if f.endswith('_history.json')]
        total_size = 0
        
        for file in cache_files:
            file_path = os.path.join(CACHE_DIR, file)
            if os.path.exists(file_path):
                total_size += os.path.getsize(file_path)
        
        return {
            "cache_dir": CACHE_DIR,
            "file_count": len(cache_files),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2)
        }
    except Exception as e:
        print(f"警告: 获取缓存信息失败: {e}")
        return {}


# 示例用法
if __name__ == "__main__":
    # 测试缓存功能
    test_user_hash = "test_user_123"
    test_history = ["你好", "我想了解一下", "谢谢"]
    
    print("保存缓存...")
    save_history(test_user_hash, test_history)
    
    print("加载缓存...")
    loaded = load_history(test_user_hash)
    print(f"加载结果: {loaded}")
    
    print("\n缓存信息:")
    info = get_cache_info()
    print(json.dumps(info, indent=2, ensure_ascii=False))
    
    print("\n清除缓存...")
    clear_cache(test_user_hash)
