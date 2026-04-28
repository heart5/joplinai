# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Joplin 智能问答系统 HTTP API 服务

# %%
# joplin_qa_api.py

# %% [markdown]
# # 导入库

# %%
import argparse
import json
import logging
import threading
import time
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, request

# 尝试导入项目核心模块
try:
    from config_manager import CONFIG_MANAGER
    from func.datatools import getkeysfromcloud
    from func.jpfuncs import (
        getinivaluefromcloud,
    )
    from func.logme import log
    from joplinai import CONFIG as CONFIG_JA
    from queryanswer import CONFIG as CONFIG_QA
    from queryanswer import JoplinQASystem, OptimizedJoplinQASystem
except ImportError as e:
    # 降级处理：配置基础日志并定义占位类
    import logging as log_module

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    log = logging.getLogger(__name__)
    log.warning(f"部分模块导入失败，API基础功能可能受限: {e}")

    # 仅为演示定义占位类，实际使用时请确保queryanswer.py可用
    class JoplinQASystem:
        def __init__(self, config):
            pass

        def ask(self, question):
            return {"answer": "模块未正确导入", "is_based_on_notes": False}

# %% [markdown]
# # 全局配置

# %%
# 默认配置 (与joplinai.py中的CONFIG保持一致)
DEFAULT_CONFIG = {**CONFIG_JA.copy(), **CONFIG_QA.copy()}

# %%
API_KEYS = getkeysfromcloud()

# %% [markdown]
# # 全局问答系统实例（单例，线程安全）

# %%
_qa_system_instances = {}
_qa_system_lock = threading.Lock()

# 会话历史存储 {session_id: [history_list]}
_session_histories = {}
_history_lock = threading.Lock()

# %% [markdown]
# # Flask应用

# %%
app = Flask(__name__)


# %% [markdown]
# # 函数库

# %% [markdown]
# ## sanitize_config(obj)

# %%
def sanitize_config(obj):
    """递归地将配置中的 Path 等对象转换为字符串"""
    # +++ 新增：关键修复 - 确保配置可JSON序列化 +++
    if isinstance(obj, dict):
        return {k: sanitize_config(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_config(item) for item in obj]
    elif isinstance(obj, Path):
        return str(obj)  # 将 Path 转换为字符串
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        # 对于其他不可序列化类型，尝试转换为字符串表示
        return str(obj)


# %% [markdown]
# ## get_qa_system_for_session(session_id: str, config_overrides: Optional[Dict] = None,) -> OptimizedJoplinQASystem

# %%
def get_qa_system_for_session(
    session_id: str, config_overrides: Optional[Dict] = None
) -> OptimizedJoplinQASystem:
    """获取或创建指定会话的问答系统实例（支持云端配置热更新）"""
    global _qa_system_instances

    # 1. 获取最新的云端配置快照及其指纹
    cloud_config_snapshot = CONFIG_MANAGER.get_config_snapshot()
    current_cloud_fingerprint = CONFIG_MANAGER.get_config_fingerprint()

    # 2. 构建本次请求的最终配置（云端快照 + 本次覆盖）
    effective_config = cloud_config_snapshot.copy()
    if config_overrides:
        effective_config.update(config_overrides)
    # 注意：这里不需要再计算指纹，直接使用云端指纹即可。
    # 因为 config_overrides 通常是会话级参数（如api_key），不影响业务逻辑配置。

    # 3. 检查是否需要为该会话创建新实例
    need_new_instance = False
    if session_id not in _qa_system_instances:
        need_new_instance = True
        log.debug(f"会话 [{session_id}] 首次请求，将创建实例。")
    else:
        # 获取已缓存实例的元信息
        cached_data = _qa_system_instances[session_id]
        cached_fingerprint = cached_data.get("cloud_fingerprint")
        # 如果云端配置指纹变了，就需要重建实例
        if cached_fingerprint != current_cloud_fingerprint:
            log.info(
                f"会话 [{session_id}] 的云端配置已更新 "
                f"({cached_fingerprint[:8]}... -> {current_cloud_fingerprint[:8]}...)，将重建问答实例。"
            )
            need_new_instance = True

    # 4. 如果需要新实例，则创建（线程安全）
    if need_new_instance:
        with _qa_system_lock:
            # 双重检查锁定
            if (
                session_id not in _qa_system_instances
                or _qa_system_instances[session_id].get("cloud_fingerprint")
                != current_cloud_fingerprint
            ):
                log.info(
                    f"为会话 [{session_id}] 创建新的问答系统实例 (配置指纹: {current_cloud_fingerprint[:8]}...)"
                )

                try:
                    # 注意：这里需要确保 OptimizedJoplinQASystem 能接受我们的 config 字典
                    qa_instance = OptimizedJoplinQASystem(effective_config)
                    # 存储实例及其关联的云端配置指纹
                    _qa_system_instances[session_id] = {
                        "instance": qa_instance,
                        "cloud_fingerprint": current_cloud_fingerprint,
                        "created_at": time.time(),
                    }
                except Exception as e:
                    log.error(f"创建会话 [{session_id}] 的问答系统实例失败: {e}")
                    raise

    # 5. 返回实例
    return _qa_system_instances[session_id]["instance"]


# %% [markdown]
# ## get_or_create_session(session_id: str) -> List[Dict]

# %%
def get_or_create_session(session_id: str) -> List[Dict]:
    """获取或创建指定会话的历史记录"""
    with _history_lock:
        if session_id not in _session_histories:
            _session_histories[session_id] = []
        return _session_histories[session_id]


# %% [markdown]
# ## update_session_history(session_id: str, question: str, answer: str, metadata: Dict)

# %%
def update_session_history(session_id: str, question: str, answer: str, metadata: Dict):
    """更新指定会话的历史记录"""
    history = get_or_create_session(session_id)
    history_entry = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer,
        "metadata": metadata,  # 可包含来源、是否基于笔记等信息
    }
    history.append(history_entry)
    # 限制历史记录长度（例如最近50条）
    if len(history) > 50:
        history.pop(0)
    log.debug(f"会话 {session_id} 历史已更新，当前长度: {len(history)}")


# %% [markdown]
# ## require_api_key(f)

# %%
def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get("X-API-Key")
        if not api_key or api_key not in API_KEYS.values():
            return jsonify({"error": "Invalid or missing API Key"}), 401
        return f(*args, **kwargs)

    return decorated_function


# %% [markdown]
# # Flask API 端点

# %% [markdown]
# ## index()

# %%
@app.route("/")
def index():
    """简单的健康检查端点"""
    return jsonify(
        {"service": "Joplin QA API", "status": "running", "version": "1.0"}
    ), 200


# %% [markdown]
# ## api_ask()

# %%
@app.route("/ask", methods=["POST"])
@require_api_key
def api_ask():
    """
    提问接口
    请求体 (JSON):
    {
        "question": "你的问题是什么？",
        "session_id": "optional_session_id",  # 可选，用于维护对话历史
        "use_history": true,                 # 可选，是否使用该会话的历史上下文
        "config_overrides": {}               # 可选，临时覆盖配置
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "请求体必须为JSON格式"}), 400

        question = data.get("question", "").strip()
        if not question:
            return jsonify({"error": "问题不能为空"}), 400

        session_id = data.get("session_id", "default_session")
        use_history = data.get("use_history", True)
        config_overrides = data.get("config_overrides", {})

        log.info(f"收到提问请求 | 会话: {session_id} | 问题: {question[:50]}...")

        # 1. 获取问答系统实例
        qa_system = get_qa_system_for_session(session_id, config_overrides)

        # 2. 获取用户身份
        user_identity = data.get("user_identity")

        # 3. 调用核心问答功能
        # result = qa_system.ask(question, use_history=use_history)
        result = qa_system.ask(
            question, use_history=use_history, user_identity=user_identity
        )

        # 4. 更新会话历史
        update_session_history(
            session_id=session_id,
            question=question,
            answer=result.get("answer", ""),
            metadata={
                "is_based_on_notes": result.get("is_based_on_notes", False),
                "relevant_notes_count": len(result.get("relevant_notes", [])),
                "sources": result.get("sources", []),
            },
        )

        # 5. 构建响应
        response_data = {
            "success": True,
            "answer": result.get("answer", ""),
            "session_id": session_id,
            "metadata": {
                "is_based_on_notes": result.get("is_based_on_notes", False),
                "relevant_notes_count": len(result.get("relevant_notes", [])),
                "sources": result.get("sources", []),
                "context_length": result.get("context_length", 0),
                "processing_time": result.get(
                    "processing_time", None
                ),  # 如果原方法返回
            },
        }
        return jsonify(response_data), 200

    except Exception as e:
        log.error(f"处理提问请求时出错: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


# %% [markdown]
# ## api_get_history()

# %%
@app.route("/history", methods=["GET"])
@require_api_key
def api_get_history():
    """
    获取指定会话的历史记录
    查询参数:
        session_id: 会话ID（默认为'default'）
        limit: 返回最近多少条记录（可选）
    """
    try:
        session_id = request.args.get("session_id", "default_session")
        limit = request.args.get("limit", type=int)

        history = get_or_create_session(session_id)
        if limit and limit > 0:
            history = history[-limit:]

        return jsonify(
            {
                "success": True,
                "session_id": session_id,
                "history": history,
                "total": len(history),
            }
        ), 200
    except Exception as e:
        log.error(f"获取历史记录时出错: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# %% [markdown]
# ## api_reload_config()

# %%
@app.route("/admin/reload_config", methods=["POST"])
@require_api_key
def api_reload_config():
    """手动触发从云端重载配置（管理员功能）"""
    try:
        updated = CONFIG_MANAGER.force_refresh()
        if updated:
            new_fingerprint = CONFIG_MANAGER.get_config_fingerprint()[:8]
            # 不清除实例，让它们按需重建（惰性更新）
            log.warning(f"手动强制刷新云端配置完成。新指纹: {new_fingerprint}...")
            return jsonify(
                {
                    "success": True,
                    "message": "云端配置已强制刷新。现有会话将在下次请求时使用新配置。",
                    "new_fingerprint": new_fingerprint,
                }
            )
        else:
            return jsonify({"success": True, "message": "云端配置未发生变化。"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# %% [markdown]
# ## api_clear_history()

# %%
@app.route("/clear_history", methods=["POST"])
@require_api_key
def api_clear_history():
    """清空指定会话的对话历史"""
    data = request.get_json()
    session_id = data.get("session_id", "default_session")

    if not session_id:
        return jsonify({"error": "缺少参数: session_id"}), 400

    try:
        # 获取该会话对应的问答系统实例
        qa_system = get_qa_system_for_session(session_id)
        # 调用清空历史的方法
        qa_system.clear_history()
        # 同时也可清空 API 层存储的该会话历史
        with _history_lock:
            if session_id in _session_histories:
                del _session_histories[session_id]
        log.info(f"已清空会话【{session_id}】的对话历史")
        return jsonify(
            {
                "success": True,
                "message": f"会话 '{session_id}' 的历史记录已清空",
                "session_id": session_id,
            }
        )
    except Exception as e:
        log.error(f"清空会话【{session_id}】历史时出错: {e}")
        return jsonify({"error": f"清空历史失败: {str(e)}"}), 500


# %% [markdown]
# ## api_get_stats()

# %%
@app.route("/stats", methods=["GET", "POST"])
@require_api_key
def api_get_stats():
    """获取指定会话的统计信息"""
    # 支持通过JSON body或查询参数传递session_id
    if request.method == "POST":
        data = request.get_json()
        session_id = data.get("session_id", "default_session")
    else:  # GET
        session_id = request.args.get("session_id", "default_session")

    if not session_id:
        return jsonify({"error": "缺少参数: session_id"}), 400

    try:
        qa_system = get_qa_system_for_session(session_id)
        stats = qa_system.get_statistics()

        log.info(f"获取会话【{session_id}】的统计信息")
        return jsonify({"success": True, "session_id": session_id, "statistics": stats})
    except Exception as e:
        log.error(f"获取会话【{session_id}】统计信息时出错: {e}")
        return jsonify({"error": f"获取统计失败: {str(e)}"}), 500


# %% [markdown]
# ## api_health()

# %%
@app.route("/health", methods=["GET"])
@require_api_key
def api_health():
    """健康检查端点"""
    try:
        qa_system = get_qa_system()
        # 简单检查向量数据库连接
        stats = (
            qa_system.get_statistics() if hasattr(qa_system, "get_statistics") else {}
        )
        return jsonify(
            {
                "status": "healthy",
                "service": "joplin_qa_api",
                "timestamp": datetime.now().isoformat(),
                "stats": stats,
            }
        ), 200
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500


# %% [markdown]
# # 主程序配置

# %% [markdown]
# ## load_config_from_file(config_path: Optional[str] = None) -> Dict

# %%
def load_config_from_file(config_path: Optional[str] = None) -> Dict:
    """从配置文件加载配置（优先级高于默认配置）"""
    config = DEFAULT_CONFIG.copy()
    if config_path and Path(config_path).exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                file_config = json.load(f)
            config.update(file_config)
            log.info(f"已从 {config_path} 加载配置")
        except Exception as e:
            log.warning(f"读取配置文件 {config_path} 失败: {e}")
    return config


# %% [markdown]
# ## main()

# %%
def main():
    """主函数：解析参数并启动API服务"""
    parser = argparse.ArgumentParser(description="Joplin 智能问答系统 HTTP API 服务")
    parser.add_argument(
        "--host", default="127.0.0.1", help="监听主机 (默认: 127.0.0.1)"
    )
    parser.add_argument("--port", type=int, default=5000, help="监听端口 (默认: 5000)")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    parser.add_argument("--config", type=str, help="配置文件路径 (JSON格式)")
    args = parser.parse_args()

    # 加载配置
    global DEFAULT_CONFIG
    if args.config:
        DEFAULT_CONFIG = load_config_from_file(args.config)

    # 预初始化问答系统（可选，加快第一个请求的响应）
    try:
        log.info("预初始化问答系统...")
        get_qa_system_for_session("default_session")
        log.info("预初始化完成")
    except Exception as e:
        log.warning(f"预初始化失败，将继续懒加载: {e}")

    # 启动Flask服务
    log.info(f"启动 Joplin QA API 服务于 http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


# %% [markdown]
# # 主函数main

# %%
if __name__ == "__main__":
    main()
