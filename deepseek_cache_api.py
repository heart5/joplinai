# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # DeepSeek 缓存集中服务

# %%
import argparse
import logging
from functools import wraps

from flask import Flask, jsonify, request

# %%
import pathmagic

with pathmagic.context():
    try:
        from aimod.cache_manager import SQLiteCacheManager
        from func.jpfuncs import getinivaluefromcloud
        from func.logme import log
    except ImportError as e:
        logging.basicConfig(level=logging.INFO)
        log = logging.getLogger(__name__)
        log.error(f"导入项目模块失败: {e}")

# %%
CACHE_API_KEY = getinivaluefromcloud("joplinai", "deepseek_cache_api_key")
cache_manager = SQLiteCacheManager(db_path="data/.deepseek_cache/deepseek_cache.db")

app = Flask(__name__)


def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        key = request.headers.get("X-API-Key")
        if not key or key != CACHE_API_KEY:
            return jsonify({"error": "Invalid or missing API Key"}), 401
        return f(*args, **kwargs)

    return decorated


@app.route("/health")
def health():
    return jsonify({"service": "DeepSeek Cache API", "status": "running"})


@app.route("/cache/get", methods=["POST"])
@require_auth
def cache_get():
    data = request.get_json(force=True)
    result = cache_manager.get(data["content_hash"], data["task"])
    return jsonify(
        {
            "content": result.content,
            "requires_validation": result.requires_validation,
            "cache_key": result.cache_key,
            "current_hit_count": result.current_hit_count,
            "total_hits": result.total_hits,
        }
    )


@app.route("/cache/set", methods=["POST"])
@require_auth
def cache_set():
    data = request.get_json(force=True)
    cache_manager.set(data["content_hash"], data["task"], data["result"])
    return jsonify({"ok": True})


@app.route("/cache/validate", methods=["POST"])
@require_auth
def cache_validate():
    data = request.get_json(force=True)
    cache_manager.update_on_validation(
        data["cache_key"],
        data.get("new_result"),
        data["validation_successful"],
    )
    return jsonify({"ok": True})


@app.route("/cache/stats")
@require_auth
def cache_stats():
    key = request.args.get("cache_key")
    return jsonify(cache_manager.get_stats(cache_key=key))


# %%
def main():
    parser = argparse.ArgumentParser(description="DeepSeek 缓存 API 服务")
    parser.add_argument("--host", default="127.0.0.1", help="监听主机")
    parser.add_argument("--port", type=int, default=5003, help="监听端口")
    args = parser.parse_args()

    log.info(f"启动 DeepSeek 缓存 API 服务于 http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
