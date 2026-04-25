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
# # 门户应用

# %% [markdown]
# # 引入库

# %%
import json
import logging
from functools import wraps
from pathlib import Path

import requests
from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    session,
    url_for,
)

# 导入自定义用户管理器
try:
    from user_manager import USER_MANAGER
except ImportError:
    # 简易回退
    USER_MANAGER = None
    print("警告: user_manager 导入失败，将使用占位符")
import pathmagic

with pathmagic.context():
    from func.datatools import getkeysfromcloud
    from func.first import getdirmain
    from func.getid import getdeviceid, getdevicename, gethostuser
    from func.jpfuncs import getinivaluefromcloud
    from func.logme import log

# %% [markdown]
# # 全局配置

# %%
app = Flask(__name__)
app.secret_key = "your-secret-key-please-change-in-production"  # 必须更改！

# 配置
QA_API_SERVER = getinivaluefromcloud("joplinai", f"joplinai_qa_server_{getdeviceid()}")
QA_API_PORT = getinivaluefromcloud("joplinai", "joplinai_qa_port")
QA_API_URL = f"http://{QA_API_SERVER}:{QA_API_PORT}"  # 指向您本地的 joplin_qa_api 服务
QA_API_KEY = getkeysfromcloud().get(
    "hc", "invalid"
)  # 从您的云端配置获取，用于内部API调用


# %% [markdown]
# # 函数集合

# %% [markdown]
# ## login_required(f)

# %%
def login_required(f):
    """装饰器：要求用户登录"""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)

    return decorated_function


# %% [markdown]
# ## admin_required(f)

# %%
def admin_required(f):
    """装饰器：要求管理员权限"""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user" not in session or session["user"].get("role") != "admin":
            return jsonify({"error": "需要管理员权限"}), 403
        return f(*args, **kwargs)

    return decorated_function


# %% [markdown]
# ## index()

# %%
@app.route("/")
@login_required
def index():
    """主问答页面"""
    return render_template("index.html", user=session["user"])


# %% [markdown]
# ## login()

# %%
@app.route("/login", methods=["GET", "POST"])
def login():
    """登录页面"""
    if request.method == "GET":
        return render_template("login.html")

    # POST 处理
    username = request.form.get("username")
    password = request.form.get("password")

    if not username or not password:
        return render_template("login.html", error="请输入用户名和密码")

    user = USER_MANAGER.verify_user(username, password)
    if user:
        # 创建会话
        session_id = USER_MANAGER.create_session(user["id"])
        session["session_id"] = session_id
        session["user"] = user
        USER_MANAGER.log_audit(user["id"], "LOGIN", ip_address=request.remote_addr)
        return redirect(url_for("index"))
    else:
        return render_template("login.html", error="用户名或密码错误")


# %% [markdown]
# ## logout()

# %%
@app.route("/logout")
@login_required
def logout():
    """登出"""
    if "session_id" in session:
        USER_MANAGER.delete_session(session["session_id"])
    session.clear()
    return redirect(url_for("login"))


# %% [markdown]
# ## api_ask()

# %%
@app.route("/api/ask", methods=["POST"])
@login_required
def api_ask():
    """提问接口 - 将请求转发给底层QA API，并附加用户身份"""
    data = request.get_json()
    question = data.get("question", "").strip()
    use_history = data.get("use_history", True)

    if not question:
        return jsonify({"error": "问题不能为空"}), 400

    # 准备转发给QA API的请求体
    qa_request_payload = {
        "question": question,
        "use_history": use_history,
        "session_id": f"web_{session['user']['username']}",  # 会话标识
        "config_overrides": {
            # 可以在这里覆盖配置，例如调整检索数量
        },
        "user_identity": {  # 核心：传递用户身份用于权限过滤
            "username": session["user"]["username"],
            "display_name": session["user"]["display_name"],
            "role": session["user"]["role"],
        },
    }

    headers = {
        "Content-Type": "application/json",
        "X-API-Key": QA_API_KEY,  # 内部API调用的密钥
    }

    try:
        response = requests.post(
            f"{QA_API_URL}/ask",
            json=qa_request_payload,
            headers=headers,
            timeout=60,  # 长超时
        )
        response.raise_for_status()
        result = response.json()

        # 记录审计日志（可脱敏）
        USER_MANAGER.log_audit(
            session["user"]["id"],
            "ASK_QUESTION",
            details=f"问题长度: {len(question)}, 来源笔记数: {len(result.get('relevant_notes', []))}",
            ip_address=request.remote_addr,
        )

        # 保存此次问答记录到数据库
        try:
            USER_MANAGER.save_qa_history(
                user_id=session["user"]["id"],
                session_id=f"web_{session['user']['username']}",
                question=question,
                answer=result.get("answer", ""),
                metadata={
                    "is_based_on_notes": result.get("metadata", {}).get(
                        "is_based_on_notes", False
                    ),
                    "relevant_notes_count": result.get("metadata", {}).get(
                        "relevant_notes_count", 0
                    ),
                    "sources": result.get("metadata", {}).get("sources", []),
                },
            )
        except Exception as e:
            log.error(f"保存问答历史到数据库失败: {e}")
            # 不影响主流程，仅记录错误

        return jsonify(result)
    except requests.exceptions.RequestException as e:
        log.error(f"调用QA API失败: {e}")
        return jsonify({"error": "问答服务暂时不可用", "details": str(e)}), 502


# %% [markdown]
# ## api_get_history()

# %%
@app.route("/api/history", methods=["GET"])
@login_required
def api_get_history():
    """获取当前登录用户的持久化问答历史"""
    try:
        limit = request.args.get("limit", default=20, type=int)
        offset = request.args.get("offset", default=0, type=int)

        history = USER_MANAGER.get_qa_history(
            user_id=session["user"]["id"],
            limit=min(limit, 100),  # 防止一次请求过多
            offset=offset,
        )

        return jsonify(
            {"success": True, "history": history, "total": len(history)}
        ), 200

    except Exception as e:
        log.error(f"获取用户历史失败: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# %% [markdown]
# ## list_users()

# %%
@app.route("/api/users", methods=["GET"])
@admin_required
def list_users():
    """管理员：获取用户列表"""
    users = USER_MANAGER.get_all_users()
    return jsonify({"users": users})


# %% [markdown]
# ## system_health()

# %%
@app.route("/api/system/health")
@login_required
def system_health():
    """系统健康检查端点"""
    health = {
        "web_app": "healthy",
        "user_db": "connected",
        "timestamp": datetime.now().isoformat(),
    }
    # 可以添加对QA API和向量库的健康检查
    try:
        qa_health = requests.get(f"{QA_API_URL}/health", timeout=5).json()
        health["qa_api"] = qa_health.get("status", "unknown")
    except:
        health["qa_api"] = "unreachable"

    return jsonify(health)


# %% [markdown]
# ## favicon()

# %%
@app.route("/favicon.ico")
def favicon():
    # 直接从 static 文件夹发送 favicon.ico 文件
    return send_from_directory(
        "static", "favicon.ico", mimetype="image/vnd.microsoft.icon"
    )


# %% [markdown]
# # main()，主函数

# %%
if __name__ == "__main__":
    # 确保模板目录存在
    template_dir = getdirmain() / "templates"
    template_dir.mkdir(exist_ok=True)

    app.run(host="127.0.0.1", port=5001, debug=False)
