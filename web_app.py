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
# ## 管理员功能路由

# %% [markdown]
# ### admin_dashboard()

# %%
@app.route("/admin")
@login_required
@admin_required
def admin_dashboard():
    """管理员仪表盘（用户管理主页）"""
    return render_template("admin/users.html", user=session["user"])


# %% [markdown]
# ### api_admin_get_users()

# %%
@app.route("/api/admin/users", methods=["GET"])
@admin_required
def api_admin_get_users():
    """API: 获取所有用户详细信息（供管理界面使用）"""
    users = USER_MANAGER.get_all_users()
    # 对敏感信息进行脱敏（如不返回密码哈希）
    safe_users = []
    for u in users:
        safe_u = {
            "id": u["id"],
            "username": u["username"],
            "display_name": u["display_name"],
            "role": u["role"],
            "is_active": bool(u["is_active"]),
            "created_at": u["created_at"],
            "last_login": u["last_login"],
        }
        safe_users.append(safe_u)
    return jsonify({"success": True, "users": safe_users})


# %% [markdown]
# ### api_admin_reset_password()

# %%
@app.route("/api/admin/user/reset_password", methods=["POST"])
@admin_required
def api_admin_reset_password():
    """API: 管理员重置用户密码"""
    data = request.get_json()
    target_username = data.get("username")
    new_password = data.get("new_password")

    if not target_username or not new_password:
        return jsonify({"success": False, "error": "用户名和新密码不能为空"}), 400

    # 可选：检查密码强度
    if len(new_password) < 6:
        return jsonify({"success": False, "error": "密码长度至少6位"}), 400

    success = USER_MANAGER.reset_user_password(
        target_username, new_password, admin_username=session["user"]["username"]
    )
    if success:
        return jsonify({"success": True, "message": "密码重置成功"})
    else:
        return jsonify({"success": False, "error": "用户不存在或操作失败"}), 400


# %% [markdown]
# ### api_admin_toggle_active()

# %%
@app.route("/api/admin/user/toggle_active", methods=["POST"])
@admin_required
def api_admin_toggle_active():
    """API: 切换用户启用/禁用状态"""
    data = request.get_json()
    target_username = data.get("username")
    is_active = data.get("is_active")  # True or False

    if target_username is None or is_active is None:
        return jsonify({"success": False, "error": "参数不完整"}), 400

    # 禁止管理员禁用自己
    if target_username == session["user"]["username"]:
        return jsonify({"success": False, "error": "不能禁用自己的账户"}), 403

    success = USER_MANAGER.update_user_active_status(
        target_username, is_active, admin_username=session["user"]["username"]
    )
    if success:
        return jsonify({"success": True, "message": "用户状态更新成功"})
    else:
        return jsonify({"success": False, "error": "操作失败"}), 400


# %% [markdown]
# ### api_admin_update_role()

# %%
@app.route("/api/admin/user/update_role", methods=["POST"])
@admin_required
def api_admin_update_role():
    """API: 更新用户角色"""
    data = request.get_json()
    target_username = data.get("username")
    new_role = data.get("new_role")

    if not target_username or new_role not in ["admin", "colleague"]:
        return jsonify({"success": False, "error": "参数无效"}), 400

    # 防止最后一个管理员被降级（可选，根据业务逻辑）
    # 此处省略，可根据实际情况添加检查

    success = USER_MANAGER.update_user_role(
        target_username, new_role, admin_username=session["user"]["username"]
    )
    if success:
        return jsonify({"success": True, "message": "用户角色更新成功"})
    else:
        return jsonify({"success": False, "error": "操作失败"}), 400


# %% [markdown]
# ### api_admin_update_display_name()

# %%
@app.route("/api/admin/user/update_display_name", methods=["POST"])
@admin_required
def api_admin_update_display_name():
    """API: 更新用户显示名称"""
    data = request.get_json()
    target_username = data.get("username")
    new_display_name = data.get("new_display_name")

    if not target_username or not new_display_name or not new_display_name.strip():
        return jsonify({"success": False, "error": "显示名称不能为空"}), 400

    success = USER_MANAGER.change_user_display_name(
        target_username, new_display_name, admin_username=session["user"]["username"]
    )
    if success:
        return jsonify({"success": True, "message": "显示名称更新成功"})
    else:
        return jsonify({"success": False, "error": "操作失败"}), 400


# %% [markdown]
# ### api_admin_create_user()

# %%
@app.route("/api/admin/user/create", methods=["POST"])
@admin_required
def api_admin_create_user():
    """API: 管理员创建新用户"""
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")
    display_name = data.get("display_name")
    role = data.get("role", "colleague")

    if not all([username, password, display_name]):
        return jsonify({"success": False, "error": "必填字段缺失"}), 400

    if role not in ["admin", "colleague"]:
        role = "colleague"

    success = USER_MANAGER.create_user(username, password, display_name, role)
    if success:
        USER_MANAGER.log_audit(
            session["user"]["id"],
            "CREATE_USER",
            details=f"创建新用户: {username} ({display_name}, {role})",
            ip_address=request.remote_addr,
        )
        return jsonify({"success": True, "message": "用户创建成功"})
    else:
        return jsonify({"success": False, "error": "用户名已存在或创建失败"}), 400


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
