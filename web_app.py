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
# # 门户应用

# %% [markdown]
# # 引入库

# %%
import json
import logging
import os
import re
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
    from func.jpfuncs import (
        getinivaluefromcloud,
        getnote,
    )
    from func.logme import log
    from user_manager import USER_DB_PATH, USER_MANAGER
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
app.secret_key = (
    getinivaluefromcloud("joplinai", "flask_secret_key") or os.urandom(32).hex()
)

# 新增：修复IP地址获取，信任反向代理
from werkzeug.middleware.proxy_fix import ProxyFix

app.wsgi_app = ProxyFix(app.wsgi_app, x_for=2, x_proto=1, x_host=1)


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
    sessions = USER_MANAGER.get_user_chat_sessions(session["user"]["id"])
    active_session = USER_MANAGER.get_active_chat_session(session["user"]["id"])
    # 将会话ID存入session以便其他api使用
    session["active_session"] = active_session
    return render_template(
        "index.html",
        user=session["user"],
        chat_sessions=sessions,
        active_session=active_session,
    )


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
    data = request.get_json()
    question = data.get("question", "").strip()
    use_history = data.get("use_history", True)

    if not question:
        return jsonify({"error": "问题不能为空"}), 400

    # 获取当前选中的会话ID（前端可通过请求传入，默认用活动会话）
    session_id = data.get("session_id") or session.get("active_session")
    if not session_id:
        # 自动获取或创建
        session_id = USER_MANAGER.get_active_chat_session(session["user"]["id"])
    # 确保该会话属于当前用户
    user_sessions = [
        s["session_id"]
        for s in USER_MANAGER.get_user_chat_sessions(session["user"]["id"])
    ]
    if session_id not in user_sessions:
        return jsonify({"error": "会话不存在"}), 404

    # 构建转发给 QA API 的请求
    qa_request_payload = {
        "question": question,
        "use_history": use_history,
        "session_id": session_id,  # 使用用户选择的会话ID
        "config_overrides": {},
        "user_identity": {
            "username": session["user"]["username"],
            "display_name": session["user"]["display_name"],
            "role": session["user"]["role"],
            "allowed_notebooks": USER_MANAGER.get_user_with_notebooks(
                session["user"]["username"]
            ).get("allowed_notebooks", ""),  # 关键！
        },
    }

    # ... 后续代码保持不变，只是 session_id 不再是固定的 f"web_{username}"
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": QA_API_KEY,  # 内部API调用的密钥
    }

    try:
        response = requests.post(
            f"{QA_API_URL}/ask",
            json=qa_request_payload,
            headers=headers,
            timeout=90,  # 长超时
        )
        response.raise_for_status()
        result = response.json()

        # 获取来源笔记数（优先使用 sources 列表，兼容旧结构）
        metadata = result.get("metadata", {})
        sources_notes = metadata.get("sources", [])
        relevant_chunks = metadata.get("relevant_chunks", [])
        is_based = metadata.get("is_based_on_notes", False)
        # 记录更详细的审计日志
        USER_MANAGER.log_audit(
            session["user"]["id"],
            "ASK_QUESTION",
            details=f"问题长度: {len(question)}, "
            f"来源笔记数: {len(sources_notes)}, "
            f"相关块数: {len(relevant_chunks)}, "
            f"基于笔记: {is_based}",
            ip_address=request.remote_addr,
        )

        # 保存此次问答记录到数据库
        try:
            USER_MANAGER.save_qa_history(
                user_id=session["user"]["id"],
                session_id=session_id,
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
        # 记录失败审计
        USER_MANAGER.log_audit(
            session["user"]["id"],
            "ASK_QUESTION_FAILED",
            details=f"问题长度: {len(question)}, 错误: {str(e)[:100]}",
            ip_address=request.remote_addr,
        )
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
        session_id = request.args.get("session_id")
        if not session_id:
            # 默认获取当前活动会话
            session_id = USER_MANAGER.get_active_chat_session(session["user"]["id"])
            if not session_id:
                return jsonify({"success": False, "history": []})
        limit = request.args.get("limit", default=20, type=int)
        offset = request.args.get("offset", default=0, type=int)

        history = USER_MANAGER.get_qa_history(
            user_id=session["user"]["id"],
            limit=min(limit, 100),  # 防止一次请求过多
            offset=offset,
            session_id=session_id,  # ✅ 传入
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
# ## 会话管理

# %% [markdown]
# ### chat_sessions_api()

# %%
@app.route("/api/chat_sessions", methods=["GET", "POST"])
@login_required
def chat_sessions_api():
    if request.method == "POST":
        data = request.get_json()
        name = data.get("name", "新对话")
        session_id = USER_MANAGER.create_chat_session(session["user"]["id"], name)
        # 自动设置为活动会话
        USER_MANAGER.set_active_chat_session(session["user"]["id"], session_id)
        return jsonify({"success": True, "session_id": session_id, "name": name})
    else:  # GET
        sessions = USER_MANAGER.get_user_chat_sessions(session["user"]["id"])
        return jsonify({"success": True, "sessions": sessions})


# %% [markdown]
# ### chat_session_detail(session_id)

# %%
@app.route("/api/chat_sessions/<session_id>", methods=["PUT", "DELETE"])
@login_required
def chat_session_detail(session_id):
    # 验证所有权（可选）
    user_sessions = [
        s["session_id"]
        for s in USER_MANAGER.get_user_chat_sessions(session["user"]["id"])
    ]
    if session_id not in user_sessions:
        return jsonify({"error": "无权操作"}), 403

    if request.method == "PUT":  # 重命名
        data = request.get_json()
        new_name = data.get("name")
        if not new_name:
            return jsonify({"error": "名称不能为空"}), 400
        USER_MANAGER.rename_chat_session(session_id, new_name)
        return jsonify({"success": True})
    elif request.method == "DELETE":
        USER_MANAGER.delete_chat_session(session_id)
        # 如果删除的是当前活动会话，自动切换到最近一个会话
        if session.get("active_session") == session_id:
            new_active = USER_MANAGER.get_active_chat_session(session["user"]["id"])
            # 前端刷新时会自动获取
        return jsonify({"success": True})


# %% [markdown]
# ### activate_chat_session(session_id)

# %%
@app.route("/api/chat_sessions/<session_id>/activate", methods=["POST"])
@login_required
def activate_chat_session(session_id):
    user_sessions = [
        s["session_id"]
        for s in USER_MANAGER.get_user_chat_sessions(session["user"]["id"])
    ]
    if session_id not in user_sessions:
        return jsonify({"error": "无权操作"}), 403
    USER_MANAGER.set_active_chat_session(session["user"]["id"], session_id)
    # 同时尝试从数据库恢复该会话的历史到 QA API 内存
    restore_history_for_session(session_id)  # 见下文
    return jsonify({"success": True, "session_id": session_id})


# %% [markdown]
# ### restore_history_for_session(session_id: str)

# %%
# web_app.py 中添加
def restore_history_for_session(session_id: str):
    """从 qa_history 数据库加载该会话的历史记录，并恢复到 QA API 内存"""
    try:
        import sqlite3

        # 从数据库获取历史（最新50条）
        conn = sqlite3.connect(USER_DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT question, answer, created_at FROM qa_history WHERE session_id = ? ORDER BY created_at ASC",
            (session_id,),
        )
        rows = cursor.fetchall()
        conn.close()
        if not rows:
            return
        history = []
        for row in rows:
            entry = {
                "timestamp": row["created_at"],
                "question": row["question"],
                "answer": row["answer"],
                "metadata": {},  # 可以根据需要从原metadata提取
            }
            history.append(entry)
        # 调用 QA API 的 restroe 端点
        requests.post(
            f"{QA_API_URL}/restore_history",
            json={"session_id": session_id, "history": history},
            headers={"X-API-Key": QA_API_KEY},
            timeout=10,
        )
    except Exception as e:
        log.warning(f"恢复会话历史失败: {e}")


# %% [markdown]
# ## 管理员功能路由

# %% [markdown]
# ### api_get_available_notebooks()

# %%
# web_app.py 中添加以下路由
@app.route("/api/admin/available-notebooks", methods=["GET"])
@login_required
@admin_required
def api_get_available_notebooks():
    """获取向量库中所有可用的笔记本标题"""
    try:
        # 提取笔记本标题
        split_ptn = re.compile(r"[,，]")
        notebooks = [
            title.strip()
            for title in split_ptn.split(
                getinivaluefromcloud("joplinai", "shared_notebook_titles")
            )
        ]

        return jsonify(
            {"success": True, "notebooks": notebooks, "count": len(notebooks)}
        )
    except Exception as e:
        log.error(f"获取笔记本列表失败: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# %% [markdown]
# ### api_get_user_details(username)

# %%
@app.route("/api/admin/user/<username>", methods=["GET"])
@login_required
@admin_required
def api_get_user_details(username):
    """获取用户详细信息（包含笔记本白名单）"""
    user = USER_MANAGER.get_user_with_notebooks(username)
    if not user:
        return jsonify({"success": False, "error": "用户不存在"}), 404

    # 移除敏感信息
    user.pop("password_hash", None)

    return jsonify({"success": True, "user": user})


# %% [markdown]
# ### api_admin_delete_user()

# %%
@app.route("/api/admin/user/delete", methods=["POST"])
@login_required
@admin_required
def api_admin_delete_user():
    """API: 管理员删除用户（物理删除）"""
    data = request.get_json()
    target_username = data.get("username")

    if not target_username:
        return jsonify({"success": False, "error": "用户名不能为空"}), 400

    # 禁止删除自己
    if target_username == session["user"]["username"]:
        return jsonify({"success": False, "error": "不能删除自己的账户"}), 403

    # 可选：防止删除最后一个管理员（根据业务需求决定）
    # 这里简单检查用户角色，如果是管理员且是最后一个，则阻止
    target_user = USER_MANAGER._get_user_by_username(target_username)
    if target_user and target_user["role"] == "admin":
        # 检查是否还有其他管理员
        all_users = USER_MANAGER.get_all_users()
        admin_count = sum(1 for u in all_users if u["role"] == "admin")
        if admin_count <= 1:
            return jsonify(
                {
                    "success": False,
                    "error": "至少需要保留一名管理员，无法删除最后一个管理员",
                }
            ), 403

    success = USER_MANAGER.delete_user(
        target_username, admin_username=session["user"]["username"]
    )
    if success:
        return jsonify({"success": True, "message": "用户已删除"})
    else:
        return jsonify({"success": False, "error": "用户不存在或操作失败"}), 400


# %% [markdown]
# ### api_update_user_permissions()

# %%
@app.route("/api/admin/user/update", methods=["POST"])
@login_required
@admin_required
def api_update_user_permissions():
    """更新用户权限配置"""
    data = request.get_json()

    required_fields = ["username", "display_name", "role"]
    for field in required_fields:
        if field not in data:
            return jsonify({"success": False, "error": f"缺少必填字段: {field}"}), 400

    # 验证角色
    valid_roles = ["admin", "team_leader", "team_member"]
    if data["role"] not in valid_roles:
        return jsonify({"success": False, "error": f"无效角色: {data['role']}"}), 400

    # 更新笔记本白名单（非管理员角色）
    if data["role"] != "admin":
        allowed_notebooks = data.get("allowed_notebooks", [])
        success = USER_MANAGER.update_user_permissions(
            username=data["username"],
            role=data["role"],
            allowed_notebooks=allowed_notebooks,
        )

        if not success:
            return jsonify({"success": False, "error": "更新笔记本白名单失败"}), 500

    USER_MANAGER.log_audit(
        session["user"]["id"],
        "UPDATE_USER",
        details=f"更新用户: {data['username']}, "
        f"角色: {data['role']}, "
        f"活跃: {data.get('is_active')}, "
        f"授权笔记本数: {len(data.get('allowed_notebooks', []))}",
        ip_address=request.remote_addr,
    )

    log.info(f"管理员更新用户权限: {data['username']} -> 角色: {data['role']}")
    return jsonify({"success": True, "message": "用户权限更新成功"})


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
        USER_MANAGER.log_audit(
            session["user"]["id"],
            "RESET_PASSWORD",
            details=f"重置用户 {data['username']} 的密码",
            ip_address=request.remote_addr,
        )
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
    role = data.get("role", "team_member")

    if not all([username, password, display_name]):
        return jsonify({"success": False, "error": "必填字段缺失"}), 400

    if role not in ["admin", "team_leader", "team_member"]:
        role = "team_member"

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
# ### admin_user_edit(username)

# %%
@app.route("/admin/users/<username>/edit", methods=["GET"])
@login_required
@admin_required
def admin_user_edit(username):
    """用户编辑独立页面"""
    user = USER_MANAGER.get_user_with_notebooks(username)
    if not user:
        abort(404)
    # 获取可用笔记本列表
    try:
        split_ptn = re.compile(r"[,，]")
        available_notebooks = [
            title.strip()
            for title in split_ptn.split(
                getinivaluefromcloud("joplinai", "shared_notebook_titles")
            )
        ]
    except:
        available_notebooks = []
    return render_template(
        "admin/user_edit.html", user=user, available_notebooks=available_notebooks
    )


# %% [markdown]
# ### admin_user_history(username)

# %%
@app.route("/admin/users/<username>/history")
@login_required
@admin_required
def admin_user_history(username):
    """用户对话历史列表页面"""
    user = USER_MANAGER._get_user_by_username(username)
    if not user:
        abort(404)
    sessions = USER_MANAGER.get_user_chat_sessions(user["id"])
    return render_template("admin/user_history.html", user=user, sessions=sessions)


# %% [markdown]
# ### api_admin_user_session_history(username, session_id)

# %%
@app.route("/api/admin/user/<username>/session/<session_id>/history")
@login_required
@admin_required
def api_admin_user_session_history(username, session_id):
    """获取指定用户某会话的详细问答历史"""
    user = USER_MANAGER._get_user_by_username(username)
    if not user:
        return jsonify({"success": False, "error": "用户不存在"}), 404
    limit = request.args.get("limit", 50, type=int)
    offset = request.args.get("offset", 0, type=int)
    history = USER_MANAGER.get_qa_history(
        user_id=user["id"], session_id=session_id, limit=limit, offset=offset
    )
    return jsonify({"success": True, "history": history})


# %% [markdown]
# ## 审计相关路由

# %%
@app.route("/admin/audit")
@login_required
@admin_required
def admin_audit_page():
    """审计日志页面"""
    return render_template("admin/audit_log.html")


@app.route("/api/admin/audit/logs")
@login_required
@admin_required
def api_admin_audit_logs():
    """审计日志数据API（分页+筛选）"""
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 20, type=int)
    username = request.args.get("username", None)
    action = request.args.get("action", None)
    start_date = request.args.get("start_date", None)
    end_date = request.args.get("end_date", None)

    result = USER_MANAGER.get_audit_logs(
        page=page,
        per_page=per_page,
        username=username,
        action=action,
        start_date=start_date,
        end_date=end_date,
    )

    return jsonify({"success": True, **result})


@app.route("/api/admin/audit/actions")
@login_required
@admin_required
def api_admin_audit_actions():
    """获取所有操作类型（用于下拉框）"""
    actions = USER_MANAGER.get_audit_actions()
    return jsonify({"success": True, "actions": actions})


@app.route("/api/admin/audit/clear", methods=["POST"])
@login_required
@admin_required
def api_admin_audit_clear():
    """清理指定天数前的审计日志"""
    data = request.get_json()
    before_days = data.get("before_days", 90)  # 默认清理90天前

    if before_days < 7:
        return jsonify({"success": False, "error": "保留天数不能少于7天"}), 400

    deleted = USER_MANAGER.clear_audit_logs(before_days)

    # 记录审计日志（本次清理操作本身也要记录）
    USER_MANAGER.log_audit(
        session["user"]["id"],
        "CLEAR_AUDIT_LOGS",
        details=f"清理了 {deleted} 条 {before_days} 天前的审计日志",
        ip_address=request.remote_addr,
    )

    return jsonify(
        {"success": True, "deleted": deleted, "message": f"已清理 {deleted} 条日志"}
    )


@app.route("/api/admin/audit/export")
@login_required
@admin_required
def api_admin_audit_export():
    """导出审计日志为CSV"""
    # 获取所有符合条件的日志（不分页）
    result = USER_MANAGER.get_audit_logs(
        page=1,
        per_page=100000,  # 大数量，实际建议限制导出范围
        username=request.args.get("username"),
        action=request.args.get("action"),
        start_date=request.args.get("start_date"),
        end_date=request.args.get("end_date"),
    )

    if not result["logs"]:
        return jsonify({"success": False, "error": "没有可导出的记录"}), 400

    # 生成CSV
    import csv
    import io

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "时间", "用户名", "显示名", "操作类型", "详情", "IP地址"])

    for log in result["logs"]:
        writer.writerow(
            [
                log["id"],
                log["timestamp"],
                log["username"],
                log["display_name"],
                log["action"],
                log["details"] or "",
                log["ip_address"] or "",
            ]
        )

    csv_content = output.getvalue()
    output.close()

    from flask import Response

    return Response(
        csv_content,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=audit_logs.csv"},
    )


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
