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
# # User Blueprint — 认证、用户、会话、问答、审计端点

# %%
import json
import secrets
import sqlite3
from datetime import datetime, timedelta

from flask import Blueprint, jsonify, request

# %%
from aimod.center_api import _init_db, log, require_auth

__all__ = ["user_bp"]

user_bp = Blueprint("user", __name__)


def _missing(data, *fields):
    """返回 data 中缺失的必填字段列表，无缺失返回 []"""
    return [f for f in fields if not data.get(f)]


# %% [markdown]
# # 认证端点
#
# %%
@user_bp.route("/auth/verify", methods=["POST"])
@require_auth
def api_auth_verify():
    data = request.get_json(force=True)
    if missing := _missing(data, "username", "password_hash"):
        return jsonify({"error": f"缺少必填字段: {', '.join(missing)}"}), 400
    conn = _init_db()
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT id, username, display_name, role FROM users WHERE username=? AND password_hash=? AND is_active=1",
        (data["username"], data["password_hash"]),
    ).fetchone()
    if row:
        user = dict(row)
        conn.execute("UPDATE users SET last_login=? WHERE id=?", (datetime.now().isoformat(), user["id"]))
        conn.commit()
        conn.close()
        log.info(f"用户登录: {user['username']} ({user['role']})")
        return jsonify({"found": True, "user": user})
    conn.close()
    log.info(f"登录失败: 用户名={data['username']} 不存在或密码错误")
    return jsonify({"found": False}), 404
#
#
@user_bp.route("/auth/create_session", methods=["POST"])
@require_auth
def api_auth_create_session():
    data = request.get_json(force=True)
    if missing := _missing(data, "user_id"):
        return jsonify({"error": f"缺少必填字段: {', '.join(missing)}"}), 400
    session_id = secrets.token_urlsafe(32)
    expires_at = datetime.now() + timedelta(hours=data.get("duration_hours", 24))
    conn = _init_db()
    conn.execute("INSERT INTO sessions (session_id, user_id, expires_at) VALUES (?,?,?)",
                 (session_id, data["user_id"], expires_at))
    conn.commit()
    conn.close()
    return jsonify({"session_id": session_id})
#
#
@user_bp.route("/auth/validate_session", methods=["POST"])
@require_auth
def api_auth_validate_session():
    data = request.get_json(force=True)
    if missing := _missing(data, "session_id"):
        return jsonify({"error": f"缺少必填字段: {', '.join(missing)}"}), 400
    conn = _init_db()
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT u.id, u.username, u.display_name, u.role FROM sessions s "
        "JOIN users u ON s.user_id=u.id "
        "WHERE s.session_id=? AND s.expires_at>? AND u.is_active=1",
        (data["session_id"], datetime.now()),
    ).fetchone()
    conn.close()
    if row:
        return jsonify({"valid": True, "user": dict(row)})
    return jsonify({"valid": False}), 404
#
#
@user_bp.route("/auth/delete_session", methods=["POST"])
@require_auth
def api_auth_delete_session():
    data = request.get_json(force=True)
    if missing := _missing(data, "session_id"):
        return jsonify({"error": f"缺少必填字段: {', '.join(missing)}"}), 400
    conn = _init_db()
    conn.execute("DELETE FROM sessions WHERE session_id=?", (data["session_id"],))
    conn.commit()
    conn.close()
    return jsonify({"ok": True})

# %% [markdown]
# # 用户 CRUD 端点
#
# %%
@user_bp.route("/users", methods=["GET"])
@require_auth
def api_users_list():
    conn = _init_db()
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, username, display_name, role, is_active, created_at, last_login FROM users ORDER BY id"
    ).fetchall()
    conn.close()
    return jsonify({"users": [dict(r) for r in rows]})
#
#
@user_bp.route("/users/<username>", methods=["GET"])
@require_auth
def api_users_get(username: str):
    conn = _init_db()
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT id, username, display_name, role, allowed_notebooks, is_active, created_at, last_login FROM users WHERE username=?",
        (username,),
    ).fetchone()
    conn.close()
    if row:
        user = dict(row)
        try:
            user["allowed_notebooks"] = json.loads(user["allowed_notebooks"] or "[]")
        except Exception:
            user["allowed_notebooks"] = []
        return jsonify({"found": True, "user": user})
    return jsonify({"found": False}), 404
#
#
@user_bp.route("/users/create", methods=["POST"])
@require_auth
def api_users_create():
    data = request.get_json(force=True)
    if missing := _missing(data, "username", "password_hash", "display_name"):
        return jsonify({"error": f"缺少必填字段: {', '.join(missing)}"}), 400
    notebooks_json = json.dumps(data.get("allowed_notebooks", []), ensure_ascii=False)
    conn = _init_db()
    try:
        conn.execute(
            "INSERT INTO users (username, password_hash, display_name, role, allowed_notebooks) VALUES (?,?,?,?,?)",
            (data["username"], data["password_hash"], data["display_name"], data.get("role", "team_member"), notebooks_json),
        )
        conn.commit()
        conn.close()
        log.info(f"用户创建: {data['username']} ({data.get('role', 'team_member')})")
        return jsonify({"ok": True})
    except sqlite3.IntegrityError:
        conn.close()
        log.info(f"用户创建失败(已存在): {data['username']}")
        return jsonify({"ok": False, "error": "用户名已存在"}), 409
#
#
@user_bp.route("/users/delete", methods=["POST"])
@require_auth
def api_users_delete():
    data = request.get_json(force=True)
    if missing := _missing(data, "target_username"):
        return jsonify({"error": f"缺少必填字段: {', '.join(missing)}"}), 400
    conn = _init_db()
    row = conn.execute("SELECT id FROM users WHERE username=?", (data["target_username"],)).fetchone()
    if not row:
        conn.close()
        return jsonify({"ok": False, "error": "用户不存在"}), 404
    user_id = row[0]
    conn.execute("DELETE FROM qa_history WHERE session_id IN (SELECT session_id FROM chat_sessions WHERE user_id=?)", (user_id,))
    conn.execute("DELETE FROM chat_sessions WHERE user_id=?", (user_id,))
    conn.execute("DELETE FROM sessions WHERE user_id=?", (user_id,))
    conn.execute("DELETE FROM users WHERE id=?", (user_id,))
    conn.commit()
    conn.close()
    log.info(f"用户删除: {data['target_username']}")
    return jsonify({"ok": True})

# %% [markdown]
# # 用户更新端点
#
# %%
@user_bp.route("/users/update_role", methods=["POST"])
@require_auth
def api_users_update_role():
    data = request.get_json(force=True)
    if missing := _missing(data, "new_role", "target_username"):
        return jsonify({"error": f"缺少必填字段: {', '.join(missing)}"}), 400
    conn = _init_db()
    cursor = conn.execute("UPDATE users SET role=? WHERE username=?", (data["new_role"], data["target_username"]))
    ok = cursor.rowcount > 0
    conn.commit()
    conn.close()
    if ok:
        log.info(f"用户角色变更: {data['target_username']} → {data['new_role']}")
    return jsonify({"ok": ok})
#
#
@user_bp.route("/users/update_permissions", methods=["POST"])
@require_auth
def api_users_update_permissions():
    data = request.get_json(force=True)
    updates, params = [], []
    if "role" in data and data["role"] is not None:
        updates.append("role=?"); params.append(data["role"])
    if "allowed_notebooks" in data and data["allowed_notebooks"] is not None:
        updates.append("allowed_notebooks=?"); params.append(json.dumps(data["allowed_notebooks"], ensure_ascii=False))
    if not updates:
        return jsonify({"ok": True})
    if missing := _missing(data, "username"):
        return jsonify({"error": f"缺少必填字段: {', '.join(missing)}"}), 400
    params.append(data["username"])
    conn = _init_db()
    conn.execute(f"UPDATE users SET {', '.join(updates)} WHERE username=?", params)
    conn.commit()
    conn.close()
    return jsonify({"ok": True})
#
#
@user_bp.route("/users/reset_password", methods=["POST"])
@require_auth
def api_users_reset_password():
    data = request.get_json(force=True)
    if missing := _missing(data, "new_password_hash", "target_username"):
        return jsonify({"error": f"缺少必填字段: {', '.join(missing)}"}), 400
    conn = _init_db()
    cursor = conn.execute("UPDATE users SET password_hash=? WHERE username=?", (data["new_password_hash"], data["target_username"]))
    ok = cursor.rowcount > 0
    conn.commit()
    conn.close()
    if ok:
        log.info(f"密码重置: {data['target_username']}")
    return jsonify({"ok": ok})
#
#
@user_bp.route("/users/toggle_active", methods=["POST"])
@require_auth
def api_users_toggle_active():
    data = request.get_json(force=True)
    if missing := _missing(data, "is_active", "target_username"):
        return jsonify({"error": f"缺少必填字段: {', '.join(missing)}"}), 400
    conn = _init_db()
    cursor = conn.execute("UPDATE users SET is_active=? WHERE username=?", (1 if data["is_active"] else 0, data["target_username"]))
    ok = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return jsonify({"ok": ok})
#
#
@user_bp.route("/users/update_display_name", methods=["POST"])
@require_auth
def api_users_update_display_name():
    data = request.get_json(force=True)
    if missing := _missing(data, "new_display_name", "target_username"):
        return jsonify({"error": f"缺少必填字段: {', '.join(missing)}"}), 400
    conn = _init_db()
    cursor = conn.execute("UPDATE users SET display_name=? WHERE username=?", (data["new_display_name"], data["target_username"]))
    ok = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return jsonify({"ok": ok})

# %% [markdown]
# # 聊天会话端点
#
# %%
@user_bp.route("/chat_sessions/<int:user_id>", methods=["GET"])
@require_auth
def api_chat_sessions_list(user_id: int):
    conn = _init_db()
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT cs.id, cs.session_id, cs.name, cs.is_active, cs.created_at, cs.updated_at, "
        "(SELECT COUNT(*) FROM qa_history WHERE session_id=cs.session_id) as message_count "
        "FROM chat_sessions cs WHERE cs.user_id=? ORDER BY cs.updated_at DESC",
        (user_id,),
    ).fetchall()
    conn.close()
    return jsonify({"sessions": [dict(r) for r in rows]})
#
#
@user_bp.route("/chat_sessions/create", methods=["POST"])
@require_auth
def api_chat_sessions_create():
    data = request.get_json(force=True)
    if missing := _missing(data, "user_id"):
        return jsonify({"error": f"缺少必填字段: {', '.join(missing)}"}), 400
    session_id = f"chat_{data['user_id']}_{secrets.token_urlsafe(16)}"
    conn = _init_db()
    conn.execute("INSERT INTO chat_sessions (user_id, session_id, name) VALUES (?,?,?)",
                 (data["user_id"], session_id, data.get("name", "新对话")))
    conn.commit()
    conn.close()
    return jsonify({"ok": True, "session_id": session_id})
#
#
@user_bp.route("/chat_sessions/create_with_id", methods=["POST"])
@require_auth
def api_chat_sessions_create_with_id():
    data = request.get_json(force=True)
    if missing := _missing(data, "user_id", "session_id"):
        return jsonify({"error": f"缺少必填字段: {', '.join(missing)}"}), 400
    conn = _init_db()
    conn.execute("INSERT OR IGNORE INTO chat_sessions (user_id, session_id, name) VALUES (?,?,?)",
                 (data["user_id"], data["session_id"], data.get("name", "默认对话")))
    conn.commit()
    conn.close()
    return jsonify({"ok": True})
#
#
@user_bp.route("/chat_sessions/rename", methods=["POST"])
@require_auth
def api_chat_sessions_rename():
    data = request.get_json(force=True)
    if missing := _missing(data, "new_name", "session_id"):
        return jsonify({"error": f"缺少必填字段: {', '.join(missing)}"}), 400
    conn = _init_db()
    cursor = conn.execute("UPDATE chat_sessions SET name=?, updated_at=? WHERE session_id=?",
                          (data["new_name"], datetime.now().isoformat(), data["session_id"]))
    ok = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return jsonify({"ok": ok})
#
#
@user_bp.route("/chat_sessions/delete", methods=["POST"])
@require_auth
def api_chat_sessions_delete():
    data = request.get_json(force=True)
    if missing := _missing(data, "session_id"):
        return jsonify({"error": f"缺少必填字段: {', '.join(missing)}"}), 400
    conn = _init_db()
    cursor = conn.execute("DELETE FROM chat_sessions WHERE session_id=?", (data["session_id"],))
    ok = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return jsonify({"ok": ok})
#
#
@user_bp.route("/chat_sessions/set_active", methods=["POST"])
@require_auth
def api_chat_sessions_set_active():
    data = request.get_json(force=True)
    if missing := _missing(data, "user_id", "session_id"):
        return jsonify({"error": f"缺少必填字段: {', '.join(missing)}"}), 400
    conn = _init_db()
    conn.execute("UPDATE chat_sessions SET is_active=0 WHERE user_id=?", (data["user_id"],))
    conn.execute("UPDATE chat_sessions SET is_active=1, updated_at=? WHERE session_id=?",
                 (datetime.now().isoformat(), data["session_id"]))
    conn.commit()
    conn.close()
    return jsonify({"ok": True})
#
#
@user_bp.route("/chat_sessions/<int:user_id>/active", methods=["GET"])
@require_auth
def api_chat_sessions_active(user_id: int):
    conn = _init_db()
    row = conn.execute("SELECT session_id FROM chat_sessions WHERE user_id=? AND is_active=1", (user_id,)).fetchone()
    if not row:
        row = conn.execute("SELECT session_id FROM chat_sessions WHERE user_id=? ORDER BY updated_at DESC LIMIT 1", (user_id,)).fetchone()
    conn.close()
    if row:
        return jsonify({"found": True, "session_id": row[0]})
    return jsonify({"found": False}), 404

# %% [markdown]
# # 问答历史端点
#
# %%
@user_bp.route("/qa/save", methods=["POST"])
@require_auth
def api_qa_save():
    data = request.get_json(force=True)
    if missing := _missing(data, "user_id", "session_id", "question", "answer"):
        return jsonify({"error": f"缺少必填字段: {', '.join(missing)}"}), 400
    metadata_json = json.dumps(data.get("metadata")) if data.get("metadata") else None
    conn = _init_db()
    conn.execute("INSERT INTO qa_history (user_id, session_id, question, answer, metadata) VALUES (?,?,?,?,?)",
                 (data["user_id"], data["session_id"], data["question"], data["answer"], metadata_json))
    conn.commit()
    conn.close()
    return jsonify({"ok": True})
#
#
@user_bp.route("/qa/<int:user_id>", methods=["GET"])
@require_auth
def api_qa_history(user_id: int):
    session_id = request.args.get("session_id")
    limit = request.args.get("limit", 50, type=int)
    offset = request.args.get("offset", 0, type=int)
    conn = _init_db()
    conn.row_factory = sqlite3.Row
    if session_id:
        rows = conn.execute(
            "SELECT session_id, question, answer, metadata, created_at FROM qa_history "
            "WHERE session_id=? ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (session_id, limit, offset),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT session_id, question, answer, metadata, created_at FROM qa_history "
            "WHERE user_id=? ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (user_id, limit, offset),
        ).fetchall()
    conn.close()
    history = []
    for row in rows:
        item = dict(row)
        if item["metadata"]:
            try:
                item["metadata"] = json.loads(item["metadata"])
            except Exception:
                item["metadata"] = {}
        history.append(item)
    return jsonify({"history": history})
#
#
@user_bp.route("/qa/by_session/<session_id>", methods=["GET"])
@require_auth
def api_qa_by_session(session_id: str):
    conn = _init_db()
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT question, answer, created_at FROM qa_history WHERE session_id=? ORDER BY created_at ASC",
        (session_id,),
    ).fetchall()
    conn.close()
    return jsonify({"history": [{"timestamp": r["created_at"], "question": r["question"], "answer": r["answer"]} for r in rows]})

# %% [markdown]
# # 审计日志端点
#
# %%
@user_bp.route("/audit/log", methods=["POST"])
@require_auth
def api_audit_log():
    data = request.get_json(force=True)
    if missing := _missing(data, "action"):
        return jsonify({"error": f"缺少必填字段: {', '.join(missing)}"}), 400
    conn = _init_db()
    conn.execute("INSERT INTO audit_log (user_id, action, details, ip_address) VALUES (?,?,?,?)",
                 (data.get("user_id"), data["action"], data.get("details", ""), data.get("ip_address", "")))
    conn.commit()
    conn.close()
    return jsonify({"ok": True})
#
#
@user_bp.route("/audit/logs", methods=["GET"])
@require_auth
def api_audit_logs():
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 50, type=int)
    username = request.args.get("username")
    action = request.args.get("action")
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")
    conn = _init_db()
    conn.row_factory = sqlite3.Row
    conditions, params = [], []
    if username:
        conditions.append("a.user_id IN (SELECT id FROM users WHERE username LIKE ?)")
        params.append(f"%{username}%")
    if action:
        conditions.append("a.action=?"); params.append(action)
    if start_date:
        conditions.append("a.timestamp >= ?"); params.append(start_date)
    if end_date:
        conditions.append("a.timestamp <= ?"); params.append(f"{end_date} 23:59:59")
    where_clause = " AND ".join(conditions) if conditions else "1=1"
    total = conn.execute(f"SELECT COUNT(*) FROM audit_log a WHERE {where_clause}", params).fetchone()[0]
    offset = (page - 1) * per_page
    rows = conn.execute(
        f"SELECT a.id, a.user_id, COALESCE(u.username, '(已删除用户)') as username, "
        f"COALESCE(u.display_name, '未知') as display_name, a.action, a.details, a.ip_address, a.timestamp "
        f"FROM audit_log a LEFT JOIN users u ON a.user_id=u.id "
        f"WHERE {where_clause} ORDER BY a.timestamp DESC LIMIT ? OFFSET ?",
        params + [per_page, offset],
    ).fetchall()
    conn.close()
    return jsonify({"total": total, "logs": [dict(r) for r in rows], "page": page, "per_page": per_page,
                    "total_pages": max(1, (total + per_page - 1) // per_page)})
#
#
@user_bp.route("/audit/actions", methods=["GET"])
@require_auth
def api_audit_actions():
    conn = _init_db()
    rows = conn.execute("SELECT DISTINCT action FROM audit_log ORDER BY action").fetchall()
    conn.close()
    return jsonify({"actions": [r[0] for r in rows]})
#
#
@user_bp.route("/audit/clear", methods=["POST"])
@require_auth
def api_audit_clear():
    data = request.get_json(force=True)
    cutoff = datetime.now() - timedelta(days=data.get("before_days", 90))
    conn = _init_db()
    cursor = conn.execute("DELETE FROM audit_log WHERE timestamp < ?", (cutoff.strftime("%Y-%m-%d %H:%M:%S"),))
    deleted = cursor.rowcount
    conn.commit()
    conn.close()
    log.info(f"审计日志清理: {deleted}条 (早于{data.get('before_days', 90)}天)")
    return jsonify({"ok": True, "deleted": deleted})
