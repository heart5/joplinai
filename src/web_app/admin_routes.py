"""Admin dashboard and user management routes."""
import csv
import io
import re

from flask import Blueprint, Response, abort, jsonify, render_template, request, session

from func.jpfuncs import getinivaluefromcloud
from func.logme import log

from src.web_app.auth import admin_required, login_required

try:
    from src.user_manager import USER_MANAGER
except ImportError:
    USER_MANAGER = None

__all__ = ["admin_bp"]

admin_bp = Blueprint("admin", __name__)


@admin_bp.route("/api/users", methods=["GET"])
@admin_required
def api_users_list():
    users = USER_MANAGER.get_all_users()
    return jsonify({"users": users})


@admin_bp.route("/admin")
@login_required
@admin_required
def admin_dashboard():
    return render_template("admin/users.html", user=session["user"])


@admin_bp.route("/api/admin/users", methods=["GET"])
@admin_required
def api_admin_get_users():
    users = USER_MANAGER.get_all_users()
    safe_users = []
    for u in users:
        safe_users.append({
            "id": u["id"],
            "username": u["username"],
            "display_name": u["display_name"],
            "role": u["role"],
            "is_active": bool(u["is_active"]),
            "created_at": u["created_at"],
            "last_login": u["last_login"],
        })
    return jsonify({"success": True, "users": safe_users})


@admin_bp.route("/api/admin/available-notebooks", methods=["GET"])
@login_required
@admin_required
def api_get_available_notebooks():
    try:
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


@admin_bp.route("/api/admin/user/<username>", methods=["GET"])
@login_required
@admin_required
def api_get_user_details(username):
    user = USER_MANAGER.get_user_with_notebooks(username)
    if not user:
        return jsonify({"success": False, "error": "用户不存在"}), 404
    user.pop("password_hash", None)
    return jsonify({"success": True, "user": user})


@admin_bp.route("/api/admin/user/delete", methods=["POST"])
@login_required
@admin_required
def api_admin_delete_user():
    data = request.get_json()
    target_username = data.get("username")

    if not target_username:
        return jsonify({"success": False, "error": "用户名不能为空"}), 400

    if target_username == session["user"]["username"]:
        return jsonify({"success": False, "error": "不能删除自己的账户"}), 403

    target_user = USER_MANAGER.get_user_by_username(target_username)
    if target_user and target_user["role"] == "admin":
        all_users = USER_MANAGER.get_all_users()
        admin_count = sum(1 for u in all_users if u["role"] == "admin")
        if admin_count <= 1:
            return jsonify({
                "success": False,
                "error": "至少需要保留一名管理员，无法删除最后一个管理员",
            }), 403

    success = USER_MANAGER.delete_user(
        target_username, admin_username=session["user"]["username"]
    )
    if success:
        return jsonify({"success": True, "message": "用户已删除"})
    else:
        return jsonify({"success": False, "error": "用户不存在或操作失败"}), 400


@admin_bp.route("/api/admin/user/update", methods=["POST"])
@login_required
@admin_required
def api_update_user_permissions():
    data = request.get_json()

    required_fields = ["username", "display_name", "role"]
    for field in required_fields:
        if field not in data:
            return jsonify({"success": False, "error": f"缺少必填字段: {field}"}), 400

    valid_roles = ["admin", "team_leader", "team_member"]
    if data["role"] not in valid_roles:
        return jsonify({"success": False, "error": f"无效角色: {data['role']}"}), 400

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


@admin_bp.route("/api/admin/user/reset_password", methods=["POST"])
@admin_required
def api_admin_reset_password():
    data = request.get_json()
    target_username = data.get("username")
    new_password = data.get("new_password")

    if not target_username or not new_password:
        return jsonify({"success": False, "error": "用户名和新密码不能为空"}), 400

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


@admin_bp.route("/api/admin/user/toggle_active", methods=["POST"])
@admin_required
def api_admin_toggle_active():
    data = request.get_json()
    target_username = data.get("username")
    is_active = data.get("is_active")

    if target_username is None or is_active is None:
        return jsonify({"success": False, "error": "参数不完整"}), 400

    if target_username == session["user"]["username"]:
        return jsonify({"success": False, "error": "不能禁用自己的账户"}), 403

    success = USER_MANAGER.update_user_active_status(
        target_username, is_active, admin_username=session["user"]["username"]
    )
    if success:
        return jsonify({"success": True, "message": "用户状态更新成功"})
    else:
        return jsonify({"success": False, "error": "操作失败"}), 400


@admin_bp.route("/api/admin/user/update_role", methods=["POST"])
@admin_required
def api_admin_update_role():
    data = request.get_json()
    target_username = data.get("username")
    new_role = data.get("new_role")

    if not target_username or new_role not in ["admin", "colleague"]:
        return jsonify({"success": False, "error": "参数无效"}), 400

    success = USER_MANAGER.update_user_role(
        target_username, new_role, admin_username=session["user"]["username"]
    )
    if success:
        return jsonify({"success": True, "message": "用户角色更新成功"})
    else:
        return jsonify({"success": False, "error": "操作失败"}), 400


@admin_bp.route("/api/admin/user/update_display_name", methods=["POST"])
@admin_required
def api_admin_update_display_name():
    data = request.get_json()
    target_username = data.get("username")
    new_display_name = data.get("new_display_name")

    if not target_username or not new_display_name or not new_display_name.strip():
        return jsonify({"success": False, "error": "显示名称不能为空"}), 400

    success = USER_MANAGER.update_user_display_name(
        target_username, new_display_name, admin_username=session["user"]["username"]
    )
    if success:
        return jsonify({"success": True, "message": "显示名称更新成功"})
    else:
        return jsonify({"success": False, "error": "操作失败"}), 400


@admin_bp.route("/api/admin/user/create", methods=["POST"])
@admin_required
def api_admin_create_user():
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


@admin_bp.route("/admin/users/<username>/edit", methods=["GET"])
@login_required
@admin_required
def admin_user_edit(username):
    user = USER_MANAGER.get_user_with_notebooks(username)
    if not user:
        abort(404)
    try:
        split_ptn = re.compile(r"[,，]")
        available_notebooks = [
            title.strip()
            for title in split_ptn.split(
                getinivaluefromcloud("joplinai", "shared_notebook_titles")
            )
        ]
    except Exception:
        available_notebooks = []
    return render_template(
        "admin/user_edit.html", user=user, available_notebooks=available_notebooks
    )


@admin_bp.route("/admin/users/<username>/history")
@login_required
@admin_required
def admin_user_history(username):
    user = USER_MANAGER.get_user_by_username(username)
    if not user:
        abort(404)
    sessions = USER_MANAGER.get_user_chat_sessions(user["id"])
    return render_template("admin/user_history.html", user=user, sessions=sessions)


@admin_bp.route("/api/admin/user/<username>/session/<session_id>/history")
@login_required
@admin_required
def api_admin_user_session_history(username, session_id):
    user = USER_MANAGER.get_user_by_username(username)
    if not user:
        return jsonify({"success": False, "error": "用户不存在"}), 404
    limit = request.args.get("limit", 50, type=int)
    offset = request.args.get("offset", 0, type=int)
    history = USER_MANAGER.get_qa_history(
        user_id=user["id"], session_id=session_id, limit=limit, offset=offset
    )
    return jsonify({"success": True, "history": history})


# --- Audit routes ---

@admin_bp.route("/admin/audit")
@login_required
@admin_required
def admin_audit_page():
    return render_template("admin/audit_log.html")


@admin_bp.route("/api/admin/audit/logs")
@login_required
@admin_required
def api_admin_audit_logs():
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 20, type=int)
    username = request.args.get("username", None)
    action = request.args.get("action", None)
    start_date = request.args.get("start_date", None)
    end_date = request.args.get("end_date", None)

    result = USER_MANAGER.get_audit_logs(
        page=page, per_page=per_page,
        username=username, action=action,
        start_date=start_date, end_date=end_date,
    )
    return jsonify({"success": True, **result})


@admin_bp.route("/api/admin/audit/actions")
@login_required
@admin_required
def api_admin_audit_actions():
    actions = USER_MANAGER.get_audit_actions()
    return jsonify({"success": True, "actions": actions})


@admin_bp.route("/api/admin/audit/clear", methods=["POST"])
@login_required
@admin_required
def api_admin_audit_clear():
    data = request.get_json()
    before_days = data.get("before_days", 90)
    if before_days < 7:
        return jsonify({"success": False, "error": "保留天数不能少于7天"}), 400

    deleted = USER_MANAGER.clear_audit_logs(before_days)
    USER_MANAGER.log_audit(
        session["user"]["id"],
        "CLEAR_AUDIT_LOGS",
        details=f"清理了 {deleted} 条 {before_days} 天前的审计日志",
        ip_address=request.remote_addr,
    )
    return jsonify(
        {"success": True, "deleted": deleted, "message": f"已清理 {deleted} 条日志"}
    )


@admin_bp.route("/api/admin/audit/export")
@login_required
@admin_required
def api_admin_audit_export():
    result = USER_MANAGER.get_audit_logs(
        page=1, per_page=100000,
        username=request.args.get("username"),
        action=request.args.get("action"),
        start_date=request.args.get("start_date"),
        end_date=request.args.get("end_date"),
    )

    if not result["logs"]:
        return jsonify({"success": False, "error": "没有可导出的记录"}), 400

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "时间", "用户名", "显示名", "操作类型", "详情", "IP地址"])
    for log_entry in result["logs"]:
        writer.writerow([
            log_entry["id"], log_entry["timestamp"],
            log_entry["username"], log_entry["display_name"],
            log_entry["action"], log_entry["details"] or "",
            log_entry["ip_address"] or "",
        ])

    csv_content = output.getvalue()
    output.close()

    return Response(
        csv_content,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=audit_logs.csv"},
    )
