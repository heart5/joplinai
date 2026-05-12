"""QA and chat session routes."""
import requests
from flask import Blueprint, jsonify, request, session

from func.logme import log

from src.web_app.auth import login_required

try:
    from src.user_manager import USER_MANAGER
except ImportError:
    USER_MANAGER = None

__all__ = ["qa_bp"]

qa_bp = Blueprint("qa", __name__)


def _get_qa_config():
    from flask import current_app
    return current_app.config["QA_API_URL"], current_app.config["QA_API_KEY"]


def _restore_history_for_session(session_id: str):
    try:
        if USER_MANAGER is None:
            return
        rows = USER_MANAGER.get_qa_history_by_session(session_id)
        if not rows:
            return
        history = []
        for row in rows:
            history.append({
                "timestamp": row["timestamp"],
                "question": row["question"],
                "answer": row["answer"],
                "metadata": {},
            })
        qa_url, qa_key = _get_qa_config()
        requests.post(
            f"{qa_url}/restore_history",
            json={"session_id": session_id, "history": history},
            headers={"X-API-Key": qa_key},
            timeout=10,
        )
    except Exception as e:
        log.warning(f"恢复会话历史失败: {e}")


@qa_bp.route("/api/ask", methods=["POST"])
@login_required
def api_ask():
    data = request.get_json()
    question = data.get("question", "").strip()
    use_history = data.get("use_history", True)

    if not question:
        return jsonify({"error": "问题不能为空"}), 400

    session_id = data.get("session_id") or session.get("active_session")
    if not session_id:
        session_id = USER_MANAGER.get_active_chat_session(session["user"]["id"])
    user_sessions = [
        s["session_id"]
        for s in USER_MANAGER.get_user_chat_sessions(session["user"]["id"])
    ]
    if session_id not in user_sessions:
        return jsonify({"error": "会话不存在"}), 404

    qa_request_payload = {
        "question": question,
        "use_history": use_history,
        "session_id": session_id,
        "config_overrides": {},
        "user_identity": {
            "username": session["user"]["username"],
            "display_name": session["user"]["display_name"],
            "role": session["user"]["role"],
            "allowed_notebooks": USER_MANAGER.get_user_with_notebooks(
                session["user"]["username"]
            ).get("allowed_notebooks", ""),
        },
    }

    qa_url, qa_key = _get_qa_config()
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": qa_key,
    }

    try:
        response = requests.post(
            f"{qa_url}/ask",
            json=qa_request_payload,
            headers=headers,
            timeout=90,
        )
        response.raise_for_status()
        result = response.json()

        metadata = result.get("metadata", {})
        sources_notes = metadata.get("sources", [])
        relevant_chunks = metadata.get("relevant_chunks", [])
        is_based = metadata.get("is_based_on_notes", False)
        USER_MANAGER.log_audit(
            session["user"]["id"],
            "ASK_QUESTION",
            details=f"问题长度: {len(question)}, "
            f"来源笔记数: {len(sources_notes)}, "
            f"相关块数: {len(relevant_chunks)}, "
            f"基于笔记: {is_based}",
            ip_address=request.remote_addr,
        )

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

        return jsonify(result)
    except requests.exceptions.RequestException as e:
        USER_MANAGER.log_audit(
            session["user"]["id"],
            "ASK_QUESTION_FAILED",
            details=f"问题长度: {len(question)}, 错误: {str(e)[:100]}",
            ip_address=request.remote_addr,
        )
        log.error(f"调用QA API失败: {e}")
        return jsonify({"error": "问答服务暂时不可用", "details": str(e)}), 502


@qa_bp.route("/api/history", methods=["GET"])
@login_required
def api_get_history():
    try:
        session_id = request.args.get("session_id")
        if not session_id:
            session_id = USER_MANAGER.get_active_chat_session(session["user"]["id"])
            if not session_id:
                return jsonify({"success": False, "history": []})
        limit = request.args.get("limit", default=20, type=int)
        offset = request.args.get("offset", default=0, type=int)

        history = USER_MANAGER.get_qa_history(
            user_id=session["user"]["id"],
            limit=min(limit, 100),
            offset=offset,
            session_id=session_id,
        )

        return jsonify(
            {"success": True, "history": history, "total": len(history)}
        ), 200

    except Exception as e:
        log.error(f"获取用户历史失败: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@qa_bp.route("/api/chat_sessions", methods=["GET", "POST"])
@login_required
def chat_sessions_api():
    if request.method == "POST":
        data = request.get_json()
        name = data.get("name", "新对话")
        session_id = USER_MANAGER.create_chat_session(session["user"]["id"], name)
        USER_MANAGER.set_active_chat_session(session["user"]["id"], session_id)
        return jsonify({"success": True, "session_id": session_id, "name": name})
    else:
        sessions = USER_MANAGER.get_user_chat_sessions(session["user"]["id"])
        return jsonify({"success": True, "sessions": sessions})


@qa_bp.route("/api/chat_sessions/<session_id>", methods=["PUT", "DELETE"])
@login_required
def chat_session_detail(session_id):
    user_sessions = [
        s["session_id"]
        for s in USER_MANAGER.get_user_chat_sessions(session["user"]["id"])
    ]
    if session_id not in user_sessions:
        return jsonify({"error": "无权操作"}), 403

    if request.method == "PUT":
        data = request.get_json()
        new_name = data.get("name")
        if not new_name:
            return jsonify({"error": "名称不能为空"}), 400
        USER_MANAGER.rename_chat_session(session_id, new_name)
        return jsonify({"success": True})
    elif request.method == "DELETE":
        USER_MANAGER.delete_chat_session(session_id)
        if session.get("active_session") == session_id:
            USER_MANAGER.get_active_chat_session(session["user"]["id"])
        return jsonify({"success": True})


@qa_bp.route("/api/chat_sessions/<session_id>/activate", methods=["POST"])
@login_required
def activate_chat_session(session_id):
    user_sessions = [
        s["session_id"]
        for s in USER_MANAGER.get_user_chat_sessions(session["user"]["id"])
    ]
    if session_id not in user_sessions:
        return jsonify({"error": "无权操作"}), 403
    USER_MANAGER.set_active_chat_session(session["user"]["id"], session_id)
    _restore_history_for_session(session_id)
    return jsonify({"success": True, "session_id": session_id})
