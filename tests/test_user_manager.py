"""UserManager 单元测试 — 直接实例化，走本地 SQLite。"""
import pytest
from src.user_manager import UserManager


@pytest.fixture
def um(user_db_path):
    """返回一个连接临时 DB 的 UserManager。"""
    return UserManager(user_db_path)


class TestUserManager:
    def test_init_creates_tables(self, um):
        """初始化后 users 表为空。"""
        users = um.get_all_users()
        assert users == []

    def test_create_user(self, um):
        ok = um.create_user("alice", "pass123", "Alice Wang", role="team_member")
        assert ok is True

        user = um.verify_user("alice", "pass123")
        assert user is not None
        assert user["username"] == "alice"
        assert user["display_name"] == "Alice Wang"
        assert user["role"] == "team_member"

    def test_verify_user_wrong_password(self, um):
        um.create_user("bob", "correct", "Bob")
        assert um.verify_user("bob", "wrong") is None

    def test_verify_user_inactive(self, um):
        um.create_user("charlie", "pass", "Charlie")
        um.update_user_active_status("charlie", False, "admin")

        assert um.verify_user("charlie", "pass") is None

    def test_duplicate_username(self, um):
        um.create_user("eve", "pass1", "Eve")
        ok = um.create_user("eve", "pass2", "Eve2")
        assert ok is False

    def test_create_and_validate_session(self, um):
        um.create_user("dave", "pass", "Dave")
        user = um.verify_user("dave", "pass")
        sid = um.create_session(user["id"])
        assert len(sid) > 20

        session_user = um.validate_session(sid)
        assert session_user is not None
        assert session_user["username"] == "dave"

    def test_validate_invalid_session(self, um):
        assert um.validate_session("nonexistent_session_id") is None

    def test_delete_session(self, um):
        um.create_user("frank", "pass", "Frank")
        user = um.verify_user("frank", "pass")
        sid = um.create_session(user["id"])

        um.delete_session(sid)
        assert um.validate_session(sid) is None

    def test_reset_password(self, um):
        um.create_user("grace", "oldpass", "Grace")
        ok = um.reset_user_password("grace", "newpass", "admin")
        assert ok is True
        assert um.verify_user("grace", "oldpass") is None
        assert um.verify_user("grace", "newpass") is not None

    def test_update_role(self, um):
        um.create_user("henry", "pass", "Henry", role="team_member")
        ok = um.update_user_role("henry", "team_leader", "admin")
        assert ok is True

        user = um.get_user_by_username("henry")
        assert user["role"] == "team_leader"

    def test_update_role_invalid(self, um):
        um.create_user("iris", "pass", "Iris")
        ok = um.update_user_role("iris", "superadmin", "admin")
        assert ok is False

    def test_update_display_name(self, um):
        um.create_user("jack", "pass", "Jack Old")
        ok = um.update_user_display_name("jack", "Jack New", "admin")
        assert ok is True

        user = um.get_user_by_username("jack")
        assert user["display_name"] == "Jack New"

    def test_delete_user(self, um):
        um.create_user("kate", "pass", "Kate")
        ok = um.delete_user("kate", "admin")
        assert ok is True
        assert um.get_user_by_username("kate") is None

    def test_delete_nonexistent_user(self, um):
        assert um.delete_user("nobody", "admin") is False

    def test_chat_session_crud(self, um):
        um.create_user("leo", "pass", "Leo")
        user = um.verify_user("leo", "pass")

        sid = um.create_chat_session(user["id"], "测试对话")
        assert sid.startswith("chat_")

        sessions = um.get_user_chat_sessions(user["id"])
        assert len(sessions) == 1
        assert sessions[0]["name"] == "测试对话"

    def test_rename_chat_session(self, um):
        um.create_user("mia", "pass", "Mia")
        user = um.verify_user("mia", "pass")
        sid = um.create_chat_session(user["id"], "旧名称")

        ok = um.rename_chat_session(sid, "新名称")
        assert ok is True

        sessions = um.get_user_chat_sessions(user["id"])
        assert sessions[0]["name"] == "新名称"

    def test_delete_chat_session(self, um):
        um.create_user("nick", "pass", "Nick")
        user = um.verify_user("nick", "pass")
        sid = um.create_chat_session(user["id"])

        ok = um.delete_chat_session(sid)
        assert ok is True
        assert um.get_user_chat_sessions(user["id"]) == []

    def test_get_user_with_notebooks(self, um):
        um.create_user("olivia", "pass", "Olivia", allowed_notebooks=["nb1", "nb2"])
        user = um.get_user_with_notebooks("olivia")
        assert user["allowed_notebooks"] == ["nb1", "nb2"]

    def test_save_and_get_qa_history(self, um):
        um.create_user("paul", "pass", "Paul")
        user = um.verify_user("paul", "pass")

        um.save_qa_history(
            user["id"], "session_x", "什么是Joplin?", "Joplin是笔记软件", {"source": "test"}
        )
        history = um.get_qa_history(user["id"])
        assert len(history) == 1
        assert history[0]["question"] == "什么是Joplin?"
        assert history[0]["metadata"]["source"] == "test"

    def test_audit_log(self, um):
        um.create_user("quinn", "pass", "Quinn")
        user = um.verify_user("quinn", "pass")
        um.log_audit(user["id"], "TEST_ACTION", "test details", "127.0.0.1")

        logs = um.get_audit_logs()
        assert logs["total"] >= 1

    def test_get_audit_actions(self, um):
        um.create_user("rachel", "pass", "Rachel")
        user = um.verify_user("rachel", "pass")
        um.log_audit(user["id"], "LOGIN", "login", "")
        um.log_audit(user["id"], "LOGOUT", "logout", "")

        actions = um.get_audit_actions()
        assert "LOGIN" in actions
        assert "LOGOUT" in actions
