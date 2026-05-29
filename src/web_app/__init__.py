"""JoplinAI Web Portal — Flask application factory."""
import os
from datetime import datetime
from pathlib import Path

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
from werkzeug.middleware.proxy_fix import ProxyFix

import pathmagic

with pathmagic.Context():
    from func.datatools import getkeysfromcloud
    from func.first import getdirmain
    from func.getid import getdeviceid
    from func.jpfuncs import getinivaluefromcloud

__all__ = ["create_app"]


def create_app():
    app = Flask(
        __name__,
        template_folder=str(getdirmain() / "templates"),
        static_folder=str(getdirmain() / "static"),
    )
    app.secret_key = (
        getinivaluefromcloud("joplinai", "flask_secret_key") or os.urandom(32).hex()
    )
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=2, x_proto=1, x_host=1)

    # Shared config for route access via current_app.config
    qa_server = getinivaluefromcloud("joplinai", f"joplinai_qa_server_{getdeviceid()}")
    qa_port = getinivaluefromcloud("joplinai", "joplinai_qa_port")
    app.config["QA_API_URL"] = f"http://{qa_server}:{qa_port}"
    app.config["QA_API_KEY"] = getkeysfromcloud().get("hc", "invalid")

    # Register core routes directly on app
    _register_core_routes(app)

    # Register Blueprints
    from src.web_app.admin_routes import admin_bp
    from src.web_app.dashboard_routes import dashboard_bp
    from src.web_app.qa_routes import qa_bp

    app.register_blueprint(qa_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(dashboard_bp)

    app.config["TC_API_URL"] = "https://api.xiloong.fans"
    app.config["TC_API_KEY"] = getkeysfromcloud().get("tc", "")

    return app


def _register_core_routes(app):
    from src.web_app.auth import login_required

    try:
        from src.user_manager import USER_MANAGER
    except ImportError:
        USER_MANAGER = None

    @app.route("/")
    @login_required
    def index():
        sessions = USER_MANAGER.get_user_chat_sessions(session["user"]["id"])
        active_session = USER_MANAGER.get_active_chat_session(session["user"]["id"])
        session["active_session"] = active_session
        return render_template(
            "index.html",
            user=session["user"],
            chat_sessions=sessions,
            active_session=active_session,
        )

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "GET":
            return render_template("login.html")

        username = request.form.get("username")
        password = request.form.get("password")

        if not username or not password:
            return render_template("login.html", error="请输入用户名和密码")

        user = USER_MANAGER.verify_user(username, password)
        if user:
            session_id = USER_MANAGER.create_session(user["id"])
            session["session_id"] = session_id
            session["user"] = user
            USER_MANAGER.log_audit(user["id"], "LOGIN", ip_address=request.remote_addr)
            return redirect(url_for("index"))
        else:
            return render_template("login.html", error="用户名或密码错误")

    @app.route("/logout")
    @login_required
    def logout():
        if "session_id" in session:
            USER_MANAGER.delete_session(session["session_id"])
        session.clear()
        return redirect(url_for("login"))

    @app.route("/api/system/health")
    @login_required
    def system_health():
        import requests

        qa_url = app.config["QA_API_URL"]
        health = {
            "web_app": "healthy",
            "user_db": "connected",
            "timestamp": datetime.now().isoformat(),
        }
        try:
            qa_health = requests.get(f"{qa_url}/health", timeout=5).json()
            health["qa_api"] = qa_health.get("status", "unknown")
        except Exception:
            health["qa_api"] = "unreachable"
        return jsonify(health)

    @app.route("/favicon.ico")
    def favicon():
        return send_from_directory(
            "static", "favicon.ico", mimetype="image/vnd.microsoft.icon"
        )


if __name__ == "__main__":
    template_dir = Path(__file__).resolve().parent.parent.parent / "templates"
    template_dir.mkdir(exist_ok=True)
    app = create_app()
    app.run(host="127.0.0.1", port=5001, debug=False)
