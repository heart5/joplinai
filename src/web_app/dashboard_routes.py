# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     split_at_heading: true
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
# # Dashboard Blueprint — 综合运行面板

# %%
import re
import subprocess
from datetime import datetime

import requests
from flask import Blueprint, current_app, jsonify, render_template, request, session

from func.jpfuncs import getinivaluefromcloud

# %%
from src.web_app.auth import admin_required, login_required

__all__ = ["dashboard_bp"]

dashboard_bp = Blueprint("dashboard", __name__, url_prefix="/admin/panel")

def _hcx_service_status():
    """动态发现恒创云本地 systemd 服务。"""
    patterns = "joplinai-* monitor-* wc-* jupyterhub* apache2* fail2ban* ssh* docker*"
    try:
        r = subprocess.run(
            f"systemctl list-unit-files --no-legend {patterns} 2>/dev/null "
            "| awk '{print $1}' | sed 's/\.service$//'",
            shell=True, capture_output=True, text=True, timeout=10,
        )
    except Exception:
        return {}
    result = {}
    for svc in r.stdout.splitlines():
        svc = svc.strip()
        if not svc or "@" in svc or svc.endswith((".timer", ".socket", ".scope")):
            continue
        try:
            r_type = subprocess.run(
                ["systemctl", "show", "-p", "Type", svc],
                capture_output=True, text=True, timeout=5,
            )
            if r_type.stdout.strip() == "Type=oneshot":
                continue
        except Exception:
            pass
        try:
            r2 = subprocess.run(
                ["systemctl", "is-active", svc],
                capture_output=True, text=True, timeout=5,
            )
            status = r2.stdout.strip() if r2.returncode == 0 else "inactive"
        except Exception:
            result[svc] = "unknown"
            continue
        result[svc] = {"status": status}
        if status == "active":
            try:
                r3 = subprocess.run(
                    ["systemctl", "show", "-p", "ActiveEnterTimestamp", svc],
                    capture_output=True, text=True, timeout=5,
                )
                ts_str = r3.stdout.strip().replace("ActiveEnterTimestamp=", "")
                ts = datetime.strptime(ts_str.rsplit(" ", 1)[0], "%a %Y-%m-%d %H:%M:%S")
                diff = datetime.now() - ts
                days, sec = diff.days, diff.seconds
                hours, minutes = sec // 3600, (sec % 3600) // 60
                if days > 0:
                    dur = f"{days}天{hours}小时"
                elif hours > 0:
                    dur = f"{hours}小时{minutes}分钟"
                else:
                    dur = f"{minutes}分钟"
                result[svc]["since"] = ts.strftime("%Y-%m-%d %H:%M")
                result[svc]["duration"] = dur
            except Exception:
                pass
    return result



def _tc_get(path, timeout=15, params=None):
    """调用 tc center_api，返回 JSON 或错误信息。"""
    url = f"{current_app.config['TC_API_URL']}{path}"
    key = current_app.config.get("TC_API_KEY", "")
    try:
        r = requests.get(url, headers={"X-API-Key": key}, timeout=timeout, params=params)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.Timeout:
        return {"status": "unreachable", "error": "连接腾讯云超时"}
    except requests.exceptions.ConnectionError:
        return {"status": "unreachable", "error": "无法连接腾讯云"}
    except requests.exceptions.HTTPError as e:
        return {"status": "error", "error": f"HTTP {e.response.status_code if e.response else '?'}"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


    return result


def _hcx_resources():
    """恒创云本地 CPU/内存/磁盘。"""
    try:
        r = subprocess.run(["uptime", "-p"], capture_output=True, text=True, timeout=5)
        uptime = r.stdout.strip().replace("up ", "")
    except Exception:
        uptime = "N/A"

    try:
        r = subprocess.run(["cat", "/proc/loadavg"], capture_output=True, text=True, timeout=5)
        parts = r.stdout.split()
        load_1 = float(parts[0]) if len(parts) > 0 else 0
    except Exception:
        load_1 = 0

    try:
        r = subprocess.run(["nproc"], capture_output=True, text=True, timeout=5)
        cores = int(r.stdout.strip())
    except Exception:
        cores = 1

    cpu_pct = round(load_1 / cores * 100, 1) if cores else 0

    try:
        r = subprocess.run(
            ["free", "-m"],
            capture_output=True, text=True, timeout=5,
        )
        for line in r.stdout.splitlines():
            if line.startswith("Mem:"):
                parts = line.split()
                mem_total = int(parts[1])
                mem_used = int(parts[2])
                mem_avail = int(parts[6]) if len(parts) > 6 else int(parts[3])
                mem_pct = round(mem_used / mem_total * 100, 1)
                break
        else:
            mem_total = mem_used = mem_avail = mem_pct = 0
    except Exception:
        mem_total = mem_used = mem_avail = mem_pct = 0

    try:
        r = subprocess.run(
            ["df", "-h", "/"],
            capture_output=True, text=True, timeout=5,
        )
        for line in r.stdout.splitlines():
            parts = line.split()
            if parts[0].startswith("/dev"):
                disk_total = parts[1]
                disk_used = parts[2]
                disk_avail = parts[3]
                disk_pct = parts[4].rstrip("%")
                break
        else:
            disk_total = disk_used = disk_avail = disk_pct = "N/A"
    except Exception:
        disk_total = disk_used = disk_avail = disk_pct = "N/A"

    return {
        "uptime": uptime,
        "cpu": {"load_1": load_1, "cores": cores, "pct": cpu_pct},
        "memory": {"total_mb": mem_total, "used_mb": mem_used, "available_mb": mem_avail, "used_pct": mem_pct},
        "disk": {"total": disk_total, "used": disk_used, "avail": disk_avail, "used_pct": disk_pct},
    }


# %% [markdown]
# # 页面路由

# %%
@dashboard_bp.route("")
@login_required
@admin_required
def panel_overview():
    return render_template("admin/panel_overview.html", user=session["user"])


@dashboard_bp.route("/monitor")
@login_required
@admin_required
def panel_monitor():
    person_list = getinivaluefromcloud("monitor", "person_list") or ""
    team_persons = [n.strip() for n in re.split(r"[,，]", person_list) if n.strip()]
    return render_template(
        "admin/panel_monitor.html",
        user=session["user"],
        team_persons=team_persons,
    )


@dashboard_bp.route("/wechat")
@login_required
@admin_required
def panel_wechat():
    return render_template("admin/panel_wechat.html", user=session["user"])


@dashboard_bp.route("/system")
@login_required
@admin_required
def panel_system():
    return render_template("admin/panel_system.html", user=session["user"])


# %% [markdown]
# # JSON API 路由（AJAX 调用）

# %%
@dashboard_bp.route("/api/status")
@login_required
@admin_required
def api_status():
    """聚合 tc + hcx 全部状态。"""
    tc_monitor = _tc_get("/monitor/status")
    tc_health = _tc_get("/system/health")
    tc_services = _tc_get("/system/services")
    tc_wechat = _tc_get("/system/wechat")
    hcx_services = _hcx_service_status()
    hcx_resources = _hcx_resources()

    return jsonify({
        "timestamp": datetime.now().isoformat(),
        "tc": {
            "monitor": tc_monitor,
            "health": tc_health,
            "services": tc_services,
            "wechat": tc_wechat,
        },
        "hcx": {
            "services": hcx_services,
            "resources": hcx_resources,
        },
    })


@dashboard_bp.route("/api/hcx/status")
@login_required
@admin_required
def api_hcx_status():
    return jsonify({
        "timestamp": datetime.now().isoformat(),
        "services": _hcx_service_status(),
        "resources": _hcx_resources(),
    })


@dashboard_bp.route("/api/monitor/heatmap")
@login_required
@admin_required
def api_monitor_heatmap():
    return jsonify(_tc_get("/monitor/heatmap", params=request.args))


@dashboard_bp.route("/api/wechat/health")
@login_required
@admin_required
def api_wechat_health():
    return jsonify(_tc_get("/system/wechat"))


@dashboard_bp.route("/api/system/info")
@login_required
@admin_required
def api_system_info():
    return jsonify(_tc_get("/system/health"))


@dashboard_bp.route("/api/spark/pool")
@login_required
@admin_required
def api_spark_pool():
    return jsonify(_tc_get("/monitor/spark/pool"))
