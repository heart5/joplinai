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
# # System Blueprint — tc 系统状态 API

# %%
import os
import re
import subprocess
from datetime import datetime, timedelta

from flask import Blueprint, jsonify

# %%
from aimod.center_api import log, require_auth

__all__ = ["system_bp"]

system_bp = Blueprint("system", __name__)

HAPPYJOPLIN = "/home/baiyefeng/codebase/happyjoplin"


def _run(cmd, timeout=15):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip(), r.returncode
    except subprocess.TimeoutExpired:
        return "", -1
    except Exception:
        return "", -1


# %% [markdown]
# # /system/health — CPU/内存/磁盘

# %%
@system_bp.route("/system/health")
@require_auth
def system_health():
    # uptime
    uptime_out, _ = _run("uptime -p")
    uptime_str = uptime_out.replace("up ", "") if uptime_out else "N/A"

    # CPU: /proc/loadavg
    load_out, _ = _run("cat /proc/loadavg")
    parts = load_out.split() if load_out else []
    load_1 = float(parts[0]) if len(parts) > 0 else 0.0
    load_5 = float(parts[1]) if len(parts) > 1 else 0.0
    load_15 = float(parts[2]) if len(parts) > 2 else 0.0

    # CPU cores
    nproc_out, _ = _run("nproc")
    cpu_cores = int(nproc_out) if nproc_out else 1
    cpu_pct = round(load_1 / cpu_cores * 100, 1)

    # Memory
    mem_out, _ = _run("free -m | awk '/^Mem:/ {print $2,$3,$4,$6}'")
    if mem_out:
        mem_parts = mem_out.split()
        mem_total = int(mem_parts[0])
        mem_used = int(mem_parts[1])
        mem_free = int(mem_parts[2])
        mem_buff_cache = int(mem_parts[3]) if len(mem_parts) > 3 else 0
        mem_available = mem_free + mem_buff_cache
        mem_used_pct = round((mem_used - mem_buff_cache) / mem_total * 100, 1)
    else:
        mem_total = mem_used = mem_free = mem_available = mem_used_pct = 0

    # Disk
    disk_out, _ = _run("df -h / | awk 'NR==2 {print $2,$3,$4,$5}'")
    if disk_out:
        disk_parts = disk_out.split()
        disk_total = disk_parts[0]
        disk_used = disk_parts[1]
        disk_avail = disk_parts[2]
        disk_pct = disk_parts[3].rstrip("%")
    else:
        disk_total = disk_used = disk_avail = "N/A"
        disk_pct = 0

    return jsonify({
        "uptime": uptime_str,
        "cpu": {
            "load_1": load_1, "load_5": load_5, "load_15": load_15,
            "cores": cpu_cores, "pct": cpu_pct,
        },
        "memory": {
            "total_mb": mem_total, "used_mb": mem_used,
            "free_mb": mem_free, "available_mb": mem_available,
            "used_pct": mem_used_pct,
        },
        "disk": {
            "total": disk_total, "used": disk_used,
            "avail": disk_avail, "used_pct": disk_pct,
        },
    })


# %% [markdown]
# # /system/wechat — webchat 进程 + 日志扫描

# %%
@system_bp.route("/system/wechat")
@require_auth
def system_wechat():
    # 进程状态
    ps_out, _ = _run(
        "ps -eo pid,args | grep 'webchat.py' | grep -v grep | head -1"
    )
    proc_ok = False
    pid = rss_mb = vsz_mb = cpu = "N/A"
    session_start = duration = "N/A"
    start_ts = None

    if ps_out:
        wpid = ps_out.split()[0]
        ps_info, _ = _run(
            f"ps -o pid,rssize,vsize,pcpu,lstart= -p {wpid} --no-headers 2>/dev/null"
        )
        if ps_info:
            parts = ps_info.split()
            try:
                pid = int(parts[0])
                rss_kb = int(parts[1])
                rss_mb = round(rss_kb / 1024, 1)
                vsz_kb = int(parts[2])
                vsz_mb = round(vsz_kb / 1024, 1)
                cpu = parts[3]
                lstart = " ".join(parts[4:])
                # Parse lstart like "Thu May 15 22:05:42 2025"
                try:
                    st = datetime.strptime(lstart, "%a %b %d %H:%M:%S %Y")
                except ValueError:
                    st = datetime.strptime(lstart, "%a %b  %d %H:%M:%S %Y")
                session_start = st.strftime("%Y-%m-%d %H:%M:%S")
                start_ts = st.timestamp()
                now_ts = datetime.now().timestamp()
                dur_sec = int(now_ts - start_ts)
                dur_d = dur_sec // 86400
                dur_h = (dur_sec % 86400) // 3600
                dur_m = (dur_sec % 3600) // 60
                duration = f"{dur_d}天{dur_h}小时{dur_m}分钟"
                proc_ok = True
            except (ValueError, IndexError):
                pass

    # 日志扫描
    logdir = os.path.join(HAPPYJOPLIN, "log")
    scan_ts = start_ts if start_ts else (datetime.now() - timedelta(days=7)).timestamp()
    all_lines = 0
    err_lines = []
    err_count = 0
    dispatch_cnt = filereply_cnt = sharing_cnt = msg_cnt = 0
    ignored_names = []
    sharing_names = []

    for logfile in ("happyjoplin.log", "happyjoplin.log.1"):
        fp = os.path.join(logdir, logfile)
        if not os.path.isfile(fp):
            continue
        with open(fp, errors="ignore") as f:
            for line in f:
                m = re.match(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
                if not m:
                    continue
                try:
                    lt = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
                except ValueError:
                    continue
                if lt < scan_ts:
                    continue
                all_lines += 1
                lower = line.lower()
                if re.search(r"error|exception|traceback|typeerror", lower):
                    err_count += 1
                    if len(err_lines) < 5:
                        err_lines.append(line.strip())
                if "dispatch" in lower:
                    dispatch_cnt += 1
                if "fileetc_reply" in lower:
                    filereply_cnt += 1
                if "sharing_reply" in lower:
                    sharing_cnt += 1
                if re.search(r"收到消息|收到文件|收到图片|收到视频|收到语音", line):
                    msg_cnt += 1
                # 未配置公众号
                m2 = re.search(r"待配置公众号.*不在ignoredmplist.*?[：:]\s*(.*)", line)
                if m2:
                    name = m2.group(1).strip()
                    if name and name not in ignored_names:
                        ignored_names.append(name)
                m3 = re.search(r"公众号信息[：:]\s*(.*)", line)
                if m3 and "待配置" not in line:
                    name = m3.group(1).strip()
                    if name and name not in sharing_names:
                        sharing_names.append(name)

    all_mp = [n for n in set(ignored_names + sharing_names) if n]

    return jsonify({
        "process": {
            "ok": proc_ok,
            "pid": pid,
            "rss_mb": rss_mb,
            "vsz_mb": vsz_mb,
            "cpu_pct": cpu,
            "session_start": session_start,
            "duration": duration,
        },
        "logs": {
            "scanned_lines": all_lines,
            "error_count": err_count,
            "recent_errors": err_lines,
            "dispatch_count": dispatch_cnt,
            "filereply_count": filereply_cnt,
            "sharing_count": sharing_cnt,
            "message_count": msg_cnt,
        },
        "unconfigured_mps": {
            "total": len(all_mp),
            "names": all_mp,
            "csv": ", ".join(all_mp) if all_mp else "无",
            "ignored": ignored_names,
            "sharing": sharing_names,
        },
        "assessment": _assess(proc_ok, err_count),
    })


def _assess(proc_ok, err_count):
    if not proc_ok:
        return "进程不存在！需要立即处理"
    if err_count > 50:
        return f"进程运行中但错误较多({err_count}条)，建议排查"
    if err_count > 10:
        return f"整体正常，有少量错误({err_count}条)"
    if err_count > 0:
        return f"运行健康，错误极少({err_count}条)"
    return "运行健康，无错误"


# %% [markdown]
# # /system/services — tc systemd 服务状态

# %%
SERVICES = [
    "joplinai-center-api", "joplinai-web-app", "joplinai-sync",
    "monitor-collect", "monitor-report", "monitor-report-full",
    "wc-health-check", "apache2", "fail2ban", "ssh", "docker",
]


@system_bp.route("/system/services")
@require_auth
def system_services():
    result = {}
    for svc in SERVICES:
        out, rc = _run(f"systemctl is-active {svc} 2>/dev/null")
        result[svc] = out if rc == 0 else "inactive"
    return jsonify(result)
