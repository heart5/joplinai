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
# ---

# %% [markdown]
# # Voice API — 语音转文字服务

# %%
"""faster-whisper 语音转文字 API，对外暴露 POST /transcribe"""

# %%
import argparse
import logging
import os
import re
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path

import pathmagic

with pathmagic.Context():
    from func.datetimetools import normalize_time_to_unix

from flask import Flask, jsonify, request

log = logging.getLogger("joplinai_voice_api")
log.propagate = False
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
log.addHandler(_handler)
log.setLevel(logging.INFO)

# %%
MODEL_SIZE = os.environ.get("VOICE_MODEL", "small")
DEVICE = os.environ.get("VOICE_DEVICE", "cpu")
COMPUTE_TYPE = os.environ.get("VOICE_COMPUTE_TYPE", "int8")
MODEL_CACHE = Path(__file__).parent.parent / "data" / "voice_models"
V4TXT_DB = Path(__file__).parent.parent / "data" / "voice_transcriptions.db"

_model = None


def _get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel

        MODEL_CACHE.mkdir(parents=True, exist_ok=True)
        log.info(f"加载 faster-whisper 模型: size={MODEL_SIZE}, device={DEVICE}, compute_type={COMPUTE_TYPE}")
        _model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE, download_root=str(MODEL_CACHE))
        log.info("模型加载完成")
    return _model


# %%
def _save_transcription(account, msg_time, sender, text, send=0, engine="ollama", source="unknown", filepath=None):
    """写入转录结果到 v4txt_v2，INSERT OR IGNORE 保证幂等。"""
    try:
        conn = sqlite3.connect(str(V4TXT_DB))
        # 确保旧列存在（向后兼容旧 schema）
        for col, default in [("source", "'unknown'"), ("send", "0")]:
            try:
                conn.execute(f"ALTER TABLE v4txt_v2 ADD COLUMN {col} TEXT DEFAULT {default}")
            except sqlite3.OperationalError:
                pass
        conn.execute(
            "INSERT OR IGNORE INTO v4txt_v2 (account, msg_time, sender, send, text, engine, source, filepath) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (account, msg_time, sender, send, text, engine, source, filepath),
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        log.error(f"写入 v4txt_v2 失败: {e}")
        return False


def _query_transcriptions(account, records):
    """批量查询 v4txt_v2。

    records 为 [(msg_time, sender), ...]。
    返回 dict: {(msg_time, sender): {text, engine, send}, ...}
    """
    if not records:
        return {}
    try:
        conn = sqlite3.connect(str(V4TXT_DB))
        result = {}
        for msg_time, sender in records:
            row = conn.execute(
                "SELECT text, engine, send FROM v4txt_v2 WHERE account=? AND msg_time=? AND sender=?",
                (account, msg_time, sender),
            ).fetchone()
            if row:
                result[(msg_time, sender)] = {"text": row[0], "engine": row[1], "send": row[2]}
        conn.close()
        return result
    except Exception as e:
        log.error(f"查询 v4txt_v2 失败: {e}")
        return {}


app = Flask(__name__)


@app.route("/health")
def health():
    return jsonify({"status": "healthy", "model": MODEL_SIZE, "device": DEVICE, "ready": _model is not None})


@app.route("/ready")
def ready():
    try:
        _get_model()
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "缺少 file 字段"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "空文件名"}), 400

    # 消息身份参数
    account = request.form.get("account", "")
    msg_time_raw = request.form.get("msg_time", "")
    msg_time = normalize_time_to_unix(msg_time_raw)
    sender = request.form.get("sender", "")
    send = int(request.form.get("send", "0"))
    source = request.form.get("source", "unknown")

    # 写入临时文件
    suffix = Path(file.filename).suffix or ".mp3"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        file.save(tmp)
        tmp_path = tmp.name

    try:
        model = _get_model()
        lang = request.form.get("language", "zh")
        segments, info = model.transcribe(tmp_path, beam_size=5, language=lang)
        text = " ".join(seg.text.strip() for seg in segments)
        log.info(
            f"转录完成: [{source}] {sender} send={send} | {file.filename} → {len(text)} 字, lang={info.language} p={info.language_probability:.2f}"
        )

        # 写入 v4txt_v2（如果提供了身份参数）
        if account and msg_time and sender:
            # 用原始时间拼路径（归一化后是unix时间戳，无法切分年月日）
            rel_path = f"img/webchat/{msg_time_raw[:4]}{msg_time_raw[5:7]}{msg_time_raw[8:10]}/{sender}_{Path(file.filename).stem}"
            _save_transcription(account, msg_time, sender, text, send=send, engine="ollama", source=source, filepath=rel_path)

        return jsonify({"text": text, "language": info.language, "probability": info.language_probability})
    except Exception as e:
        log.error(f"转录失败: {file.filename} — {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        os.unlink(tmp_path)


@app.route("/transcription", methods=["GET"])
def transcription():
    """单条查询已转录记录。?account=白晔峰&time=...&sender=..."""
    account = request.args.get("account", "")
    msg_time = request.args.get("time", "")
    sender = request.args.get("sender", "")
    if not account or not msg_time or not sender:
        return jsonify({"error": "缺少 account/time/sender 参数"}), 400

    result = _query_transcriptions(account, [(msg_time, sender)])
    if result:
        key = (msg_time, sender)
        return jsonify({"found": True, **result[key]})
    return jsonify({"found": False})


@app.route("/transcriptions/batch", methods=["POST"])
def batch_transcription():
    """批量查询已转录记录。

    请求体: {"account": "白晔峰", "records": [["time1", "sender1"], ...]}
    返回: {"results": {"time1#sender1": {"text":"...", "engine":"..."}, ...}}
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "需要 JSON 请求体"}), 400

    account = data.get("account", "")
    records_raw = data.get("records", [])
    if not account or not records_raw:
        return jsonify({"error": "缺少 account/records 字段"}), 400

    records = [(r[0], r[1]) for r in records_raw if len(r) >= 2]
    hits = _query_transcriptions(account, records)
    results = {f"{t}#{s}": v for (t, s), v in hits.items()}
    return jsonify({"results": results})


@app.route("/chat/sync", methods=["POST"])
def chat_sync():
    """接收手机端推送的增量聊天记录。

    请求体: {"account": "白晔峰", "records": [{"time":"...","send":false,"sender":"...","type":"Text","content":"..."}, ...]}
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "需要 JSON 请求体"}), 400

    account = data.get("account", "")
    records = data.get("records", [])
    if not account or not records:
        return jsonify({"error": "缺少 account/records 字段"}), 400

    # 写入 hcx 合并库（合并库未就绪时暂存到独立文件）
    merged_db = V4TXT_DB.parent / "wcitemsall_merged.db"
    if not merged_db.exists():
        return jsonify({"status": "pending", "received": len(records), "message": "合并库尚未就绪，数据未写入"})

    device_source = data.get("source", "unknown")
    inserted = 0
    try:
        conn = sqlite3.connect(str(merged_db))
        table_name = f"wc_{account}"
        for r in records:
            t, sd, sr, tp, ct = (
                r.get("time", ""),
                r.get("send", False),
                r.get("sender", ""),
                r.get("type", ""),
                r.get("content", ""),
            )
            try:
                # SELECT 先查是否存在，避免 INSERT OR IGNORE 消耗 AUTOINCREMENT id
                cur = conn.execute(
                    f"SELECT 1 FROM [{table_name}] WHERE time=? AND send=? AND sender=? AND type=? AND content IS ?",
                    (t, sd, sr, tp, ct),
                )
                if cur.fetchone():
                    continue
                conn.execute(
                    f"INSERT INTO [{table_name}] (time, send, sender, type, content, source) VALUES (?, ?, ?, ?, ?, ?)",
                    (t, sd, sr, tp, ct, device_source),
                )
                inserted += 1
            except sqlite3.IntegrityError:
                pass  # 竞态条件，另一请求刚好插入了同一条
        conn.commit()
        conn.close()
        return jsonify({"status": "ok", "received": len(records), "inserted": inserted})
    except Exception as e:
        log.error(f"chat/sync 失败: {e}")
        return jsonify({"error": str(e)}), 500


# ── WeChat 数据查询端点 ──

_ACCOUNT_RE = re.compile(r"^[\w一-鿿]+$")
MERGED_DB = V4TXT_DB.parent / "wcitemsall_merged.db"


def _ensure_merged_db() -> sqlite3.Connection:
    if not MERGED_DB.exists():
        raise FileNotFoundError(f"合并库不存在: {MERGED_DB}")
    conn = sqlite3.connect(str(MERGED_DB))
    conn.row_factory = sqlite3.Row
    return conn


def _validate_account(account: str):
    if not _ACCOUNT_RE.match(account):
        raise ValueError(f"非法的账号名: {account!r}")


def _normalize_time(val) -> str:
    """将混合格式的 time 统一为 ISO 格式。"""
    if val is None or val == "":
        return ""
    if isinstance(val, (int, float)):
        try:
            return datetime.fromtimestamp(val).strftime("%Y-%m-%d %H:%M:%S")
        except (OSError, ValueError):
            return str(val)
    s = str(val).strip()
    if re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$", s):
        return s
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return s + " 00:00:00"
    try:
        return datetime.fromtimestamp(int(float(s))).strftime("%Y-%m-%d %H:%M:%S")
    except (OSError, ValueError, TypeError):
        return s


def _row_to_dict(r) -> dict:
    """sqlite3.Row → dict，统一 time 格式。"""
    return {
        "id": r["id"],
        "time": _normalize_time(r["time"]),
        "send": bool(r["send"]),
        "sender": r["sender"],
        "type": r["type"],
        "content": r["content"],
        "source": r["source"],
    }


@app.route("/wechat/health")
def wechat_health():
    """合并库健康检查。"""
    try:
        conn = _ensure_merged_db()
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'wc_%'").fetchall()
        info = {}
        for t in tables:
            name = t["name"]
            cnt = conn.execute(f"SELECT COUNT(*) FROM [{name}]").fetchone()[0]
            info[name] = cnt
        conn.close()
        return jsonify({
            "status": "ok",
            "db_path": str(MERGED_DB),
            "db_size_mb": round(MERGED_DB.stat().st_size / 1024 / 1024, 1),
            "tables": info,
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/wechat/contacts")
def wechat_contacts():
    """联系人活跃列表。"""
    account = request.args.get("account", "")
    days = int(request.args.get("days", 30))
    try:
        _validate_account(account)
        conn = _ensure_merged_db()
        table = f"wc_{account}"
        rows = conn.execute(
            f"""SELECT sender, COUNT(*) as msg_count,
                       SUM(CASE WHEN send=1 THEN 1 ELSE 0 END) as sent_count,
                       MAX(time) as last_time
                FROM [{table}]
                WHERE time >= datetime('now', ? || ' days')
                GROUP BY sender
                ORDER BY msg_count DESC
                LIMIT 200""",
            (str(days),),
        ).fetchall()
        conn.close()
        return jsonify({"account": account, "contacts": [dict(r) for r in rows]})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/wechat/stats")
def wechat_stats():
    """时段统计。"""
    account = request.args.get("account", "")
    date_from = request.args.get("date_from", "")
    date_to = request.args.get("date_to", "")
    try:
        _validate_account(account)
        conn = _ensure_merged_db()
        table = f"wc_{account}"
        total = conn.execute(
            f"SELECT COUNT(*) FROM [{table}] WHERE time >= ? AND time <= ?",
            (date_from, date_to),
        ).fetchone()[0]
        type_dist = conn.execute(
            f"SELECT type, COUNT(*) as cnt FROM [{table}] WHERE time >= ? AND time <= ? GROUP BY type ORDER BY cnt DESC",
            (date_from, date_to),
        ).fetchall()
        daily = conn.execute(
            f"SELECT substr(time,1,10) as d, COUNT(*) as cnt FROM [{table}] WHERE time >= ? AND time <= ? GROUP BY d ORDER BY d",
            (date_from, date_to),
        ).fetchall()
        conn.close()
        return jsonify({
            "account": account,
            "date_from": date_from,
            "date_to": date_to,
            "total": total,
            "type_distribution": [{"type": r[0], "count": r[1]} for r in type_dist],
            "daily": [{"date": r[0], "count": r[1]} for r in daily],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/wechat/conversation")
def wechat_conversation():
    """与特定联系人的对话详情。"""
    account = request.args.get("account", "")
    sender = request.args.get("sender", "")
    date_from = request.args.get("date_from", "")
    date_to = request.args.get("date_to", "")
    limit = int(request.args.get("limit", 100))
    try:
        _validate_account(account)
        conn = _ensure_merged_db()
        table = f"wc_{account}"
        conditions = ["sender LIKE ?"]
        params = [f"%{sender}%"]
        if date_from:
            conditions.append("time >= ?")
            params.append(date_from)
        if date_to:
            conditions.append("time <= ?")
            params.append(date_to)
        where = " AND ".join(conditions)
        rows = conn.execute(
            f"SELECT id, time, send, sender, type, content, source FROM [{table}] WHERE {where} ORDER BY id ASC LIMIT ?",
            params + [limit],
        ).fetchall()
        conn.close()
        return jsonify({
            "account": account,
            "sender": sender,
            "records": [_row_to_dict(r) for r in rows],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/wechat/query")
def wechat_query():
    """通用查询。"""
    account = request.args.get("account", "")
    date_from = request.args.get("date_from")
    date_to = request.args.get("date_to")
    sender = request.args.get("sender")
    keyword = request.args.get("keyword")
    type_filter = request.args.get("type")
    after_id = request.args.get("after_id")
    limit = int(request.args.get("limit", 1000))
    try:
        _validate_account(account)
        conn = _ensure_merged_db()
        table = f"wc_{account}"

        conditions = []
        params = []
        if after_id:
            conditions.append("id > ?")
            params.append(int(after_id))
        if date_from:
            conditions.append("time >= ?")
            params.append(date_from)
        if date_to:
            conditions.append("time <= ?")
            params.append(date_to)
        if sender:
            conditions.append("sender LIKE ?")
            params.append(f"%{sender}%")
        if keyword:
            conditions.append("content LIKE ?")
            params.append(f"%{keyword}%")
        if type_filter:
            types = [t.strip() for t in type_filter.split(",")]
            placeholders = ",".join("?" for _ in types)
            conditions.append(f"type IN ({placeholders})")
            params.extend(types)

        where = " AND ".join(conditions) if conditions else "1"
        rows = conn.execute(
            f"SELECT id, time, send, sender, type, content, source FROM [{table}] WHERE {where} ORDER BY id ASC LIMIT ?",
            params + [limit],
        ).fetchall()
        conn.close()
        return jsonify({
            "account": account,
            "records": [_row_to_dict(r) for r in rows],
            "returned": len(rows),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/wechat/finance")
def wechat_finance():
    """财务相关消息过滤。"""
    account = request.args.get("account", "")
    date_from = request.args.get("date_from", "")
    date_to = request.args.get("date_to", "")
    try:
        _validate_account(account)
        conn = _ensure_merged_db()
        table = f"wc_{account}"
        rows = conn.execute(
            f"""SELECT id, time, send, sender, type, content, source FROM [{table}]
                WHERE time >= ? AND time <= ?
                  AND (sender LIKE '%支付%' OR sender LIKE '%银行%' OR sender LIKE '%信用卡%'
                       OR content LIKE '%￥%' OR content LIKE '%消费%'
                       OR content LIKE '%转账%' OR content LIKE '%红包%')
                ORDER BY id ASC""",
            (date_from, date_to),
        ).fetchall()
        conn.close()
        return jsonify({
            "account": account,
            "records": [_row_to_dict(r) for r in rows],
            "returned": len(rows),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ── SMS 接收端点 ──

_SMS_DB = Path(__file__).parent.parent / "data" / "sms_received.db"


def _ensure_sms_db():
    conn = sqlite3.connect(str(_SMS_DB))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sms_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sms_id INTEGER UNIQUE,
            number TEXT,
            body TEXT,
            received TEXT,
            uploaded_at TEXT,
            source TEXT DEFAULT 'termux'
        )
    """)
    conn.commit()
    conn.close()


def _get_sms_api_key():
    try:
        from func.jpfuncs import getinivaluefromcloud
        return getinivaluefromcloud("sms_collector", "api_key") or ""
    except Exception:
        return ""


@app.route("/sms/upload", methods=["POST"])
def sms_upload():
    """接收 Termux 上传的短信数据。

    请求体: {"messages": [{"_id": 123, "number": "95555", "body": "...", "received": "..."}], "source": "termux"}
    """
    expected_key = _get_sms_api_key()
    if expected_key:
        got_key = request.headers.get("X-API-Key", "")
        if got_key != expected_key:
            return jsonify({"error": "unauthorized"}), 401

    data = request.get_json(silent=True)
    if not data or "messages" not in data:
        return jsonify({"error": "need JSON with messages field"}), 400

    messages = data["messages"]
    if not messages:
        return jsonify({"imported": 0, "errors": 0, "total": 0})

    _ensure_sms_db()
    conn = sqlite3.connect(str(_SMS_DB))
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    imported = 0
    errors = 0

    for m in messages:
        try:
            conn.execute(
                "INSERT OR IGNORE INTO sms_messages (sms_id, number, body, received, uploaded_at, source) VALUES (?,?,?,?,?,?)",
                (int(m["_id"]), str(m.get("number", "")), str(m.get("body", "")), str(m.get("received", "")), now, data.get("source", "termux"))
            )
            if conn.total_changes:
                imported += 1
        except Exception as e:
            log.warning(f"sms 处理失败: {e}")
            errors += 1

    conn.commit()
    conn.close()
    log.info(f"sms 接收: {imported} 条入库, {errors} 条错误")
    return jsonify({"imported": imported, "errors": errors, "total": len(messages)})


@app.route("/sms/query", methods=["GET"])
def sms_query():
    """查询短信数据（供 happyjoplin 消费端 API 调用）。

    参数: date_from, date_to, limit, number
    """
    expected_key = _get_sms_api_key()
    if expected_key:
        got_key = request.headers.get("X-API-Key", "")
        if got_key != expected_key:
            return jsonify({"error": "unauthorized"}), 401

    date_from = request.args.get("date_from", "")
    date_to = request.args.get("date_to", "")
    number = request.args.get("number", "")
    limit = int(request.args.get("limit", 50000))

    if not _SMS_DB.exists():
        return jsonify({"error": "sms db not found", "records": []})

    conn = sqlite3.connect(str(_SMS_DB))
    conditions = []
    params = []
    if date_from:
        conditions.append("received >= ?")
        params.append(date_from)
    if date_to:
        conditions.append("received < ?")
        params.append(date_to)
    if number:
        conditions.append("number = ?")
        params.append(number)

    where = " AND ".join(conditions) if conditions else "1"
    rows = conn.execute(
        f"SELECT sms_id, number, body, received FROM sms_messages WHERE {where} ORDER BY received ASC LIMIT ?",
        params + [limit],
    ).fetchall()
    conn.close()

    records = [
        {"_id": r[0], "number": r[1], "body": r[2], "received": r[3]}
        for r in rows
    ]
    log.info(f"sms 查询: {len(records)} 条 (date_from={date_from}, date_to={date_to})")
    return jsonify({"records": records, "total": len(records)})



# %%
def main():
    parser = argparse.ArgumentParser(description="Joplinai Voice API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5004)
    args = parser.parse_args()
    log.info(f"启动 Voice API 于 http://{args.host}:{args.port}, model={MODEL_SIZE}, device={DEVICE}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
