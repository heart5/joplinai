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
import sqlite3
import tempfile
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
