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
# # State Blueprint — 笔记处理状态端点

# %%
import json
from datetime import datetime

from flask import Blueprint, jsonify, request

# %%
from aimod.center_api import _init_db, log, require_auth

__all__ = ["state_bp"]

state_bp = Blueprint("state", __name__)

# %% [markdown]
# # Flask 端点

# %%
@state_bp.route("/state/batch_load", methods=["POST"])
@require_auth
def api_state_batch_load():
    data = request.get_json(force=True)
    model_name = data["model_name"]
    conn = _init_db()
    rows = conn.execute(
        "SELECT note_id, state_json FROM note_process_state WHERE model_name=?",
        (model_name,),
    ).fetchall()
    conn.close()
    states = {}
    virtual_collections = {}
    for note_id, state_json in rows:
        state = json.loads(state_json)
        if note_id == "__virtual_collections__":
            virtual_collections = state
        else:
            states[note_id] = state
    result = {"states": states}
    if virtual_collections:
        result["virtual_collections"] = virtual_collections
    log.info(f"状态加载: model={model_name}, {len(states)}条笔记, {len(virtual_collections)}个虚拟集合")
    return jsonify(result)


@state_bp.route("/state/batch_save", methods=["POST"])
@require_auth
def api_state_batch_save():
    data = request.get_json(force=True)
    model_name = data["model_name"]
    states = data.get("states", {})
    virtual_collections = data.get("virtual_collections", {})
    now = datetime.now().isoformat()
    conn = _init_db()
    count = 0
    for note_id, note_state in states.items():
        conn.execute(
            "INSERT INTO note_process_state (model_name, note_id, state_json, updated_at) "
            "VALUES (?,?,?,?) ON CONFLICT(model_name, note_id) DO UPDATE SET "
            "state_json=excluded.state_json, updated_at=excluded.updated_at",
            (model_name, note_id, json.dumps(note_state, ensure_ascii=False), now),
        )
        count += 1
    if virtual_collections:
        conn.execute(
            "INSERT INTO note_process_state (model_name, note_id, state_json, updated_at) "
            "VALUES (?,?,?,?) ON CONFLICT(model_name, note_id) DO UPDATE SET "
            "state_json=excluded.state_json, updated_at=excluded.updated_at",
            (model_name, "__virtual_collections__", json.dumps(virtual_collections, ensure_ascii=False), now),
        )
        count += 1
    conn.commit()
    conn.close()
    log.info(f"状态保存: model={model_name}, {count}条记录")
    return jsonify({"ok": True, "count": count})


@state_bp.route("/state/<model_name>/<note_id>", methods=["GET"])
@require_auth
def api_state_get_note(model_name: str, note_id: str):
    conn = _init_db()
    row = conn.execute(
        "SELECT state_json FROM note_process_state WHERE model_name=? AND note_id=?",
        (model_name, note_id),
    ).fetchone()
    conn.close()
    if row:
        return jsonify({"found": True, "state": json.loads(row[0])})
    return jsonify({"found": False}), 404


@state_bp.route("/state/delete_model", methods=["POST"])
@require_auth
def api_state_delete_model():
    data = request.get_json(force=True)
    model_name = data["model_name"]
    conn = _init_db()
    cursor = conn.execute("DELETE FROM note_process_state WHERE model_name=?", (model_name,))
    deleted = cursor.rowcount
    conn.commit()
    conn.close()
    return jsonify({"ok": True, "deleted": deleted})


# %% [markdown]
# # Run State 端点 — checkpoint/batch_progress 等运行时标记

# %%
@state_bp.route("/state/run_state/load", methods=["POST"])
@require_auth
def api_run_state_load():
    data = request.get_json(force=True)
    model_name = data["model_name"]
    key = data["key"]
    conn = _init_db()
    row = conn.execute(
        "SELECT state_json FROM note_process_state WHERE model_name=? AND note_id=?",
        (model_name, f"__{key}__"),
    ).fetchone()
    conn.close()
    if row:
        return jsonify({"found": True, "value": json.loads(row[0])})
    return jsonify({"found": False, "value": None})


@state_bp.route("/state/run_state/save", methods=["POST"])
@require_auth
def api_run_state_save():
    data = request.get_json(force=True)
    model_name = data["model_name"]
    key = data["key"]
    value = data["value"]
    conn = _init_db()
    conn.execute(
        "INSERT OR REPLACE INTO note_process_state (model_name, note_id, state_json, updated_at) "
        "VALUES (?,?,?,?)",
        (model_name, f"__{key}__", json.dumps(value, ensure_ascii=False), datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()
    log.info(f"run_state 保存: model={model_name}, key={key}")
    return jsonify({"ok": True})


@state_bp.route("/state/run_state/delete", methods=["POST"])
@require_auth
def api_run_state_delete():
    data = request.get_json(force=True)
    model_name = data["model_name"]
    key = data["key"]
    conn = _init_db()
    conn.execute(
        "DELETE FROM note_process_state WHERE model_name=? AND note_id=?",
        (model_name, f"__{key}__"),
    )
    conn.commit()
    conn.close()
    return jsonify({"ok": True})
