import sqlite3, os, sys, json, glob
from datetime import datetime


def migrate_one(base):
    old_ds = os.path.join(base, "data/.deepseek_cache/deepseek_cache.db")
    old_hist = os.path.join(base, "data/joplinai_history.db")
    new_db = os.path.join(base, "data/joplinai_center.db")

    conn = sqlite3.connect(new_db)

    if os.path.exists(old_ds):
        old = sqlite3.connect(old_ds)
        rows = old.execute("SELECT * FROM processing_cache").fetchall()
        if rows:
            cols = [c[0] for c in old.execute("PRAGMA table_info(processing_cache)")]
            ph = ",".join(["?"] * len(cols))
            for row in rows:
                try: conn.execute(f"INSERT OR IGNORE INTO deepseek_cache ({','.join(cols)}) VALUES ({ph})", row)
                except: pass
            conn.commit()
            print(f"  deepseek_cache: {len(rows)} 条")
        old.close()

    if os.path.exists(old_hist):
        old = sqlite3.connect(old_hist)
        for table in ("notebook_history", "global_run_history"):
            try:
                rows = old.execute(f"SELECT * FROM {table}").fetchall()
                if rows:
                    cols = [c[0] for c in old.execute(f"PRAGMA table_info({table})")]
                    ph = ",".join(["?"] * len(cols))
                    for row in rows:
                        try: conn.execute(f"INSERT OR IGNORE INTO {table} ({','.join(cols)}) VALUES ({ph})", row)
                        except: pass
                    conn.commit()
                    print(f"  {table}: {len(rows)} 条")
            except Exception as e:
                print(f"  {table}: skip - {e}")
        old.close()

    # 迁移笔记处理状态 JSON → note_process_state 表
    migrate_process_state(base, conn)

    # 迁移用户数据库 → center.db
    migrate_users(base, conn)

    conn.close()


def migrate_process_state(base, conn):
    """将本地 JSON 状态文件迁移到 note_process_state 表"""
    state_dir = os.path.join(base, "data")
    pattern = os.path.join(state_dir, "joplin_process_state_*.json")
    files = glob.glob(pattern)
    if not files:
        print("  process_state: 无状态文件，跳过")
        return

    now = datetime.now().isoformat()
    for state_file in files:
        basename = os.path.basename(state_file)
        model_name = basename.replace("joplin_process_state_", "").replace(".json", "")

        try:
            with open(state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"  {basename}: 读取失败 - {e}")
            continue

        count = 0
        for note_id, note_state in data.items():
            sid = "__virtual_collections__" if note_id == "_virtual_collections" else note_id
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO note_process_state (model_name, note_id, state_json, updated_at) VALUES (?, ?, ?, ?)",
                    (model_name, sid, json.dumps(note_state, ensure_ascii=False), now),
                )
                count += 1
            except Exception as e:
                print(f"  {basename}/{note_id}: 写入失败 - {e}")
        conn.commit()
        print(f"  {basename}: {count} 条")


def migrate_users(base, conn):
    """将 joplinai_users.db 迁移到 center.db"""
    old_users_db = os.path.join(base, "data", "joplinai_users.db")
    if not os.path.exists(old_users_db):
        print("  joplinai_users.db: 未找到，跳过")
        return

    old = sqlite3.connect(old_users_db)
    tables_to_migrate = ["users", "sessions", "audit_log", "qa_history", "chat_sessions"]

    for table in tables_to_migrate:
        try:
            rows = old.execute(f"SELECT * FROM {table}").fetchall()
            if not rows:
                print(f"  {table}: 0 条（空表）")
                continue
            cols = [c[0] for c in old.execute(f"PRAGMA table_info({table})")]
            ph = ",".join(["?"] * len(cols))
            for row in rows:
                try:
                    conn.execute(f"INSERT OR IGNORE INTO {table} ({','.join(cols)}) VALUES ({ph})", row)
                except Exception as e:
                    print(f"    {table}: 行跳过 - {e}")
            conn.commit()
            print(f"  {table}: {len(rows)} 条")
        except Exception as e:
            print(f"  {table}: 跳过 - {e}")

    old.close()


if __name__ == "__main__":
    migrate_one(sys.argv[1])
