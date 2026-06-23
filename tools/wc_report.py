#!/usr/bin/env python3
"""微信聊天记录 & 语音转录收集情况报告（数据集中至恒创云的时间）。

用法:
    python tools/wc_report.py --date 2026-06-02          # 日报告
    python tools/wc_report.py --date 今天                 # 中文日期
    python tools/wc_report.py --month 2026-06             # 月报告
    python tools/wc_report.py --account 白晔峰
"""
import argparse
import re
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
MERGED_DB = PROJ / "data" / "wcitemsall_merged.db"
VOICE_DB = PROJ / "data" / "voice_transcriptions.db"

# ── 中文日期解析 ──

_WEEKDAYS = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]


def parse_date(text: str) -> str:
    """将中文/自然语言日期转为 YYYY-MM-DD，无法解析返回原字符串。"""
    text = text.strip()
    today = datetime.now()

    # 已是标准格式
    if re.match(r"^\d{4}-\d{2}-\d{2}$", text):
        return text

    # 今天 / 昨天 / 前天
    if text in ("今天", "今日"):
        return today.strftime("%Y-%m-%d")
    if text in ("昨天", "昨日"):
        return (today - timedelta(days=1)).strftime("%Y-%m-%d")
    if text in ("前天", "前日"):
        return (today - timedelta(days=2)).strftime("%Y-%m-%d")

    # 周X → 最近的一个该星期几
    for i, wd in enumerate(_WEEKDAYS):
        if text in (wd, f"星期{['一','二','三','四','五','六','日'][i]}"):
            target_wday = i  # 0=Mon
            delta = (today.weekday() - target_wday) % 7
            if delta == 0:
                delta = 0  # 今天就是
            return today.strftime("%Y-%m-%d") if delta == 0 else (
                today - timedelta(days=delta)
            ).strftime("%Y-%m-%d")

    # N月D日
    m = re.match(r"^(\d{1,2})月(\d{1,2})日?$", text)
    if m:
        month, day = int(m.group(1)), int(m.group(2))
        return f"{today.year}-{month:02d}-{day:02d}"

    # YYYY年N月D日
    m = re.match(r"^(\d{4})年(\d{1,2})月(\d{1,2})日?$", text)
    if m:
        year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return f"{year}-{month:02d}-{day:02d}"

    return text


def parse_date_or_month(text: str) -> tuple[str | None, str | None]:
    """从用户输入中解析出 --date 和 --month 值。"""
    # 先看是否匹配 YYYY-MM 月份格式
    if re.match(r"^\d{4}-\d{2}$", text.strip()):
        return None, text.strip()

    # 中文月份：2026年6月
    m = re.match(r"^(\d{4})年(\d{1,2})月$", text.strip())
    if m:
        year, month = int(m.group(1)), int(m.group(2))
        return None, f"{year}-{month:02d}"

    # 本月 / 上个月
    if text.strip() in ("本月", "这个月"):
        return None, today.strftime("%Y-%m")
    if text.strip() in ("上个月", "上月"):
        first_of_this_month = today.replace(day=1)
        last_month = first_of_this_month - timedelta(days=1)
        return None, last_month.strftime("%Y-%m")

    # 其他都当作日期
    return parse_date(text), None


# ── 报告函数 ──


def _type_source_pivot(conn, table: str, like_val: str) -> str:
    """生成 类型×来源 交叉表 markdown。"""
    rows = conn.execute(
        f'SELECT type, source, COUNT(*) FROM [{table}] WHERE time LIKE ? GROUP BY type, source ORDER BY type, COUNT(*) DESC',
        (like_val,),
    ).fetchall()
    if not rows:
        return ""

    # 收集所有来源列（按总量降序）
    src_totals: dict[str, int] = defaultdict(int)
    pivot: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    type_order: list[str] = []
    seen_types = set()
    for tp, src, cnt in rows:
        src_totals[src] += cnt
        pivot[tp][src] = cnt
        if tp not in seen_types:
            seen_types.add(tp)
            type_order.append(tp)

    sources = sorted(src_totals, key=src_totals.get, reverse=True)

    lines = []
    header = "| 类型 | " + " | ".join(sources) + " | 合计 |"
    lines.append(header)
    lines.append("|------|" + "|".join(["------"] * (len(sources) + 1)) + "|")

    for tp in type_order:
        cells = [str(pivot[tp].get(src, 0)) for src in sources]
        total = sum(pivot[tp].values())
        lines.append(f"| {tp} | " + " | ".join(cells) + f" | {total} |")

    return "\n".join(lines) + "\n"


def report_day(account: str, date: str) -> str:
    conn = sqlite3.connect(str(MERGED_DB))
    table = f"wc_{account}"

    total = conn.execute(
        f"SELECT COUNT(*) FROM [{table}] WHERE time LIKE ?", (f"{date}%",)
    ).fetchone()[0]

    lines = [
        f"## {date} 微信数据收集日报",
        f"账号: {account}",
        "",
        "> 时间为数据集中至恒创云的时间，非消息原始时间。",
        "",
    ]

    # ── 聊天记录 ──
    lines.append("### 聊天记录")
    lines.append(f"总量: {total} 条")
    lines.append("")

    # 来源分布
    src_rows = conn.execute(
        f"SELECT source, COUNT(*) FROM [{table}] WHERE time LIKE ? GROUP BY source ORDER BY COUNT(*) DESC",
        (f"{date}%",),
    ).fetchall()
    if src_rows:
        lines.append("| 来源 | 数量 |")
        lines.append("|------|------|")
        for src, cnt in src_rows:
            lines.append(f"| {src} | {cnt} |")
        lines.append("")

    # 类型×来源交叉表
    lines.append("#### 类型分布（按来源）")
    lines.append("")
    lines.append(_type_source_pivot(conn, table, f"{date}%"))
    lines.append("")

    # 收集时间按小时分布
    hour_rows = conn.execute(
        f"SELECT substr(time,12,2), COUNT(*) FROM [{table}] WHERE time LIKE ? GROUP BY 1 ORDER BY 1",
        (f"{date}%",),
    ).fetchall()
    if hour_rows:
        max_cnt = max(c for _, c in hour_rows)
        lines.append("#### 收集时间分布（小时）")
        lines.append("")
        lines.append("```")
        for h, cnt in hour_rows:
            bar = "█" * max(1, int(cnt / max(max_cnt, 1) * 30))
            lines.append(f"  {h}:00  {cnt:>5}  {bar}")
        lines.append("```")
        lines.append("")

    conn.close()

    # ── 语音转录 ──
    lines.append("### 语音转录")
    if VOICE_DB.exists():
        conn2 = sqlite3.connect(str(VOICE_DB))
        v_total = conn2.execute(
            "SELECT COUNT(*) FROM v4txt_v2 WHERE datetime(msg_time, 'unixepoch') LIKE ?",
            (f"{date}%",),
        ).fetchone()[0]
        lines.append(f"总量: {v_total} 条")
        lines.append("")

        src_rows = conn2.execute(
            "SELECT source, COUNT(*) FROM v4txt_v2 WHERE datetime(msg_time, 'unixepoch') LIKE ? GROUP BY source ORDER BY COUNT(*) DESC",
            (f"{date}%",),
        ).fetchall()
        if src_rows:
            lines.append("| 来源 | 数量 |")
            lines.append("|------|------|")
            for src, cnt in src_rows:
                lines.append(f"| {src} | {cnt} |")
            lines.append("")

        # 引擎
        eng_rows = conn2.execute(
            "SELECT engine, COUNT(*) FROM v4txt_v2 WHERE datetime(msg_time, 'unixepoch') LIKE ? GROUP BY engine",
            (f"{date}%",),
        ).fetchall()
        if eng_rows:
            lines.append("| 引擎 | 数量 |")
            lines.append("|------|------|")
            for eng, cnt in eng_rows:
                lines.append(f"| {eng} | {cnt} |")
            lines.append("")

        conn2.close()
    else:
        lines.append("(语音转录库不存在)")
        lines.append("")

    return "\n".join(lines)


def report_month(account: str, month: str) -> str:
    conn = sqlite3.connect(str(MERGED_DB))
    table = f"wc_{account}"

    lines = [
        f"## {month} 微信数据收集月报",
        f"账号: {account}",
        "",
        "> 时间为数据集中至恒创云的时间，非消息原始时间。",
        "",
    ]

    total = conn.execute(
        f"SELECT COUNT(*) FROM [{table}] WHERE time LIKE ?", (f"{month}%",)
    ).fetchone()[0]
    lines.append("### 聊天记录")
    lines.append(f"月总量: {total:,} 条")
    lines.append("")

    # 来源
    src_rows = conn.execute(
        f"SELECT source, COUNT(*) FROM [{table}] WHERE time LIKE ? GROUP BY source ORDER BY COUNT(*) DESC",
        (f"{month}%",),
    ).fetchall()
    if src_rows:
        lines.append("| 来源 | 数量 |")
        lines.append("|------|------|")
        for src, cnt in src_rows:
            lines.append(f"| {src} | {cnt:,} |")
        lines.append("")

    # 类型×来源交叉表
    lines.append("#### 类型分布（按来源）")
    lines.append("")
    lines.append(_type_source_pivot(conn, table, f"{month}%"))
    lines.append("")

    # 按日分布
    day_rows = conn.execute(
        f"SELECT substr(time,1,10), COUNT(*) FROM [{table}] WHERE time LIKE ? GROUP BY 1 ORDER BY 1",
        (f"{month}%",),
    ).fetchall()
    if day_rows:
        max_cnt = max(c for _, c in day_rows)
        lines.append("#### 收集时间分布（日）")
        lines.append("")
        lines.append("```")
        for d, cnt in day_rows:
            bar = "█" * max(1, int(cnt / max(max_cnt, 1) * 30))
            lines.append(f"  {d[-5:]}  {cnt:>5}  {bar}")
        lines.append("```")
        lines.append("")

    conn.close()

    # ── 语音转录 ──
    lines.append("### 语音转录")
    if VOICE_DB.exists():
        conn2 = sqlite3.connect(str(VOICE_DB))
        v_total = conn2.execute(
            "SELECT COUNT(*) FROM v4txt_v2 WHERE datetime(msg_time, 'unixepoch') LIKE ?",
            (f"{month}%",),
        ).fetchone()[0]
        lines.append(f"月总量: {v_total:,} 条")
        lines.append("")

        src_rows = conn2.execute(
            "SELECT source, COUNT(*) FROM v4txt_v2 WHERE datetime(msg_time, 'unixepoch') LIKE ? GROUP BY source ORDER BY COUNT(*) DESC",
            (f"{month}%",),
        ).fetchall()
        if src_rows:
            lines.append("| 来源 | 数量 |")
            lines.append("|------|------|")
            for src, cnt in src_rows:
                lines.append(f"| {src} | {cnt:,} |")
            lines.append("")

        conn2.close()
    else:
        lines.append("(语音转录库不存在)")
        lines.append("")

    return "\n".join(lines)


# ── CLI ──

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="微信数据收集情况报告")
    parser.add_argument("--date", help="日期 YYYY-MM-DD 或中文（今天/昨天/6月2日/2026年6月2日）")
    parser.add_argument("--month", help="月份 YYYY-MM")
    parser.add_argument("--account", default="白晔峰", help="微信账号")
    args = parser.parse_args()

    if args.date:
        date = parse_date(args.date)
        print(report_day(args.account, date))
    elif args.month:
        print(report_month(args.account, args.month))
    else:
        today = datetime.now().strftime("%Y-%m-%d")
        print(report_day(args.account, today))
