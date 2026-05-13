#!/usr/bin/env python3
"""检测 jupytext percent 格式 bug：markdown cell 后缺少 # %% 导致代码被注释。

扫描 aimod/ 和 src/ 下所有 .py 文件，查找被注释掉的 Flask route 或
endpoint 函数定义（# @.*route / # def api_），这通常是 jupytext sync
误将代码识别为 markdown 并注释掉的信号。

用法:
    python tools/check_jupytext_comment.py          # 检查全部文件
    python tools/check_jupytext_comment.py <file>   # 检查指定文件
"""
import re
import sys
from pathlib import Path

PATTERNS = [
    # 被注释掉的 Flask/Blueprint route 装饰器 (锚定行首)
    (r"^[ \t]*# @\w+\.route\(", "commented-out route decorator"),
    # 被注释掉的端点函数定义
    (r"^[ \t]*# def api_\w+\(", "commented-out api endpoint function"),
]

EXCLUDE_DIRS = {"func", ".git", "__pycache__", ".claude", "tools"}


def check_file(filepath: Path) -> list[str]:
    errors = []
    try:
        content = filepath.read_text()
    except Exception:
        return errors

    for lineno, line in enumerate(content.splitlines(), 1):
        for pattern, desc in PATTERNS:
            if re.search(pattern, line):
                errors.append(f"{filepath}:{lineno}: {desc}: {line.strip()}")
    return errors


def main() -> int:
    args = sys.argv[1:]

    if args:
        files = [Path(f) for f in args]
    else:
        root = Path(__file__).resolve().parent.parent
        files = []
        for pyfile in root.rglob("*.py"):
            parts = pyfile.parts
            if any(d in parts for d in EXCLUDE_DIRS):
                continue
            files.append(pyfile)

    all_errors = []
    for fp in sorted(files):
        all_errors.extend(check_file(fp))

    if all_errors:
        print(
            "jupytext 代码被误注释 — 请检查 markdown cell (# %% [markdown]) "
            "后是否缺少 '# %%' 分隔符：",
            file=sys.stderr,
        )
        for err in all_errors:
            print(f"  {err}", file=sys.stderr)
        print(f"\n共 {len(all_errors)} 处问题。", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
