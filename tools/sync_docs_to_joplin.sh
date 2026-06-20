#!/bin/bash
# post-commit hook / 手动工具: .md 文件变更 → md2note 同步到 Joplin → TC joplin sync
#
# md2note 内置 mtime 检测，未变更的文件自动跳过，无需额外 diff 判断。
#
# 依赖：
#   - /home/baiyefeng/bin/md2note（happyjoplin/etc/md2note.py 包装）
#   - SSH 免密登录 tc（~/.ssh/config 配置主机别名为 tc）

PROJ_ROOT="/data/codebase/joplinai"
MD2NOTE="/home/baiyefeng/bin/md2note"
NOTEBOOK="joplinai"
LOGGER_TAG="sync-docs-joplin"

# ── 扫描需要同步的 .md 文件（md2note 自动 mtime 过滤） ──
FILES=""

for f in "$PROJ_ROOT"/docs/*.md; do
    [ -f "$f" ] && FILES="$FILES $f"
done

for f in "$PROJ_ROOT"/*.md; do
    [ -f "$f" ] && FILES="$FILES $f"
done

if [ -z "$FILES" ]; then
    exit 0
fi

# ── 1. md2note 同步到 Joplin ──
echo "[$LOGGER_TAG] 同步 .md 文件到 Joplin..."

for full_path in $FILES; do
    rel="${full_path#$PROJ_ROOT/}"
    if "$MD2NOTE" --notebook "$NOTEBOOK" --quiet "$full_path" 2>/dev/null; then
        echo "[$LOGGER_TAG] 已同步: $rel"
    else
        echo "[$LOGGER_TAG] 同步失败: $rel（md2note 已记录错误）" >&2
    fi
done

# ── 2. 远程 TC 执行 joplin sync ──
echo "[$LOGGER_TAG] 远程 TC joplin sync..."
if ssh tc \
    "source /usr/miniconda3/etc/profile.d/conda.sh && conda activate newlsp && joplin sync" \
    2>/dev/null; then
    echo "[$LOGGER_TAG] TC joplin sync 完成"
else
    echo "[$LOGGER_TAG] TC joplin sync 失败（非阻塞警告）" >&2
fi
