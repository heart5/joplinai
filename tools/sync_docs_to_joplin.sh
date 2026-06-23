#!/bin/bash
# post-commit hook / 手动工具: .md 文件变更 → md2note 同步到 Joplin → TC joplin sync
#
# 委托给 func/tools/md2note.py（--find-files 自动扫描，--ssh-sync 远程同步）
#
# 依赖：
#   - func 子模块（func/tools/md2note.py）
#   - SSH 免密登录 tc（~/.ssh/config 配置主机别名为 tc）

PROJ_ROOT="/data/codebase/joplinai"
LOGGER_TAG="sync-docs-joplin"

echo "[$LOGGER_TAG] 同步 .md 文件到 Joplin..."
cd "$PROJ_ROOT" || exit 1
python -m func.tools.md2note --find-files --notebook joplinai --quiet --ssh-sync
echo "[$LOGGER_TAG] 完成"
