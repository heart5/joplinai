#!/usr/bin/env bash
# Joplinai 统一部署脚本
# Usage:
#   ./deploy/deploy.sh hcx     — 部署到恒创云（本地），重启 QA_API + Web_App
#   ./deploy/deploy.sh tc      — 部署到腾讯云，rsync + 重启 center_api
#   ./deploy/deploy.sh hcx --dry-run   — 仅显示将执行的操作，不实际执行
#   ./deploy/deploy.sh tc --dry-run    — 同上
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---------- 配置 ----------
TC_HOST="tc"
TC_PATH="/home/baiyefeng/work/joplinai"
TC_SERVICE="joplinai-center-api"

HCX_SERVICES=("joplin-qa-api" "joplin-web-app")

RSYNC_EXCLUDE=(
    --exclude="func/"
    --exclude=".git/"
    --exclude="__pycache__/"
    --exclude="*.pyc"
    --exclude=".ipynb_checkpoints/"
    --exclude="data/"
    --exclude="log/"
    --exclude="*.ipynb"
    --exclude="node_modules/"
    --exclude=".pytest_cache/"
)

# ---------- 工具函数 ----------
red()   { echo -e "\033[31m$*\033[0m"; }
green() { echo -e "\033[32m$*\033[0m"; }
yellow(){ echo -e "\033[33m$*\033[0m"; }

systemctl_cmd() {
    local host="$1" svc="$2" action="$3"
    if [ "$host" = "local" ]; then
        sudo systemctl "$action" "$svc"
    else
        ssh "$host" "sudo systemctl $action $svc"
    fi
}

# ---------- TC 部署 ----------
deploy_tc() {
    local dry="$1"
    green "=== 部署到腾讯云 (tc) ==="

    # 1. rsync
    echo "rsync $PROJECT_DIR/ → $TC_HOST:$TC_PATH/"
    if [ "$dry" = "false" ]; then
        rsync -avz --delete "${RSYNC_EXCLUDE[@]}" \
            "$PROJECT_DIR/" "$TC_HOST:$TC_PATH/"
        green "rsync 完成"
    else
        yellow "[dry-run] 跳过 rsync"
    fi

    # 2. 重启 center_api
    echo "重启远程服务: $TC_SERVICE"
    if [ "$dry" = "false" ]; then
        systemctl_cmd "$TC_HOST" "$TC_SERVICE" restart
        green "$TC_SERVICE 重启完成"
    else
        yellow "[dry-run] 跳过 systemctl restart $TC_SERVICE"
    fi
}

# ---------- HCX 部署 ----------
deploy_hcx() {
    local dry="$1"
    green "=== 部署到恒创云 (hcx, 本地) ==="

    for svc in "${HCX_SERVICES[@]}"; do
        echo "重启本地服务: $svc"
        if [ "$dry" = "false" ]; then
            systemctl_cmd "local" "$svc" restart
            green "$svc 重启完成"
        else
            yellow "[dry-run] 跳过 systemctl restart $svc"
        fi
    done
}

# ---------- 主入口 ----------
main() {
    cd "$PROJECT_DIR"

    local target="${1:-}"
    local dry="false"
    if [ "${2:-}" = "--dry-run" ]; then
        dry="true"
    fi

    # 验证工作树干净
    if [ "$dry" = "false" ] && [ -n "$(git status --porcelain 2>/dev/null)" ]; then
        yellow "警告: 工作树有未提交的更改"
        git status --short
        echo -n "是否继续？[y/N] "
        read -r ans
        [ "$ans" = "y" ] || [ "$ans" = "Y" ] || exit 0
    fi

    case "$target" in
        tc)  deploy_tc "$dry" ;;
        hcx) deploy_hcx "$dry" ;;
        *)
            echo "Usage: $0 {hcx|tc} [--dry-run]"
            echo "  hcx — 恒创云（本地），重启 QA_API + Web_App"
            echo "  tc  — 腾讯云（远程），rsync + 重启 center_api"
            exit 1
            ;;
    esac

    green "部署完成。"
}

main "$@"
