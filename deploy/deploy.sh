#!/usr/bin/env bash
# Joplinai 统一部署脚本
# Usage:
#   ./deploy/deploy.sh hcx     — 部署到恒创云（本地），重启 QA_API + Web_App
#   ./deploy/deploy.sh tc      — 部署到腾讯云，git push→git pull (直连→代理→rsync兜底)→重启 center_api
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
        ssh "$host" "sudo systemctl $action --no-block $svc"
    fi
}

# ---------- TC 部署 ----------
deploy_tc() {
    local dry="$1"
    green "=== 部署到腾讯云 (tc) ==="

    # 0. 推送 HCX 本地提交到 GitHub（TC git pull 的前提）
    if [ "$dry" = "false" ]; then
        local ahead
        ahead=$(git rev-list origin/main..HEAD --count 2>/dev/null || echo 0)
        if [ "$ahead" -gt 0 ]; then
            echo "推送本地 $ahead 个提交到 GitHub..."
            if git push origin main; then
                green "git push 完成"
            else
                yellow "警告: git push 失败，TC 将无法通过 git pull 获取最新提交"
            fi
        else
            echo "本地与 origin/main 同步，跳过 push"
        fi
    else
        yellow "[dry-run] 跳过 git push"
    fi

    # 1. 优先 git pull（直连）
    if [ "$dry" = "false" ]; then
        echo "--- TC: git pull (直连) ---"
        if ssh "$TC_HOST" "cd $TC_PATH && git pull origin main"; then
            green "git pull (直连) 完成"
        else
            # 2. 直连失败 → 开代理重试
            yellow "直连失败，尝试 clash 代理..."
            if ssh "$TC_HOST" "
                source /etc/profile.d/clash.sh 2>/dev/null
                proxy_on 2>/dev/null
                cd $TC_PATH
                git pull origin main
                ret=\$?
                proxy_off 2>/dev/null
                exit \$ret
            "; then
                green "git pull (代理) 完成"
            else
                # 3. 代理也失败 → rsync 兜底 + 重置 TC git
                red "git pull (代理) 也失败，回退到 rsync..."
                rsync -avz --delete "${RSYNC_EXCLUDE[@]}" \
                    "$PROJECT_DIR/" "$TC_HOST:$TC_PATH/"
                green "rsync 兜底完成"
                # rsync 后重置 TC git 对齐 origin/main，避免历史发散
                echo "同步 TC git 历史..."
                if ssh "$TC_HOST" "cd $TC_PATH && git fetch origin && git reset --hard origin/main && git submodule update --init"; then
                    green "TC git 历史已对齐 origin/main"
                else
                    yellow "TC git 对齐失败，下次部署前需手动修复"
                fi
            fi
        fi
    else
        yellow "[dry-run] 跳过 TC git pull"
    fi

    # 4. 检查 sync 是否正在运行
    if [ "$dry" = "false" ]; then
        if ssh "$TC_HOST" "systemctl is-active --quiet joplinai-sync.service" 2>/dev/null; then
            yellow "注意: joplinai-sync 正在运行中，center_api 重启期间其状态保存可能暂时失败（有错误处理兜底）"
        fi
    else
        yellow "[dry-run] 跳过 sync 运行状态检查"
    fi

    # 5. 重启 center_api
    echo "重启远程服务: $TC_SERVICE"
    if [ "$dry" = "false" ]; then
        systemctl_cmd "$TC_HOST" "$TC_SERVICE" restart
        echo -n "等待 $TC_SERVICE 就绪"
        for i in $(seq 1 30); do
            if ssh "$TC_HOST" "curl -sf http://localhost:5003/health" 2>/dev/null; then
                echo ""
                green "$TC_SERVICE 健康检查通过 (${i}x3s)"
                break
            fi
            if [ $i -eq 30 ]; then
                echo ""
                red "$TC_SERVICE 健康检查失败: 90s 内未就绪"
            fi
            echo -n "."
            sleep 3
        done
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
            echo "  tc  — 腾讯云（远程），git pull (直连→代理→rsync兜底) + 重启 center_api"
            exit 1
            ;;
    esac

    green "部署完成。"
}

main "$@"
