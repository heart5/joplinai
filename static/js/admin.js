// 全局辅助函数
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// 切换用户启用/禁用（保持与原有模态框一致）
async function toggleUserActive(username, newState) {
    const action = newState ? '启用' : '禁用';
    if (!confirm(`确定要${action}用户 ${username} 吗？`)) return;
    try {
        const resp = await fetch('/api/admin/user/toggle_active', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({username, is_active: newState})
        });
        const result = await resp.json();
        if (result.success) {
            showToast(`用户已${action}`, 'success');
            loadUsers();  // 需引入 loadUsers 函数，此处假设加载由页面内联脚本定义
        } else {
            showToast('操作失败: ' + result.error, 'danger');
        }
    } catch (e) {
        showToast('网络错误', 'danger');
    }
}

// 简化版 toast（如果已有可忽略）
function showToast(message, type = 'success') {
    const toast = document.createElement('div');
    toast.className = `toast align-items-center text-bg-${type} border-0 position-fixed bottom-0 end-0 m-3`;
    toast.innerHTML = `<div class="d-flex"><div class="toast-body">${message}</div><button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button></div>`;
    document.body.appendChild(toast);
    const bsToast = new bootstrap.Toast(toast);
    bsToast.show();
    setTimeout(() => toast.remove(), 3000);
}
