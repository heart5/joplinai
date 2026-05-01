// ===== 全局变量 =====
let currentPage = 1;
let totalPages = 1;
const perPage = 20;

// ===== 页面初始化 =====
document.addEventListener('DOMContentLoaded', function() {
    // 加载操作类型下拉框
    loadActionOptions();
    // 加载第一页日志
    loadLogs(currentPage);
    
    // 绑定搜索表单提交事件
    document.getElementById('filterForm').addEventListener('submit', function(e) {
        e.preventDefault();
        currentPage = 1;
        loadLogs(1);
    });
});

// ===== 加载操作类型 =====
async function loadActionOptions() {
    try {
        const resp = await fetch('/api/admin/audit/actions');
        const data = await resp.json();
        if (data.success) {
            const select = document.getElementById('filterAction');
            data.actions.forEach(action => {
                const option = document.createElement('option');
                option.value = action;
                option.textContent = action;
                select.appendChild(option);
            });
        }
    } catch (e) {
        console.error('加载操作类型失败', e);
    }
}

// ===== 加载日志数据 =====
async function loadLogs(page) {
    const tbody = document.getElementById('logsTableBody');
    
    try {
        // 构建查询参数
        const params = new URLSearchParams({
            page: page,
            per_page: perPage
        });
        
        const username = document.getElementById('filterUsername').value.trim();
        const action = document.getElementById('filterAction').value;
        const startDate = document.getElementById('filterStartDate').value;
        const endDate = document.getElementById('filterEndDate').value;
        
        if (username) params.append('username', username);
        if (action) params.append('action', action);
        if (startDate) params.append('start_date', startDate);
        if (endDate) params.append('end_date', endDate);
        
        const resp = await fetch(`/api/admin/audit/logs?${params}`);
        const data = await resp.json();
        
        if (!data.success) {
            throw new Error('加载失败');
        }
        
        // 更新统计
        document.getElementById('totalCount').textContent = data.total;
        document.getElementById('currentPage').textContent = data.page;
        currentPage = data.page;
        totalPages = data.total_pages;
        
        // 渲染表格
        if (data.logs.length === 0) {
            tbody.innerHTML = '<tr><td colspan="7" class="text-center text-muted">暂无匹配的日志记录</td></tr>';
        } else {
            let html = '';
            for (const log of data.logs) {
                // 修复时区问题：强制将无时区标记的时间视为UTC
                let time = '-';
                if (log.timestamp) {
                    const utcString = log.timestamp.endsWith('Z') || log.timestamp.includes('+') 
                        ? log.timestamp 
                        : log.timestamp + 'Z';
                    time = new Date(utcString).toLocaleString('zh-CN', { 
                        timeZone: 'Asia/Shanghai', 
                        year: 'numeric', month: '2-digit', day: '2-digit',
                        hour: '2-digit', minute: '2-digit', second: '2-digit',
                        hour12: false 
                    });
                }
                html += `<tr>
                    <td>${log.id}</td>
                    <td><small>${time}</small></td>
                    <td><code>${escapeHtml(log.username)}</code></td>
                    <td>${escapeHtml(log.display_name)}</td>
                    <td><span class="badge bg-info">${escapeHtml(log.action)}</span></td>
                    <td><small>${escapeHtml(log.details || '-')}</small></td>
                    <td><small>${escapeHtml(log.ip_address || '-')}</small></td>
                </tr>`;
            }
            tbody.innerHTML = html;
        }
        
        // 渲染分页
        renderPagination();
        
    } catch (e) {
        tbody.innerHTML = `<tr><td colspan="7" class="text-center text-danger">
            加载失败：${e.message}
            <button class="btn btn-sm btn-outline-primary ms-2" onclick="loadLogs(${currentPage})">重试</button>
        </td></tr>`;
    }
}

// ===== 渲染分页 =====
function renderPagination() {
    const ul = document.getElementById('pagination');
    let html = '';
    
    // 上一页
    html += `<li class="page-item ${currentPage <= 1 ? 'disabled' : ''}">
        <a class="page-link" href="#" onclick="loadLogs(${currentPage - 1}); return false;">«</a>
    </li>`;
    
    // 页码
    const startPage = Math.max(1, currentPage - 2);
    const endPage = Math.min(totalPages, currentPage + 2);
    
    if (startPage > 1) {
        html += `<li class="page-item"><a class="page-link" href="#" onclick="loadLogs(1); return false;">1</a></li>`;
        if (startPage > 2) {
            html += `<li class="page-item disabled"><span class="page-link">...</span></li>`;
        }
    }
    
    for (let i = startPage; i <= endPage; i++) {
        html += `<li class="page-item ${i === currentPage ? 'active' : ''}">
            <a class="page-link" href="#" onclick="loadLogs(${i}); return false;">${i}</a>
        </li>`;
    }
    
    if (endPage < totalPages) {
        if (endPage < totalPages - 1) {
            html += `<li class="page-item disabled"><span class="page-link">...</span></li>`;
        }
        html += `<li class="page-item"><a class="page-link" href="#" onclick="loadLogs(${totalPages}); return false;">${totalPages}</a></li>`;
    }
    
    // 下一页
    html += `<li class="page-item ${currentPage >= totalPages ? 'disabled' : ''}">
        <a class="page-link" href="#" onclick="loadLogs(${currentPage + 1}); return false;">»</a>
    </li>`;
    
    ul.innerHTML = html;
}

// ===== 重置筛选 =====
function resetFilter() {
    document.getElementById('filterUsername').value = '';
    document.getElementById('filterAction').value = '';
    document.getElementById('filterStartDate').value = '';
    document.getElementById('filterEndDate').value = '';
    currentPage = 1;
    loadLogs(1);
}

// ===== 清理旧日志 =====
function showClearModal() {
    new bootstrap.Modal(document.getElementById('clearModal')).show();
}

async function confirmClear() {
    const days = parseInt(document.getElementById('clearDays').value) || 90;
    
    if (days < 7) {
        alert('保留天数不能少于7天');
        return;
    }
    
    if (!confirm(`确定要删除 ${days} 天前的审计日志吗？此操作不可恢复！`)) {
        return;
    }
    
    try {
        const resp = await fetch('/api/admin/audit/clear', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({before_days: days})
        });
        const data = await resp.json();
        
        if (data.success) {
            bootstrap.Modal.getInstance(document.getElementById('clearModal')).hide();
            showToast(`已清理 ${data.deleted} 条日志`, 'success');
            loadLogs(1);
        } else {
            showToast('清理失败: ' + data.error, 'danger');
        }
    } catch (e) {
        showToast('网络错误', 'danger');
    }
}

// ===== 导出CSV =====
function exportLogs() {
    // 构建当前筛选参数
    const params = new URLSearchParams();
    const username = document.getElementById('filterUsername').value.trim();
    const action = document.getElementById('filterAction').value;
    const startDate = document.getElementById('filterStartDate').value;
    const endDate = document.getElementById('filterEndDate').value;
    
    if (username) params.append('username', username);
    if (action) params.append('action', action);
    if (startDate) params.append('start_date', startDate);
    if (endDate) params.append('end_date', endDate);
    
    // 使用当前筛选条件导出
    window.open(`/api/admin/audit/export?${params}`, '_blank');
}

// ===== 辅助函数 =====
function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = String(text);
    return div.innerHTML;
}

function showToast(message, type = 'success') {
    // 复用 admin.js 中的 showToast（如果存在）
    if (typeof window.showToast === 'function') {
        window.showToast(message, type);
        return;
    }
    
    const toast = document.createElement('div');
    toast.className = `toast align-items-center text-bg-${type} border-0 position-fixed bottom-0 end-0 m-3`;
    toast.innerHTML = `<div class="d-flex"><div class="toast-body">${message}</div><button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button></div>`;
    document.body.appendChild(toast);
    const bsToast = new bootstrap.Toast(toast);
    bsToast.show();
    setTimeout(() => toast.remove(), 3000);
}
