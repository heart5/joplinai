// ========== static/js/app.js ==========
// 功能：门户网站自定义JavaScript脚本，支持多会话问答系统

// ===== 全局变量（供控制台调试和提问请求使用）=====
window.currentSessionId = null;

// ===== 简易 Markdown 解析器（增强版）=====
window.marked = {
    parse: function(text) {
        try {
            if (!text) return '';
            // 先 HTML 转义，防止 XSS
            let safeText = String(text)
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;');
            
            // 然后进行 Markdown 语法转换
            return safeText
                // 代码块（必须优先处理，避免内部被其他规则解析）
                .replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code class="language-$1">$2</code></pre>')
                .replace(/`([^`]+)`/g, '<code>$1</code>')
                // 引用块
                .replace(/^> (.*$)/gm, '<blockquote>$1</blockquote>')
                // 标题（从大到小，避免混淆）
                .replace(/^##### (.*$)/gm, '<h5>$1</h5>')
                .replace(/^#### (.*$)/gm, '<h4>$1</h4>')
                .replace(/^### (.*$)/gm, '<h3>$1</h3>')
                .replace(/^## (.*$)/gm, '<h2>$1</h2>')
                .replace(/^# (.*$)/gm, '<h1>$1</h1>')
                // 加粗和斜体
                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                .replace(/\*(.*?)\*/g, '<em>$1</em>')
                // 链接（确保URL完整）
                .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>')
                // 图片（替换为链接）
                .replace(/!\[([^\]]*)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">📷 图片：$1</a>')
                // 换行处理（注意顺序）
                .replace(/\n\n/g, '</p><p>')   // 段落
                .replace(/\n/g, '<br>')         // 普通换行
                // 包裹段落（避免重复包裹）
                .replace(/^/, '<p>')
                .replace(/$/, '</p>')
                // 清理空段落
                .replace(/<p><\/p>/g, '');
        } catch (e) {
            console.warn('[Markdown] 解析错误，返回原始文本:', e);
            return String(text || '');
        }
    }
};


// ===== 工具函数：HTML转义（防止XSS）=====
function escapeHtml(text) {
    if (text == null) return '';
    return String(text)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

// ===== 工具函数：格式化时间 =====
function formatMessageTime(date) {
    if (!(date instanceof Date) || isNaN(date.getTime())) return '';
    // ★★★ 关键修复：如果日期字符串不包含时区信息，则视为 UTC 时间 ★★★
    if (typeof date === 'string' && !date.endsWith('Z') && !date.includes('+') && !date.includes('-')) {
        date = date + 'Z';  // 添加 UTC 标记
    }
    const now = new Date();
    const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    const yesterday = new Date(today.getTime() - 86400000);
    const messageDay = new Date(date.getFullYear(), date.getMonth(), date.getDate());

    let dayStr;
    if (messageDay.getTime() === today.getTime()) {
        dayStr = '今天';
    } else if (messageDay.getTime() === yesterday.getTime()) {
        dayStr = '昨天';
    } else {
        dayStr = `${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')}`;
    }
    const timeStr = `${String(date.getHours()).padStart(2, '0')}:${String(date.getMinutes()).padStart(2, '0')}`;
    return `${dayStr} ${timeStr}`;
}

// ===== 渲染参考笔记到指定容器 =====
function renderSourcesIntoContainer(notesArray, container) {
    if (!container || !notesArray || !Array.isArray(notesArray)) return;
    container.innerHTML = '';
    if (notesArray.length === 0) {
        container.innerHTML = '<li><em>本次回答未引用特定笔记。</em></li>';
        return;
    }
    notesArray.forEach((note, idx) => {
        const li = document.createElement('li');
        li.className = 'source-item';
        const title = note.title || note.note_id || `参考笔记 ${idx + 1}`;
        const notebook = note.notebook || (note.tags ? (Array.isArray(note.tags) ? note.tags.join(', ') : String(note.tags)) : '未知');
        const score = typeof note.similarity === 'number' ? note.similarity : (typeof note.score === 'number' ? note.score : 0);
        li.innerHTML = `
            <div class="source-title">${escapeHtml(title)}</div>
            <div class="source-meta">来源：${escapeHtml(notebook)} | 相关性：${score.toFixed(3)}</div>
        `;
        container.appendChild(li);
    });
}

// ===== 添加消息（最新置顶 + 时间标签 + 嵌入式参考笔记）=====
function appendMessage(sender, text, timestamp = null, attachedSources = null) {
    // 确保聊天容器存在
    const chat = document.getElementById('chatHistory');
    if (!chat) {
        console.warn('[appendMessage] 忽略：chatHistory 容器不存在。');
        return;
    }

    const msgDiv = document.createElement('div');
    msgDiv.className = sender === 'user' ? 'user-message' : 'bot-message';

    const msgTime = timestamp instanceof Date ? timestamp : new Date();
    const timeStr = formatMessageTime(msgTime);
    const safeText = String(text || '');
    let contentHtml = '';

    if (sender === 'bot') {
        try {
            // 使用增强版 Markdown 解析
            const renderedText = window.marked && typeof window.marked.parse === 'function'
                ? window.marked.parse(safeText)
                : escapeHtml(safeText);
            const sourceContainerId = 'src_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);

            contentHtml = `
                <strong>助手：</strong>
                <div class="message-content">${renderedText}</div>
                <div class="source-collapse-section">
                    <div class="source-toggle collapsed" data-bs-toggle="collapse" data-bs-target="#${sourceContainerId}" aria-expanded="false">
                        <span class="icon">▾</span>
                        <strong>📄 查看答案参考笔记</strong>
                    </div>
                    <div id="${sourceContainerId}" class="collapse">
                        <ul class="source-list" id="${sourceContainerId}_list">
                            <li><em>等待加载或本次回答未引用特定笔记。</em></li>
                        </ul>
                    </div>
                </div>
                <span class="message-time">${timeStr}</span>
            `;

            if (attachedSources && Array.isArray(attachedSources)) {
                msgDiv.dataset.sources = JSON.stringify(attachedSources);
                msgDiv.dataset.sourceContainerId = sourceContainerId;
            }
        } catch (mdError) {
            console.warn('[appendMessage] Markdown渲染失败，降级为纯文本:', mdError);
            contentHtml = `
                <strong>助手：</strong>
                <div class="message-content">${escapeHtml(safeText)}</div>
                <span class="message-time">${timeStr}</span>
            `;
        }
    } else {
        contentHtml = `
            <strong>您：</strong>
            <div class="message-content">${escapeHtml(safeText)}</div>
            <span class="message-time">${timeStr}</span>
        `;
    }

    msgDiv.innerHTML = contentHtml;

    // 插入到顶部（最新消息置顶）
    if (chat.firstChild) {
        chat.insertBefore(msgDiv, chat.firstChild);
    } else {
        chat.appendChild(msgDiv);
    }

    // 渲染参考笔记（如果存在）
    if (sender === 'bot' && msgDiv.dataset.sources) {
        try {
            const sources = JSON.parse(msgDiv.dataset.sources);
            const container = document.getElementById(msgDiv.dataset.sourceContainerId + '_list');
            if (container) {
                renderSourcesIntoContainer(sources, container);
            }
        } catch (e) {
            console.warn('[appendMessage] 渲染嵌入式参考笔记失败:', e);
        }
    }

    // 自动滚动到顶部
    chat.scrollTop = 0;
}

// ===== 更新参考笔记面板（侧边栏）=====
function updateSources(apiResult) {
    const sourcesList = document.getElementById('sourcesList');
    if (!sourcesList) return;
    sourcesList.innerHTML = '';
    if (!apiResult || typeof apiResult !== 'object') {
        sourcesList.innerHTML = '<li><em>本次回答未引用特定笔记。</em></li>';
        return;
    }
    let notesArray = [];
    try {
        if (Array.isArray(apiResult.relevant_notes)) {
            notesArray = apiResult.relevant_notes;
        } else if (apiResult.metadata && Array.isArray(apiResult.metadata.sources)) {
            notesArray = apiResult.metadata.sources;
        } else if (Array.isArray(apiResult.sources)) {
            notesArray = apiResult.sources;
        }
    } catch (e) {
        notesArray = [];
    }
    if (!notesArray.length) {
        sourcesList.innerHTML = '<li><em>本次回答未引用特定笔记。</em></li>';
        return;
    }
    for (let i = 0; i < notesArray.length; i++) {
        const safeItem = notesArray[i] || {};
        const title = safeItem.title || safeItem.note_id || `参考 ${i + 1}`;
        const notebook = safeItem.notebook || (safeItem.tags ? (Array.isArray(safeItem.tags) ? safeItem.tags.join(', ') : String(safeItem.tags)) : '未知');
        const score = typeof safeItem.similarity === 'number' ? safeItem.similarity : (typeof safeItem.score === 'number' ? safeItem.score : 0);
        const li = document.createElement('li');
        li.className = 'source-item';
        li.innerHTML = `
            <div class="source-title">${escapeHtml(title)}</div>
            <div class="source-meta">来源：${escapeHtml(notebook)} | 相关性：${score.toFixed(3)}</div>
        `;
        sourcesList.appendChild(li);
    }
}

// ===== 切换会话：加载指定会话的历史（全局函数）=====
window.loadSessionHistory = async function(sessionId) {
    console.log('[Session] 开始切换至:', sessionId);
    if (!sessionId) {
        console.warn('[Session] 无效的 sessionId');
        return;
    }
    
    // 1. 激活会话（后端标记为当前活动会话）
    try {
        const actResp = await fetch(`/api/chat_sessions/${sessionId}/activate`, { method: 'POST' });
        const actData = await actResp.json();
        console.log('[Session] 激活结果:', actData);
    } catch (e) {
        console.error('[Session] 激活请求失败:', e);
    }

    // 2. 获取该会话的历史记录
    let historyData = null;
    try {
        const resp = await fetch(`/api/history?session_id=${sessionId}&limit=50`);
        historyData = await resp.json();
        console.log('[Session] 获取历史响应:', historyData);
    } catch (e) {
        console.error('[Session] 获取历史请求失败:', e);
        return;
    }

    // 3. 清空聊天框（直接操作DOM，不依赖 elements 变量）
    const chatEl = document.getElementById('chatHistory');
    if (!chatEl) {
        console.error('[Session] 找不到 chatHistory 元素');
        return;
    }
    chatEl.innerHTML = '';

    // 4. 检查数据是否有效
    if (!historyData || !historyData.success || !Array.isArray(historyData.history)) {
        console.warn('[Session] 获取历史失败或数据格式异常');
        // 显示默认欢迎语
        appendMessage('bot', '这是一个新的对话，您可以开始提问。', new Date());
        // 仍然更新当前会话ID和高亮
        window.currentSessionId = sessionId;
        updateSessionHighlight(sessionId);
        return;
    }

    const records = historyData.history;

    // 5. 反向遍历（从旧到新），使用 appendMessage 渲染
    for (let i = records.length - 1; i >= 0; i--) {
        const item = records[i];
        const question = item.question || '';
        const answer = item.answer || '';
        // 将 item.created_at 标准化为带时区标记的字符串
        const normalizedTime = item.created_at
            ? (item.created_at.endsWith('Z') || item.created_at.includes('+')
                ? item.created_at
                : item.created_at + 'Z')
            : null;
        const createdAt = normalizedTime ? new Date(normalizedTime) : new Date();
        let sources = [];
        if (item.metadata && Array.isArray(item.metadata.sources)) {
            sources = item.metadata.sources;
        }

        appendMessage('user', question, createdAt);
        appendMessage('bot', answer, createdAt, sources);
    }

    // 6. 如果历史为空，显示欢迎语
    if (records.length === 0) {
        appendMessage('bot', '这是一个新的对话，您可以开始提问。', new Date());
    }

    // 7. 更新全局当前会话ID
    window.currentSessionId = sessionId;

    // 8. 高亮当前会话列表项
    updateSessionHighlight(sessionId);

    console.log('[Session] ✅ 切换完成，当前会话:', sessionId);
};

// ===== 辅助函数：更新会话列表高亮 =====
function updateSessionHighlight(sessionId) {
    document.querySelectorAll('.session-item').forEach(el => el.classList.remove('active'));
    const target = document.querySelector(`.session-item[data-session-id="${sessionId}"]`);
    if (target) target.classList.add('active');
}

// ===== 页面初始化 =====
document.addEventListener('DOMContentLoaded', function() {
    console.log('[Init] DOM已加载，开始初始化...');

    // 1. 初始化当前会话ID（从第一个活跃的会话元素获取）
    const activeItem = document.querySelector('.session-item.active');
    if (activeItem) {
        window.currentSessionId = activeItem.dataset.sessionId;
        console.log('[Init] 初始会话ID:', window.currentSessionId);
    }

    // 2. 绑定左侧会话列表点击事件（使用全局 loadSessionHistory）
    document.querySelectorAll('.session-item').forEach(item => {
        item.addEventListener('click', function(e) {
            if (e.target.closest('.rename-btn') || e.target.closest('.delete-btn')) return;
            const sessionId = this.dataset.sessionId;
            // 调用全局函数
            window.loadSessionHistory(sessionId);
        });
    });

    // 3. 绑定新建会话按钮
    const newSessionBtn = document.getElementById('newSessionBtn');
    if (newSessionBtn) {
        newSessionBtn.addEventListener('click', async function() {
            const name = prompt('请输入新对话名称：', '新对话');
            if (!name) return;
            try {
                const resp = await fetch('/api/chat_sessions', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ name })
                });
                const result = await resp.json();
                if (result.success) {
                    // 切换到新会话（无需刷新页面）
                    await window.loadSessionHistory(result.session_id);
                    // 动态添加列表项
                    const sessionList = document.getElementById('sessionList');
                    const li = document.createElement('li');
                    li.className = 'list-group-item session-item active';
                    li.dataset.sessionId = result.session_id;
                    li.innerHTML = `
                        <span class="session-name">${escapeHtml(name)}</span>
                        <span class="badge bg-light text-muted rounded-pill message-count">0</span>
                        <div class="session-actions">
                            <button class="btn btn-sm btn-outline-secondary rename-btn" title="重命名">✏️</button>
                            <button class="btn btn-sm btn-outline-danger delete-btn" title="删除">🗑️</button>
                        </div>
                    `;
                    if (sessionList.firstChild) {
                        sessionList.insertBefore(li, sessionList.firstChild);
                    } else {
                        sessionList.appendChild(li);
                    }
                }
            } catch (e) {
                console.error('[NewSession] 创建失败:', e);
            }
        });
    }

    // 4. 绑定重命名按钮（事件委托）
    document.getElementById('sessionList')?.addEventListener('click', async function(e) {
        const renameBtn = e.target.closest('.rename-btn');
        if (renameBtn) {
            e.stopPropagation();
            const li = renameBtn.closest('.session-item');
            const sessionId = li.dataset.sessionId;
            const currentName = li.querySelector('.session-name').textContent;
            const newName = prompt('请输入新名称：', currentName);
            if (!newName || newName === currentName) return;
            try {
                const resp = await fetch(`/api/chat_sessions/${sessionId}`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ name: newName })
                });
                if (resp.ok) {
                    li.querySelector('.session-name').textContent = newName;
                }
            } catch (e) {
                console.error('[Rename] 重命名失败:', e);
            }
        }
    });

    // 5. 绑定删除按钮（事件委托）
    document.getElementById('sessionList')?.addEventListener('click', async function(e) {
        const deleteBtn = e.target.closest('.delete-btn');
        if (deleteBtn) {
            e.stopPropagation();
            if (!confirm('确定要删除该对话及其所有历史记录吗？')) return;
            const li = deleteBtn.closest('.session-item');
            const sessionId = li.dataset.sessionId;
            try {
                const resp = await fetch(`/api/chat_sessions/${sessionId}`, { method: 'DELETE' });
                if (resp.ok) {
                    li.remove();
                    if (sessionId === window.currentSessionId) {
                        // 自动切换到第一个会话
                        const first = document.querySelector('.session-item');
                        if (first) {
                            window.loadSessionHistory(first.dataset.sessionId);
                        } else {
                            location.reload(); // 没有会话了，刷新页面重建
                        }
                    }
                }
            } catch (e) {
                console.error('[Delete] 删除失败:', e);
            }
        }
    });

    // 6. 绑定提问表单提交
    const askForm = document.getElementById('askForm');
    const questionInput = document.getElementById('questionInput');
    const submitBtn = document.getElementById('submitBtn');
    const loadingIndicator = document.getElementById('loadingIndicator');

    if (askForm && questionInput && submitBtn) {
        askForm.addEventListener('submit', async function(event) {
            event.preventDefault();
            const question = questionInput.value.trim();
            if (!question) {
                alert('请输入问题内容。');
                return;
            }

            // 显示用户问题
            appendMessage('user', question);
            questionInput.value = '';
            submitBtn.disabled = true;
            if (loadingIndicator) loadingIndicator.style.display = 'block';

            // 构造请求数据
            const payload = {
                question: question,
                use_history: document.getElementById('useHistorySwitch')?.checked || true,
                session_id: window.currentSessionId  // 使用全局会话ID
            };
            console.log('[Submit] 发送请求:', payload);

            try {
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), 150000);

                const response = await fetch('/api/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                    signal: controller.signal
                });
                clearTimeout(timeoutId);

                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }

                const result = await response.json();
                console.log('[Submit] 响应:', result);

                // 更新参考笔记侧边栏
                updateSources(result);

                // 提取回答内容
                let answer = '';
                if (result.error) {
                    answer = `⚠️ 抱歉，处理时遇到错误：${result.error}`;
                } else if (result.answer && typeof result.answer === 'string') {
                    answer = result.answer;
                } else {
                    answer = result.response || result.text || result.content || '（服务已响应，但未能解析出答案内容。）';
                }

                // 提取参考笔记
                let notes = [];
                if (Array.isArray(result.relevant_notes)) {
                    notes = result.relevant_notes;
                } else if (result.metadata && Array.isArray(result.metadata.sources)) {
                    notes = result.metadata.sources;
                }

                // 显示AI回答
                appendMessage('bot', answer, null, notes);

                // 更新左侧会话列表的消息数量（+1）
                const activeListItem = document.querySelector(`.session-item[data-session-id="${window.currentSessionId}"]`);
                if (activeListItem) {
                    const countBadge = activeListItem.querySelector('.message-count');
                    if (countBadge) {
                        let currentCount = parseInt(countBadge.textContent, 10) || 0;
                        countBadge.textContent = currentCount + 1;
                        // 更新样式：从0变成>0时，改变徽标颜色（可选）
                        countBadge.className = 'badge bg-secondary rounded-pill message-count';
                    }
                }

            } catch (error) {
                if (error.name === 'AbortError') {
                    appendMessage('bot', '请求超时（150秒），请稍后再试。');
                } else {
                    console.error('[Submit] 请求失败:', error);
                    appendMessage('bot', `请求失败：${error.message}`);
                }
            } finally {
                submitBtn.disabled = false;
                if (loadingIndicator) loadingIndicator.style.display = 'none';
                questionInput.focus();
            }
        });
    }

    // 7. 加载初始会话历史（使用全局函数）
    if (window.currentSessionId) {
        window.loadSessionHistory(window.currentSessionId);
    } else {
        // 如果没有找到会话，显示欢迎语
        appendMessage('bot', '您好！我可以基于您的Joplin笔记库回答问题。请在上方输入您的问题。', new Date());
    }

    // 8. 初始焦点
    if (questionInput) questionInput.focus();
    console.log('[Init] 🎉 初始化完成');
});
