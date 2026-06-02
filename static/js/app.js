// ========== static/js/app.js ==========
// 功能：门户网站自定义JavaScript脚本，支持多会话问答系统

// ===== 全局变量（供控制台调试和提问请求使用）=====
window.currentSessionId = null;
// 聊天分页：全局消息数组 + 分页状态
window.allMessages = [];
window.chatPageSize = 10;   // 每页10个问答对
window.chatCurrentPage = 1;

// ===== Markdown 解析器：marked.js → 手写兜底 → 纯文本（三级降级）=====
try { mermaid.initialize({ startOnLoad: false, theme: 'default' }); } catch(e) {}

// 保存 marked.js CDN 库的引用（不被覆盖）
var _realMarked = (typeof marked !== 'undefined' && marked.parse) ? marked : null;

// 手写简易解析器（_realMarked 不可用时的兜底）
function _fallbackMarkdown(text) {
    if (!text) return '';
    var safe = String(text)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
    return safe
        .replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code class="language-$1">$2</code></pre>')
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        .replace(/^##### (.*$)/gm, '<h5>$1</h5>')
        .replace(/^#### (.*$)/gm, '<h4>$1</h4>')
        .replace(/^### (.*$)/gm, '<h3>$1</h3>')
        .replace(/^## (.*$)/gm, '<h2>$1</h2>')
        .replace(/^# (.*$)/gm, '<h1>$1</h1>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>')
        .replace(/^> (.*$)/gm, '<blockquote>$1</blockquote>')
        .replace(/\n\n/g, '</p><p>')
        .replace(/\n/g, '<br>')
        .replace(/^/, '<p>').replace(/$/, '</p>')
        .replace(/<p><\/p>/g, '');
}

// ===== Mermaid 中文标点规范化 =====
// AI 模型中文回答习惯性使用中文标点，mermaid 语法只认 ASCII。
// 此函数做系统化归一，顺序不可随意调整。
function normalizeMermaid(code) {
    // -1. 清除泄漏进mermaid代码的markdown fence残片
    //     LLM常在```后加。等标点，破坏fence闭合 → marked.js把```吞入代码块
    var fixed = code.replace(/^```[^\n]*$/gm, '').trim();
    // 0. 仅解码 &quot; → "：步骤4-6的正则需匹配字面量双引号
    //    不碰 &lt; &gt; &amp; —— 浏览器 innerHTML 自动解码，且保护 <br> 标签
    fixed = fixed.replace(/&quot;/g, '"');
    // 1. 中文分号→ASCII（全局安全：；仅 mermaid 语法符，不会出现在标签内）
    fixed = fixed.replace(/；/g, ';');

    // 2. 边界位置中文弯引号→ASCII 双引号
    //    弯引号紧随语法符（[{(|subgraph）后 → 左双引号
    //    弯引号在语法符（]})|行尾）前 → 右双引号
    //    注意：$1" 中的 \" 是 ASCII U+0022，不是中文弯引号 U+201D
    fixed = fixed.replace(/([\[{\(|]|subgraph\s+)“/g, '$1"');
    fixed = fixed.replace(/”(?=[\]\}\)|]|\s*$)/gm, '"');

    // 3. 标签内部残留中文弯引号→直角引号（避免 mermaid 解析错误）
    fixed = fixed.replace(/“/g, '「'); // " → 「
    fixed = fixed.replace(/”/g, '」'); // " → 」

    // 4. 含特殊字符的 subgraph 标题→英文双引号包裹（兜底）
    fixed = fixed.replace(
        /^([ \t]*subgraph\s+)([^\n"{]+[()（）[\]{}<>"'][^\n"]*?)(\s*)$/gm,
        function(m, prefix, title, suffix) {
            return prefix + '"' + title.trim() + '"' + suffix;
        }
    );

    // 5. 方括号标签内ASCII双引号→直角引号
    //    ["完整引用"]格式保留不动；[标签内含"quote"]→替换为「」
    //    正则尊重mermaid引用语法："..."内可含任意字符(含[ ])，非引用标签不含嵌套[ ]
    fixed = fixed.replace(/\[("(?:[^"\\]|\\.)*"|[^\]\[]*)\]/g, function(m, content) {
        if (/^\s*"[^"]*"\s*$/.test(content)) return m;
        if (content.indexOf('"') === -1) return m;
        var i = 0;
        var replaced = content.replace(/"/g, function() {
            return (i++ % 2 === 0) ? '「' : '」';
        });
        return '[' + replaced + ']';
    });

    // 6. 圆角矩形括号标签内ASCII双引号→直角引号
    //    ()内"会破坏mermaid解析——"被当作字符串定界符，导致)无法闭合
    fixed = fixed.replace(/\(([^)]+)\)/g, function(m, content) {
        if (content.indexOf('"') === -1) return m;
        var i = 0;
        var replaced = content.replace(/"/g, function() {
            return (i++ % 2 === 0) ? '「' : '」';
        });
        return '(' + replaced + ')';
    });

    // 7. subgraph ID [...] → subgraph "..."（消除ID与节点名冲突）
    //    mermaid中subgraph和节点共享命名空间，AI常将同一ID同时用于节点和子图
    fixed = fixed.replace(
        /^([ \t]*subgraph\s+)(\w+)\s+\["([^"]*)"\]\s*$/gm,
        function(m, prefix, id, title) {
            return prefix + '"' + title + '"';
        }
    );

    // 8. gantt任务行处理：|→:（mermaid 10.x语法）+ ASCII双引号→直角引号
    //    mermaid 9.x用|分隔gantt任务，10.x改为: —— LLM训练数据仍产出旧语法
    //    仅对gantt图表生效：flowchart/sequence用|作链接文本分隔符(如-->|text|)
    //    步骤5/6已处理[]/()内引号，本步骤仅处理gantt任务行(含|分隔符)
    if (/^\s*gantt\b/m.test(fixed)) {
        fixed = fixed.replace(
            /^([ \t]*)([^|\n]+?)[ \t]*\|[ \t]*(.+)$/gm,
            function(m, space, task, data) {
                var i = 0;
                var fixedTask = task.indexOf('"') === -1
                    ? task
                    : task.replace(/"/g, function() { return (i++ % 2 === 0) ? '「' : '」'; });
                return space + fixedTask + ' : ' + data;
            }
        );
    }

    return fixed;
}

// ===== Mermaid SVG 导出 PNG =====
function downloadMermaidAsPng(svgEl) {
    var clone = svgEl.cloneNode(true);
    var rect = svgEl.getBoundingClientRect();
    var w = Math.ceil(rect.width) || parseInt(svgEl.getAttribute('width')) || 800;
    var h = Math.ceil(rect.height) || parseInt(svgEl.getAttribute('height')) || 600;

    // 注入页面背景色的 rect，避免透明底导出后看不清
    var bg = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    bg.setAttribute('width', '100%');
    bg.setAttribute('height', '100%');
    bg.setAttribute('fill', '#ffffff');
    clone.insertBefore(bg, clone.firstChild);

    clone.setAttribute('width', w);
    clone.setAttribute('height', h);

    var data = new XMLSerializer().serializeToString(clone);
    var blob = new Blob([data], { type: 'image/svg+xml;charset=utf-8' });
    var url = URL.createObjectURL(blob);
    var img = new Image();
    img.onload = function() {
        var scale = 2;
        var canvas = document.createElement('canvas');
        canvas.width = w * scale;
        canvas.height = h * scale;
        var ctx = canvas.getContext('2d');
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        URL.revokeObjectURL(url);
        var a = document.createElement('a');
        a.href = canvas.toDataURL('image/png');
        a.download = 'mermaid_' + new Date().toISOString().slice(0, 10) + '.png';
        a.click();
    };
    img.onerror = function() {
        URL.revokeObjectURL(url);
        console.warn('[Mermaid] SVG→PNG 转换失败');
    };
    img.src = url;
}

function addMermaidExportButtons() {
    var containers = document.querySelectorAll('.mermaid');
    containers.forEach(function(el) {
        var svg = el.querySelector('svg');
        if (!svg || el.querySelector('.mermaid-export-btn')) return;

        var wrapper = document.createElement('div');
        wrapper.className = 'mermaid-wrapper';
        el.insertBefore(wrapper, svg);
        wrapper.appendChild(svg);

        var btn = document.createElement('button');
        btn.className = 'mermaid-export-btn';
        btn.innerHTML = '&#x2913;';
        btn.title = '下载为 PNG';
        btn.onclick = function(e) {
            e.stopPropagation();
            downloadMermaidAsPng(svg);
        };
        wrapper.appendChild(btn);
    });
}

// 三级降级渲染入口
window.parseMarkdown = function(text) {
    if (!text) return '';
    // 第1级：marked.js（CDN 本地化）
    if (_realMarked) {
        try {
            var html = _realMarked.parse(String(text));
            html = html.replace(
                /<pre><code class="language-mermaid">([\s\S]*?)<\/code><\/pre>/g,
                function(_, mermaidCode) {
                    return '<div class="mermaid">' + normalizeMermaid(mermaidCode) + '</div>';
                }
            );
            return html;
        } catch (e) {
            console.warn('[Markdown] marked.js 失败，降级手写解析器:', e.message);
        }
    }
    // 第2级：手写解析器
    try {
        return _fallbackMarkdown(String(text));
    } catch (e2) {
        console.warn('[Markdown] 手写解析器也失败，降级纯文本:', e2.message);
        // 第3级：纯文本
        return String(text).replace(/</g, '&lt;').replace(/>/g, '&gt;');
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

// ===== 翻页栏 =====
function createPaginationBar(page, totalPages, totalPairs) {
    var bar = document.createElement('div');
    bar.className = 'pagination-bar';
    bar.innerHTML =
        '<button class="page-btn" data-page="first" ' + (page <= 1 ? 'disabled' : '') + '>首页</button>' +
        '<button class="page-btn" data-page="prev" ' + (page <= 1 ? 'disabled' : '') + '>上一页</button>' +
        '<span class="page-info">第 ' + page + '/' + totalPages + ' 页（共 ' + totalPairs + ' 个问答）</span>' +
        '<button class="page-btn" data-page="next" ' + (page >= totalPages ? 'disabled' : '') + '>下一页</button>' +
        '<button class="page-btn" data-page="last" ' + (page >= totalPages ? 'disabled' : '') + '>末页</button>';
    return bar;
}

// ===== 渲染聊天页面 =====
function renderChatPage(page) {
    var totalPairs = Math.ceil(window.allMessages.length / 2);
    var totalPages = Math.ceil(totalPairs / window.chatPageSize);
    if (totalPages < 1) totalPages = 1;
    if (page < 1) page = 1;
    if (page > totalPages) page = totalPages;
    window.chatCurrentPage = page;

    var chat = document.getElementById('chatHistory');
    if (!chat) return;
    chat.innerHTML = '';

    // 0条消息 → 仅显示欢迎语
    if (window.allMessages.length === 0) {
        var welcome = document.createElement('div');
        welcome.className = 'bot-message';
        welcome.innerHTML = '<strong>助手：</strong> 您好！我可以基于您的Joplin笔记库回答问题。请在上方输入您的问题。';
        chat.appendChild(welcome);
        return;
    }

    // 顶部翻页栏
    if (totalPages > 1) {
        var topBar = createPaginationBar(page, totalPages, totalPairs);
        topBar.classList.add('top');
        chat.appendChild(topBar);
    }

    // 当前页消息范围（从数组末尾取，page1=最新）
    var endMsg = window.allMessages.length - (page - 1) * window.chatPageSize * 2;
    var startMsg = Math.max(0, endMsg - window.chatPageSize * 2);

    for (var i = endMsg - 1; i >= startMsg; i--) {
        var el = _buildMessageEl(window.allMessages[i]);
        // bot消息：关联对应的user提问（数组中bot的前一条即为配对提问）
        if (window.allMessages[i].sender === 'bot' && i > 0 && window.allMessages[i - 1].sender === 'user') {
            el.dataset.questionText = window.allMessages[i - 1].text;
        }
        chat.appendChild(el);
    }

    // 底部翻页栏
    if (totalPages > 1) {
        var bottomBar = createPaginationBar(page, totalPages, totalPairs);
        bottomBar.classList.add('bottom');
        chat.appendChild(bottomBar);
    }

    // 更新分享按钮状态
    _updateShareButtons();

    // 渲染 mermaid 图表 → 挂载导出按钮
    if (chat.querySelector('.mermaid')) {
        setTimeout(function() {
            try {
                mermaid.run().then(function() {
                    addMermaidExportButtons();
                }).catch(function(e) {
                    console.warn('[Mermaid] 渲染失败:', e);
                });
            } catch(e) { console.warn('[Mermaid] 同步错误:', e); }
        }, 100);
    }

    chat.scrollTop = 0;
}

// ===== 构建单条消息DOM（内部辅助）=====
function _buildMessageEl(msg) {
    var msgDiv = document.createElement('div');
    msgDiv.className = msg.sender === 'user' ? 'user-message' : 'bot-message';

    var timeStr = formatMessageTime(msg.timestamp instanceof Date ? msg.timestamp : new Date());
    var safeText = String(msg.text || '');
    var contentHtml = '';

    if (msg.sender === 'bot') {
        try {
            var renderedText = window.parseMarkdown
                ? window.parseMarkdown(safeText)
                : escapeHtml(safeText);
            var sourceContainerId = 'src_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);

            contentHtml =
                '<strong>助手：</strong>' +
                '<button class="share-btn" title="分享此回答">🔗</button>' +
                '<div class="message-content">' + renderedText + '</div>' +
                '<div class="source-collapse-section">' +
                    '<div class="source-toggle collapsed" data-bs-toggle="collapse" data-bs-target="#' + sourceContainerId + '" aria-expanded="false">' +
                        '<span class="icon">▾</span>' +
                        '<strong>📄 查看答案参考笔记</strong>' +
                    '</div>' +
                    '<div id="' + sourceContainerId + '" class="collapse">' +
                        '<ul class="source-list" id="' + sourceContainerId + '_list">' +
                            '<li><em>等待加载或本次回答未引用特定笔记。</em></li>' +
                        '</ul>' +
                    '</div>' +
                '</div>' +
                '<span class="message-time">' + timeStr + '</span>';

            if (msg.sources && Array.isArray(msg.sources)) {
                msgDiv.dataset.sources = JSON.stringify(msg.sources);
                msgDiv.dataset.sourceContainerId = sourceContainerId;
            }
        } catch (mdError) {
            console.warn('[appendMessage] Markdown渲染失败，降级为纯文本:', mdError);
            contentHtml =
                '<strong>助手：</strong>' +
                '<div class="message-content">' + escapeHtml(safeText) + '</div>' +
                '<span class="message-time">' + timeStr + '</span>';
        }
    } else {
        contentHtml =
            '<strong>您：</strong>' +
            '<div class="message-content">' + escapeHtml(safeText) + '</div>' +
            '<span class="message-time">' + timeStr + '</span>';
    }

    msgDiv.innerHTML = contentHtml;
    msgDiv.dataset.rawText = safeText;

    // 渲染参考笔记
    if (msg.sender === 'bot' && msgDiv.dataset.sources) {
        try {
            var sources = JSON.parse(msgDiv.dataset.sources);
            // 延迟到DOM插入后渲染，此处先存下引用
            setTimeout(function() {
                var container = document.getElementById(msgDiv.dataset.sourceContainerId + '_list');
                if (container) {
                    renderSourcesIntoContainer(sources, container);
                }
            }, 0);
        } catch (e) {
            console.warn('[appendMessage] 渲染嵌入式参考笔记失败:', e);
        }
    }

    return msgDiv;
}

// ===== 添加消息（push到数组 + 重新渲染）=====
function appendMessage(sender, text, timestamp = null, attachedSources = null) {
    // 加载历史时跳过DOM重建，仅填充数组
    if (arguments[4] === true) {
        window.allMessages.push({
            sender: sender,
            text: String(text || ''),
            timestamp: timestamp instanceof Date ? timestamp : new Date(),
            sources: attachedSources
        });
        return;
    }

    window.allMessages.push({
        sender: sender,
        text: String(text || ''),
        timestamp: timestamp instanceof Date ? timestamp : new Date(),
        sources: attachedSources
    });

    // 新消息 → 跳到首页（最新页）
    renderChatPage(1);
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

    // 3. 重置全局消息数组
    window.allMessages = [];

    // 4. 检查数据是否有效
    if (!historyData || !historyData.success || !Array.isArray(historyData.history)) {
        console.warn('[Session] 获取历史失败或数据格式异常');
        window.currentSessionId = sessionId;
        updateSessionHighlight(sessionId);
        renderChatPage(1);  // 显示空状态
        return;
    }

    const records = historyData.history;

    // 5. 反向遍历（从旧到新），填充消息数组（skipRender=true）
    for (let i = records.length - 1; i >= 0; i--) {
        const item = records[i];
        const question = item.question || '';
        const answer = item.answer || '';
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

        appendMessage('user', question, createdAt, null, true);
        appendMessage('bot', answer, createdAt, sources, true);
    }

    // 6. 渲染首页
    renderChatPage(1);

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

// ===== 分享状态管理（localStorage 持久化）=====
var SHARE_STORE_KEY = 'joplinai_shares';

function _loadShareStore() {
    try {
        return JSON.parse(localStorage.getItem(SHARE_STORE_KEY)) || {};
    } catch (e) { return {}; }
}

function _saveShareStore(store) {
    localStorage.setItem(SHARE_STORE_KEY, JSON.stringify(store));
}

function _qaHash(question, answer) {
    // 取问答首尾各40字符做简易指纹
    var q = (question || '').slice(0, 40) + (question || '').slice(-20);
    var a = (answer || '').slice(0, 40) + (answer || '').slice(-20);
    return btoa(unescape(encodeURIComponent(q + '|' + a))).replace(/[+/=]/g, '').slice(0, 32);
}

function _getShareForQA(question, answer) {
    var store = _loadShareStore();
    return store[_qaHash(question, answer)] || null;
}

function _saveShareForQA(question, answer, shareData) {
    var store = _loadShareStore();
    store[_qaHash(question, answer)] = {
        share_id: shareData.share_id,
        share_url: shareData.share_url,
        expires_at: shareData.expires_at
    };
    _saveShareStore(store);
}

function _removeShareForQA(question, answer) {
    var store = _loadShareStore();
    delete store[_qaHash(question, answer)];
    _saveShareStore(store);
}

// 遍历所有分享按钮，根据 localStorage 设置 shared 样式
function _updateShareButtons() {
    var buttons = document.querySelectorAll('.share-btn');
    buttons.forEach(function(btn) {
        var botMsg = btn.closest('.bot-message');
        if (!botMsg) return;
        var question = botMsg.dataset.questionText || '';
        var answer = botMsg.dataset.rawText || '';
        if (!question || !answer) return;
        if (_getShareForQA(question, answer)) {
            btn.classList.add('shared');
            btn.title = '已分享 — 点击查看/管理分享链接';
        } else {
            btn.classList.remove('shared');
            btn.title = '分享此回答';
        }
    });
}

// ===== 分享弹窗 =====
function showShareModal(shareData, opts) {
    opts = opts || {};
    var isExisting = !!opts.isExisting;
    var question = opts.question || '';
    var answer = opts.answer || '';

    // 移除已有弹窗
    var existing = document.querySelector('.share-modal-overlay');
    if (existing) existing.remove();

    var overlay = document.createElement('div');
    overlay.className = 'share-modal-overlay';
    overlay.innerHTML =
        '<div class="share-modal">' +
            '<div class="share-modal-header">' +
                '<h5>' + (isExisting ? '管理分享' : '分享此回答') + '</h5>' +
                '<button class="share-modal-close">&times;</button>' +
            '</div>' +
            '<div class="share-modal-body">' +
                '<label>分享链接</label>' +
                '<div class="share-link-row">' +
                    '<input type="text" class="share-link-input" value="' + escapeHtml(shareData.share_url) + '" readonly>' +
                    '<button class="share-copy-btn">复制</button>' +
                '</div>' +
                '<div class="share-expiry">有效期至：' + escapeHtml(shareData.expires_at ? shareData.expires_at.slice(0, 16) : '') + '</div>' +
                '<div class="share-actions">' +
                    (isExisting ? '<button class="share-new-btn">新建分享</button>' : '') +
                    '<button class="share-revoke-btn">撤销分享</button>' +
                '</div>' +
            '</div>' +
        '</div>';

    document.body.appendChild(overlay);

    // 关闭
    overlay.querySelector('.share-modal-close').onclick = function() { overlay.remove(); };
    overlay.addEventListener('click', function(e) { if (e.target === overlay) overlay.remove(); });

    // 复制
    overlay.querySelector('.share-copy-btn').onclick = function() {
        var input = overlay.querySelector('.share-link-input');
        input.select();
        document.execCommand('copy');
        var btn = overlay.querySelector('.share-copy-btn');
        btn.textContent = '已复制';
        setTimeout(function() { btn.textContent = '复制'; }, 2000);
    };

    // 撤销 — 清除 localStorage + 重置按钮状态
    overlay.querySelector('.share-revoke-btn').onclick = function() {
        if (!confirm('撤销后该分享链接将立即失效，确定？')) return;
        fetch('/api/share/' + encodeURIComponent(shareData.share_id), { method: 'DELETE' })
            .then(function(r) { return r.json(); })
            .then(function(data) {
                if (data.ok) {
                    _removeShareForQA(question, answer);
                    _updateShareButtons();
                    overlay.querySelector('.share-modal-body').innerHTML =
                        '<div class="share-revoked-msg">该分享已被撤销，链接已失效。</div>';
                } else {
                    alert('撤销失败，请稍后重试。');
                }
            })
            .catch(function(e) {
                console.error('[Share] 撤销失败:', e);
                alert('撤销请求失败，请检查网络后重试。');
            });
    };

    // 新建分享（仅在"管理分享"模式下出现）
    var newBtn = overlay.querySelector('.share-new-btn');
    if (newBtn) {
        newBtn.onclick = function() {
            overlay.remove();
            _doCreateShare(question, answer);
        };
    }
}

// ===== 辅助函数：获取分享按钮对应的问答内容 =====
function getShareContent(shareBtn) {
    var botMsg = shareBtn.closest('.bot-message');
    if (!botMsg) return null;
    var question = botMsg.dataset.questionText || '';
    var answer = botMsg.dataset.rawText || '';
    if (!question || !answer) return null;
    return { question: question, answer: answer };
}

// 执行分享创建（供点击和"新建分享"复用）
function _doCreateShare(question, answer) {
    fetch('/api/share', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: question, answer: answer })
    })
    .then(function(r) { return r.json(); })
    .then(function(data) {
        if (data.ok) {
            _saveShareForQA(question, answer, data);
            _updateShareButtons();
            showShareModal(data, { question: question, answer: answer });
        } else {
            alert('创建分享失败：' + (data.error || '未知错误'));
        }
    })
    .catch(function(e) {
        console.error('[Share] 创建分享失败:', e);
        alert('网络请求失败，请稍后重试。');
    });
}
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

    // 7.5 翻页按钮点击委托
    document.getElementById('chatHistory').addEventListener('click', function(e) {
        var btn = e.target.closest('.page-btn');
        if (!btn || btn.disabled) return;
        var target = btn.dataset.page;
        var totalPairs = Math.ceil(window.allMessages.length / 2);
        var totalPages = Math.ceil(totalPairs / window.chatPageSize) || 1;
        var newPage = window.chatCurrentPage;
        if (target === 'first') newPage = 1;
        else if (target === 'prev') newPage = window.chatCurrentPage - 1;
        else if (target === 'next') newPage = window.chatCurrentPage + 1;
        else if (target === 'last') newPage = totalPages;
        if (newPage !== window.chatCurrentPage) renderChatPage(newPage);
    });

    // 7.6 分享按钮点击委托
    document.getElementById('chatHistory').addEventListener('click', function(e) {
        var shareBtn = e.target.closest('.share-btn');
        if (!shareBtn) return;
        e.stopPropagation();
        var content = getShareContent(shareBtn);
        if (!content) {
            alert('无法获取问答内容，请刷新页面后重试。');
            return;
        }
        // 检查是否已有分享记录
        var existing = _getShareForQA(content.question, content.answer);
        if (existing) {
            showShareModal(existing, {
                isExisting: true,
                question: content.question,
                answer: content.answer
            });
            return;
        }
        _doCreateShare(content.question, content.answer);
    });

    // 8. 初始焦点
    if (questionInput) questionInput.focus();
    console.log('[Init] 🎉 初始化完成');
});
