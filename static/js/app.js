// ========== 【超级全局安全守卫 - 置于所有JS之前】 ==========
(function() {
    'use strict';
    
    // 1. 拦截全局错误事件（最关键）
    window.addEventListener('error', function(event) {
        // 检查错误是否来自 all.js 或包含 'reading' 的类型错误
        if (event.filename && (
            event.filename.includes('all.js') ||
            event.filename.includes('extension')
        )) {
            console.warn(
                '[安全守卫] 已静默一个外部脚本错误，页面功能不受影响。',
                '错误:', event.message,
                '来自:', event.filename
            );
            event.preventDefault(); // 阻止错误上报
            event.stopImmediatePropagation();
            return true; // 表示错误已处理
        }
        // 如果是“读取未定义属性0”这类泛型错误，也静默（覆盖 all.js 的错误）
        if (event.error instanceof TypeError && /reading\s+'?\d'?/.test(event.message)) {
            console.warn('[安全守卫] 已静默一个类型错误（疑似外部脚本引发）。');
            event.preventDefault();
            return true;
        }
    }, true); // 使用捕获阶段，尽可能早拦截
    
    // 2. 保护 addEventListener（您原有的逻辑，保留）
    const originalAddEventListener = EventTarget.prototype.addEventListener;
    EventTarget.prototype.addEventListener = function(type, listener, options) {
        const wrappedListener = function(...args) {
            try {
                return listener.apply(this, args);
            } catch (error) {
                if (error instanceof TypeError && error.message.includes("reading")) {
                    console.warn('[安全守卫] 事件监听器错误已静默。');
                    return;
                }
                throw error;
            }
        };
        return originalAddEventListener.call(this, type, wrappedListener, options);
    };
    
    console.log('[安全守卫] 超级守卫已激活，将抵御外部脚本干扰。');
})();
// ========== 守卫代码结束 ==========

<!-- 简易markdown解析器 -->
window.marked = {
    parse: function(text) {
        if (!text) return '';
        return text
            .replace(/^##### (.*$)/gm, '<h5>$1</h5>')
            .replace(/^#### (.*$)/gm, '<h4>$1</h4>')
            .replace(/^### (.*$)/gm, '<h3>$1</h3>')
            .replace(/^## (.*$)/gm, '<h2>$1</h2>')
            .replace(/^# (.*$)/gm, '<h1>$1</h1>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`([^`]+)`/g, '<code>$1</code>')
            .replace(/^> (.*$)/gm, '<blockquote>$1</blockquote>')
            .replace(/\n\n/g, '</p><p>')
            .replace(/\n/g, '<br/>')
            .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>')
            .replace(/^/, '<p>').replace(/$/, '</p>');
    }
};

<!-- 【核心】修复后的JavaScript -->
document.addEventListener('DOMContentLoaded', function() {
    console.log('[Init] DOM已加载，开始初始化...');

    // 1. 安全获取DOM元素函数
    function safeGet(id) {
        const el = document.getElementById(id);
        if (!el) console.error(`[Error] 未找到ID为 "${id}" 的元素`);
        return el;
    }

    const elements = {
        form: safeGet('askForm'),
        input: safeGet('questionInput'),
        button: safeGet('submitBtn'),
        loading: safeGet('loadingIndicator'),
        chat: safeGet('chatHistory'),
        sources: safeGet('sourcesList'),
    };


    // 2. 关键检查：如果表单不存在，则停止初始化
    if (!elements.form) {
        alert('页面初始化失败：关键组件缺失。请刷新页面。');
        return;
    }
    console.log('[Init] ✅ 所有DOM元素已就绪。');

    /**
     * 添加消息（最新消息置顶 + 时间标签 + 嵌入式参考笔记）
     * @param {string} sender - 'user' 或 'bot'
     * @param {string} text - 消息内容
     * @param {Date} timestamp - 可选，消息时间
     * @param {Array} attachedSources - 可选，附属于此条答案的参考笔记数组
     */
    function appendMessage(sender, text, timestamp = null, attachedSources = null) {
        // 1. 防御：确保聊天容器存在
        if (!elements.chat) {
            console.warn('[appendMessage] 忽略：chatHistory 容器不存在。');
            return;
        }
    
        // 2. 创建消息容器
        const msgDiv = document.createElement('div');
        msgDiv.className = sender === 'user' ? 'user-message' : 'bot-message';
    
        // 3. 生成时间字符串
        const msgTime = timestamp instanceof Date ? timestamp : new Date();
        const timeStr = formatMessageTime(msgTime);
    
        // 4. 构建消息内容
        const safeText = String(text || '');
        let contentHtml = '';
    
        if (sender === 'bot') {
            // AI 消息：包含答案主体和可选的参考笔记区块
            try {
                const renderedText = window.marked && window.marked.parse ? window.marked.parse(safeText) : escapeHtml(safeText);
                // 为这条答案生成一个唯一的ID，用于关联其参考笔记
                const sourceContainerId = 'src_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
                
                contentHtml = `
                    <strong>助手：</strong>
                    <div class="message-content">${renderedText}</div>
                    <!-- 嵌入式参考笔记区块（默认折叠） -->
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
                
                // 将 attachedSources 暂存到消息元素的 dataset 中，稍后渲染
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
            // 用户消息
            contentHtml = `
                <strong>您：</strong>
                <div class="message-content">${escapeHtml(safeText)}</div>
                <span class="message-time">${timeStr}</span>
            `;
        }
    
        msgDiv.innerHTML = contentHtml;
    
        // 5. 【关键】将新消息插入到历史列表的最顶部（最新在最上）
        if (elements.chat.firstChild) {
            elements.chat.insertBefore(msgDiv, elements.chat.firstChild);
        } else {
            elements.chat.appendChild(msgDiv);
        }
    
        // 6. 如果这是一条AI消息且有附带的参考笔记，立即渲染它们
        if (sender === 'bot' && msgDiv.dataset.sources) {
            try {
                const sources = JSON.parse(msgDiv.dataset.sources);
                const container = document.getElementById(msgDiv.dataset.sourceContainerId + '_list');
                if (container) {
                    renderSourcesIntoContainer(sources, container);
                }
            } catch(e) {
                console.warn('[appendMessage] 渲染嵌入式参考笔记失败:', e);
            }
        }
    
        // 7. 自动滚动到顶部（因为最新消息已置顶）
        elements.chat.scrollTop = 0;
    }
    
    /**
     * 将参考笔记列表渲染到指定的容器中
     * @param {Array} notesArray - 笔记数组
     * @param {HTMLElement} container - 目标UL容器
     */
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
            // 安全提取信息
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

    /**
     * 终极安全的更新参考笔记函数
     * 设计原则：即使传入 undefined/null，或内部数据结构任意变化，也绝不崩溃。
     */
    function updateSources(apiResult) {
        // 1. 终极防御：确保操作对象存在
        if (!elements.sources) {
            return; // 静默退出
        }
        
        // 清空现有内容
        elements.sources.innerHTML = '';
        
        // 2. 验证输入 - 如果输入无效，直接显示无来源并退出
        if (!apiResult || typeof apiResult !== 'object') {
            elements.sources.innerHTML = '<li><em>本次回答未引用特定笔记。</em></li>';
            return;
        }
        
        // 3. 【安全数据提取】不假设任何数据结构
        let notesArray = [];
        try {
            // 尝试所有可能的数据路径
            if (Array.isArray(apiResult.relevant_notes)) {
                notesArray = apiResult.relevant_notes;
            } else if (apiResult.metadata && Array.isArray(apiResult.metadata.sources)) {
                notesArray = apiResult.metadata.sources;
            } else if (Array.isArray(apiResult.sources)) {
                notesArray = apiResult.sources;
            }
            // 如果都不是数组，notesArray 保持为空数组
        } catch (e) {
            // 即使在属性访问时出错，也保持空数组
            notesArray = [];
        }
        
        // 4. 安全渲染
        if (!notesArray.length) {
            elements.sources.innerHTML = '<li><em>本次回答未引用特定笔记。</em></li>';
            return;
        }
        
        // 5. 使用 for 循环替代 forEach，更可控
        for (let i = 0; i < notesArray.length; i++) {
            const item = notesArray[i];
            // 即使 item 是 null/undefined，也安全处理
            const safeItem = item || {};
            const title = safeItem.title || safeItem.note_id || `参考 ${i + 1}`;
            const notebook = safeItem.notebook || (safeItem.tags ? (Array.isArray(safeItem.tags) ? safeItem.tags.join(', ') : String(safeItem.tags)) : '未知');
            const score = typeof safeItem.similarity === 'number' ? safeItem.similarity : (typeof safeItem.score === 'number' ? safeItem.score : 0);
            
            const li = document.createElement('li');
            li.className = 'source-item';
            li.innerHTML = `
                <strong>${escapeHtml(title)}</strong><br>
                <small>来源：${escapeHtml(notebook)} | 相关性：${score.toFixed(3)}</small>
            `;
            elements.sources.appendChild(li);
        }
    }

    /**
     * 格式化消息时间为友好格式
     */
    function formatMessageTime(date) {
        const now = new Date();
        const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
        const yesterday = new Date(today);
        yesterday.setDate(yesterday.getDate() - 1);
        const messageDay = new Date(date.getFullYear(), date.getMonth(), date.getDate());
    
        let dayStr = '';
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

    /**
     * HTML 转义（防止 XSS）
     */
    function escapeHtml(text) {
        if (text == null) return '';
        return String(text)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    function updateRawResponse(data) {
        if (!elements.raw) return;
        try {
            // 使用安全序列化，避免循环引用等问题
            const safeData = structuredClone ? structuredClone(data) : JSON.parse(JSON.stringify(data));
            elements.raw.textContent = JSON.stringify(safeData, null, 2);
        } catch (e) {
            console.warn('[updateRawResponse] 无法序列化响应数据，显示简化信息:', e);
            elements.raw.textContent = `{ “status”: “数据已接收，但包含不可序列化内容” }`;
        }
    }

    /**
     * 页面加载时，从服务器获取用户的历史问答记录并显示（确保倒序）
     */
    async function loadUserHistory() {
        const loadingIndicator = document.getElementById('loadingIndicator');
        if (loadingIndicator) loadingIndicator.style.display = 'block';
    
        try {
            const response = await fetch('/api/history?limit=50');
            const result = await response.json();
    
            if (result.success && Array.isArray(result.history)) {
                console.log(`[Init] 加载到 ${result.history.length} 条历史记录`);
                
                // 清空现有的欢迎消息（如果需要）
                elements.chat.innerHTML = '';

                // 关键：反向遍历数组，确保最新的记录（数组的第一项）最先被插入到聊天顶部
                for (let i = result.history.length - 1; i >= 0; i--) {
                    const item = result.history[i];
                    // 从历史记录的 metadata 中提取当时答案的参考笔记
                    let sourcesForThisAnswer = [];
                    if (item.metadata && Array.isArray(item.metadata.sources)) {
                        sourcesForThisAnswer = item.metadata.sources;
                    }

                    // 先添加用户问题（无来源）
                    appendMessage('user', item.question, new Date(item.created_at));
                    // 再添加AI答案，并传入当时的参考笔记
                    appendMessage('bot', item.answer, new Date(item.created_at), sourcesForThisAnswer);
                }

                // 如果没有历史，显示欢迎语
                if (result.history.length === 0) {
                    appendMessage('bot', '您好！我可以基于您的Joplin笔记库回答问题。请在上方输入您的问题。', new Date());
                }

            } else {
                console.warn('[Init] 加载历史记录失败:', result.error);
                appendMessage('bot', '您好！我可以基于您的Joplin笔记库回答问题。', new Date());
            }
        } catch (error) {
            console.error('[Init] 请求历史记录API失败:', error);
            appendMessage('bot', '您好！我可以基于您的Joplin笔记库回答问题。', new Date());
        } finally {
            if (loadingIndicator) loadingIndicator.style.display = 'none';
        }
    }

    // 在页面初始化完成后调用
    loadUserHistory();

    // 4. 绑定表单提交事件（核心）
    elements.form.addEventListener('submit', async function(event) {
        event.preventDefault();
        console.log('[Submit] 表单提交事件触发。');

        const question = (elements.input.value || '').trim();
        if (!question) {
            alert('请输入问题内容。');
            return;
        }

        // UI状态更新
        appendMessage('user', question);
        elements.input.value = '';
        elements.button.disabled = true;
        if (elements.loading) elements.loading.style.display = 'block';
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 90000); // 90秒超时

        try {
            const payload = {
                question: question,
                use_history: document.getElementById('useHistorySwitch')?.checked || true
            };
            console.log('[Fetch] 发送请求，载荷:', payload);

            // 关键：根据您的日志，后端路由可能是 /ask，但代码中是 /api/ask。这里做兼容尝试。
            let response;
            let apiError = null;

            // 先尝试 /api/ask（根据知识库代码）
            response = await fetch('/api/ask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
                signal: controller.signal
            }).catch(e => { apiError = e; });
            clearTimeout(timeoutId);

            if (!response || !response.ok) {
                const errMsg = apiError ? apiError.message : `HTTP ${response?.status}`;
                throw new Error(`请求失败: ${errMsg}`);
            }

            const result = await response.json();
            console.log('[API] 原始响应:', result);
            
            // ===================== 【核心修复区：绝对安全的响应处理】 =====================
            try {
                // 1. 验证响应基本结构
                if (!result || typeof result !== 'object') {
                    throw new Error('API返回的响应不是有效的JSON对象。');
                }
            
                // 2. 安全更新调试信息（原始响应）
                updateRawResponse(result);
            
                // 3. 安全更新参考笔记（传入整个result对象）
                updateSources(result);
            
                // 4. 安全提取并显示答案
                let finalAnswer = '';
                if (result.error) {
                    // 优先显示错误信息
                    finalAnswer = `⚠️ 抱歉，处理时遇到错误：${result.error}`;
                } else if (result.answer && typeof result.answer === 'string') {
                    // 正常答案
                    finalAnswer = result.answer;
                } else {
                    // 后备：尝试从其他可能字段提取
                    finalAnswer = result.response || result.text || result.content || '（服务已响应，但未能解析出答案内容。）';
                    console.warn('[API] 从非标准字段提取答案:', finalAnswer.substring(0, 100));
                }

                // 5. 渲染答案（appendMessage内部已做Markdown安全处理）
                // 安全提取 notesArray（复用之前的智能提取逻辑）
                let notesArray = [];
                if (Array.isArray(result.relevant_notes)) {
                    notesArray = result.relevant_notes;
                } else if (result.metadata && Array.isArray(result.metadata.sources)) {
                    notesArray = result.metadata.sources;
                }
                // 调用 appendMessage 时，将 notesArray 作为第四个参数传入
                appendMessage('bot', finalAnswer, null, notesArray);

            } catch (processingError) {
                // 捕获响应处理过程中的任何意外错误
                console.error('[API] 处理响应数据时发生内部错误:', processingError);
                appendMessage('bot', `🤖 答案生成成功，但在界面处理时遇到技术问题。详情请查看控制台。`);
            }
            // ===================== 修复结束 =====================
        } catch (error) {
            clearTimeout(timeoutId);
            if (error.name === 'AbortError') {
                appendMessage('bot', '请求超时（一分半钟过去了），请稍后再试。');
            }
            console.error('[Fetch] 请求过程出错:', error);
            appendMessage('bot', `请求失败：${error.message}`);
        } finally {
            // 恢复UI
            elements.button.disabled = false;
            if (elements.loading) elements.loading.style.display = 'none';
            elements.input.focus();
        }
    });

    // 5. 初始焦点
    if (elements.input) elements.input.focus();
    console.log('[Init] 🎉 初始化完成，等待提问。');
});