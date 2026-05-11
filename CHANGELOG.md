# joplinai 项目变更日志

<!--
格式约定：
- 日期倒序（最新在上）
- 日期分隔符：### YYYY年M月D日
- 当天无变更则不添加条目
- 同一日有多个提交则按分类归入该日条目
-->

本文档记录 Claude Code 协助下的所有项目变更。

### 2026年5月11日

**user_manager.py 重构** — 代码质量改进，不影响对外接口和前端：

- P0 死代码：删除空的 `update_user_role` 空壳方法（已有正确实现在后）
- P1 角色校验修复：`update_user_role` 白名单 `["admin", "colleague"]` → `["admin", "team_leader", "team_member"]`，与 DB CHECK 约束一致
- P2 审计日志模板抽取：5 个方法的重复审计模式抽为 `_audit_admin_action()` 辅助方法
- P3 命名统一：`change_user_display_name` → `update_user_display_name`；`_get_user_by_username` → `get_user_by_username`
- P4 方法拆分：`get_active_chat_session` 中 `web_{username}` 旧格式迁移逻辑抽为 `_migrate_legacy_chat_session()`
- P5 一致性：`get_qa_history_by_session` 返回格式统一含 `metadata`；数据库连接管理统一为 `with` 语句
- 远程优先模式下不再预创建本地 SQLite 数据库（延迟初始化）
- 影响文件：`src/user_manager.py`、`aimod/center_client.py`、`joplin_web_app.py`（净减 76 行）

**笔记状态与用户管理集中化** — 将笔记处理状态和用户管理从本地存储迁移到 TC 的 center_api，统一 remote-first + local fallback 模式：

- `note_process_state` 表 + 6 张用户管理表新增至 `joplinai_center.db`（现共 10 张表）
- `ProcessStateClient`：远程优先，本地 JSON 回退
- `UserManagerClient`：实现 `UserManager` 全部公开接口（~22 方法），远程优先，本地 SQLite 回退
- `joplinai_center_api.py` 新增 `/state/*`、`/auth/*`、`/users/*`、`/chat_sessions/*`、`/qa/*`、`/audit/*` 共 24 个端点
- `joplinai.py` 通过 `state_client` 接入远程状态服务
- 数据迁移：1029 条状态 + 3 用户 + 29 会话 + 260 审计 + 159 问答 + 6 会话
- 迁移脚本列索引 bug 修复（`c[0]` → `c[1]`）
- 部署架构确认：TC 数据核心 / HCX 推理+门户

**集中式数据中心扩展** — 将原 `joplinai_cache_api` 扩展为统一数据中心 `joplinai_center_api`，所有主机共享同一份数据：

- 统一数据库 4 表 → 10 表（新增历史、状态、用户系列表）
- 全面重命名 `cache` → `center`（文件、变量名、cloud 配置 key）
- URL 发现逻辑：生产主机不配 cloud URL 自动走 localhost；其他主机指定 remote URL

**腾讯云部署修复**：

- 绑定地址 `127.0.0.1` → `0.0.0.0:5003`，允许外部主机访问
- systemd：Group=club、MemoryMax=800M、ExecStart 路径适配
- API Key 加载：TC 侧 Joplin 不可用，本地 `data/joplinai.ini` 提供 fallback

**严重 Bug 修复**：

- `_init_db()` 仅 `main()` 调用 → gunicorn 导入时模块级调用建表
- 数据损失：旧 DB 文件已删除，迁移必须先验证成功再删旧数据

**文档更新**：CLAUDE.md / README / CHANGELOG 同步更新部署架构与重构记录。

### 2026年5月10日

**集中式缓存服务**：

- DeepSeek 摘要/标签缓存迁移至 `joplinai_center_api`，多主机共享
- `DeepSeekCacheClient` 实现 remote-first + local SQLite fallback
- 缓存 LRU 淘汰策略：超过 50000 条时淘汰最旧 1000 条
- 探测缓存迁移至中心服务，`ProbeCacheClient` 同上模式

**功能增强**：

- `joplinai.py` 支持 `--note_ids` 按指定笔记 ID 向量化（虚拟笔记集）
- 合并 `OptimizedJoplinQASystem` 到 `JoplinQASystem`，删除死代码

**代码质量**：

- 日志语句统一标记字符/token 单位及模块前缀
- API Key 优先从云端配置读取，导入安全加固
- 缓存服务完全独立于 Joplin，避免 `getapi()` 级联失败
- 缓存服务 URL 按 deviceid 区分，与 Ollama/Chroma 配置模式一致

### 2026年5月9日

**分块与嵌入优化**：

- 新增 `--batch-size` 分批处理，force_update 支持笔记本级分批推进
- 移除重分块机制，嵌入超长时改用安全值截取回退
- 嵌入阶段区分长度错误和瞬态错误，消除无意义重分块
- HTTP 500 也算长度错误信号，避免重分块兜底漏掉
- 优化自适应分块探测策略
- 统一迭代分块策略，废弃二次拆分和防过碎逻辑
- 高命中分块建议按推荐尺寸聚合，去除无意义哈希

**缓存分析报告优化**：

- 报告表头与数据内容对应优化
- DeepSeek 缓存报告标题和日期更新修复
- 多项 bug 修复与代码完善

**项目布局**：

- jupytext 文件布局重组，ipynb 与 py 同目录配对
- 移除自适应分块探测的内存+持久化缓存，简化代码
- 分块阶段预生成嵌入并缓存复用

### 2026年5月8日

- DeepSeek 增强失败时标记 `deepseek_missing`，下次运行自动重试
- 更新 func 子模块，修复远程回退 Path 未导入问题

### 2026年5月7日

**Joplin 服务连通性增强**：

- `jpfuncs.getapi()` 支持远程 Joplin server 回退（通过 `fallback_url`/`fallback_token` 配置）
- 远程回退同样需要 ping 验证连通性
- 用 `joplin server status` 替代 ping 验证本地状态
- 避免 `getcfpoptionvalue` 循环依赖，修复误导性错误信息

**项目结构规范化**：

- 将 `func/` 注册为正式的 git 子模块
- 统一命名：`web_app.py` → `joplin_web_app.py`
- 三个服务入口文件回迁到根目录，适配 systemd 服务路径
- pathmagic 结构修正：根目录保留配对，src/ 独立实现，统一 func 子模块模式

### 2026年5月6日

- 添加 CLAUDE.md 项目文档，完善 .gitignore 排除规则
- 重构项目结构：.py 迁入 src/，.ipynb 迁入 notebooks/，jupytext.toml 映射
- 统一 jupytext 版本号为 1.19.1，优化导入方式使用 pathmagic 上下文管理
- 更新 README.md 文档
