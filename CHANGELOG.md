# Changelog

## 2026-05-11

### user_manager.py 重构

**代码质量改进**，不影响对外接口和前端：

- **P0 死代码**：删除空的 `update_user_role` 空壳方法（已有正确实现在后）
- **P1 角色校验修复**：`update_user_role` 白名单 `["admin", "colleague"]` → `["admin", "team_leader", "team_member"]`，与 DB CHECK 约束一致
- **P2 审计日志模板抽取**：`reset_user_password`、`update_user_active_status`、`update_user_role`、`update_user_display_name`、`delete_user` 五个方法的重复审计模式抽为 `_audit_admin_action()` 辅助方法
- **P3 命名统一**：
  - `change_user_display_name` → `update_user_display_name`（与类内其他 `update_*` 方法风格一致）
  - `_get_user_by_username` → `get_user_by_username`（去下划线，正式公开，3 个 web_app 路由直接调用）
- **P4 方法拆分**：`get_active_chat_session` 中 `web_{username}` 旧格式迁移逻辑抽为 `_migrate_legacy_chat_session()`
- **P5 一致性**：`get_qa_history_by_session` 返回格式统一含 `metadata` 字段；数据库连接管理统一为 `with` 语句

影响文件：`src/user_manager.py`、`aimod/center_client.py`、`joplin_web_app.py`（净减 76 行）。所有 Flask 路由路径、前端模板/JS 均不受影响。

### 部署架构确认

- **腾讯云**：Joplin Server、ChromaDB 向量库、`joplinai_center_api`（0.0.0.0:5003）— 数据核心
- **恒创云**：Ollama LLM、`joplin_qa_api`（127.0.0.1:5000）、`joplin_web_app`（127.0.0.1:5001）— 推理+门户

### 集中式数据中心（joplinai_center_api）

**重大架构变更**：将原 `joplinai_cache_api` 扩展为统一数据中心 `joplinai_center_api`，所有主机共享同一份数据。

- **统一数据库**：4 张表合并到 `data/joplinai_center.db`（`deepseek_cache`、`probe_cache`、`notebook_history`、`global_run_history`）
- **历史数据库集中化**：`aitaskreporter.py` 新增 `HistoryClient` 远程模式，`joplinai.py` 初始化时注入 client，写入统一 DB
- **全面重命名 `cache` → `center`**：文件、变量名、cloud 配置 key 统一更名
  - `joplinai_cache_api.py` → `joplinai_center_api.py`
  - `aimod/cache_client.py` → `aimod/center_client.py`
  - `joplinai_cache_url` → `joplinai_center_url`
  - `joplinai_cache_api_key` → `joplinai_center_api_key`
  - systemd 服务 `joplinai-cache-api.service` → `joplinai-center-api.service`
- **URL 发现逻辑反转**：生产主机（腾讯云）不配 cloud URL 自动走 `127.0.0.1:5003`；其他主机指定 `joplinai_center_url` 指向 TC 公网 IP

### 腾讯云部署修复

- **绑定地址**：`127.0.0.1` → `0.0.0.0:5003`，允许外部主机访问
- **systemd 配置修正**：Group=club、MemoryMax=800M、ExecStart 路径适配 `/usr/miniconda3/envs/newlsp/bin/gunicorn`
- **API Key 加载**：TC 侧 Joplin 不可用，创建本地 `data/joplinai.ini` 提供 fallback 配置

### 严重 Bug 修复

- **`_init_db()` 仅 `main()` 中调用 → gunicorn 导入不执行**：模块级新增 `_init_db()` 调用，确保 worker 导入时即建表。新增 `/health` 端点 `db_ok` 检查。
- **数据损失**：迁移脚本因上述 bug INSERT 全部静默失败，旧 DB 文件已删除，数据无法恢复。教训：迁移必须先验证成功再删旧数据。

## 2026-05-10

### 集中式探测缓存

- 探测缓存迁移至 `joplinai_center_api`，多主机共享，避免重复探测
- `ProbeCacheClient` 实现 remote-first + local-fallback 模式

### 其他

- `joplinai.py` 支持 `--note_ids` 按指定笔记 ID 向量化
- 合并 `OptimizedJoplinQASystem` 到 `JoplinQASystem`，移除死代码
- API Key 优先从云端配置读取，导入安全加固
- 缓存服务完全独立于 Joplin，避免 `getapi()` 级联失败
- 日志语句统一标记字符/token 单位及模块前缀
