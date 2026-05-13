---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# joplinai 项目变更日志

<!--
格式约定：
- 日期倒序（最新在上）
- 日期分隔符：### YYYY年M月D日
- 当天无变更则不添加条目
- 同一日有多个提交则按分类归入该日条目
-->

本文档记录 Claude Code 协助下的所有项目变更。


### 2026年5月13日


**Bug 修复**：

- `joplin_web_app.py` gunicorn 不兼容：Phase 3C 重构后 `app = create_app()` 被放入 `if __name__ == "__main__":` 块内，gunicorn 导入 `joplin_web_app:app` 时找不到 Flask 实例。修复：将 `app` 提升到模块级别。
- jupytext hook 代码注释化：`src/cli.py`、`aimod/center_api/*_routes.py` 中 markdown cell (`# %% [markdown]`) 后缺少显式 `# %%` code cell 标记，导致 jupytext `--sync` 将后续代码识别为 markdown 并注释掉。修复：在各 markdown cell 后的代码块前补充 `# %%`，删除残留的损坏 `.ipynb`。

**技术文档体系建设**：

- 创建 `docs/TECHNICAL_MANUAL.md` 技术手册（10 张 Mermaid 图）：部署拓扑、服务架构、数据流全景、向量化/Q&A/Remote-First 序列图、数据库 ER 图、配置链路、部署流程、模块依赖
- 修正手册中 ChromaDB 部署位置：腾讯云 Docker `8009→8000`，HCX 通过 `chromadb.HttpClient` 远程连接
- `CLAUDE.md`、`README.md`、`CHANGELOG.md` 全面更新至 Phase 5 完成态（架构、源布局、新模块、工程规范）
- 4 份重要 `.md` 文档（`CLAUDE.md`、`README.md`、`docs/CHANGELOG.md`、`docs/TECHNICAL_MANUAL.md`）添加 jupytext YAML frontmatter 关联 `.ipynb`，支持 JupyterLab 双向同步
- `CHANGELOG.md` 移至 `docs/` 目录，与 TECHICAL_MANUAL.md 集中管理

**TC 部署策略优化**：

- TC 代码同步从 rsync 改为 git pull 优先三级策略：直连→clash 代理→rsync 兜底
- `deploy/deploy.sh` 更新：自动 git push、TC git pull 直连/代理/rsync 回退、rsync 后重置 TC git 历史 + 子模块更新
- GitHub Personal Access Token 补充 `workflow` scope
- TC git 历史对齐修复（`git reset --hard origin/main` + `pull.rebase = false`）
- 全链路验证通过：git push → TC git pull 直连 → service restart


### 2026年5月12日


**Phase 5 — CI/CD + 部署标准化**：

- 新建 `.pre-commit-config.yaml`：flake8 自动检查（同 CI 参数）
- 新建 `.github/workflows/ci.yml`：push/PR 触发 flake8 lint + pytest
- 新建 `docker-compose.yml`：本地开发拓扑（center_api、qa_api、web_app、chromadb）
- 新建 `deploy/deploy.sh`：统一部署脚本（`hcx` 本地重启 / `tc` rsync+远程重启，支持 `--dry-run`）
- systemd 更新：`joplinai-center-api.service` gunicorn 命令从 `joplinai_center_api:app` 改为 `"aimod.center_api:create_app()"`
- `.gitignore` 补充 `.env`

**Phase 4 — 工程化增强**：

- **`__all__` 全覆盖**：30+ 文件添加 `__all__` 明确导出列表（`aimod/`、`src/`、`aimod/center_api/`、`aimod/center_client/`）
- **`__repr__` 关键类**：`CacheResult`、`VectorDBManager`、`EmbeddingGenerator`、`RunTracker`、`QASystem`
- **返回类型标注**：所有公开函数补充返回类型 (`-> None` / `-> str` / `-> Optional[str]` 等)
- **导入规范化**：全项目统一包前缀（`src.` / `aimod.` / `func.`），消除隐式相对导入
- **日志统一**：`aimod/__init__.py` 新增 `get_logger(name)`，自动继承 `func/logme` 的 handler/level
- **`src/config_manager.py`**：新增 `get_all()` 别名方法
- **函数签名规范化**：`days` 参数默认值统一，部分函数重命名
- **`.ipynb_checkpoints/`**：清理 20 个残留文件

**Phase 3 — 大文件拆分收尾**：

- `aimod/center_client.py` → `aimod/center_client/` 包（5 个独立客户端：`deepseek_cache.py`、`probe_cache.py`、`history.py`、`process_state.py`、`user_client.py`）
- `aimod/embedding_generator.py` → `embedding_generator.py` + `chunk_optimizer.py` + `text_splitter.py`
- `src/queryanswer.py` → `queryanswer.py` + `qa_config.py` + `prompt_manager.py` + `qa_system.py`
- `joplinai_center_api.py` → `aimod/center_api/` 包（`cache_routes.py`、`history_routes.py`、`state_routes.py`、`user_routes.py`），app factory 模式 `create_app()`
- `src/` 新增 `cli.py`（从 `joplinai.py` 拆分 CLI 参数解析）
- 类名去 Joplin 前缀：`JoplinAIRunTracker` → `RunTracker`、`JoplinQASystem` → `QASystem`
- `pathmagic.context()` → `pathmagic.Context()` 全项目更新，用 `rootfile` 标记定位取代 cwd 相对路径
- `aimod/` 和 `src/` 添加 `__init__.py`，转为正则 Python 包

**报告生成统一化** — 将所有缓存/探测报告迁移至 center_api stats 端点，消除本地数据库依赖：

- 新增 `/cache/deepseek/report` 端点（5 维度缓存统计：按模型/操作/日期/命中率/趋势）
- 新增 `/cache/probe/report` 端点（按模型/块大小/安全长度/趋势 4 维度统计）
- 新建 `src/report_writer.py` 统一报告模块（格式化 + Joplin 写入 + CLI 入口）
- `aitaskreporter.py` → `run_tracker.py`，`JoplinAIRunTracker` 回归数据采集本职
- 删除 `CacheStatsAnalyzer` + `CacheReportGenerator`（~800 行），净减 1500+ 行
- `write_to_joplin()` 支持 `config_key` 参数，不同报告类型独立缓存 note_id
- CLI 用法：`python src/report_writer.py [--type deepseek_cache|probe_cache|all] [--output joplin|stdout]`
- 缓存报告不再依赖本地 `deepseek_cache.db`，完全 remote-first

**Bug 修复**：

- `deepseek_cache_report` 连接未设 `row_factory=sqlite3.Row`，导致 `dict()` 转换失败
- `report_writer.py` 中 `func` 子模块导入缺少 `pathmagic.context()` 包裹，子目录/systemd 下找不到模块
- `getinivaluefromcloud` 应从 `func.jpfuncs` 导入而非 `func.configpr`

**jupytext 工作流规范化**：

- 新建 `jupytext.toml`：`formats = "ipynb,py:percent"`，py 与 ipynb 同目录配对
- `.gitignore`：`*.ipynb` 全面屏蔽，ipynb 永不再入库
- git pre-commit hook：staged `.py` 文件自动 `jupytext --sync` 生成对应 ipynb
- git filter-branch：从历史中清除全部 ipynb 文件，避免调试信息泄露
- 工作流：编辑 `.py` → 自动生成 `.ipynb`（仅本地阅览，不入库）

**代码大清理（5批次）** — 删函数/清导入/去注释/移迁移/正命名：

- 删除 8 个零引用死函数（484 行）：`cleanup_old_entries`、`get_deepseek_embedding`、`get_ollama_embedding_other`、`migrate_content_hash`、`refresh_estimated_date`、`get_existing_chunk_hashes_for_note_other`、`_get_user_by_username`、`_generate_answer_with_deepseek`、`sanitize_config`、`get_common_prompt_variants`
- 清理 7 个文件 50+ 个未使用导入（83 行），如 `argparse`、`hashlib`、`chromadb`、大量未用 `jpfuncs`/`configpr`/`getid` 导入
- 清理 10 处注释掉的无效代码块（40 行），含 PersistentClient、hash 修复逻辑、旧正则、孤立方法引用
- 移除一次性迁移代码（294 行）：删除 `_migrate_db.py`、`migrate_add_notebook_id()`、`migrate_all_chunks_with_author()` 及 joplinai.py 中注释迁移块
- `_generate_answer_with_deepseek_optimized` → `_generate_answer_with_deepseek`（原非优化版已删）


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
