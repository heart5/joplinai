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


### 2026年5月17日

**文档全量更新至当前架构**（commit `f801188`）：

- CLAUDE.md：服务表拆分向量化引擎(TC)+嵌入推理(HCX)，数据流更新，已知问题补充 ChromaDB Docker/Ollama 公网暴露
- README.md：分布式架构重新描述，修正文件名，刷新最近更新
- TECHNICAL_MANUAL.md：9处修正 — 拓扑图/序列图/协议表/ER schema/ChromaDB集合名和元数据字段/健康检查端口
- CHANGELOG.md：拆分5月16-17日合并条目为独立日期


### 2026年5月16日

**架构重构：DeepSeek增强功能命名 → provider-agnostic AI增强命名**（commit `05a64a9`）：

- **文件重命名**（`git mv` 保留历史）：`aimod/deepseek_enhancer.py` → `aimod/note_enhancer.py`、`aimod/deepseek_client.py` → `aimod/cache_client.py`、`tests/test_deepseek.py` → `tests/test_note_enhancer.py`
- **类重命名**：`DeepSeekCacheClient` → `CacheClient`
- **函数重命名**：`deepseek_describe_images()` → `describe_images()`、`enhance_by_deepseek_for_summary_tags()` → `enhance_chunk_metadata()`、缓存操作 `deepseek_cache_*()` → `enhance_cache_*()`
- **DB 表重命名**（TC 生产）：`ALTER TABLE deepseek_cache RENAME TO enhance_cache`（5572条记录完整迁移）
- **ChromaDB 元数据字段**：`deepseek_enhanced` → `enhanced`（新 chunk 生效，存量无此字段无需迁移）
- **URL 路由重命名**：`/cache/deepseek/*` → `/cache/enhance/*`（5个端点）
- **CLI 参数**：`--use-deepseek` → `--use-cloud`
- **日志清理**：所有 "deepseek增强" → "AI增强"/"增强"、"DeepSeek生成答案" → "云端模型生成答案"
- **保留不动**：真正涉及 DeepSeek API 服务的命名（API Key、URL、模型名 `deepseek-chat`、`_call_deepseek_api_directly()` 等）

**Bug 修复**（commit `ff9a748`）：
- `get_call_stats()`/`reset_call_stats()` 被 jupytext sync 误放入 markdown cell 导致注释化，TC 验证测试时才暴露

**技术收获**：
- TC ChromaDB 通过 Docker 运行（端口 8009→8000），`vector_db_manager.py` 默认 8000 无效，依赖云端配置覆盖
- 向量化是分布式执行：引擎在 TC（分块/编排/入库），嵌入推理在 HCX（Ollama `149.30.242.156:11434`）
- jupytext `# %% [markdown]` 后缺 `# %%` 会把后续代码当 markdown 注释掉——`tools/check_jupytext_comment.py` 可自动检测


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


### 2026年5月14日


**Bug 修复（Phase 3 重构回归）**：

- `TemplateNotFound: login.html` — Phase 3 将 `Flask(__name__)` 从项目根移到 `src/web_app/__init__.py`，默认 `template_folder`/`static_folder` 变为子目录。修复：`create_app()` 显式传入绝对路径。
- `BuildError: admin_dashboard` — Blueprint 自动前缀 `admin.`，4 个模板文件仍用旧端点名。修复：统一改为 `admin.admin_dashboard`。
- `NameError: 'model_name' is not defined` (`joplinai.py:459`) — 变量未定义。修复：改为 `config["embedding_model"]`。
- `AttributeError: 'text_prep'` (`embedding_generator.py`) — `_set_chunk_size()` 在 `self.text_prep` 初始化前被调用。修复：调整 `__init__` 中初始化顺序。
- `KeyError: 'content'` (`deepseek_client.py:76`) — center_api 缓存未命中返回 `{"found": False}` 不含 `content` 键。修复：增加 `data.get("found")` 检查，远程未命中回退本地 SQLite。
- `UnboundLocalError: 'user_role'` (`qa_system.py`) — 直接 API 调用时 `user_identity` 为 None，`user_role`/`user_display_name`/`sys_prompt` 未初始化。修复：`ask()` 和 `api_ask()` 入口校验身份，无身份拒绝请求（400）。
- `meta_hash` 不反映增强结果 — `meta_hash` 仅含 `source_note_tags + source_notebook_title`，与 DeepSeek 增强结果无关。增强失败的 chunk 入库后，下次 meta_hash 不变被跳过，永远得不到补增强。修复：`meta_hash` 纳入 `chunk_summary` 和增强后 `tags`，增强状态变化时 hash 不同触发重处理。

**服务运维**：

- HCX `joplin-qa-api` / `joplin-web-app` 重启后加载最新代码（gunicorn 进程 8h+ 未重启仍运行旧版本）
- TC git pull 直连失败时通过 `HTTPS_PROXY` 代理连接
- 全链路验证通过：sync 25 个笔记本无 bug 报错，之前 `deepseek_missing=True` 的笔记成功重处理

**center_api 日志增强**：

- 4 个蓝图文件全部关键端点补充业务日志：
  - `state_routes.py`：`batch_load`/`batch_save` 记录模型名与条数
  - `history_routes.py`：`notebook_record` 记录笔记本处理统计（更新/失败/总数，块增/跳/清）
  - `cache_routes.py`：`deepseek set` 记录缓存写入；命中日志静默（量太大）
  - `user_routes.py`：登录/创建/删除/角色变更/密码重置/审计清理 记录操作事件
- 修复日志重复输出：`center_api/__init__.py` 去掉 `logging.basicConfig()`，handler 直接挂在 `joplinai_center_api` logger，设 `propagate=False` 避免 `func/logme` root handler 双重输出
- TC gunicorn 移除 `--capture-output`，worker 日志改回 journal 输出（`journalctl -u joplinai-center-api`）

**Bug 修复（jupytext 代码注释，再次发生）**：

- 4 个路由文件 markdown cell (`# %% [markdown]`) 后缺少 `# %%` code cell 分隔符，jupytext sync 将全部 45 个 Flask 端点注释掉
- 修复：各文件 `# # Flask 端点` markdown cell 后补充 `# %%`，取消注释全部端点代码

**jupytext 代码注释自动检测**：

- 新增 `tools/check_jupytext_comment.py` 检测脚本，扫描被注释掉的 `@route`/`def api_` 模式
- 挂载到 `.git/hooks/pre-commit`：jupytext sync 后立即检测，有问题拒绝提交
- 挂载到 `.pre-commit-config.yaml` 和 CI（`.github/workflows/ci.yml`），三层防线

**图片 Vision 处理功能**：

- 新增 `aimod/image_processor.py`：从 Joplin API 获取图片资源，通过 magic bytes 检测格式（JPEG/PNG/WebP/GIF/BMP），并发拉取（ThreadPoolExecutor 5 并发），限制 8MB
- `aimod/text_preprocessor.py` 新增 `extract_resource_ids()`：从笔记 body 提取 `![...](:/resource_id)` 中的 32 位 hex resource_id
- `aimod/deepseek_enhancer.py` 新增 `ollama_vision_describe()`：调用本地 Ollama vision 模型描述图片，结合上下文生成中文描述
- `aimod/embedding_generator.py` `split_into_semantic_chunks()` 接受 `image_descriptions` 参数，图片描述自动拼接为 chunk 上下文前缀
- `joplinai.py` pipeline 集成：`process_note_chunks()` 提取 resource_id → 拉取图片 → Ollama vision 描述 → 合并到分块文本
- Vision 模型选型：DeepSeek 云 API 不支持图片输入 → 回退本地 Ollama；HCX CPU-only 测试后选定 `minicpm-v`（8B/5.5GB，中文 OCR 可用 57-121s/图），`qwen3-vl:2b` 超时不可用
- 图片获取/vision 调用失败时静默回退纯文本模式，不影响无图片笔记
- Vision 开关：`data/joplinai.ini` 的 `[vision] enabled` 控制，默认开启


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
