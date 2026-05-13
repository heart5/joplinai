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

# joplinai 技术手册


面向维护人员的技术参考文档，覆盖部署拓扑、服务架构、数据流、数据库结构、配置链路与运维操作。

---


## 1. 部署拓扑


```mermaid
flowchart TB
    subgraph TC["腾讯云 (TC)"]
        direction TB
        TC_JOPLIN["Joplin Server<br/>端口 41184"]
        TC_CHROMA["ChromaDB<br/>Docker 端口 8009→8000"]
        TC_CENTER["center_api<br/>端口 5003 (0.0.0.0)"]
        TC_SQLITE[("joplinai_center.db")]
        TC_TIMER_SYNC["sync.timer (每8小时)"]
        TC_TIMER_REPORT["cache-report.timer (每日)"]

        TC_CENTER --> TC_SQLITE
        TC_TIMER_SYNC -.-> TC_CENTER
        TC_TIMER_REPORT -.-> TC_CENTER
    end

    subgraph HCX["恒创云 (HCX)"]
        direction TB
        HCX_OLLAMA["Ollama LLM"]
        HCX_QA["qa_api<br/>端口 5000 (127.0.0.1)"]
        HCX_WEB["web_app<br/>端口 5001 (127.0.0.1)"]
        HCX_CLI["joplinai.py (手动 CLI)"]

        HCX_QA --> HCX_OLLAMA
        HCX_WEB --> HCX_QA
    end

    DEEPSEEK["DeepSeek API (外部)"]
    USERS["用户浏览器"]

    USERS --> HCX_WEB
    HCX_QA -.-> TC_CENTER
    HCX_QA -.-> TC_CHROMA
    HCX_WEB -.-> TC_CENTER
    HCX_CLI -.-> TC_CHROMA
    HCX_CLI -.-> DEEPSEEK
    HCX_CLI -.-> TC_CENTER
```

**要点：**

| 项目 | 腾讯云 (TC) | 恒创云 (HCX) |
|------|------------|-------------|
| 角色 | 数据核心 | 推理 + 门户 |
| 运行服务 | center_api, ChromaDB, Joplin Server | qa_api, web_app, Ollama |
| 对外端口 | 5003 (公网), 8009 (ChromaDB Docker) | 无公网端口 (均 127.0.0.1) |
| 定时任务 | sync (8h), cache-report (daily) | 无 |
| 数据库 | joplinai_center.db (10表), ChromaDB向量库 | 无本地数据库 |

---


## 2. 服务架构


```mermaid
flowchart LR
    subgraph ENTRY["入口层"]
        direction TB
        A1["joplin_web_app.py<br/>joplin_web_app:app<br/>gunicorn (2 workers)"]
        A2["joplin_qa_api.py<br/>joplin_qa_api:app<br/>gunicorn (2 workers)"]
        A3["joplinai.py<br/>手动 CLI 执行"]
    end

    subgraph SRC["src/ 核心模块"]
        direction TB
        B1["web_app/ 包<br/>create_app() 工厂"]
        B2["config_manager.py"]
        B3["user_manager.py"]
        B4["queryanswer.py"]
        B5["report_writer.py"]
        B6["cli.py"]
        B7["qa_system.py"]
        B8["prompt_manager.py"]
        B9["qa_config.py"]
    end

    subgraph AIMOD["aimod/ AI 模块"]
        direction TB
        C1["center_api/ 包<br/>create_app() 工厂"]
        C2["center_client/ 包<br/>(5 个客户端)"]
        C3["embedding_generator.py"]
        C4["chunk_optimizer.py"]
        C5["text_splitter.py"]
        C6["vector_db_manager.py"]
        C7["deepseek_enhancer.py"]
        C8["run_tracker.py"]
        C9["cache_manager.py"]
    end

    subgraph INFRA["基础设施"]
        direction TB
        D1[("ChromaDB")]
        D2[("joplinai_center.db")]
        D3["Ollama"]
        D4["Joplin Server"]
    end

    A1 --> B1
    A2 --> B4
    A3 --> C3
    A3 --> C7
    A3 --> C8

    B1 --> B2
    B1 --> B3
    B4 --> B7
    B4 --> B8
    B4 --> B9

    C2 --> C1
    C3 --> C4
    C3 --> C5
    C1 --> D2
    C6 --> D1
```

**Gunicorn 入口速查：**

| 服务 | systemd 命令 | 模式 |
|------|-------------|------|
| center_api | `"aimod.center_api:create_app()"` | app factory |
| web_app | `joplin_web_app:app` | 模块级实例 |
| qa_api | `joplin_qa_api:app` | 模块级实例 |

---


## 3. 数据流全景


```mermaid
flowchart LR
    subgraph FLOW1["向量化流水线"]
        direction LR
        V1["joplinai.py"] --> V2["text_splitter.py<br/>文本切分"]
        V2 --> V3["chunk_optimizer.py<br/>自适应分块"]
        V3 --> V4["embedding_generator.py<br/>生成嵌入向量"]
        V4 --> V5["center_client/<br/>查 DeepSeek 缓存"]
        V5 --> V6["deepseek_enhancer.py<br/>生成摘要/标签"]
        V6 --> V7["vector_db_manager.py<br/>写入 ChromaDB (TC)"]
        V7 --> V8["run_tracker.py<br/>记录运行历史"]
        V8 --> V9["center_api<br/>(/history/*)"]
    end

    subgraph FLOW2["Q&A 链路"]
        direction LR
        Q1["用户提问"] --> Q2["web_app<br/>POST /qa/ask"]
        Q2 --> Q3["qa_api<br/>向量检索 + LLM"]
        Q3 --> Q4["ChromaDB (TC)<br/>语义搜索 top-k"]
        Q4 --> Q5["Ollama<br/>LLM 生成回答"]
        Q5 --> Q6["center_api<br/>记录 QA 历史"]
        Q6 --> Q7["返回用户"]
    end

    subgraph FLOW3["管理链路"]
        direction LR
        M1["管理员"] --> M2["web_app<br/>管理面板"]
        M2 --> M3["center_api<br/>用户/会话/审计"]
        M3 --> M4[("joplinai_center.db")]
    end
```

**协议总览：**

| 调用方 | 被调用方 | 协议 | 认证 |
|--------|---------|------|------|
| web_app → qa_api | HCX 本地 | HTTP | X-API-Key (云端配置) |
| web_app → center_api | HCX → TC | HTTP | X-API-Key |
| qa_api → center_api | HCX → TC | HTTP | X-API-Key |
| joplinai.py → center_api | HCX → TC | HTTP | X-API-Key |
| joplinai.py → ChromaDB | HCX → TC | chromadb.HttpClient (HTTP) | 无 |
| qa_api → ChromaDB | HCX → TC | chromadb.HttpClient (HTTP) | 无 |
| joplinai.py → DeepSeek | 外部 API | HTTPS | API Key (配置) |

---


## 4. 核心流程序列图


### 4a. 向量化流程

```mermaid
sequenceDiagram
    participant CLI as "joplinai.py (CLI)"
    participant JP as "Joplin API"
    participant TS as "TextSplitter"
    participant CO as "ChunkOptimizer"
    participant EG as "EmbeddingGenerator"
    participant DS as "DeepSeekEnhancer"
    participant CC as "DeepSeekCacheClient"
    participant CA as "center_api (TC)"
    participant VDB as "ChromaDB (TC)"
    participant RT as "RunTracker"

    CLI->>JP: "获取笔记列表 (imp_nbs)"
    JP-->>CLI: "笔记 JSON"
    CLI->>CLI: "parse_args() 过滤"
    CLI->>CC: "查询 DeepSeek 缓存"
    CC->>CA: "HTTP GET /cache/deepseek/query"
    CA-->>CC: "缓存命中/未命中"
    CLI->>TS: "文本切分"
    TS->>CO: "自适应分块探测"
    CO-->>TS: "最优块大小"
    CLI->>EG: "生成嵌入向量"
    EG->>VDB: "批量写入向量"
    CLI->>DS: "生成摘要/标签"
    DS-->>CLI: "摘要/标签 JSON"
    CLI->>CC: "写入 DeepSeek 缓存"
    CC->>CA: "HTTP POST /cache/deepseek/upsert"
    CLI->>RT: "记录运行历史"
    RT->>CA: "HTTP POST /history/record"
    CLI-->>CLI: "输出报告"
```

### 4b. Q&A 问答流程

```mermaid
sequenceDiagram
    participant USER as "用户浏览器"
    participant WEB as "web_app (HCX)"
    participant QA as "qa_api (HCX)"
    participant VDB as "ChromaDB (TC)"
    participant OLLAMA as "Ollama"
    participant CA as "center_api (TC)"

    USER->>WEB: "POST /qa/ask (问题文本)"
    WEB->>WEB: "session 验证"

    WEB->>QA: "POST /qa (X-API-Key 认证)"
    QA->>QA: "构建查询向量"
    QA->>VDB: "语义搜索 (top-k chunks)"
    VDB-->>QA: "相关文本块 + 元数据"

    QA->>QA: "组装 prompt"
    QA->>OLLAMA: "生成回答 (流式/非流式)"
    OLLAMA-->>QA: "LLM 回答"

    QA->>CA: "POST /qa/record (记录问答)"
    CA-->>QA: "OK"

    QA-->>WEB: "回答 + 引用来源"
    WEB-->>USER: "渲染回答页面"
```

### 4c. Remote-First 决策流程

```mermaid
flowchart TD
    START["客户端发起请求"] --> CHECK_URL["cloud 配置中<br/>joplinai_center_url<br/>有值?"]
    CHECK_URL -->|"有值"| REMOTE["调用远程 center_api<br/>HTTP + X-API-Key"]
    CHECK_URL -->|"无值"| IS_TC["本机是生产主机?<br/>(TC 不配 URL)"]
    IS_TC -->|"是"| LOCALHOST["调用 localhost:5003<br/>本机 center_api"]
    IS_TC -->|"否"| FALLBACK["使用本地 SQLite/JSON<br/>回退存储"]

    REMOTE --> RESULT_OK{"请求成功?"}
    RESULT_OK -->|"是"| RETURN["返回结果"]
    RESULT_OK -->|"否"| RETRY{"可重试错误?"}
    RETRY -->|"是"| REMOTE
    RETRY -->|"否"| USE_FALLBACK["降级到本地回退"]
    USE_FALLBACK --> RETURN
    LOCALHOST --> RETURN
    FALLBACK --> RETURN
```

---


## 5. 数据库结构


### 5a. SQLite 中心数据库

```mermaid
erDiagram
    users {
        int id PK
        text username UK
        text password_hash
        text display_name
        text role "admin/team_leader/team_member"
        text allowed_notebooks "JSON array"
        text created_at
        text updated_at
    }

    sessions {
        int id PK
        text session_id UK
        text username FK
        text created_at
        text expires_at
        text ip_address
        text user_agent
    }

    audit_log {
        int id PK
        text username
        text action
        text target
        text details "JSON"
        text timestamp
    }

    qa_history {
        int id PK
        text username FK
        text question
        text answer
        text session_id
        text metadata "JSON"
        text created_at
    }

    chat_sessions {
        int id PK
        text session_id UK
        text username FK
        text title
        text notebook_id
        text created_at
        text updated_at
    }

    deepseek_cache {
        int id PK
        text model
        text operation
        text input_hash
        text output "JSON"
        text created_at
        text last_accessed
    }

    probe_cache {
        int id PK
        text model
        int chunk_size
        int safe_length
        int hit_count
        text created_at
        text last_accessed
    }

    notebook_history {
        int id PK
        text notebook_id
        text notebook_title
        int note_count
        int chunk_count
        text timestamp
    }

    global_run_history {
        int id PK
        text run_id
        text started_at
        text finished_at
        text status
        text summary "JSON"
    }

    note_process_state {
        int id PK
        text note_id UK
        text notebook_id
        text content_hash
        text process_status
        text error_message
        int retry_count
        text last_processed
        text next_retry
    }

    users ||--o{ sessions : "username"
    users ||--o{ qa_history : "username"
    users ||--o{ chat_sessions : "username"
    users ||--o{ audit_log : "username"
```

### 5b. ChromaDB 向量集合

部署于腾讯云（Docker `chromadb/chroma`，端口映射 8009→8000），HCX 通过 `chromadb.HttpClient` 远程连接。

| 集合名称 | 用途 | 元数据字段 |
|---------|------|-----------|
| `joplin_chunks` | 笔记文本块的向量索引 | `note_id`, `notebook_id`, `chunk_index`, `content_hash`, `author`, `source_url` |

---


## 6. 配置链路


```mermaid
flowchart TD
    CLOUD["Joplin 云笔记<br/>(INI 格式配置)"]
    LOCAL["data/joplinai.ini<br/>(本地覆盖)"]

    CLOUD --> CM["ConfigManager 单例<br/>src/config_manager.py"]
    LOCAL --> CM

    CM --> HOT["热更新检查<br/>(每5分钟)"]
    HOT --> CM

    CM --> GET_ALL["get_all() /<br/>get_config_snapshot()"]
    GET_ALL --> VALUE1["joplin_token"]
    GET_ALL --> VALUE2["ollama_model"]
    GET_ALL --> VALUE3["embedding_model"]
    GET_ALL --> VALUE4["joplinai_center_url"]
    GET_ALL --> VALUE5["joplinai_center_api_key"]
    GET_ALL --> VALUE6["qa_api_key"]
    GET_ALL --> VALUE7["imp_nbs / imp_note_ids"]
    GET_ALL --> VALUE8["probe_cache_limit"]

    VALUE1 --> JP["jpfuncs.getapi()"]
    VALUE2 --> OLLAMA["Ollama 调用"]
    VALUE3 --> EG["EmbeddingGenerator"]
    VALUE4 --> CC["CenterClient 各子客户端"]
    VALUE5 --> CC
    VALUE6 --> QA["qa_api 认证"]
    VALUE7 --> CLI["joplinai.py 过滤"]
    VALUE8 --> CO["ChunkOptimizer"]
```

**配置优先级：** 命令行参数 > 本地 `data/joplinai.ini` > 云端 Joplin 笔记 INI > 代码硬编码默认值

---


## 7. 部署流程


```mermaid
flowchart TD
    START(["deploy/deploy.sh {hcx|tc}"])
    START --> DRY{"--dry-run?"}
    DRY -->|"是"| DRY_RUN["仅打印操作<br/>不实际执行"]
    DRY -->|"否"| CHECK["git status 检查<br/>工作树干净?"]

    CHECK -->|"有未提交更改"| ASK["提示确认 (y/N)"]
    ASK -->|"N"| ABORT(["退出"])
    ASK -->|"y"| TARGET{"目标?"}

    CHECK -->|"干净"| TARGET

    TARGET -->|"hcx"| HCX["恒创云部署"]
    TARGET -->|"tc"| TC["腾讯云部署"]

    subgraph HCX_STEPS["HCX 部署步骤"]
        direction TB
        H1["systemctl restart joplin-qa-api"]
        H2["systemctl restart joplin-web-app"]
        H1 --> H2
    end

    subgraph TC_STEPS["TC 部署步骤 (三级同步)"]
        direction TB
        T0["git push origin main<br/>(推送 HCX 提交到 GitHub)"]
        T1["TC: git pull 直连"]
        T2["失败? → 开 clash 代理<br/>TC: git pull 代理"]
        T3["也失败? → rsync 兜底"]
        T4["ssh tc systemctl restart<br/>joplinai-center-api"]
        T0 --> T1
        T1 -->|"成功"| T4
        T1 -->|"失败"| T2
        T2 -->|"成功"| T4
        T2 -->|"失败"| T3
        T3 --> T4
    end

    HCX --> HCX_STEPS
    TC --> TC_STEPS

    HCX_STEPS --> DONE(["部署完成"])
    TC_STEPS --> DONE
```

**TC 同步三级策略：**

1. git pull 直连 — 先试，大多数情况能通
2. git pull + clash 代理 — 直连失败时 `proxy_on` 后重试
3. rsync — 前两步都失败才用，跳过 func/ data/ log/

**部署前检查清单：**

- [ ] 所有更改已提交
- [ ] `gunicorn` 入口正确（module:app 或 "module:create_app()"）
- [ ] HCX 本地提交已 push 到 GitHub（deploy.sh 自动执行）
- [ ] systemd service 文件已同步到 `/etc/systemd/system/`

---


## 8. 工程规范速查


### 8a. Jupytext 工作流

```
编辑 .py 源文件 → git add → git commit → pre-commit hook
                                              ├── flake8 检查
                                              └── jupytext --sync (自动生成 .ipynb)
```

| 规则 | 说明 |
|------|------|
| 源文件 | `.py` (percent 格式)，提交到 git |
| 生成文件 | `.ipynb` (自动生成，不入库，`.gitignore` 已屏蔽) |
| Cell 标记 | `# %%` 代码 cell，`# %% [markdown]` 文档 cell |
| 关键陷阱 | markdown cell 后的代码前必须有显式 `# %%`，否则代码会被注释 |
| 手动同步 | `jupytext --sync <file>.py` |
| 恢复 ipynb | `jupytext --to notebook <file>.py` |

### 8b. 代码约定

| 约定 | 适用范围 | 说明 |
|------|---------|------|
| `__all__` | 所有公有模块 | 明确导出列表 |
| `__repr__` | 5 个数据类 | CacheResult, VectorDBManager, EmbeddingGenerator, RunTracker, QASystem |
| 类型标注 | 所有公开函数 | `-> None`, `-> str`, `-> Optional[str]` 等 |
| 日志 | 所有模块 | `from aimod import get_logger; logger = get_logger(__name__)` |
| 导入 | 全项目 | 统一包前缀 `src.` / `aimod.` / `func.` |
| pathmagic | 任意入口 | `with pathmagic.Context():` 包裹项目导入 |


### 8c. 代码质量命令

```bash
# Lint
flake8 . --max-line-length=100 --ignore=E402,W503,E203,E501 --exclude=func/

# 测试
python -m pytest tests/ -v --tb=short

# Pre-commit (手动触发)
pre-commit run --all-files
```

---

## 9. 维护操作速查

### 9a. 服务管理

```bash
# === 恒创云 (HCX) ===
sudo systemctl status joplin-qa-api       # 查看 QA API 状态
sudo systemctl status joplin-web-app      # 查看 Web 门户状态
sudo systemctl restart joplin-qa-api      # 重启 QA API
sudo systemctl restart joplin-web-app     # 重启 Web 门户
sudo journalctl -u joplin-qa-api -f       # QA API 实时日志
sudo journalctl -u joplin-web-app -f      # Web 门户实时日志

# === 腾讯云 (TC) ===
ssh tc "sudo systemctl status joplinai-center-api"
ssh tc "sudo systemctl restart joplinai-center-api"
ssh tc "sudo journalctl -u joplinai-center-api -f"

# ChromaDB (TC)
ssh tc "docker ps | grep chroma"
ssh tc "docker restart chroma-server"

# 定时器
ssh tc "sudo systemctl status joplinai-sync.timer"
ssh tc "sudo systemctl status joplinai-cache-report.timer"
```

### 9b. 健康检查

```bash
# center_api
curl http://127.0.0.1:5003/health          # 本地
curl http://122.51.102.233:5003/health     # 外部 (TC 公网)

# qa_api
curl -H "X-API-Key: <key>" http://127.0.0.1:5000/health

# web_app
curl -I http://127.0.0.1:5001/             # 应返回 302 → /login

# ChromaDB (TC)
ssh tc "curl -s http://127.0.0.1:8000/api/v1/heartbeat"
```

### 9c. 日志文件

| 服务 | 访问日志 | 错误日志 |
|------|---------|---------|
| center_api | `/var/log/joplin-qa/cache_access.log` | `/var/log/joplin-qa/cache_error.log` |
| qa_api | `/var/log/joplin-qa/qa_access.log` | `/var/log/joplin-qa/qa_error.log` |
| web_app | `/var/log/joplin-qa/web_access.log` | `/var/log/joplin-qa/web_error.log` |

### 9d. 常见排查路径

| 症状 | 检查步骤 |
|------|---------|
| Web 页面 502/500 | `journalctl -u joplin-web-app -f` → 检查 gunicorn error log |
| Q&A 返回空 | 检查 TC ChromaDB 是否运行 → 检查 ChromaDB 是否有数据 → 检查 Ollama 是否运行 |
| center_api 无响应 | `ssh tc` → `systemctl status joplinai-center-api` → 检查数据库 |
| 定时同步失败 | `ssh tc` → `journalctl -u joplinai-sync.service` → 检查 Joplin API 连通性 |
| 配置不生效 | 检查云端 Joplin 笔记 INI → 等 5 分钟热更新或重启服务 |
| jupytext 代码被注释 | 检查 markdown cell 后是否有显式 `# %%` → `jupytext --sync` 修复 |

### 9e. 手动操作

```bash
# 手动触发向量同步
ssh tc "sudo systemctl start joplinai-sync.service"

# 手动生成缓存报告
ssh tc "sudo systemctl start joplinai-cache-report.service"

# 手动执行报告 (HCX)
cd /data/codebase/joplinai && python src/report_writer.py --type all --output joplin

# 重新同步 ipynb
jupytext --sync path/to/file.py

# rsync 到 TC (跳过 func/ data/ log/)
cd /data/codebase/joplinai && rsync -avz --delete \
  --exclude="func/" --exclude=".git/" --exclude="__pycache__/" \
  --exclude="*.pyc" --exclude="data/" --exclude="log/" \
  --exclude="*.ipynb" --exclude="node_modules/" \
  /data/codebase/joplinai/ tc:/home/baiyefeng/work/joplinai/
```

---

## 10. 文件清单

### 10a. 入口文件

| 文件 | 用途 | 部署位置 |
|------|------|---------|
| `joplin_web_app.py` | Web 门户入口 | HCX |
| `joplin_qa_api.py` | Q&A API 入口 | HCX |
| `joplinai.py` | 向量化 CLI | HCX (手动) |
| `aimod/center_api/__init__.py` | 数据中心入口 (app factory) | TC |
| `src/report_writer.py` | 报告生成 CLI | HCX (手动) |
| `src/cli.py` | 共享 CLI 参数解析 | HCX (被 joplinai.py 调用) |

### 10b. 关键模块依赖方向

```mermaid
flowchart TD
    subgraph LAYER0["基础设施"]
        F1["func/ (工具子模块)"]
        F2["pathmagic.py"]
    end

    subgraph LAYER1["数据与配置"]
        M1["src/config_manager.py"]
        M2["aimod/cache_manager.py"]
        M3["aimod/vector_db_manager.py"]
    end

    subgraph LAYER2["AI 核心"]
        M4["aimod/embedding_generator.py"]
        M5["aimod/deepseek_enhancer.py"]
        M6["aimod/run_tracker.py"]
    end

    subgraph LAYER3["客户端"]
        M7["aimod/center_client/"]
    end

    subgraph LAYER4["服务端"]
        M8["aimod/center_api/"]
        M9["src/web_app/"]
        MA["src/qa_system.py"]
    end

    subgraph LAYER5["入口"]
        E1["joplinai.py"]
        E2["joplin_qa_api.py"]
        E3["joplin_web_app.py"]
    end

    F1 --> F2
    LAYER1 --> LAYER0
    LAYER2 --> LAYER1
    LAYER3 --> LAYER4
    E1 --> LAYER2
    E1 --> LAYER3
    E2 --> MA
    E2 --> M4
    E2 --> M3
    E3 --> M9
    E3 --> M7
```

**依赖原则：** 上层依赖下层，同层可互调，禁止下层反向依赖上层。`func/` 为最底层，不依赖项目任何模块。

---

## 附录 A：Mermaid 渲染兼容性说明

本手册中所有 mermaid 图的节点标签均使用双引号包裹（`["标签"]`），避免 Joplin 客户端因特殊字符（`/`、`:`、`-`、`(`、`)`）渲染失败。如在其他渲染器中打开，去掉双引号即可获得更简洁的显示效果。

---

> 最后更新：2026-05-13 | joplinai main 分支
