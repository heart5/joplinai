# joplinai

通过 AI 深化 Joplin 笔记使用 — 向量化知识检索与多用户问答系统

## 概述

joplinai 将 Joplin 笔记向量化存储到 ChromaDB，利用本地 Ollama 模型实现语义搜索和 LLM 问答，并提供多用户 Web 界面和管理后台。

## 架构

两个服务器，四个独立 Flask 服务：

| 服务 | 文件 | 端口 | 主机 | 职责 |
|------|------|------|------|------|
| Web 门户 | `joplin_web_app.py` | 127.0.0.1:5001 | 恒创云 | 用户登录、问答 UI、管理面板 |
| Q&A API | `joplin_qa_api.py` | 127.0.0.1:5000 | 恒创云 | 向量搜索 + LLM 问答的 HTTP API |
| 数据中心 | `joplinai_center_api.py` | 0.0.0.0:5003 | 腾讯云 | DeepSeek 缓存、探测缓存、历史DB、笔记状态、用户管理 |
| 向量化 CLI | `joplinai.py` | — | 恒创云 | 将 Joplin 笔记分块、生成嵌入、存入 ChromaDB |

部署分工：
- **腾讯云**：Joplin Server、ChromaDB 向量库、数据中心 API（统一 SQLite 数据库，含缓存/历史/状态/用户）
- **恒创云**：Ollama 大模型推理、Q&A API、Web 门户

数据流：

```
浏览器 → joplin_web_app.py (Flask session) → HTTP (API Key 认证) → joplin_qa_api.py → ChromaDB + Ollama
                    │                                                                                  
                    └──────── HTTP (API Key 认证) ────────→ joplinai_center_api.py (TC)               
                             用户管理 / 笔记状态 / 缓存                                           
```

## 项目结构

```
src/              # Python 源码（jupytext paired with .ipynb in same dir）
├── config_manager.py    + config_manager.ipynb
├── queryanswer.py       + queryanswer.ipynb
├── user_manager.py      + user_manager.ipynb
└── pathmagic.py         + pathmagic.ipynb
joplinai.py           # 向量化 CLI         + joplinai.ipynb
joplin_qa_api.py      # Q&A API 服务      + joplin_qa_api.ipynb
joplin_web_app.py     # Web 门户          + joplin_web_app.ipynb
aimod/                # AI 核心模块
func/                 # 工具子模块（heart5/func）
static/               # 前端静态资源
templates/            # Jinja2 模板
data/                 # 运行时数据（SQLite、ChromaDB、配置）
log/                  # 日志文件
```

## 快速开始

所有命令在项目根目录下执行：

```bash
# 1. 向量化笔记（首次运行需先执行，填充 ChromaDB）
python joplinai.py

# 2. 启动 Q&A API 服务（在配置的端口上启动）
python joplin_qa_api.py

# 3. 启动 Web 门户
python joplin_web_app.py
```

## 关键模块

- `aimod/` — AI 核心：文本分块与嵌入（`embedding_generator.py`）、ChromaDB 操作（`vector_db_manager.py`）、AI 缓存（`cache_manager.py`）、DeepSeek 增强（`deepseek_enhancer.py`）、运行追踪（`run_tracker.py`）
- `src/` — 核心模块：用户管理（`user_manager.py`）、配置管理（`config_manager.py`）、统一报告（`report_writer.py`）
- `func/` — 工具子模块（git submodule `heart5/func`）：Joplin API 封装、配置读取、日志、路径检测
- `src/user_manager.py` — SQLite 用户系统，三角色权限（admin / team_leader / team_member），笔记本级访问控制
- `src/config_manager.py` — 云端配置热更新单例，定时从 Joplin 笔记拉取配置
- `jupytext.toml` — 已弃用（jupytext 不支持跨目录映射），py 与 ipynb 现已同目录配对

## 技术栈

- **后端**: Flask, requests, Jinja2
- **向量库**: ChromaDB
- **LLM**: Ollama（本地）+ 可选 DeepSeek API
- **配置**: Joplin 云端笔记（INI 格式），支持热更新
- **编辑模式**: Jupytext 配对笔记本（py 与 ipynb 同目录配对，percent 格式，`jupytext --sync <file>.py`）

## 最近更新

```
48f72f3  refactor: user_manager.py 清理死代码、修复角色校验、统一命名与连接管理
d940c5c  fix: _migrate_db.py PRAGMA table_info 列索引修复
1aa6917  feat: 笔记状态与用户管理集中化到 center_api
03561b7  fix: force_update 模式下仅实际有块变更的笔记才计入 updated_count
98a24f6  docs: 新增 CHANGELOG.md 记录数据中心重构及部署变更
540d909  fix: 模块级调用 _init_db() 确保 gunicorn 导入时建表
```

## 许可证

本项目基于 [MIT License](LICENSE) 许可。
