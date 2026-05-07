# joplinai

通过 AI 深化 Joplin 笔记使用 — 向量化知识检索与多用户问答系统

## 概述

joplinai 将 Joplin 笔记向量化存储到 ChromaDB，利用本地 Ollama 模型实现语义搜索和 LLM 问答，并提供多用户 Web 界面和管理后台。

## 架构

三个独立 Flask 服务：

| 服务 | 文件 | 端口 | 职责 |
|------|------|------|------|
| Web 门户 | `joplin_web_app.py` | 127.0.0.1:5001 | 用户登录、问答 UI、管理面板 |
| Q&A API | `src/joplin_qa_api.py` | 动态（从配置读取） | 向量搜索 + LLM 问答的 HTTP API |
| 向量化 CLI | `src/joplinai.py` | — | 将 Joplin 笔记分块、生成嵌入、存入 ChromaDB |

数据流：

```
浏览器 → joplin_web_app.py (Flask session) → HTTP (API Key 认证) → src/joplin_qa_api.py → ChromaDB + Ollama
```

## 项目结构

```
src/              # Python 源码（与 notebooks/ 通过 jupytext.toml 配对）
├── config_manager.py
├── joplinai.py
├── joplin_qa_api.py
├── queryanswer.py
├── joplin_web_app.py
├── user_manager.py
└── pathmagic.py
notebooks/        # Jupyter notebooks（jupytext 配对文件）
aimod/            # AI 核心模块
func/             # 工具子模块（heart5/func）
static/           # 前端静态资源
templates/        # Jinja2 模板
data/             # 运行时数据（SQLite、ChromaDB、配置）
log/              # 日志文件
```

## 快速开始

所有命令在项目根目录下执行：

```bash
# 1. 向量化笔记（首次运行需先执行，填充 ChromaDB）
python src/joplinai.py

# 2. 启动 Q&A API 服务（在配置的端口上启动）
python src/joplin_qa_api.py

# 3. 启动 Web 门户
python joplin_web_app.py
```

## 关键模块

- `aimod/` — AI 核心：文本分块与嵌入（`embedding_generator.py`）、ChromaDB 操作（`vector_db_manager.py`）、AI 缓存（`cache_manager.py`）、DeepSeek 增强（`deepseek_enhancer.py`）、向量化报告（`aitaskreporter.py`）
- `func/` — 工具子模块（git submodule `heart5/func`）：Joplin API 封装、配置读取、日志、路径检测
- `src/user_manager.py` — SQLite 用户系统，三角色权限（admin / team_leader / team_member），笔记本级访问控制
- `src/config_manager.py` — 云端配置热更新单例，定时从 Joplin 笔记拉取配置
- `jupytext.toml` — 定义 `src/` 与 `notebooks/` 之间的文件配对映射

## 技术栈

- **后端**: Flask, requests, Jinja2
- **向量库**: ChromaDB
- **LLM**: Ollama（本地）+ 可选 DeepSeek API
- **配置**: Joplin 云端笔记（INI 格式），支持热更新
- **编辑模式**: Jupytext 配对笔记本（`jupytext.toml` 映射 `src/` ↔ `notebooks/`，percent 格式）

## 许可证

本项目基于 [MIT License](LICENSE) 许可。
