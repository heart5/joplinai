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

# joplinai

通过 AI 深化 Joplin 笔记使用 — 向量化知识检索与多用户问答系统

## 概述

joplinai 将 Joplin 笔记向量化存储到 ChromaDB，利用本地 Ollama 模型实现语义搜索和 LLM 问答，并提供多用户 Web 界面和管理后台。

## 架构

两个服务器，分布式服务：

| 服务 | 入口 | 端口 | 主机 | 职责 |
|------|------|------|------|------|
| Web 门户 | `joplin_web_app.py` | 127.0.0.1:5001 | 恒创云 | 用户登录、问答 UI、管理面板 |
| Q&A API | `joplin_qa_api.py` | 127.0.0.1:5000 | 恒创云 | 向量搜索 + LLM 问答的 HTTP API |
| 数据中心 | `aimod/center_api/` | 0.0.0.0:5003 | 腾讯云 | 增强缓存 / 探测缓存 / 历史DB / 笔记状态 / 用户管理 |
| 向量化引擎 | `joplinai.py` | — | 腾讯云 | 分块编排 / AI增强 / 写 ChromaDB / 报告。每8小时定时触发 |
| 嵌入推理 | Ollama | 11434 | 恒创云 | 文本嵌入向量生成，TC 经公网 IP 远程调用 |

部署分工：
- **腾讯云**：Joplin Server、ChromaDB 向量库、数据中心 API (joplinai_center.db)、向量化引擎 (joplinai-sync.timer)、用户管理
- **恒创云**：Ollama 推理 (embedding + LLM)、Q&A API、Web 门户

数据流：

```
浏览器 → web_app (Flask session) → HTTP (X-API-Key) → qa_api → ChromaDB (TC) + Ollama (HCX)
                │
                └──── HTTP (X-API-Key) ────→ center_api (TC)
                                             用户/会话/审计/QA历史/缓存/状态
```

## 项目结构

```
src/                    # 核心模块
├── config_manager.py       # 云端配置热更新
├── queryanswer.py          # QA 入口
├── user_manager.py         # 多用户管理
├── pathmagic.py            # 项目路径上下文
├── report_writer.py        # 统一报告
├── qa_system.py            # Q&A 核心（拆分自 queryanswer）
├── prompt_manager.py       # Prompt 管理（拆分自 queryanswer）
├── qa_config.py            # QA 配置常量（拆分自 queryanswer）
├── cli.py                  # CLI 参数解析
└── web_app/                # Web 门户包
    ├── __init__.py         # create_app() 工厂
    ├── routes.py           # 页面路由
    ├── auth.py             # 认证
    ├── admin.py            # 管理面板
    └── api.py              # 内部 API
joplinai.py             # 向量化 CLI 入口
joplin_qa_api.py        # Q&A API 服务入口
joplin_web_app.py       # Web 门户入口
aimod/                  # AI 核心包
├── center_api/             # 数据中心 API（5 个路由蓝图）
├── center_client/          # 数据中心客户端（5 个独立类）
├── embedding_generator.py  # 嵌入生成
├── chunk_optimizer.py      # 自适应分块（拆分自 embedding_generator）
├── text_splitter.py        # 文本切分（拆分自 embedding_generator）
├── vector_db_manager.py    # ChromaDB CRUD
├── cache_manager.py        # 本地缓存回退
├── note_enhancer.py         # AI增强
├── run_tracker.py          # 运行追踪
└── runner_config.py        # 运行器配置
func/                   # 工具子模块 (git submodule heart5/func)
deploy/                 # systemd 服务文件 + deploy.sh
tests/                  # 测试
static/                 # 前端静态资源
templates/              # Jinja2 模板
data/                   # 运行时数据 (gitignored)
log/                    # 日志 (gitignored)
.github/workflows/      # CI (flake8 + pytest)
```

## 快速开始

所有命令在项目根目录下执行：

```bash
# 1. 向量化笔记（首次运行需先执行，填充 ChromaDB）
python joplinai.py

# 2. 启动 Q&A API 服务
python joplin_qa_api.py

# 3. 启动 Web 门户
python joplin_web_app.py
```

部署：

```bash
./deploy/deploy.sh hcx           # 恒创云：重启 QA_API + Web_App
./deploy/deploy.sh tc            # 腾讯云：git push → git pull(直连→代理→rsync兜底) → 重启 center_api
./deploy/deploy.sh hcx --dry-run # 仅预览
```

## 关键模块

- `aimod/` — AI 核心：文本分块与嵌入（`embedding_generator.py`、`chunk_optimizer.py`、`text_splitter.py`）、ChromaDB 操作（`vector_db_manager.py`）、AI 缓存（`cache_manager.py`）、AI增强（`note_enhancer.py`）、运行追踪（`run_tracker.py`）
- `aimod/center_api/` — 数据中心 Flask 包（`cache_routes`、`history_routes`、`state_routes`、`user_routes`）
- `aimod/center_client/` — 数据中心客户端（`CacheClient`、`ProbeCacheClient`、`HistoryClient`、`ProcessStateClient`、`UserManagerClient`），全部 remote-first + local fallback
- `src/` — 核心模块：用户管理（`user_manager.py`）、配置管理（`config_manager.py`）、统一报告（`report_writer.py`）、Q&A 系统（`qa_system.py`）
- `src/web_app/` — Web 门户包（`create_app()` 工厂 + `routes`、`auth`、`admin`、`api`）
- `func/` — 工具子模块（git submodule `heart5/func`）：Joplin API 封装、配置读取、日志、路径检测
- `src/user_manager.py` — SQLite 用户系统，三角色权限（admin / team_leader / team_member），笔记本级访问控制
- `src/config_manager.py` — 云端配置热更新单例，`get_all()` 获取完整快照
- `aimod/__init__.py` — `get_logger(name)` 统一日志工厂

## 工程规范

- **Jupytext**: `.py` → `.ipynb` 自动配对（percent 格式），ipynb 不入库
- **`__all__`**: 所有公开模块明确导出列表
- **`__repr__`**: 关键数据类（CacheResult、VectorDBManager、EmbeddingGenerator、RunTracker、QASystem）
- **类型标注**: 所有公开函数有返回类型标注
- **Lint**: flake8 (max-line-length=100, ignore=E402,W503,E203,E501), pre-commit + CI
- **Test**: pytest, CI 自动运行
- **Git**: pre-commit hook (flake8 + jupytext sync)

## 技术栈

- **后端**: Flask, requests, Jinja2
- **向量库**: ChromaDB
- **LLM**: Ollama（本地 embedding + LLM）+ DeepSeek API（云端对话/增强）
- **配置**: Joplin 云端笔记（INI 格式），支持热更新
- **部署**: systemd + rsync，`deploy/deploy.sh` 一键部署
- **CI/CD**: GitHub Actions (lint + test)

## 最近更新

```
ff9a748 fix: 修复 jupytext 导致 get_call_stats/reset_call_stats 被注释掉的问题
05a64a9 refactor: DeepSeek增强功能命名 → provider-agnostic AI增强命名 (文件/类/函数/表/路由/元数据)
23daf5e refactor: provider-agnostic增强架构 — 统一入口+模型级缓存+配置重命名
4370bfc fix: JPEG检测bug修复 + vision结果多层过滤防止污染
224bef9 fix: vision 流程三处改进 — 清理图片语法/过滤无效缓存/alt_text回退
2fce603 feat: ollama_vision_describe 接入缓存层，避免重复 vision 调用
d2064e0 docs: CHANGELOG/CLAUDE 更新至 May 14 最新变更
ed31f8c fix: 将 Flask app 实例提升到模块级别以兼容 gunicorn
30e4e45 fix: 修复 jupytext hook 导致 cli.py 和 center_api 路由文件代码被注释
dda7c94 feat: Phase 5 CI/CD + 部署标准化
```

## 许可证

本项目基于 [MIT License](LICENSE) 许可。
