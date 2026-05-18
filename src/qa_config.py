# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # QA 配置

# %% [markdown]
# ## 导入库

# %%
import logging

import pathmagic

with pathmagic.Context():
    try:
        from func.first import getdirmain
        from func.jpfuncs import getinivaluefromcloud
    except ImportError as e:
        logging.basicConfig(level=logging.INFO)
        getinivaluefromcloud = lambda s, k: None
        getdirmain = lambda: None
        logging.error(f"导入项目模块失败: {e}")


# %%
__all__ = ["CONFIG"]

CONFIG = {
    "ollama_embedding_model": getinivaluefromcloud("joplinai", "ollama_embedding_model") or "dengcao/bge-large-zh-v1.5",
    "qa_ollama_chat_model": qa_ollama_chat_model
    if (qa_ollama_chat_model := getinivaluefromcloud("joplinai", "qa_ollama_chat_model"))
    else "qwen2.5:1.5b",
    "db_path": getdirmain() / "data" / "joplin_vector_db",
    "max_retrieved_notes": 10,
    "max_retrieved_chunks": max_retrieved_chunks
    if (
        max_retrieved_chunks := getinivaluefromcloud("joplinai", "max_retrieved_chunks")
    )
    else 20,
    "similarity_threshold": 0.5,
    "cloud_api_key": cloud_api_key
    if (cloud_api_key := getinivaluefromcloud("joplinai", "cloud_api_key"))
    else getinivaluefromcloud("joplinai", "deepseek_token"),
    "cloud_api_url": cloud_api_url
    if (cloud_api_url := getinivaluefromcloud("joplinai", "cloud_api_url"))
    else "https://api.deepseek.com/v1/chat/completions",
    "cloud_model": cloud_model
    if (cloud_model := getinivaluefromcloud("joplinai", "cloud_model"))
    else "deepseek-chat",
    "context_max_length": context_max_length
    if (context_max_length := getinivaluefromcloud("joplinai", "context_max_length"))
    else 4000,
    "min_answer_length": 50,
}

# %%
model_name = (
    CONFIG.get("ollama_embedding_model").replace(":", "_").replace("/", "_").replace("-", "_")
)
CONFIG["collection_name"] = f"joplin_{model_name}"
