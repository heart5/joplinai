# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     split_at_heading: true
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
    "embedding_provider": getinivaluefromcloud("joplinai", "embedding_provider") or "ollama",
    "ollama_host": getinivaluefromcloud("joplinai", "ollama_host") or "",
    "siliconflow_api_key": getinivaluefromcloud("joplinai", "siliconflow_api_key") or "",
    "siliconflow_embedding_model": getinivaluefromcloud("joplinai", "siliconflow_embedding_model") or "",
    "siliconflow_embedding_chunk_size": siliconflow_embedding_chunk_size
    if (siliconflow_embedding_chunk_size := getinivaluefromcloud("joplinai", "siliconflow_embedding_chunk_size"))
    else 1500,
    "siliconflow_embedding_dimension": siliconflow_embedding_dimension
    if (siliconflow_embedding_dimension := getinivaluefromcloud("joplinai", "siliconflow_embedding_dimension"))
    else 1024,
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
    else "deepseek-v4-flash",
    "max_context_chars": max_context_chars
    if (max_context_chars := getinivaluefromcloud("joplinai", "context_max_length"))
    else 4000,
    "max_output_tokens": max_output_tokens
    if (max_output_tokens := getinivaluefromcloud("joplinai", "max_output_tokens"))
    else 4096,
    "min_answer_length": 50,
}

# VectorDBManager 通过 normalize_collection_name() 自行派生集合名，无需此键

