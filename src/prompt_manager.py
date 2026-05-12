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
# # PromptManager
# 集中管理从云端获取的系统提示词，杜绝硬编码。

# %%
import logging
import re
from typing import Dict, Optional

import pathmagic

with pathmagic.Context():
    try:
        from func.jpfuncs import getinivaluefromcloud
        from func.logme import log
    except ImportError as e:
        logging.basicConfig(level=logging.INFO)
        log = logging.getLogger(__name__)
        log.error(f"导入项目模块失败: {e}")


# %%
__all__ = ["PromptManager"]

class PromptManager:
    """集中管理从云端获取的系统提示词，杜绝硬编码。"""

    @staticmethod
    def get_sys_prompt_for_role(user_identity: Optional[Dict]) -> str:
        """
        根据用户身份，从云端获取对应的系统提示词。
        如果云端未配置，则返回一个极简的、安全的通用提示。
        """
        from func.jpfuncs import getinivaluefromcloud

        if not user_identity:
            base_prompt = getinivaluefromcloud("joplinai", "sys_prompt_base")
            if base_prompt:
                return base_prompt
            else:
                return "请根据提供的笔记内容回答问题。如果笔记中没有相关信息，请说明。"

        user_role = user_identity.get("role")
        user_display_name = user_identity.get("display_name", "")

        default_personal_author = (
            getinivaluefromcloud("joplinai", "default_personal_author") or "用户"
        )

        split_ptn = re.compile(r"[,，]")
        if colleague_str := getinivaluefromcloud("joplinai", "colleague"):
            colleagues = [title.strip() for title in split_ptn.split(colleague_str)]
        else:
            colleagues = []
        colleague_str_for_prompt = "，".join([f"“{person}”" for person in colleagues])

        prompt_key = ""
        if user_role == "admin":
            prompt_key = "sys_prompt"
        elif user_role == "colleague":
            prompt_key = "sys_colleague_prompt"
        else:
            prompt_key = "sys_prompt_base"

        prompt_template = getinivaluefromcloud("joplinai", prompt_key)

        if prompt_template:
            prompt = prompt_template.replace(
                "{default_personal_author}", default_personal_author
            )
            prompt = prompt.replace("{colleague_str}", colleague_str_for_prompt)
            prompt = prompt.replace("{user_display_name}", user_display_name)
            return prompt
        else:
            log.warning(f"云端未配置提示词键: {prompt_key}，将使用内置通用模板。")
            if user_role == "admin":
                return f"""你是我（{default_personal_author}）的笔记助手，基于笔记回答问题。笔记可能包含个人记录、团队共享信息或收藏文章。请根据笔记片段的【类型】和【作者】元数据，客观、准确地回答。如果笔记中没有相关信息，请说明。"""
            elif user_role == "colleague":
                return f"""你是{user_display_name}的笔记助手，基于共享笔记库回答问题。你只能访问作者为"团队_共同维护"或"同事_{user_display_name}"的笔记片段。请基于这些内容回答，如果无相关信息，请说明。"""
            else:
                return "请根据提供的笔记内容回答问题。如果笔记中没有相关信息，请说明。"
