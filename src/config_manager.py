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
# # 全局配置管理器

# %% [markdown]
# ## 导入库

# %%
# config_manager.py
import hashlib
import json
import threading
import time
from typing import Any, Dict, Optional

# %%
import pathmagic

with pathmagic.context():
    from func.jpfuncs import getinivaluefromcloud
    from func.logme import log


# %% [markdown]
# ## ConfigManager类

# %%
class ConfigManager:
    """
    云端配置热更新管理器（单例）。
    定期检查云端配置变化，并提供一致的配置快照。
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialize()
            return cls._instance

    def _initialize(self):
        """初始化管理器状态"""
        self.current_config_snapshot = {}
        self.config_fingerprint = None
        self.last_check_time = 0
        self.check_interval = 300  # 每5分钟检查一次（秒）
        self.lock = threading.Lock()

        # 定义需要监控的云端配置键列表（这是关键！）
        self.monitored_keys = [
            # 来自 joplinai 和 queryanswer 的核心配置
            "sys_prompt",
            "sys_colleague_prompt",
            "sys_prompt_base",
            "default_personal_author",
            "colleague",
            "similarity_threshold",
            "context_max_length",
            "max_retrieved_chunks",
            "enable_deepseek_summary",
            "enable_deepseek_tags",
            # 可以根据需要无限扩展...
        ]
        self._fetch_and_update()  # 初始加载
        log.info(
            f"ConfigManager 初始化完成，监控 {len(self.monitored_keys)} 个云端配置项。"
        )

    def _fetch_config_from_cloud(self) -> Dict[str, Any]:
        """
        从云端获取所有被监控配置项的当前值。
        这是与云端交互的唯一入口。
        """
        config_snapshot = {}
        for key in self.monitored_keys:
            value = getinivaluefromcloud("joplinai", key)
            # 处理特殊类型的值（如逗号分隔的字符串）
            if value is not None:
                config_snapshot[key] = value
            else:
                config_snapshot[key] = None  # 显式记录None，便于检测删除
        return config_snapshot

    def _compute_fingerprint(self, config: Dict) -> str:
        """计算配置字典的MD5指纹，用于快速比较"""
        # 序列化时排序，确保相同配置产生相同指纹
        config_str = json.dumps(config, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(config_str.encode("utf-8")).hexdigest()

    def _fetch_and_update(self) -> bool:
        """
        从云端拉取配置，如果发现变化则更新内部快照。
        返回 True 表示配置已更新。
        """
        with self.lock:
            fresh_config = self._fetch_config_from_cloud()
            fresh_fingerprint = self._compute_fingerprint(fresh_config)

            if fresh_fingerprint != self.config_fingerprint:
                old_fp = (
                    self.config_fingerprint[:8] if self.config_fingerprint else "None"
                )
                new_fp = fresh_fingerprint[:8]

                self.current_config_snapshot = fresh_config
                self.config_fingerprint = fresh_fingerprint

                log.warning(
                    f"🔁 云端配置已变更！指纹: {old_fp}... -> {new_fp}...\n"
                    f"变更摘要: {self._generate_change_summary(fresh_config, self.current_config_snapshot)}"
                )
                return True
            return False

    def _generate_change_summary(self, new_config, old_config):
        """生成配置变更的简明摘要，用于日志"""
        changes = []
        all_keys = set(new_config.keys()) | set(old_config.keys())
        for key in all_keys:
            old_val = old_config.get(key)
            new_val = new_config.get(key)
            if old_val != new_val:
                # 简略显示，避免日志过长
                changes.append(f"{key}: {repr(old_val)[:50]} -> {repr(new_val)[:50]}")
        return "; ".join(changes) if changes else "（无明显变更）"

    def get_config_snapshot(self) -> Dict[str, Any]:
        """
        获取当前配置的快照。
        如果距上次检查已超过间隔，会触发一次云端检查。
        """
        current_time = time.time()
        if current_time - self.last_check_time >= self.check_interval:
            self.last_check_time = current_time
            self._fetch_and_update()  # 惰性检查

        with self.lock:
            # 返回一个深拷贝，防止外部修改内部状态
            return self.current_config_snapshot.copy()

    def get_config_fingerprint(self) -> str:
        """获取当前配置的指纹"""
        with self.lock:
            return self.config_fingerprint

    def force_refresh(self) -> bool:
        """
        强制立即从云端刷新配置。
        返回 True 表示配置已更新。
        """
        self.last_check_time = 0  # 强制下次get_config_snapshot时检查
        return self._fetch_and_update()


# %% [markdown]
# ## 全局单例实例

# %%
# 创建全局单例实例
CONFIG_MANAGER = ConfigManager()
