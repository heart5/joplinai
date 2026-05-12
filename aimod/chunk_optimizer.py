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
# # AdaptiveChunkOptimizer
# 自适应分块优化器 — 通过实际调用嵌入接口，探测文本在特定模型下的最大安全处理长度。

# %%
import hashlib
import logging
from typing import List, Optional, Tuple

# %%
import pathmagic

with pathmagic.Context():
    try:
        from func.logme import log
    except ImportError as e:
        logging.basicConfig(level=logging.INFO)
        log = logging.getLogger(__name__)
        log.error(f"导入项目模块失败: {e}")


# %%
class AdaptiveChunkOptimizer:
    """
    自适应分块优化器。
    原理：通过实际调用嵌入接口，探测给定文本在特定模型下的最大安全处理长度。
    利用本地模型零成本的优势，动态调整分块大小，避免过碎分块。
    """

    def __init__(self, embedding_generator, enabled=False, probe_client=None):
        self.embedding_generator = embedding_generator
        self.enabled = enabled
        self.probe_client = probe_client  # ProbeCacheClient 或 None

    @staticmethod
    def _is_length_error(exception: Exception) -> bool:
        """判断异常是否为模型token上下文超限"""
        msg = str(exception).lower()
        return any(
            kw in msg
            for kw in ["context length", "input length", "too long", "exceed", "500"]
        )

    def _probe_at(self, text: str, length: int) -> Tuple[bool, Optional[List[float]]]:
        """探测指定长度是否安全。返回 (成功?, 嵌入向量或None)。"""
        try:
            emb = self.embedding_generator.get_ollama_embedding(text[:length])
            return True, emb
        except Exception as e:
            if self._is_length_error(e):
                return False, None
            raise  # 非长度错误向上抛

    def probe_max_safe_length(
        self, text: str, model_name: str, start_len: int = None
    ) -> int:
        """
        双向探测模型能安全处理的最大字符数。
        - start_len 通过 → 指数增长向上探索
        - start_len 失败 → 二分搜索向下探索
        支持 start_len 温启动。
        """
        if not self.enabled:
            return self.embedding_generator.chunk_size

        # 查探测缓存（远程集中式）
        if self.probe_client is not None:
            text_md5 = hashlib.md5(text.encode("utf-8")).hexdigest()
            cached = self.probe_client.get(text_md5)
            if cached is not None:
                log.debug(f"[自适应探测] 缓存命中: {text_md5[:8]}... → {cached}字符")
                return cached

        chunk_size = self.embedding_generator.chunk_size
        if start_len is None:
            # 温启动未提供：取 chunk_size 的 88%，接近上限减少探测次数
            start_len = int(chunk_size * 0.88)
        max_len = len(text)

        log.info(
            f"[自适应探测] 模型={model_name}, 文本长度={max_len}字符, "
            f"起始探测={start_len}字符"
        )

        current_safe_len = start_len

        # Phase 1: 测试起始点
        try:
            ok, _ = self._probe_at(text, min(start_len, max_len))
        except Exception as e:
            log.warning(f"  起始探测非长度错误，回退默认chunk_size={chunk_size}字符: {e}")
            current_safe_len = chunk_size
            ok = False  # 触发下方回退

        if ok:
            # 起始点通过 → 指数增长向上探索
            current_safe_len = min(start_len, max_len)
            test_len = int(current_safe_len * 1.1)
            while test_len <= max_len:
                try:
                    ok, _ = self._probe_at(text, test_len)
                except Exception as e:
                    log.warning(f"  探测非长度错误，保留current_safe_len={current_safe_len}字符: {e}")
                    break
                if ok:
                    current_safe_len = test_len
                    log.debug(f"  探测通过: {test_len}字符 未超模型token上下文限制")
                    test_len = int(test_len * 1.1)
                else:
                    log.debug(
                        f"  探测失败(token超限): {test_len}字符 → 超出模型token上下文限制"
                    )
                    break
        else:
            # 起始点失败 → 二分搜索向下探索
            lo = int(chunk_size * 0.5)  # 绝对下限
            hi = min(start_len, max_len)
            log.debug(f"  起始探测超限，二分搜索字符范围: [{lo}字符, {hi}字符]")
            while lo < hi:
                mid = (lo + hi) // 2
                try:
                    ok_mid, _ = self._probe_at(text, mid)
                except Exception:
                    break  # 非长度错误，中止搜索
                if ok_mid:
                    current_safe_len = mid
                    lo = mid + 1
                else:
                    hi = mid
            log.debug(f"  二分搜索完成: 安全上限={current_safe_len}字符")

        # 3. 确保结果在合理范围内
        final_safe_len = max(
            int(chunk_size * 0.5),
            min(current_safe_len, chunk_size * 2),
        )
        log.info(
            f"[自适应探测完成] 建议块大小={final_safe_len}字符 "
            f"(chunk_size={chunk_size}字符)"
        )

        # 写入探测缓存
        if self.probe_client is not None:
            text_md5 = hashlib.md5(text.encode("utf-8")).hexdigest()
            self.probe_client.set(
                text_md5=text_md5,
                safe_len=final_safe_len,
                snippet=text[:20],
                model_name=model_name,
                chunk_size=chunk_size,
            )

        return final_safe_len
